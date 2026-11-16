"""Masked categorical helpers shared by every head (ported from v2 model.py)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

NEG = -1e9


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.float().masked_fill(~mask, NEG)


def safe_mask(mask: torch.Tensor) -> torch.Tensor:
    """Rows with nothing allowed allow everything (never an all -inf softmax)."""
    return mask | ~mask.any(-1, keepdim=True)


def log_prob_of(logits: torch.Tensor, mask: torch.Tensor, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Log-prob of `index` (clamped to >= 0), entropy, and whether the label had to be rescued.

    A label the mask excludes is let through ("rescued") instead of giving -inf:
    teacher labels are ground truth, a mask is only a hint about legality."""
    idx = index.clamp(min=0)
    hit = torch.zeros_like(mask).scatter_(-1, idx.unsqueeze(-1), True)
    rescued = ~(mask & hit).any(-1)
    m = safe_mask(mask | hit)
    logp = F.log_softmax(masked_logits(logits, m), dim=-1)
    chosen = logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    p = logp.exp()
    entropy = -(p * logp.masked_fill(~m, 0.0)).sum(-1)
    return chosen, entropy, rescued


def gumbel_sample(logits: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0,
                  generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked sample by Gumbel-max (NaN-proof: a masked logit plus noise never beats an open one).

    temperature 0 = argmax. Returns (index, log-prob under the tempered distribution)."""
    m = safe_mask(mask)
    t = max(float(temperature), 1e-6)
    scaled = masked_logits(logits, m) / t if temperature > 0 else masked_logits(logits, m)
    if temperature <= 0:
        idx = scaled.argmax(-1)
    else:
        u = torch.rand(scaled.shape, generator=generator, device=scaled.device if generator is None else generator.device)
        u = u.to(scaled.device).clamp_(1e-20, 1.0 - 1e-7)
        idx = (scaled - torch.log(-torch.log(u))).argmax(-1)
    logp = F.log_softmax(scaled, dim=-1).gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    return idx, logp


def topk_probs(logits: torch.Tensor, mask: torch.Tensor, k: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    p = F.softmax(masked_logits(logits, safe_mask(mask)), dim=-1)
    v, i = p.topk(min(k, p.shape[-1]), dim=-1)
    return i, v
