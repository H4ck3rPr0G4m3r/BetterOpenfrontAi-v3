"""A checkpoint playing any number of seats (M3 Task 8): batched core + sample,
one LSTM state per row."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ofai import checkpoints as ck
from ofai.layout import Layout
from ofai.model.config import ModelConfig
from ofai.model.planes import to_tensors
from ofai.model.policy import Policy


def load_policy(checkpoint: Path, device: torch.device | str = "cpu", layout: Layout | None = None) -> Policy:
    d = ck.load_checkpoint(Path(checkpoint), map_location="cpu")
    lay = layout or Layout.from_json(d["layout"])
    pol = Policy(lay, ModelConfig.from_dict(d["config"]))
    pol.load_state_dict(d["model"])
    return pol.to(device).eval()


class PolicyAgent:
    def __init__(self, checkpoint: Path | None = None, temperature: float = 1.0, device: str | None = None,
                 policy: Policy | None = None, seed: int = 0, max_batch: int = 64) -> None:
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)
        self.policy = policy.to(self.device).eval() if policy is not None else load_policy(checkpoint, self.device)
        self.temperature = temperature
        self.gen = torch.Generator(self.device).manual_seed(seed)
        self.state: tuple[torch.Tensor, torch.Tensor] | None = None
        self.max_batch = max_batch

    @property
    def layout(self) -> Layout:
        return self.policy.layout

    @torch.no_grad()
    def act(self, batch: dict, rows: np.ndarray, fresh: np.ndarray) -> tuple[np.ndarray, dict]:
        """Actions [len(rows), K, A] for the given rows of a ROWS batch; `fresh` resets those rows' state.

        Returns the actions and mind info per acting row: value and top-5 per head per action."""
        lay = self.layout
        total = batch["flags"].shape[0]
        if self.state is None or self.state[0].shape[0] != total:
            self.state = self.policy.initial_state(total, self.device)
        acts = np.full((len(rows), lay.K, lay.A), -1, dtype=np.int32)
        values = np.zeros(len(rows), dtype=np.float32)
        tops: list[list[dict]] = [[] for _ in rows]
        for s in range(0, len(rows), self.max_batch):
            r = rows[s : s + self.max_batch]
            obs = to_tensors({f.name: batch[f.name][r] for f in lay.fields}, self.device)
            idx = torch.as_tensor(r, device=self.device, dtype=torch.long)
            state = (self.state[0][idx], self.state[1][idx])
            fr = torch.as_tensor(fresh[s : s + self.max_batch], device=self.device)
            h, ctx, (h1, c1) = self.policy.core(obs, state, fr)
            self.state[0][idx], self.state[1][idx] = h1, c1
            a, _, top = self.policy.sample(h, ctx, self.temperature, self.gen)
            acts[s : s + len(r)] = a.cpu().numpy()
            values[s : s + len(r)] = self.policy.value(h).cpu().numpy()
            for j in range(len(r)):
                tops[s + j] = [{head: (i[j].tolist(), p[j].tolist()) for head, (i, p) in k.items()} for k in top]
        return acts, {"value": values, "top": tops}
