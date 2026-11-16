"""Weighted behaviour-cloning loss and its metrics (M3 Task 3)."""

from __future__ import annotations

import torch

from ofai.model.geometry import minimap_cell_to_tile, window_cell_to_tile
from ofai.model.policy import CoreCtx, Policy


def bc_loss(
    policy: Policy,
    h: torch.Tensor,  # [N, lstm]
    ctx: CoreCtx,
    obs: dict[str, torch.Tensor],
    labels: torch.Tensor,  # [N, K, A]
    weight: torch.Tensor,  # [N]
    value_target: torch.Tensor,  # [N]
    valid: torch.Tensor,  # [N] bool
    stop_weight: float = 0.3,
    value_coef: float = 0.5,
    metrics: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Mean over valid rows of weight * -log_prob(labels) (rows whose step is only STOP
    count `stop_weight` as much), plus value_coef * MSE(value, value_target).

    An all-invalid batch returns a zero loss that needs no backward (`loss.requires_grad`
    is False); the trainer skips that step."""
    rows = valid.bool().nonzero(as_tuple=True)[0]
    if rows.numel() == 0:
        return torch.zeros((), device=h.device), {"rows": 0, "skipped": True}
    lay = policy.layout
    t_type = lay.af("type")
    h, labels = h[rows], policy.clean(labels[rows].long())
    sub = ctx.rows(rows)
    lp, ent, info = policy.log_prob(h, sub, labels)
    stop_only = labels[:, 0, t_type] == policy.t_stop
    w = weight[rows].float() * torch.where(stop_only, torch.full_like(lp, stop_weight), torch.ones_like(lp))
    wsum = w.sum().clamp(min=1e-8)
    nll = (-(lp) * w).sum() / wsum
    value = policy.value(h)
    vt = value_target[rows].float()
    vloss = torch.mean((value - vt) ** 2)
    loss = nll + value_coef * vloss
    out = {"rows": int(rows.numel()), "loss": float(loss.detach()), "nll": float(nll.detach()),
           "value_mse": float(vloss.detach()), "entropy": float(ent.mean().detach()),
           "rescued": float(info["rescued"]) / max(1, rows.numel()), "stop_rows": float(stop_only.float().mean())}
    if metrics:
        out.update(_metrics(policy, info, labels, value, vt, obs, rows))
    return loss, out


@torch.no_grad()
def _metrics(policy: Policy, info: dict, labels: torch.Tensor, value: torch.Tensor, vt: torch.Tensor,
             obs: dict[str, torch.Tensor], rows: torch.Tensor) -> dict:
    lay = policy.layout
    t_type = lay.af("type")
    out: dict = {"value_mae": float((value - vt).abs().mean())}
    # Type head, first action: top-1 / top-5 and STOP precision / recall.
    logits = info["logits"][0]["type"]
    label = labels[:, 0, t_type]
    ok = label >= 0
    if ok.any():
        masked = logits.masked_fill(~_type_mask(policy, info, logits), -1e9)
        top5 = masked.topk(5, -1).indices
        pred = top5[:, 0]
        out["type_top1"] = float((pred[ok] == label[ok]).float().mean())
        out["type_top5"] = float((top5[ok] == label[ok, None]).any(-1).float().mean())
        ps, ls = pred == policy.t_stop, label == policy.t_stop
        out["stop_precision"] = float((ps & ls).sum() / ps.sum().clamp(min=1))
        out["stop_recall"] = float((ps & ls).sum() / ls.sum().clamp(min=1))
    # Location error in tiles (argmax vs label), through the sim's geometry.
    gm = list(lay.raw["gameMetaFields"])
    wm = list(lay.raw["winMetaFields"])
    meta = obs["game_meta"][rows].double()
    near_err, far_err = [], []
    for r, win, pred, lab in info["near"]:
        wmeta = obs["win_meta"][rows][r, win].double()
        x0, y0, tpc = wmeta[:, wm.index("x0")], wmeta[:, wm.index("y0")], wmeta[:, wm.index("tpc")]
        a = window_cell_to_tile(x0, y0, tpc, pred, int(lay.sizes["WIN"]))
        b = window_cell_to_tile(x0, y0, tpc, lab.clamp(min=0), int(lay.sizes["WIN"]))
        near_err.append((a - b).norm(dim=-1))
    for r, pcell, pfine, lcell, lfine in info["far"]:
        m = meta[r]
        args = (m[:, gm.index("map_w")], m[:, gm.index("map_h")], m[:, gm.index("mm_tpc")], m[:, gm.index("off_x")],
                m[:, gm.index("off_y")], int(lay.sizes["MM"]), int(lay.sizes["FINE"]))
        a = minimap_cell_to_tile(pcell, pfine, *args)
        b = minimap_cell_to_tile(lcell.clamp(min=0), lfine, *args)
        far_err.append((a - b).norm(dim=-1))
    if near_err:
        e = torch.cat(near_err)
        out["loc_near_tiles_p50"] = float(e.median())
    if far_err:
        e = torch.cat(far_err)
        out["loc_far_tiles_p50"] = float(e.median())
    return out


def _type_mask(policy: Policy, info: dict, logits: torch.Tensor) -> torch.Tensor:
    return info.get("type_mask", torch.ones_like(logits, dtype=torch.bool))
