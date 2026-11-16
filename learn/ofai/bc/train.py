"""Resumable BC trainer (M3 Task 6).

AdamW, linear warmup then cosine to 10 % over `total_samples`, grad clip,
bf16 autocast for encoders, truncated BPTT over `seq_len`. Lanes are split
into micro-batches (`micro_lanes`) whose gradients accumulate, so memory is
bounded by micro_lanes x seq_len rows. Stops at `until`, on `stop`, when
`run_dir/STOP` exists, or at `total_samples`; checkpoints every
`checkpoint_minutes` and at every stop.

A sim worker that dies is restarted and every lane's LSTM state restarts;
three deaths within 10 minutes stop the run with an `error` event carrying
the worker's stderr tail.
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ofai import checkpoints as ck
from ofai.bc.loss import bc_loss
from ofai.bridge.pool import SimWorkerDied
from ofai.layout import Layout
from ofai.model.config import ModelConfig
from ofai.model.planes import to_tensors
from ofai.model.policy import Policy, parameter_count
from ofai.runlog import RunLog

DEFAULTS: dict[str, Any] = {
    # Lanes = workers x games_per_worker x seats_per_game (per pool, summed over commits).
    "seq_len": 64,
    "micro_lanes": 4,
    "workers": 14,
    "games_per_worker": 2,
    "seats_per_game": 2,
    "split": "bc-v1",
    "lr": 3e-4,
    "weight_decay": 0.01,
    "warmup_steps": 1000,
    "grad_clip": 1.0,
    "total_samples": 40_000_000,
    "eval_every": 2_000_000,
    "eval_seqs": 8,
    "checkpoint_minutes": 30,
    "keep_checkpoints": 3,
    "stop_weight": 0.3,
    "value_coef": 0.5,
    "log_every": 10,
    "seed": 0,
    "device": "cuda",
    "model": {},
}

DEATHS_WINDOW_S = 600
MAX_DEATHS = 3


def lr_at(cfg: dict, step: int, samples: int) -> float:
    warm = min(1.0, (step + 1) / max(1, cfg["warmup_steps"]))
    prog = min(1.0, samples / max(1, cfg["total_samples"]))
    return cfg["lr"] * warm * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))


def _rows(seq: dict, keys: list[str], l0: int, l1: int) -> dict[str, np.ndarray]:
    out = {}
    for k in keys:
        a = seq[k][:, l0:l1]
        out[k] = a.reshape(a.shape[0] * a.shape[1], *a.shape[2:])
    return out


class Trainer:
    def __init__(self, cfg: dict, run_dir: Path, layout: Layout, make_stream: Callable[[str], Any],
                 log: RunLog | None = None) -> None:
        self.cfg = {**DEFAULTS, **cfg}
        self.run_dir = Path(run_dir)
        self.layout = layout
        self.make_stream = make_stream
        self.log = log or RunLog(self.run_dir)
        dev = self.cfg["device"]
        self.device = torch.device(dev if (dev != "cuda" or torch.cuda.is_available()) else "cpu")
        torch.manual_seed(self.cfg["seed"])
        random.seed(self.cfg["seed"])
        self.model_cfg = ModelConfig.from_dict(self.cfg["model"])
        self.policy = Policy(layout, self.model_cfg).to(self.device)
        self.opt = torch.optim.AdamW(self.policy.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"])
        self.step = 0
        self.samples = 0
        self.next_eval = self.cfg["eval_every"]
        self.deaths: list[float] = []
        self.obs_keys = [f.name for f in layout.fields]

    # -- checkpoints ------------------------------------------------------------------

    def save(self, stream: Any, reason: str) -> Path:
        path = self.run_dir / f"step-{self.step:08d}.pt"
        ck.save_checkpoint(path, {
            "model": self.policy.state_dict(), "config": self.model_cfg.to_dict(), "trainConfig": self.cfg,
            "layoutVersion": self.layout.version, "layout": self.layout.raw, "step": self.step, "samples": self.samples, "nextEval": self.next_eval,
            "optimizer": self.opt.state_dict(), "stream": stream.state_dict() if stream is not None else None,
            "rng": {"torch": torch.get_rng_state(), "python": random.getstate()}, "parent": None,
            "createdAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        keep = sorted(self.run_dir.glob("step-*.pt"))[: -self.cfg["keep_checkpoints"]]
        for old in keep:
            old.unlink(missing_ok=True)
            old.with_name(old.name + ".sha256").unlink(missing_ok=True)
        self.log.event("checkpoint", path=str(path), step=self.step, samples=self.samples, reason=reason)
        return path

    def resume(self) -> dict | None:
        path = ck.latest_checkpoint(self.run_dir)
        if path is None:
            return None
        d = ck.load_checkpoint(path, map_location=self.device)
        self.policy.load_state_dict(d["model"])
        self.opt.load_state_dict(d["optimizer"])
        self.step, self.samples = int(d["step"]), int(d["samples"])
        self.next_eval = int(d.get("nextEval", self.next_eval))
        rng = d.get("rng") or {}
        if "torch" in rng:
            torch.set_rng_state(rng["torch"].cpu() if hasattr(rng["torch"], "cpu") else rng["torch"])
        if "python" in rng:
            random.setstate(rng["python"])
        self.log.event("resume", path=str(path), step=self.step, samples=self.samples)
        return d

    # -- one sequence batch -------------------------------------------------------------

    def train_seq(self, seq: dict, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[dict, tuple]:
        T, lanes = seq["flags"].shape
        micro = max(1, min(self.cfg["micro_lanes"], lanes))
        groups = [(l0, min(lanes, l0 + micro)) for l0 in range(0, lanes, micro)]
        total_valid = int(seq["valid"].sum())
        for g in self.opt.param_groups:
            g["lr"] = lr_at(self.cfg, self.step, self.samples)
        if total_valid == 0:
            return {"skipped": True}, state
        self.opt.zero_grad(set_to_none=True)
        h_all, c_all = state
        new_h, new_c = h_all.clone(), c_all.clone()
        agg: dict[str, float] = {}
        for l0, l1 in groups:
            n_valid = int(seq["valid"][:, l0:l1].sum())
            rows = _rows(seq, self.obs_keys + ["labels", "weight", "value", "valid", "fresh"], l0, l1)
            obs = to_tensors({k: rows[k] for k in self.obs_keys}, self.device)
            emb, ctx = self.policy.encode(obs)
            Lg = l1 - l0
            fresh = torch.from_numpy(seq["fresh"][:, l0:l1]).to(self.device)
            hs, (h1, c1) = self.policy.unroll(emb.view(T, Lg, -1), (h_all[l0:l1], c_all[l0:l1]), fresh)
            new_h[l0:l1], new_c[l0:l1] = h1.detach(), c1.detach()
            if n_valid == 0:
                continue
            t = lambda k, dt=None: torch.from_numpy(np.ascontiguousarray(rows[k])).to(self.device)  # noqa: E731
            loss, m = bc_loss(self.policy, hs.reshape(T * Lg, -1), ctx, obs, t("labels"), t("weight"), t("value"),
                              t("valid"), self.cfg["stop_weight"], self.cfg["value_coef"],
                              metrics=self.step % self.cfg["log_every"] == 0)
            if loss.requires_grad:
                (loss * (n_valid / total_valid)).backward()
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    agg[k] = agg.get(k, 0.0) + float(v) * (n_valid / total_valid)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg["grad_clip"])
        if torch.isfinite(grad_norm):
            self.opt.step()
        else:
            self.log.event("nonfinite_grad", step=self.step)
        self.step += 1
        self.samples += total_valid
        agg.update(grad_norm=float(grad_norm), lr=self.opt.param_groups[0]["lr"], valid=total_valid)
        return agg, (new_h, new_c)

    @torch.no_grad()
    def evaluate(self) -> dict:
        stream = self.make_stream("heldout")
        if stream is None:
            return {}
        try:
            state = self.policy.initial_state(stream.lanes, self.device)
            tot: dict[str, float] = {}
            n = 0
            for _ in range(self.cfg["eval_seqs"]):
                seq = stream.next_seq()
                T, lanes = seq["flags"].shape
                for l0 in range(0, lanes, self.cfg["micro_lanes"]):
                    l1 = min(lanes, l0 + self.cfg["micro_lanes"])
                    rows = _rows(seq, self.obs_keys + ["labels", "weight", "value", "valid"], l0, l1)
                    obs = to_tensors({k: rows[k] for k in self.obs_keys}, self.device)
                    emb, ctx = self.policy.encode(obs)
                    fresh = torch.from_numpy(seq["fresh"][:, l0:l1]).to(self.device)
                    hs, (h1, c1) = self.policy.unroll(emb.view(T, l1 - l0, -1), (state[0][l0:l1], state[1][l0:l1]), fresh)
                    state[0][l0:l1], state[1][l0:l1] = h1, c1
                    t = lambda k: torch.from_numpy(np.ascontiguousarray(rows[k])).to(self.device)  # noqa: E731
                    _, m = bc_loss(self.policy, hs.reshape(-1, hs.shape[-1]), ctx, obs, t("labels"), t("weight"),
                                   t("value"), t("valid"), self.cfg["stop_weight"], self.cfg["value_coef"])
                    w = m.get("rows", 0)
                    for k, v in m.items():
                        if isinstance(v, (int, float)) and k != "rows":
                            tot[k] = tot.get(k, 0.0) + float(v) * w
                    n += w
            return {f"eval_{k}": v / max(1, n) for k, v in tot.items()}
        finally:
            stream.close()

    # -- the loop -------------------------------------------------------------------------

    def run(self, until: datetime | None = None, stop: threading.Event | None = None,
            max_steps: int | None = None) -> dict:
        resumed = self.resume()
        stream = self.make_stream("train")
        if resumed is not None and resumed.get("stream"):
            stream.load_state_dict(resumed["stream"])
        if resumed is None:
            self.log.event("start", params=parameter_count(self.policy), device=str(self.device), config=self.cfg)
        last_ckpt = time.monotonic()
        state = self.policy.initial_state(stream.lanes, self.device)
        reason = "done"
        try:
            while True:
                if self.samples >= self.cfg["total_samples"]:
                    reason = "total_samples"
                    break
                if max_steps is not None and self.step >= max_steps:
                    reason = "max_steps"
                    break
                if stop is not None and stop.is_set():
                    reason = "stop_event"
                    break
                if (self.run_dir / "STOP").exists():
                    reason = "stop_file"
                    break
                if until is not None and datetime.now(until.tzinfo) >= until:
                    reason = "until"
                    break
                try:
                    seq = stream.next_seq()
                except SimWorkerDied as e:
                    now = time.monotonic()
                    self.deaths = [d for d in self.deaths if now - d < DEATHS_WINDOW_S] + [now]
                    if len(self.deaths) >= MAX_DEATHS:
                        self.log.event("error", what="sim workers keep dying", deaths=len(self.deaths),
                                       stderr=e.stderr_tail[-4000:])
                        reason = "worker_deaths"
                        break
                    self.log.event("worker_died", worker=e.index, commit=getattr(e, "commit", None),
                                   stderr=e.stderr_tail[-2000:])
                    stream.restart(e)
                    state = self.policy.initial_state(stream.lanes, self.device)
                    continue
                t0 = time.monotonic()
                m, state = self.train_seq(seq, state)
                if self.step % self.cfg["log_every"] == 0 and not m.get("skipped"):
                    self.log.metric(step=self.step, samples=self.samples, sec=round(time.monotonic() - t0, 3), **m)
                if self.samples >= self.next_eval:
                    self.next_eval += self.cfg["eval_every"]
                    ev = self.evaluate()
                    if ev:
                        self.log.metric(step=self.step, samples=self.samples, **ev)
                        self.log.event("eval", step=self.step, **ev)
                if time.monotonic() - last_ckpt >= self.cfg["checkpoint_minutes"] * 60:
                    self.save(stream, "periodic")
                    last_ckpt = time.monotonic()
        finally:
            path = self.save(stream, reason)
            self.log.event("stop", reason=reason, step=self.step, samples=self.samples)
            stream.close()
        return {"reason": reason, "step": self.step, "samples": self.samples, "checkpoint": str(path)}


def default_stream_factory(cfg: dict) -> tuple[Callable[[str], Any], Layout]:
    """Real pools: one SimPool per commit that has games in the split, workers split by game share."""
    from ofai.bc.split import load_split
    from ofai.bc.stream import TeacherStream
    from ofai.bridge.pool import SimPool
    from ofai.data.index import Index
    from ofai.engines import ensure_bundle

    cfg = {**DEFAULTS, **cfg}
    index = Index()
    split = load_split(cfg["split"])
    share: dict[str, int] = {}
    for gid in split["train"]:
        g = index.game(gid)
        if g is not None:
            share[g["commit"]] = share.get(g["commit"], 0) + 1
    if not share:
        raise RuntimeError(f"split {cfg['split']} has no indexed games")
    for c in share:
        ensure_bundle(c)
    total = sum(share.values())
    workers = {c: max(1, round(cfg["workers"] * n / total)) for c, n in share.items()}
    layout_box: list[Layout] = []

    def make(part: str):
        n_workers = workers if part == "train" else {c: 1 for c in share}
        pools = {c: SimPool(c, "replay", n, games=cfg["games_per_worker"], seats=cfg["seats_per_game"])
                 for c, n in n_workers.items()}
        if not layout_box:
            layout_box.append(next(iter(pools.values())).layout)
        return TeacherStream(index, split, part, pools, seq_len=cfg["seq_len"], seed=cfg["seed"])

    probe = make("heldout")
    probe.close()
    return make, layout_box[0]


def train_bc(cfg: dict, run_dir: Path, *, until: datetime | None = None, stop: threading.Event | None = None,
             make_stream: Callable[[str], Any] | None = None, layout: Layout | None = None,
             max_steps: int | None = None) -> dict:
    if make_stream is None or layout is None:
        make_stream, layout = default_stream_factory(cfg)
    return Trainer(cfg, run_dir, layout, make_stream).run(until=until, stop=stop, max_steps=max_steps)
