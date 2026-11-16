"""Recurrent policy (M3 Task 2, spec §11).

encode: planes (GPU expansion) -> minimap CNN + shared window CNN, player
transformer with edge-biased attention, entity/global MLPs, fusion
transformer -> one embedding per row (bf16 autocast on CUDA).
core: LSTM (state zeroed where `fresh`).
heads (fp32): K autoregressive actions per step; within an action
type -> (player | entity | build) -> troops -> locMode -> (window -> cell) | (mmCell -> fine).

Sampling and `log_prob` use the same masks: the sim's legality masks plus
location hints (unowned land for SPAWN, own land for structures, water for
warships; any valid cell when a hint leaves nothing). In `log_prob` a teacher
label a mask excludes is let through ("rescued") and counted instead of
giving -inf: minimap cells are coarse (plurality owner), so a legal coastal
build can sit in a cell the hint calls water.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ofai.layout import Layout
from ofai.model.config import ModelConfig
from ofai.model.encoders import FusionEncoder, MinimapEncoder, PlayerEncoder, WindowEncoder, mlp
from ofai.model.heads import gumbel_sample, log_prob_of, topk_probs
from ofai.model.planes import expand_minimap, expand_windows

KIND_GLOBAL, KIND_MM, KIND_WIN, KIND_PLAYER, KIND_STRUCT, KIND_MOBILE, KIND_ATTACK = range(7)
# Entity table per pointer kind: 0 structs, 1 mobile, 2 attacks, 3 structs+mobile.
ENTITY_KIND = {"structs": 0, "mobile": 1, "attacks": 2, "structs+mobile": 3}


@dataclass
class CoreCtx:
    players: torch.Tensor  # [N, 34, W]
    structs: torch.Tensor  # [N, 64, W]
    mobile: torch.Tensor  # [N, 32, W]
    attacks: torch.Tensor  # [N, 32, W]
    windows: torch.Tensor  # [N, 8, W]
    win_keys: torch.Tensor  # [N, 8, ptr, 32, 32]
    mm_keys: torch.Tensor  # [N, ptr, 32, 32]
    mm_fine: torch.Tensor  # [N, fine, 128, 128]
    masks: dict[str, torch.Tensor] = field(default_factory=dict)

    def rows(self, idx: torch.Tensor) -> CoreCtx:
        return CoreCtx(*(getattr(self, f)[idx] for f in ("players", "structs", "mobile", "attacks", "windows",
                                                          "win_keys", "mm_keys", "mm_fine")),
                       masks={k: v[idx] for k, v in self.masks.items()})


class Policy(nn.Module):
    def __init__(self, layout: Layout, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.layout = layout
        self.cfg = cfg = cfg or ModelConfig()
        s = layout.sizes
        W, P = cfg.width, len(layout.planes)
        self.n_types, self.n_build = int(s["NUM_TYPES"]), int(s["NUM_BUILD"])
        self.K, self.A = layout.K, layout.A
        self.MM, self.WIN, self.NW = int(s["MM"]), int(s["WIN"]), int(s["NUM_WINDOWS"])
        self.FINE = int(s["FINE"])
        self.n_tokens, self.n_slots = int(s["NUM_PLAYER_TOKENS"]), int(s["ENTITY_SLOTS"])
        self.n_troops = int(s["TROOP_BUCKETS"])
        feats = layout.features

        self.minimap = MinimapEncoder(P, cfg.cnn_channels, cfg.blocks_per_stage, W, cfg.ptr_dim, cfg.fine_dim,
                                      cfg.mm_stem_stride)
        self.window = WindowEncoder(P, cfg.win_channels, cfg.blocks_per_stage, W, cfg.ptr_dim, int(s["WIN_META"]),
                                    cfg.win_stem_stride)
        self.player_enc = PlayerEncoder(len(feats["players"]), len(feats["edges"]), W, cfg.heads, cfg.player_layers, cfg.ff_mult)
        self.struct_enc = mlp(len(feats["structs"]), W, W, last_act=False)
        self.mobile_enc = mlp(len(feats["mobile"]), W, W, last_act=False)
        self.attack_enc = mlp(len(feats["attacks"]), W, W, last_act=False)
        self.global_enc = mlp(len(feats["globals"]), W, W, last_act=False)
        self.fusion = FusionEncoder(W, cfg.heads, cfg.fusion_layers, cfg.ff_mult, 7)
        self.pool = mlp(2 * W, W, W, last_act=False)
        self.lstm = nn.LSTMCell(W, cfg.lstm)

        # Heads.
        L, D = cfg.lstm, cfg.ptr_dim
        self.k_emb = nn.Parameter(torch.zeros(self.K, W))
        self.query = mlp(L + W, 2 * W, W, last_act=False)
        self.type_head = nn.Linear(W, self.n_types)
        self.type_emb = nn.Embedding(self.n_types, W)
        self.build_head = nn.Linear(W, self.n_build)
        self.build_emb = nn.Embedding(self.n_build + 1, W)
        self.troop_head = nn.Linear(W, self.n_troops)
        self.troop_emb = nn.Embedding(self.n_troops + 1, W)
        self.loc_head = nn.Linear(W, 2)
        self.loc_emb = nn.Embedding(3, W)
        self.q_player, self.k_player = nn.Linear(W, D), nn.Linear(W, D)
        self.q_entity, self.k_entity = nn.Linear(W, D), nn.Linear(W, D)
        self.q_window, self.k_window = nn.Linear(W, D), nn.Linear(W, D)
        self.q_cell = mlp(2 * W, W, D, last_act=False)
        self.q_mm, self.q_fine = nn.Linear(W, D), nn.Linear(W, cfg.fine_dim)
        self.fine_head = mlp(W + D + cfg.fine_dim, W, self.FINE * self.FINE, last_act=False)
        self.player_proj = nn.Linear(W, W)
        self.value_head = mlp(L, W, 1, last_act=False)

        # Static tables from the layout.
        T, Fi = layout.action_types, layout.action_fields
        uses = torch.zeros(self.n_types, self.A, dtype=torch.bool)
        ent_kind = torch.full((self.n_types,), -1, dtype=torch.long)
        for ti, name in enumerate(T):
            for f in layout.type_args[name]:
                uses[ti, Fi.index(f)] = True
            if name in layout.entity_table:
                ent_kind[ti] = ENTITY_KIND[layout.entity_table[name]]
        self.register_buffer("uses", uses, persistent=False)
        self.register_buffer("ent_kind", ent_kind, persistent=False)
        self.f = {n: Fi.index(n) for n in Fi}
        self.t_stop, self.t_spawn, self.t_build = T.index("STOP"), T.index("SPAWN"), T.index("BUILD")
        self.t_boat, self.t_move = T.index("BOAT"), T.index("MOVE_WARSHIP")
        nukes = set(layout.raw["nukeBuilds"])
        self.structure_builds = [i for i, b in enumerate(layout.build_types) if b not in ("Warship",) and i not in nukes]
        self.warship_build = layout.build_types.index("Warship")
        self.plane = {p["name"]: i for i, p in enumerate(layout.planes)}
        self.fv = {t: layout.feat(t, "valid") for t in ("players", "structs", "mobile", "attacks")}
        self.gm = list(layout.raw["gameMetaFields"])
        self.wm = list(layout.raw["winMetaFields"])

    # -- encoding -------------------------------------------------------------------

    def initial_state(self, batch: int, device: torch.device | str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        dev = device or next(self.parameters()).device
        z = torch.zeros(batch, self.cfg.lstm, device=dev)
        return z, z.clone()

    def _autocast(self, device: torch.device):
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                              enabled=self.cfg.amp and device.type == "cuda")

    def encode(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, CoreCtx]:
        dev = obs["mm_owner"].device
        lay = self.layout
        # bf16 planes when the encoders run under autocast (no fp32 -> bf16 copy, half the memory).
        pdt = torch.bfloat16 if (self.cfg.amp and dev.type == "cuda") else torch.float32
        mm_planes = expand_minimap(obs, lay, pdt)
        win_planes = expand_windows(obs, lay, pdt)
        players = obs["players"].float()
        structs, mobile, attacks = obs["structs"].float(), obs["mobile"].float(), obs["attacks"].float()
        pv = players[..., self.fv["players"]] > 0.5
        sv = structs[..., self.fv["structs"]] > 0.5
        mv = mobile[..., self.fv["mobile"]] > 0.5
        av = attacks[..., self.fv["attacks"]] > 0.5
        meta = obs["win_meta"].float().clone()
        gmeta = obs["game_meta"].float()
        mw = gmeta[:, self.gm.index("map_w")].clamp(min=1)[:, None]
        mh = gmeta[:, self.gm.index("map_h")].clamp(min=1)[:, None]
        meta[..., self.wm.index("x0")] /= mw
        meta[..., self.wm.index("y0")] /= mh
        meta[..., self.wm.index("tpc")] /= 4.0
        wv = obs["win_meta"][..., self.wm.index("valid")] > 0.5
        n = players.shape[0]
        ckpt = self.cfg.checkpoint_cnn and self.training and torch.is_grad_enabled()
        with self._autocast(dev):
            if ckpt:
                mm_tok, mm_keys, mm_fine = checkpoint(self.minimap, mm_planes, use_reentrant=False)
                win_tok, win_keys = checkpoint(self.window, win_planes, meta, use_reentrant=False)
            else:
                mm_tok, mm_keys, mm_fine = self.minimap(mm_planes)
                win_tok, win_keys = self.window(win_planes, meta)
            p_tok = self.player_enc(players, obs["edges"].float(), pv)
            tokens = torch.cat([
                self.global_enc(obs["globals"].float()).unsqueeze(1), mm_tok.unsqueeze(1), win_tok, p_tok,
                self.struct_enc(structs), self.mobile_enc(mobile), self.attack_enc(attacks),
            ], 1)
            kinds = torch.cat([torch.full((c,), k, device=dev, dtype=torch.long) for k, c in (
                (KIND_GLOBAL, 1), (KIND_MM, 1), (KIND_WIN, self.NW), (KIND_PLAYER, self.n_tokens),
                (KIND_STRUCT, structs.shape[1]), (KIND_MOBILE, mobile.shape[1]), (KIND_ATTACK, attacks.shape[1]))])
            ones = torch.ones(n, 2, dtype=torch.bool, device=dev)
            valid = torch.cat([ones, wv, pv, sv, mv, av], 1)
            fused = self.fusion(tokens, kinds, valid)
            vf = valid.unsqueeze(-1).to(fused.dtype)
            mean = (fused * vf).sum(1) / vf.sum(1).clamp(min=1)
            emb = self.pool(torch.cat([fused[:, 0], mean], 1))
        o = 2
        win_out = fused[:, o:o + self.NW]
        o += self.NW
        p_out = fused[:, o:o + self.n_tokens]
        o += self.n_tokens
        s_out = fused[:, o:o + structs.shape[1]]
        o += structs.shape[1]
        m_out = fused[:, o:o + mobile.shape[1]]
        o += mobile.shape[1]
        a_out = fused[:, o:]

        masks = self._masks(obs, mm_planes, win_planes, wv)
        # Token outputs in fp32 for the heads; the big key maps stay in the encoder dtype
        # (bf16 under autocast) and are cast per use on the rows that need them.
        ctx = CoreCtx(p_out.float(), s_out.float(), m_out.float(), a_out.float(), win_out.float(),
                      win_keys, mm_keys, mm_fine, masks)
        return emb.float(), ctx

    def _masks(self, obs, mm_planes, win_planes, win_valid) -> dict[str, torch.Tensor]:
        n = mm_planes.shape[0]
        p = self.plane
        mm = lambda name: mm_planes[:, p[name]].reshape(n, -1) > 0.5  # noqa: E731  (0/1 planes: exact in bf16)
        wn = lambda name: win_planes[:, :, p[name]].reshape(n, self.NW, -1) > 0.5  # noqa: E731
        return {
            "type": obs["type_mask"].bool(),
            "build": obs["build_mask"].bool(),
            "player": obs["player_mask"].bool(),
            "entity": obs["entity_mask"].bool(),
            "win": win_valid,
            "mm_valid": mm("valid"), "mm_spawn": mm("unowned"), "mm_own": mm("own"),
            "mm_water": mm("water") | mm("coast"),
            "win_valid": wn("valid"), "win_spawn": wn("unowned"), "win_own": wn("own"),
            "win_water": wn("water") | wn("coast"),
        }

    def lstm_step(self, emb: torch.Tensor, state, fresh: torch.Tensor | None = None):
        h, c = state
        if fresh is not None:
            keep = (~fresh.bool()).float().unsqueeze(-1)
            h, c = h * keep, c * keep
        h, c = self.lstm(emb, (h, c))
        return h, (h, c)

    def core(self, obs: dict[str, torch.Tensor], state, fresh: torch.Tensor | None = None):
        emb, ctx = self.encode(obs)
        h, state = self.lstm_step(emb, state, fresh)
        return h, ctx, state

    def unroll(self, emb: torch.Tensor, state, fresh: torch.Tensor):
        """emb [T, B, W], fresh [T, B] -> h [T, B, lstm] (truncated BPTT over T)."""
        hs = []
        for t in range(emb.shape[0]):
            h, state = self.lstm_step(emb[t], state, fresh[t])
            hs.append(h)
        return torch.stack(hs), state

    def value(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.value_head(h.float()).squeeze(-1))

    # -- heads ------------------------------------------------------------------------

    def _ptr(self, q: torch.Tensor, keys: torch.Tensor, wq: nn.Linear, wk: nn.Linear) -> torch.Tensor:
        return torch.einsum("nd,ntd->nt", wq(q), wk(keys)) / math.sqrt(self.cfg.ptr_dim)

    def _entity_keys(self, ctx: CoreCtx, kind: torch.Tensor) -> torch.Tensor:
        """Keys [n, 96, W] of each row's entity table."""
        n, W = ctx.structs.shape[0], ctx.structs.shape[-1]
        pad = lambda x: F.pad(x, (0, 0, 0, self.n_slots - x.shape[1]))  # noqa: E731
        tables = torch.stack([pad(ctx.structs), pad(ctx.mobile), pad(ctx.attacks),
                              torch.cat([ctx.structs, ctx.mobile], 1)[:, : self.n_slots]], 1)
        return tables[torch.arange(n, device=kind.device), kind.clamp(min=0)]

    def _cell_logits(self, q3: torch.Tensor, ctx: CoreCtx, win: torch.Tensor) -> torch.Tensor:
        n = q3.shape[0]
        ar = torch.arange(n, device=q3.device)
        qc = self.q_cell(torch.cat([q3, ctx.windows[ar, win.clamp(min=0)]], 1))
        keys = ctx.win_keys[ar, win.clamp(min=0)].reshape(n, self.cfg.ptr_dim, -1).float()
        return torch.einsum("nd,ndc->nc", qc, keys) / math.sqrt(self.cfg.ptr_dim)

    def _mm_logits(self, q3: torch.Tensor, ctx: CoreCtx) -> torch.Tensor:
        n = q3.shape[0]
        coarse = torch.einsum("nd,ndhw->nhw", self.q_mm(q3), ctx.mm_keys.float()) / math.sqrt(self.cfg.ptr_dim)
        scale = self.MM // coarse.shape[-1]
        coarse = coarse.repeat_interleave(scale, 1).repeat_interleave(scale, 2)
        fine = torch.einsum("nd,ndhw->nhw", self.q_fine(q3), ctx.mm_fine.float())
        return (coarse + fine).reshape(n, -1)

    def _fine_logits(self, q3: torch.Tensor, ctx: CoreCtx, cell: torch.Tensor) -> torch.Tensor:
        n = q3.shape[0]
        ar = torch.arange(n, device=q3.device)
        c = cell.clamp(min=0)
        scale = self.MM // ctx.mm_keys.shape[-1]
        cy, cx = c // self.MM, c % self.MM
        ck = ctx.mm_keys[ar, :, cy // scale, cx // scale].float()
        fk = ctx.mm_fine[ar, :, cy, cx].float()
        return self.fine_head(torch.cat([q3, ck, fk], 1))

    def _loc_hint(self, ctx: CoreCtx, t: torch.Tensor, b: torch.Tensor, prefix: str) -> torch.Tensor:
        """Sampling-time location mask: valid cells narrowed by type, falling back to valid."""
        m = ctx.masks
        valid = m[f"{prefix}_valid"]
        expand = (lambda x: x[:, None]) if prefix == "mm" else (lambda x: x[:, None, None])
        is_spawn = expand(t == self.t_spawn)
        is_struct = expand((t == self.t_build) & torch.isin(b, torch.tensor(self.structure_builds, device=b.device)))
        is_water = expand((t == self.t_move) | ((t == self.t_build) & (b == self.warship_build)))
        hint = torch.where(is_spawn, m[f"{prefix}_spawn"], valid)
        hint = torch.where(is_struct, m[f"{prefix}_own"], hint)
        hint = torch.where(is_water, m[f"{prefix}_water"], hint) & valid
        empty = ~hint.reshape(hint.shape[0], -1).any(-1)
        return torch.where(expand(empty), valid, hint)

    def _query(self, h: torch.Tensor, prev: torch.Tensor, k: int) -> torch.Tensor:
        return self.query(torch.cat([h, prev + self.k_emb[k]], 1))

    def _embed_action(self, ctx: CoreCtx, a: torch.Tensor) -> torch.Tensor:
        f = self.f
        n = a.shape[0]
        ar = torch.arange(n, device=a.device)
        e = self.type_emb(a[:, f["type"]].clamp(min=0))
        e = e + self.build_emb(a[:, f["build"]] + 1) + self.troop_emb(a[:, f["troops"]] + 1) + self.loc_emb(a[:, f["locMode"]] + 1)
        pl = a[:, f["player"]]
        e = e + self.player_proj(ctx.players[ar, pl.clamp(min=0)]) * (pl >= 0).float().unsqueeze(-1)
        return e

    def clean(self, actions: torch.Tensor) -> torch.Tensor:
        """-1 in every field an action's type (and locMode) doesn't use, and in every action after STOP."""
        f = self.f
        a = actions.clone()
        t = a[..., f["type"]]
        uses = self.uses[t.clamp(min=0)] & (t >= 0).unsqueeze(-1)
        lm = a[..., f["locMode"]]
        for name in ("window", "cell"):
            uses[..., f[name]] &= lm == 0
        for name in ("mmCell", "fine"):
            uses[..., f[name]] &= lm == 1
        uses[..., f["type"]] = True
        a = torch.where(uses, a, torch.full_like(a, -1))
        stopped = torch.zeros_like(t, dtype=torch.bool)
        for k in range(1, a.shape[-2]):
            stopped[..., k] = stopped[..., k - 1] | (a[..., k - 1, f["type"]] == self.t_stop) | (a[..., k - 1, f["type"]] < 0)
        return torch.where(stopped.unsqueeze(-1), torch.full_like(a, -1), a)

    def log_prob(self, h: torch.Tensor, ctx: CoreCtx, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Sum of log-probs of the fields each label action uses. actions [N, K, A] (-1 = unused)."""
        h = h.float()
        actions = self.clean(actions.long())
        n = h.shape[0]
        dev = h.device
        f, m = self.f, ctx.masks
        lp = torch.zeros(n, device=dev)
        ent = torch.zeros(n, device=dev)
        prev = torch.zeros(n, self.cfg.width, device=dev)
        alive = torch.ones(n, dtype=torch.bool, device=dev)
        info: dict = {"rescued": torch.zeros((), device=dev), "logits": [], "near": [], "far": [], "type_mask": m["type"]}

        def acc(rows: torch.Tensor, chosen: torch.Tensor, entropy: torch.Tensor, rescued: torch.Tensor) -> None:
            lp.index_add_(0, rows, chosen)
            ent.index_add_(0, rows, entropy)
            info["rescued"] = info["rescued"] + rescued.sum()

        for k in range(self.K):
            a = actions[:, k]
            t = a[:, f["type"]]
            active = alive & (t >= 0)
            q = self._query(h, prev, k)
            tl = self.type_head(q)
            rows = active.nonzero(as_tuple=True)[0]
            if rows.numel():
                acc(rows, *log_prob_of(tl[rows], m["type"][rows], t[rows]))
            info["logits"].append({"type": tl})
            tc = t.clamp(min=0)
            q2 = q + self.type_emb(tc)
            uses = self.uses[tc] & active.unsqueeze(1)

            r = uses[:, f["player"]].nonzero(as_tuple=True)[0]
            if r.numel():
                logits = self._ptr(q2[r], ctx.players[r], self.q_player, self.k_player)
                acc(r, *log_prob_of(logits, m["player"][r, tc[r]], a[r, f["player"]]))
            r = uses[:, f["entity"]].nonzero(as_tuple=True)[0]
            if r.numel():
                logits = self._ptr(q2[r], self._entity_keys(ctx.rows(r), self.ent_kind[tc[r]]), self.q_entity, self.k_entity)
                acc(r, *log_prob_of(logits, m["entity"][r, tc[r]], a[r, f["entity"]]))
            r = uses[:, f["build"]].nonzero(as_tuple=True)[0]
            if r.numel():
                acc(r, *log_prob_of(self.build_head(q2[r]), m["build"][r], a[r, f["build"]]))
            q3 = q2 + self.build_emb(a[:, f["build"]] + 1)
            r = uses[:, f["troops"]].nonzero(as_tuple=True)[0]
            if r.numel():
                logits = self.troop_head(q3[r])
                acc(r, *log_prob_of(logits, torch.ones_like(logits, dtype=torch.bool), a[r, f["troops"]]))
            r = uses[:, f["locMode"]].nonzero(as_tuple=True)[0]
            if r.numel():
                lm = torch.stack([m["win"][r].any(-1), torch.ones(r.numel(), dtype=torch.bool, device=dev)], 1)
                acc(r, *log_prob_of(self.loc_head(q3[r]), lm, a[r, f["locMode"]]))
                near = r[a[r, f["locMode"]] == 0]
                far = r[a[r, f["locMode"]] == 1]
                if near.numel():
                    q4 = q3[near] + self.loc_emb.weight[1]
                    acc(near, *log_prob_of(self._ptr(q4, ctx.windows[near], self.q_window, self.k_window), m["win"][near], a[near, f["window"]]))
                    win = a[near, f["window"]].clamp(min=0)
                    sub = ctx.rows(near)
                    cm = self._loc_hint(sub, t[near], a[near, f["build"]], "win")[torch.arange(near.numel(), device=dev), win]
                    cl = self._cell_logits(q4, sub, win)
                    acc(near, *log_prob_of(cl, cm, a[near, f["cell"]]))
                    info["near"].append((near, win, cl.detach().masked_fill(~cm, -1e9).argmax(-1), a[near, f["cell"]]))
                if far.numel():
                    q4 = q3[far] + self.loc_emb.weight[2]
                    sub = ctx.rows(far)
                    hint = self._loc_hint(sub, t[far], a[far, f["build"]], "mm")
                    ml = self._mm_logits(q4, sub)
                    acc(far, *log_prob_of(ml, hint, a[far, f["mmCell"]]))
                    fl = self._fine_logits(q4, sub, a[far, f["mmCell"]])
                    acc(far, *log_prob_of(fl, torch.ones_like(fl, dtype=torch.bool), a[far, f["fine"]]))
                    info["far"].append((far, ml.detach().masked_fill(~hint, -1e9).argmax(-1), fl.detach().argmax(-1),
                                        a[far, f["mmCell"]], a[far, f["fine"]]))
            prev = prev + self._embed_action(ctx, a) * active.float().unsqueeze(-1)
            alive = active & (t != self.t_stop)
        return lp, ent, info

    @torch.no_grad()
    def sample(self, h: torch.Tensor, ctx: CoreCtx, temperature: float = 1.0,
               generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
        """Samples K actions per row (stops at STOP). Returns actions [N, K, A], log-prob [N], top-5 per head per k."""
        h = h.float()
        n = h.shape[0]
        dev = h.device
        f, m = self.f, ctx.masks
        out = torch.full((n, self.K, self.A), -1, dtype=torch.long, device=dev)
        lp = torch.zeros(n, device=dev)
        prev = torch.zeros(n, self.cfg.width, device=dev)
        alive = torch.ones(n, dtype=torch.bool, device=dev)
        tops: list[dict] = []
        smp = lambda logits, mask: gumbel_sample(logits, mask, temperature, generator)  # noqa: E731

        for k in range(self.K):
            q = self._query(h, prev, k)
            tl = self.type_head(q)
            tmask = m["type"].clone()
            t, l0 = smp(tl, tmask)
            t = torch.where(alive, t, torch.full_like(t, -1))
            lp += torch.where(alive, l0, torch.zeros_like(l0))
            top = {"type": topk_probs(tl, tmask)}
            a = torch.full((n, self.A), -1, dtype=torch.long, device=dev)
            a[:, f["type"]] = t
            tc = t.clamp(min=0)
            q2 = q + self.type_emb(tc)
            uses = self.uses[tc] & alive.unsqueeze(1) & (t >= 0).unsqueeze(1)

            def take(field: str, rows: torch.Tensor, logits: torch.Tensor, mask: torch.Tensor) -> None:
                idx, l = smp(logits, mask)
                a[rows, f[field]] = idx
                lp.index_add_(0, rows, l)

            r = uses[:, f["player"]].nonzero(as_tuple=True)[0]
            if r.numel():
                logits = self._ptr(q2[r], ctx.players[r], self.q_player, self.k_player)
                take("player", r, logits, m["player"][r, tc[r]])
            r = uses[:, f["entity"]].nonzero(as_tuple=True)[0]
            if r.numel():
                logits = self._ptr(q2[r], self._entity_keys(ctx.rows(r), self.ent_kind[tc[r]]), self.q_entity, self.k_entity)
                take("entity", r, logits, m["entity"][r, tc[r]])
            r = uses[:, f["build"]].nonzero(as_tuple=True)[0]
            if r.numel():
                bl = self.build_head(q2[r])
                take("build", r, bl, m["build"][r])
            q3 = q2 + self.build_emb(a[:, f["build"]] + 1)
            r = uses[:, f["troops"]].nonzero(as_tuple=True)[0]
            if r.numel():
                logits = self.troop_head(q3[r])
                take("troops", r, logits, torch.ones_like(logits, dtype=torch.bool))
            r = uses[:, f["locMode"]].nonzero(as_tuple=True)[0]
            if r.numel():
                lm = torch.stack([m["win"][r].any(-1), torch.ones(r.numel(), dtype=torch.bool, device=dev)], 1)
                take("locMode", r, self.loc_head(q3[r]), lm)
                near = r[a[r, f["locMode"]] == 0]
                far = r[a[r, f["locMode"]] == 1]
                if near.numel():
                    q4 = q3[near] + self.loc_emb.weight[1]
                    take("window", near, self._ptr(q4, ctx.windows[near], self.q_window, self.k_window), m["win"][near])
                    win = a[near, f["window"]]
                    hint = self._loc_hint(ctx.rows(near), t[near], a[near, f["build"]], "win")
                    hint = hint[torch.arange(near.numel(), device=dev), win]
                    take("cell", near, self._cell_logits(q4, ctx.rows(near), win), hint)
                if far.numel():
                    q4 = q3[far] + self.loc_emb.weight[2]
                    sub = ctx.rows(far)
                    hint = self._loc_hint(sub, t[far], a[far, f["build"]], "mm")
                    take("mmCell", far, self._mm_logits(q4, sub), hint)
                    fl = self._fine_logits(q4, sub, a[far, f["mmCell"]])
                    take("fine", far, fl, torch.ones_like(fl, dtype=torch.bool))
            out[:, k] = a
            tops.append(top)
            prev = prev + self._embed_action(ctx, a) * alive.float().unsqueeze(-1)
            alive = alive & (t != self.t_stop)
        return out, lp, tops


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
