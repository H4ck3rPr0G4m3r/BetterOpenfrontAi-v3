"""GPU plane expansion (M3 Task 1, roadmap §3.5).

Builds the PLANES of the minimap and the zoom windows from the compact wire
fields, as pure tensor ops. The meaning of each plane is defined by the CPU
reference `sim/src/obs/reference/expandPlanes.ts`; `test_planes.py` checks
both agree exactly. Geometry that decides which cell a unit or SAM disk
lands in is computed in float64, like the JavaScript reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ofai.layout import Layout

LETTERBOX = 0xFFFF


@dataclass
class PlaneTables:
    """Per-plane lookup tables built once from the layout."""

    kinds: list[str]  # per plane: terrain | owner | density | units | sam_cover | valid
    terrain_bit: torch.Tensor  # [P] int (0 = water)
    rel_table: torch.Tensor  # [P, NREL] bool: owner relation codes the plane takes
    owner_unowned: torch.Tensor  # [P] bool
    unit_table: torch.Tensor  # [P, NUNIT] bool: unit type codes the plane marks
    land_bit: int
    n_units: int


_CACHE: dict[tuple[int, str], PlaneTables] = {}


def plane_tables(layout: Layout, device: torch.device | str = "cpu") -> PlaneTables:
    key = (id(layout), str(device))
    if key in _CACHE:
        return _CACHE[key]
    n_rel = max(layout.raw["relation"].values()) + 1
    unit_types = list(layout.raw["unitTypes"])
    P = len(layout.planes)
    bits = torch.zeros(P, dtype=torch.int32)
    rel = torch.zeros(P, n_rel, dtype=torch.bool)
    unowned = torch.zeros(P, dtype=torch.bool)
    units = torch.zeros(P, len(unit_types), dtype=torch.bool)
    kinds = []
    for i, p in enumerate(layout.planes):
        kinds.append(p["from"])
        if p.get("bit") is not None:
            bits[i] = int(p["bit"])
        for r in p.get("rel") or []:
            rel[i, int(r)] = True
        unowned[i] = bool(p.get("ownerUnowned"))
        for u in p.get("unit") or []:
            units[i, unit_types.index(u)] = True
    t = PlaneTables(kinds, bits.to(device), rel.to(device), unowned.to(device), units.to(device),
                    int(layout.raw["terrainBits"]["land"]), len(unit_types))
    _CACHE[key] = t
    return t


def to_tensors(batch: dict[str, np.ndarray], device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """Wire arrays -> tensors. u16 becomes int32 (torch has little uint16 support), i16 int32, f16 stays half."""
    out: dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if not isinstance(v, np.ndarray):
            continue
        if v.dtype == np.uint16 or v.dtype == np.int16:
            v = v.astype(np.int32)
        out[k] = torch.from_numpy(np.array(v, copy=True)).to(device, non_blocking=True)
    return out


def _gm(layout: Layout, obs: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    return obs["game_meta"][:, list(layout.raw["gameMetaFields"]).index(name)]


def _expand(
    t: PlaneTables,
    owner: torch.Tensor,  # [N, H, W] int32, 0xFFFF letterbox
    terrain: torch.Tensor,  # [N, H, W] uint8/int
    relation: torch.Tensor,  # [N, 1024] uint8 (per view's row)
    density: torch.Tensor,  # [N, 1024] float
    unit_cell: torch.Tensor,  # [N, U] int64 cell index or -1
    unit_type: torch.Tensor,  # [N, U] int64, -1 empty
    unit_rel: torch.Tensor,  # [N, U] int64 relation of the unit's owner
    sam: torch.Tensor | None,  # [N, H*W] bool SAM-covered cells
) -> torch.Tensor:
    N, H, W = owner.shape
    P = len(t.kinds)
    dev = owner.device
    valid = owner != LETTERBOX
    own_idx = torch.where(owner < relation.shape[1], owner, torch.zeros_like(owner)).long()
    rel = torch.gather(relation.long(), 1, own_idx.view(N, -1)).view(N, H, W)
    rel = torch.where(owner < relation.shape[1], rel, torch.zeros_like(rel))
    dens = torch.gather(density.float(), 1, own_idx.view(N, -1)).view(N, H, W)
    terr = terrain.int()
    is_land = (terr & t.land_bit) != 0
    owned = valid & (owner != 0)
    out = torch.zeros(N, P, H, W, dtype=torch.float32, device=dev)
    for p, kind in enumerate(t.kinds):
        if kind == "valid":
            out[:, p] = valid.float()
        elif kind == "terrain":
            bit = int(t.terrain_bit[p])
            out[:, p] = (valid & ~is_land).float() if bit == 0 else ((terr & bit) != 0).float()
        elif kind in ("owner", "density"):
            if bool(t.owner_unowned[p]):
                out[:, p] = (valid & (owner == 0) & is_land).float()
            else:
                hit = owned & t.rel_table[p][rel]
                out[:, p] = torch.where(hit, dens if kind == "density" else torch.ones_like(dens), torch.zeros_like(dens))
        elif kind == "units":
            ok = (unit_type >= 0) & (unit_cell >= 0)
            ok &= t.unit_table[p][unit_type.clamp(min=0)] & t.rel_table[p][unit_rel]
            n_idx, u_idx = ok.nonzero(as_tuple=True)
            if n_idx.numel():
                plane = torch.zeros(N, H * W, dtype=torch.float32, device=dev)
                plane[n_idx, unit_cell[n_idx, u_idx]] = 1.0
                out[:, p] = plane.view(N, H, W)
        elif kind == "sam_cover" and sam is not None:
            out[:, p] = (sam.view(N, H, W) & valid).float()  # letterbox cells are never covered
    return out


def _unit_fields(layout: Layout, obs: dict[str, torch.Tensor]):
    units = obs["units"].long()  # [B, U, 8]
    cols = list(layout.raw["unitColumns"])
    utype = units[..., cols.index("type")]
    uowner = units[..., cols.index("owner")]
    ux = units[..., cols.index("x")].double()
    uy = units[..., cols.index("y")].double()
    rel = obs["relation"].long()
    own_ok = (uowner >= 0) & (uowner < rel.shape[1])
    urel = torch.gather(rel, 1, uowner.clamp(0, rel.shape[1] - 1))
    urel = torch.where(own_ok, urel, torch.zeros_like(urel))
    return utype, urel, ux, uy


def _sam_cover(
    t: PlaneTables, sam_p: int, utype, urel, ux, uy, n_views: int, views_per_row: int,
    centre_x: torch.Tensor, centre_y: torch.Tensor, tpc: torch.Tensor, sam_range: torch.Tensor,
) -> torch.Tensor | None:
    """[n_views, C] bool: cells within SAM range of a SAM the plane marks. View v belongs to row v // views_per_row.

    centre_*: [n_views, C] float64 cell centres in tiles; tpc, sam_range: [n_views]. Same float64 ops as the TS
    reference, so the disk edge agrees exactly."""
    ok = (utype >= 0) & t.unit_table[sam_p][utype.clamp(min=0)] & t.rel_table[sam_p][urel]
    b_idx, u_idx = ok.nonzero(as_tuple=True)
    if b_idx.numel() == 0:
        return None
    V = views_per_row
    dev = centre_x.device
    v_idx = (b_idx[:, None] * V + torch.arange(V, device=dev)[None, :]).reshape(-1)
    sx = ux[b_idx, u_idx].repeat_interleave(V)[:, None]
    sy = uy[b_idx, u_idx].repeat_interleave(V)[:, None]
    out = torch.zeros(n_views, centre_x.shape[1], dtype=torch.float32, device=dev)
    for c in range(0, v_idx.numel(), 2048):  # bounded memory on busy late-game batches
        v = v_idx[c : c + 2048]
        tp = tpc[v][:, None]
        r = sam_range[v][:, None] / tp
        dx = (centre_x[v] - sx[c : c + 2048]) / tp
        dy = (centre_y[v] - sy[c : c + 2048]) / tp
        out.index_add_(0, v, ((dx * dx + dy * dy) <= r * r).float())
    return out > 0


def expand_minimap(obs: dict[str, torch.Tensor], layout: Layout, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """[B, P, MM, MM] float planes of the minimap."""
    dev = obs["mm_owner"].device
    t = plane_tables(layout, dev)
    MM = int(layout.sizes["MM"])
    B = obs["mm_owner"].shape[0]
    tpc = _gm(layout, obs, "mm_tpc").double()
    off_x = _gm(layout, obs, "off_x").double()
    off_y = _gm(layout, obs, "off_y").double()
    sam_range = _gm(layout, obs, "sam_range").double()
    utype, urel, ux, uy = _unit_fields(layout, obs)
    cx = off_x[:, None] + torch.floor(ux / tpc[:, None])
    cy = off_y[:, None] + torch.floor(uy / tpc[:, None])
    inside = (cx >= 0) & (cy >= 0) & (cx < MM) & (cy < MM)
    cell = torch.where(inside, (cy * MM + cx).long(), torch.full_like(utype, -1))
    sam = None
    if "sam_cover" in t.kinds:
        grid = torch.arange(MM, device=dev, dtype=torch.float64)
        gx = grid[None, :].expand(MM, MM).reshape(-1)
        gy = grid[:, None].expand(MM, MM).reshape(-1)
        centre_x = (gx[None, :] - off_x[:, None] + 0.5) * tpc[:, None]
        centre_y = (gy[None, :] - off_y[:, None] + 0.5) * tpc[:, None]
        sam = _sam_cover(t, t.kinds.index("sam_cover"), utype, urel, ux, uy, B, 1, centre_x, centre_y, tpc, sam_range)
    return _expand(t, obs["mm_owner"].int(), obs["mm_terrain"], obs["relation"], obs["density"], cell, utype, urel,
                   sam).to(dtype)


def expand_windows(obs: dict[str, torch.Tensor], layout: Layout, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """[B, NUM_WINDOWS, P, WIN, WIN] float planes of the zoom windows (zeros for invalid windows)."""
    dev = obs["win_owner"].device
    t = plane_tables(layout, dev)
    WIN = int(layout.sizes["WIN"])
    NW = int(layout.sizes["NUM_WINDOWS"])
    B = obs["win_owner"].shape[0]
    wf = list(layout.raw["winMetaFields"])
    meta = obs["win_meta"].double()  # [B, NW, 6]
    x0 = meta[..., wf.index("x0")].reshape(-1)
    y0 = meta[..., wf.index("y0")].reshape(-1)
    tpc = meta[..., wf.index("tpc")].reshape(-1)
    wvalid = meta[..., wf.index("valid")].reshape(-1) != 0
    safe_tpc = torch.where(wvalid, tpc, torch.ones_like(tpc))
    view_of_row = torch.arange(B, device=dev).repeat_interleave(NW)
    utype, urel, ux, uy = _unit_fields(layout, obs)
    # Per view (row, window): unit cells.
    uxv = ux[view_of_row]
    uyv = uy[view_of_row]
    cx = torch.floor((uxv - x0[:, None]) / safe_tpc[:, None])
    cy = torch.floor((uyv - y0[:, None]) / safe_tpc[:, None])
    inside = (cx >= 0) & (cy >= 0) & (cx < WIN) & (cy < WIN) & wvalid[:, None]
    utv = utype[view_of_row]
    cell = torch.where(inside, (cy * WIN + cx).long(), torch.full_like(utv, -1))
    owner = obs["win_owner"].int().reshape(B * NW, WIN, WIN)
    owner = torch.where(wvalid[:, None, None], owner, torch.full_like(owner, LETTERBOX))
    terrain = obs["win_terrain"].reshape(B * NW, WIN, WIN)
    terrain = torch.where(wvalid[:, None, None], terrain, torch.zeros_like(terrain))
    sam = None
    if "sam_cover" in t.kinds:
        grid = torch.arange(WIN, device=dev, dtype=torch.float64)
        gx = grid[None, :].expand(WIN, WIN).reshape(-1)
        gy = grid[:, None].expand(WIN, WIN).reshape(-1)
        centre_x = x0[:, None] + (gx[None, :] + 0.5) * safe_tpc[:, None]
        centre_y = y0[:, None] + (gy[None, :] + 0.5) * safe_tpc[:, None]
        sam_range = _gm(layout, obs, "sam_range").double()[view_of_row]
        sam = _sam_cover(t, t.kinds.index("sam_cover"), utype, urel, ux, uy, B * NW, NW, centre_x, centre_y,
                         safe_tpc, sam_range)
    out = _expand(t, owner, terrain, obs["relation"][view_of_row], obs["density"][view_of_row], cell, utv,
                  urel[view_of_row], sam)
    return out.view(B, NW, len(t.kinds), WIN, WIN).to(dtype)
