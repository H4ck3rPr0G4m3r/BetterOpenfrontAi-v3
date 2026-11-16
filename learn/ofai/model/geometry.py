"""Tensor versions of the sim's location geometry (sim/src/obs/Geometry.ts), for
metrics that turn pointer outputs back into tiles."""

from __future__ import annotations

import torch


def minimap_cell_to_tile(cell: torch.Tensor, fine: torch.Tensor, map_w: torch.Tensor, map_h: torch.Tensor,
                         tpc: torch.Tensor, off_x: torch.Tensor, off_y: torch.Tensor, mm: int, fine_n: int) -> torch.Tensor:
    """Centre tile [n, 2] of fine spot `fine` inside minimap cell `cell` (MinimapGeometry.fineToTile)."""
    cell, fine = cell.long(), fine.long().clamp(min=0)
    cx = (cell % mm).double() - off_x
    cy = torch.div(cell, mm, rounding_mode="floor").double() - off_y
    x0, y0 = cx * tpc, cy * tpc
    x1 = torch.minimum(map_w, (cx + 1) * tpc)
    y1 = torch.minimum(map_h, (cy + 1) * tpc)
    fx = (fine % fine_n).double()
    fy = torch.div(fine, fine_n, rounding_mode="floor").double()
    x = torch.minimum(x1 - 1, x0 + torch.floor((fx + 0.5) * (x1 - x0) / fine_n))
    y = torch.minimum(y1 - 1, y0 + torch.floor((fy + 0.5) * (y1 - y0) / fine_n))
    return torch.stack([x, y], -1)


def window_cell_to_tile(x0: torch.Tensor, y0: torch.Tensor, tpc: torch.Tensor, cell: torch.Tensor, win: int) -> torch.Tensor:
    """Tile [n, 2] a window cell decodes to (windowCellToTile, without the map-edge check)."""
    cell = cell.long()
    x = x0 + (cell % win).double() * tpc + torch.floor(tpc / 2)
    y = y0 + torch.div(cell, win, rounding_mode="floor").double() * tpc + torch.floor(tpc / 2)
    return torch.stack([x, y], -1)
