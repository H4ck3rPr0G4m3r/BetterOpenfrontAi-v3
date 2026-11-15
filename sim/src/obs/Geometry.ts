/**
 * Map-shape-agnostic geometry (spec §9.2-9.3, §10.3).
 *
 * Minimap: the whole map at a whole number of tiles per cell
 * (tpc = ceil(max(w, h) / MM)), centred in the MM x MM grid. The shape is kept
 * (portrait maps stay portrait) and the cells outside the map are letterbox.
 * Far pointer: a minimap cell plus a FINE x FINE spot inside its tile rect.
 * Windows: WIN x WIN cells at 2-4 tiles per cell, always shifted inside the map.
 */
import { FINE, MM, WIN } from "./Layout";

export interface TileRect {
  x0: number; // inclusive
  y0: number;
  x1: number; // exclusive
  y1: number;
}

export class MinimapGeometry {
  readonly tpc: number;
  /** Map size in minimap cells. */
  readonly cellsW: number;
  readonly cellsH: number;
  /** Letterbox offset in cells. */
  readonly offX: number;
  readonly offY: number;

  constructor(readonly width: number, readonly height: number) {
    if (width <= 0 || height <= 0) throw new Error(`bad map size ${width}x${height}`);
    this.tpc = Math.max(1, Math.ceil(Math.max(width, height) / MM));
    this.cellsW = Math.ceil(width / this.tpc);
    this.cellsH = Math.ceil(height / this.tpc);
    this.offX = Math.floor((MM - this.cellsW) / 2);
    this.offY = Math.floor((MM - this.cellsH) / 2);
  }

  cellOfTile(x: number, y: number): number {
    return (this.offY + Math.floor(y / this.tpc)) * MM + this.offX + Math.floor(x / this.tpc);
  }

  isLetterbox(cell: number): boolean {
    const cx = (cell % MM) - this.offX;
    const cy = Math.floor(cell / MM) - this.offY;
    return cx < 0 || cy < 0 || cx >= this.cellsW || cy >= this.cellsH;
  }

  tileRectOfCell(cell: number): TileRect | null {
    if (cell < 0 || cell >= MM * MM || this.isLetterbox(cell)) return null;
    const cx = (cell % MM) - this.offX;
    const cy = Math.floor(cell / MM) - this.offY;
    return {
      x0: cx * this.tpc,
      y0: cy * this.tpc,
      x1: Math.min(this.width, (cx + 1) * this.tpc),
      y1: Math.min(this.height, (cy + 1) * this.tpc),
    };
  }

  /** Centre tile of fine spot `fine` (row-major FINE x FINE) inside `cell`. */
  fineToTile(cell: number, fine: number): { x: number; y: number } | null {
    const r = this.tileRectOfCell(cell);
    if (r === null || fine < 0 || fine >= FINE * FINE) return null;
    const fx = fine % FINE;
    const fy = Math.floor(fine / FINE);
    const w = r.x1 - r.x0;
    const h = r.y1 - r.y0;
    return {
      x: Math.min(r.x1 - 1, r.x0 + Math.floor(((fx + 0.5) * w) / FINE)),
      y: Math.min(r.y1 - 1, r.y0 + Math.floor(((fy + 0.5) * h) / FINE)),
    };
  }

  /** The fine spot of a tile inside its own minimap cell. */
  tileToFine(x: number, y: number): { cell: number; fine: number } {
    const cell = this.cellOfTile(x, y);
    const r = this.tileRectOfCell(cell)!;
    const fx = Math.min(FINE - 1, Math.floor(((x - r.x0) * FINE) / (r.x1 - r.x0)));
    const fy = Math.min(FINE - 1, Math.floor(((y - r.y0) * FINE) / (r.y1 - r.y0)));
    return { cell, fine: fy * FINE + fx };
  }

  /** Largest distance in tiles between a tile and the centre of its fine spot. */
  fineSpan(): number {
    return Math.max(1, Math.ceil(this.tpc / FINE));
  }
}

export type WindowKind = "core" | "front" | "coast" | "nuke";
export const WINDOW_KIND_CODE: Record<WindowKind, number> = { core: 0, front: 1, coast: 2, nuke: 3 };

export interface WindowSpec {
  x0: number;
  y0: number;
  tpc: 2 | 3 | 4;
  kind: WindowKind;
  priority: number;
}

export function windowSpan(w: Pick<WindowSpec, "tpc">): number {
  return WIN * w.tpc;
}

/** Shifts a window inside the map (never clipped). A map smaller than the window pins it to 0. */
export function clampWindow<W extends WindowSpec>(w: W, mapW: number, mapH: number): W {
  const span = windowSpan(w);
  return {
    ...w,
    x0: Math.max(0, Math.min(Math.round(w.x0), Math.max(0, mapW - span))),
    y0: Math.max(0, Math.min(Math.round(w.y0), Math.max(0, mapH - span))),
  };
}

/** Window centred on a tile, then clamped. */
export function windowAround(cx: number, cy: number, tpc: 2 | 3 | 4, kind: WindowKind, priority: number, mapW: number, mapH: number): WindowSpec {
  const half = windowSpan({ tpc }) / 2;
  return clampWindow({ x0: cx - half, y0: cy - half, tpc, kind, priority }, mapW, mapH);
}

/** Centre tile of a window cell, or null when that cell lies outside a map smaller than the window. */
export function windowCellToTile(w: WindowSpec, cell: number, mapW: number, mapH: number): { x: number; y: number } | null {
  if (cell < 0 || cell >= WIN * WIN) return null;
  const x = w.x0 + (cell % WIN) * w.tpc + Math.floor(w.tpc / 2);
  const y = w.y0 + Math.floor(cell / WIN) * w.tpc + Math.floor(w.tpc / 2);
  if (x >= mapW || y >= mapH) return null;
  return { x, y };
}

export function tileToWindowCell(w: WindowSpec, x: number, y: number): number {
  const cx = Math.floor((x - w.x0) / w.tpc);
  const cy = Math.floor((y - w.y0) / w.tpc);
  if (cx < 0 || cy < 0 || cx >= WIN || cy >= WIN) return -1;
  return cy * WIN + cx;
}

export function windowContains(w: WindowSpec, x: number, y: number): boolean {
  return tileToWindowCell(w, x, y) >= 0;
}
