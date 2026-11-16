/**
 * CPU reference for the GPU plane expansion (learn/ofai/model/planes.py).
 * It defines what each plane means; the Python version must match it exactly
 * (M3 Task 1 equivalence test, fixture from `sim.mjs dump-obs`).
 *
 * A "view" is the minimap or one zoom window: an owner raster, a terrain
 * raster, and how a tile maps to a view cell. Unit planes mark the cell
 * holding each unit; SAM cover marks every cell whose centre lies within the
 * SAM range (tiles / view tpc) of a hostile SAM's cell centre.
 */
import { fromF16 } from "../f16";
import { MAX_SMALL_ID, PLANES, TERRAIN_BITS, UNIT_COLS, UNIT_TYPES } from "../Layout";

export interface PlaneView {
  w: number; // cells
  h: number;
  owner: Uint16Array; // w*h, 0xFFFF = invalid
  terrain: Uint8Array; // w*h
  tpc: number;
  /** View cell of a tile, or -1 when outside the view. */
  cellOf: (x: number, y: number) => number;
  /** Centre of a view cell in tile coordinates (float, not floored). */
  centre: (cx: number, cy: number) => { x: number; y: number };
}

/** Minimap view: cell (cx, cy) covers tiles [(cx - offX) * tpc, ...); centre = (cx - offX + 0.5) * tpc. */
export function minimapView(owner: Uint16Array, terrain: Uint8Array, mm: number, tpc: number, offX: number, offY: number): PlaneView {
  return {
    w: mm,
    h: mm,
    owner,
    terrain,
    tpc,
    cellOf: (x, y) => (offY + Math.floor(y / tpc)) * mm + offX + Math.floor(x / tpc),
    centre: (cx, cy) => ({ x: (cx - offX + 0.5) * tpc, y: (cy - offY + 0.5) * tpc }),
  };
}

/** Window view: cell (cx, cy) covers tiles [x0 + cx * tpc, ...); centre = x0 + (cx + 0.5) * tpc. */
export function windowView(owner: Uint16Array, terrain: Uint8Array, win: number, x0: number, y0: number, tpc: number): PlaneView {
  return {
    w: win,
    h: win,
    owner,
    terrain,
    tpc,
    cellOf: (x, y) => {
      const cx = Math.floor((x - x0) / tpc);
      const cy = Math.floor((y - y0) / tpc);
      return cx < 0 || cy < 0 || cx >= win || cy >= win ? -1 : cy * win + cx;
    },
    centre: (cx, cy) => ({ x: x0 + (cx + 0.5) * tpc, y: y0 + (cy + 0.5) * tpc }),
  };
}

export function expandView(
  view: PlaneView,
  relation: Uint8Array,
  densityF16: Uint16Array,
  units: Int16Array,
  samRangeTiles: number,
): Float32Array {
  const n = view.w * view.h;
  const out = new Float32Array(PLANES.length * n);
  const rel = (owner: number) => (owner < MAX_SMALL_ID ? relation[owner] : 0);
  PLANES.forEach((p, pi) => {
    const base = pi * n;
    if (p.from === "terrain" || p.from === "owner" || p.from === "density" || p.from === "valid") {
      for (let c = 0; c < n; c++) {
        const o = view.owner[c];
        const valid = o !== 0xffff;
        const t = view.terrain[c];
        let v = 0;
        if (p.from === "valid") v = valid ? 1 : 0;
        else if (p.from === "terrain") {
          if (p.bit === 0) v = valid && !(t & TERRAIN_BITS.land) ? 1 : 0;
          else v = t & p.bit! ? 1 : 0;
        } else if (!valid) v = 0;
        else if (p.ownerUnowned) v = o === 0 && t & TERRAIN_BITS.land ? 1 : 0;
        else if (o !== 0 && p.rel!.includes(rel(o))) v = p.from === "density" ? fromF16(densityF16[o]) : 1;
        out[base + c] = v;
      }
      return;
    }
    const codes = new Set(p.unit!.map((u) => UNIT_TYPES.indexOf(u as (typeof UNIT_TYPES)[number])));
    for (let i = 0; i < units.length / UNIT_COLS; i++) {
      const o = i * UNIT_COLS;
      const type = units[o];
      if (type < 0 || !codes.has(type)) continue;
      if (!p.rel!.includes(rel(units[o + 1]))) continue;
      if (p.from === "units") {
        const c = view.cellOf(units[o + 2], units[o + 3]);
        if (c >= 0) out[base + c] = 1;
        continue;
      }
      // sam_cover: disk around the SAM's position, radius in view cells.
      const r = samRangeTiles / view.tpc;
      const r2 = r * r;
      const sx = units[o + 2];
      const sy = units[o + 3];
      for (let c = 0; c < n; c++) {
        const cx = c % view.w;
        const cy = Math.floor(c / view.w);
        if (view.owner[c] === 0xffff) continue;
        const center = view.centre(cx, cy);
        const dx = (center.x - sx) / view.tpc;
        const dy = (center.y - sy) / view.tpc;
        if (dx * dx + dy * dy <= r2) out[base + c] = 1;
      }
    }
  });
  return out;
}

