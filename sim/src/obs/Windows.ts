/**
 * Deterministic zoom-window placement (spec §9.3). Same game state, same
 * windows, so record replay and live play see identical views.
 *
 * Order (= priority): 1 core territory; 2 each active front, most troops
 * first; 3 the largest own coastal stretch; 4 hostile nukes heading for our
 * land. Windows whose centres are closer than MERGE_TILES are merged (the
 * earlier one wins); all are shifted inside the map; at most NUM_WINDOWS.
 * No windows before the seat has land (spawn uses the far pointer).
 */
import type { Attack, EngineGame, Player } from "../engine/Adapter";
import { UnitType } from "../engine/Adapter";
import type { GameObsCache } from "./GameObsCache";
import { windowAround, type WindowKind, type WindowSpec } from "./Geometry";
import { MM, NUM_WINDOWS, TERRAIN_BITS } from "./Layout";

export const MERGE_TILES = 16;
const NUKE_TYPES = [UnitType.AtomBomb, UnitType.HydrogenBomb, UnitType.MIRV, UnitType.MIRVWarhead];

interface Candidate {
  cx: number;
  cy: number;
  tpc: 2 | 3 | 4;
  kind: WindowKind;
}

function tpcForSpan(span: number): 2 | 3 | 4 {
  return span <= 64 ? 2 : span <= 96 ? 3 : 4;
}

/** Centroid and span of a tile set, in tiles. */
function tileStats(tiles: Iterable<number>, mapW: number): { cx: number; cy: number; span: number } | null {
  let n = 0;
  let sx = 0;
  let sy = 0;
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const t of tiles) {
    const x = t % mapW;
    const y = (t - x) / mapW;
    n++;
    sx += x;
    sy += y;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  if (n === 0) return null;
  return { cx: sx / n, cy: sy / n, span: Math.max(maxX - minX, maxY - minY) + 1 };
}

/** Own minimap cells adjacent to `other`'s cells (other 0 = unowned land). */
function frontCells(cache: GameObsCache, self: number, other: number): number[] {
  const own = cache.ownerCells.get(self) ?? [];
  const out: number[] = [];
  for (const c of own) {
    const col = c % MM;
    const n = [col > 0 ? c - 1 : -1, col < MM - 1 ? c + 1 : -1, c - MM, c + MM];
    for (const d of n) {
      if (d < 0 || d >= MM * MM) continue;
      if (cache.mmOwner[d] === other && cache.mmTerrain[d] & TERRAIN_BITS.land) {
        out.push(c);
        break;
      }
    }
  }
  return out;
}

function cellsStats(cache: GameObsCache, cells: number[]): { cx: number; cy: number; span: number } | null {
  if (cells.length === 0) return null;
  const geo = cache.geometry;
  let sx = 0;
  let sy = 0;
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const c of cells) {
    const x = c % MM;
    const y = Math.floor(c / MM);
    sx += x;
    sy += y;
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  const cx = (sx / cells.length - geo.offX + 0.5) * geo.tpc;
  const cy = (sy / cells.length - geo.offY + 0.5) * geo.tpc;
  return { cx, cy, span: (Math.max(maxX - minX, maxY - minY) + 1) * geo.tpc };
}

function attackFront(cache: GameObsCache, a: Attack, self: Player, outgoing: boolean, mapW: number): Candidate | null {
  const border = tileStats(a.borderTiles(), mapW);
  if (border !== null) return { cx: border.cx, cy: border.cy, tpc: border.span <= 64 ? 2 : 4, kind: "front" };
  const other = outgoing ? a.target() : a.attacker();
  const otherId = other.isPlayer() ? other.smallID() : 0;
  const s = cellsStats(cache, frontCells(cache, self.smallID(), otherId));
  if (s === null) return null;
  return { cx: s.cx, cy: s.cy, tpc: s.span <= 64 ? 2 : 4, kind: "front" };
}

/** Largest 4-connected cluster of own coastal minimap cells. */
function coastCluster(cache: GameObsCache, self: number): number[] {
  const own = cache.ownerCells.get(self) ?? [];
  const coastal = new Set(own.filter((c) => cache.mmTerrain[c] & TERRAIN_BITS.coast));
  const seen = new Set<number>();
  let best: number[] = [];
  for (const start of [...coastal].sort((a, b) => a - b)) {
    if (seen.has(start)) continue;
    const comp: number[] = [];
    const stack = [start];
    seen.add(start);
    while (stack.length) {
      const c = stack.pop()!;
      comp.push(c);
      const col = c % MM;
      for (const d of [col > 0 ? c - 1 : -1, col < MM - 1 ? c + 1 : -1, c - MM, c + MM]) {
        if (d >= 0 && coastal.has(d) && !seen.has(d)) {
          seen.add(d);
          stack.push(d);
        }
      }
    }
    if (comp.length > best.length) best = comp;
  }
  return best;
}

export function mergeWindows(windows: WindowSpec[], mergeTiles = MERGE_TILES): WindowSpec[] {
  const out: WindowSpec[] = [];
  const centre = (w: WindowSpec) => ({ x: w.x0 + 16 * w.tpc, y: w.y0 + 16 * w.tpc });
  for (const w of windows) {
    const c = centre(w);
    if (out.some((o) => Math.hypot(centre(o).x - c.x, centre(o).y - c.y) < mergeTiles)) continue;
    out.push(w);
  }
  return out;
}

export function placeWindows(cache: GameObsCache, g: EngineGame, selfSmallId: number): WindowSpec[] {
  const game = g.game;
  const mapW = game.width();
  const mapH = game.height();
  const self = cache.bySmallId.get(selfSmallId);
  if (self === undefined || !self.isAlive() || self.numTilesOwned() === 0) return [];
  const cands: Candidate[] = [];

  // 1. core: the largest cluster's bounding box when the engine has it, else the minimap cells.
  const box = self.largestClusterBoundingBox;
  if (box) {
    const span = Math.max(box.max.x - box.min.x, box.max.y - box.min.y) + 1;
    cands.push({ cx: (box.min.x + box.max.x) / 2, cy: (box.min.y + box.max.y) / 2, tpc: tpcForSpan(span), kind: "core" });
  } else {
    const s = cellsStats(cache, cache.ownerCells.get(selfSmallId) ?? []);
    const spawn = self.spawnTile();
    if (s !== null) cands.push({ cx: s.cx, cy: s.cy, tpc: tpcForSpan(s.span), kind: "core" });
    else if (spawn !== undefined) cands.push({ cx: spawn % mapW, cy: Math.floor(spawn / mapW), tpc: 2, kind: "core" });
  }

  // 2. fronts, most troops first (ties: outgoing first, then attack id).
  const attacks = [
    ...self.outgoingAttacks().map((a) => ({ a, out: true })),
    ...self.incomingAttacks().map((a) => ({ a, out: false })),
  ].filter(({ a }) => a.isActive());
  attacks.sort((x, y) => y.a.troops() - x.a.troops() || Number(y.out) - Number(x.out) || (x.a.id() < y.a.id() ? -1 : 1));
  for (const { a, out } of attacks) {
    const c = attackFront(cache, a, self, out, mapW);
    if (c !== null) cands.push(c);
  }

  // 3. coast.
  const coast = cellsStats(cache, coastCluster(cache, selfSmallId));
  if (coast !== null) cands.push({ cx: coast.cx, cy: coast.cy, tpc: 3, kind: "coast" });

  // 4. hostile nukes aimed within 2 minimap cells of our land.
  const geo = cache.geometry;
  for (const u of game.units(...NUKE_TYPES)) {
    if (!u.isActive() || u.owner() === self || self.isFriendly(u.owner())) continue;
    const target = u.targetTile();
    if (target === undefined) continue;
    const tx = target % mapW;
    const ty = Math.floor(target / mapW);
    const tc = geo.cellOfTile(tx, ty);
    let near = false;
    for (let dy = -2; dy <= 2 && !near; dy++) {
      for (let dx = -2; dx <= 2 && !near; dx++) {
        const c = tc + dy * MM + dx;
        if (c >= 0 && c < MM * MM && cache.mmOwner[c] === selfSmallId) near = true;
      }
    }
    if (near) cands.push({ cx: tx, cy: ty, tpc: 2, kind: "nuke" });
  }

  const placed = cands.map((c, i) => windowAround(c.cx, c.cy, c.tpc, c.kind, i, mapW, mapH));
  return mergeWindows(placed)
    .slice(0, NUM_WINDOWS)
    .map((w, i) => ({ ...w, priority: i }));
}
