/**
 * The seat-independent part of the observation, computed once per decision
 * step per game and shared by every seat in it (spec §9.6): minimap owner
 * and terrain rasters, per-player troop density, the unit list, minimap
 * border adjacency between owners, and per-owner cell lists.
 */
import type { EngineGame, Player, Unit } from "../engine/Adapter";
import { TerrainType, UnitType } from "../engine/Adapter";
import { toF16 } from "./f16";
import { MinimapGeometry } from "./Geometry";
import { MAX_SMALL_ID, MAX_UNITS, MM, TERRAIN_BITS, UNIT_COLS, UNIT_FLAGS, UNIT_TYPES } from "./Layout";

export const LETTERBOX = 0xffff;
// Owner per cell: plurality among the *owned* land samples of a SUB x SUB grid (ties to the lower
// smallID); unowned only when no sample is owned. A plain majority hid small players on big maps
// (35 tiles per cell on Sol: a fresh spawn never outvotes the unowned land around it).
const SUB = 6;
const TERRAIN_SUB = 4;

const UNIT_CODE = new Map<string, number>(UNIT_TYPES.map((t, i) => [t, i]));
const NUKES = new Set<string>([UnitType.AtomBomb, UnitType.HydrogenBomb, UnitType.MIRV, UnitType.MIRVWarhead]);
const STRUCTURES = new Set<string>([
  UnitType.City, UnitType.Port, UnitType.Factory, UnitType.DefensePost, UnitType.SAMLauncher, UnitType.MissileSilo,
]);
const LISTED_TYPES = [
  UnitType.AtomBomb, UnitType.HydrogenBomb, UnitType.MIRV, UnitType.MIRVWarhead,
  UnitType.Warship, UnitType.TransportShip,
  UnitType.City, UnitType.Port, UnitType.Factory, UnitType.DefensePost, UnitType.SAMLauncher, UnitType.MissileSilo,
];

/** Pair key for two smallIDs, order-free. */
export function pairKey(a: number, b: number): number {
  return a < b ? a * 4096 + b : b * 4096 + a;
}

export interface PairBorder {
  /** Number of adjacent minimap cell pairs. */
  count: number;
  /** Land terrain of the cells along the border: plains, highland, mountain. */
  terrain: [number, number, number];
}

export function clampSmallId(id: number): number {
  return id >= MAX_SMALL_ID ? MAX_SMALL_ID - 1 : id;
}

export class GameObsCache {
  readonly geometry: MinimapGeometry;
  readonly mmOwner = new Uint16Array(MM * MM);
  readonly mmTerrain = new Uint8Array(MM * MM);
  /** Per-cell majority land terrain: 0 plains, 1 highland, 2 mountain, 255 water/letterbox. */
  readonly mmLandType = new Uint8Array(MM * MM);
  readonly density = new Uint16Array(MAX_SMALL_ID); // f16 bits
  readonly densityF32 = new Float32Array(MAX_SMALL_ID);
  readonly units = new Int16Array(MAX_UNITS * UNIT_COLS);
  unitCount = 0;
  /** Units in list order (unit list row -> engine unit). */
  listedUnits: Unit[] = [];
  /** smallID -> Map(other smallID (0 = unowned land) -> border). */
  adjacency = new Map<number, Map<number, PairBorder>>();
  ownerCells = new Map<number, number[]>();
  /** smallID -> centroid in tiles. */
  centroids = new Map<number, { x: number; y: number }>();
  players: Player[] = [];
  bySmallId = new Map<number, Player>();
  totalLand = 1;
  ticks = -1;
  private readonly sampleOffsets: number[][] = [];

  constructor(readonly g: EngineGame) {
    const game = g.game;
    this.geometry = new MinimapGeometry(game.width(), game.height());
    this.computeTerrain();
    this.totalLand = Math.max(1, game.map().numLandTiles());
  }

  private computeTerrain(): void {
    const map = this.g.game.map();
    const w = map.width();
    const geo = this.geometry;
    this.mmLandType.fill(255);
    for (let cell = 0; cell < MM * MM; cell++) {
      const r = geo.tileRectOfCell(cell);
      if (r === null) {
        this.mmTerrain[cell] = 0;
        this.mmOwner[cell] = LETTERBOX;
        this.sampleOffsets.push([]);
        continue;
      }
      let land = 0;
      let water = 0;
      let coast = false;
      let shoreWater = false;
      let ocean = 0;
      const kinds = [0, 0, 0];
      const tw = r.x1 - r.x0;
      const th = r.y1 - r.y0;
      const nx = Math.min(TERRAIN_SUB, tw);
      const ny = Math.min(TERRAIN_SUB, th);
      for (let sy = 0; sy < ny; sy++) {
        for (let sx = 0; sx < nx; sx++) {
          const x = r.x0 + Math.floor(((sx + 0.5) * tw) / nx);
          const y = r.y0 + Math.floor(((sy + 0.5) * th) / ny);
          const t = y * w + x;
          if (map.isLand(t)) {
            land++;
            if (map.isShoreline(t)) coast = true;
            const tt = map.terrainType(t);
            if (tt === TerrainType.Highland) kinds[1]++;
            else if (tt === TerrainType.Mountain) kinds[2]++;
            else kinds[0]++;
          } else {
            water++;
            if (map.isShoreline(t)) shoreWater = true;
            if (map.isOcean(t)) ocean++;
          }
        }
      }
      let bits = 0;
      if (land >= water && land > 0) {
        bits |= TERRAIN_BITS.land;
        if (coast) bits |= TERRAIN_BITS.coast;
        const k = kinds[2] > kinds[1] && kinds[2] > kinds[0] ? 2 : kinds[1] > kinds[0] ? 1 : 0;
        this.mmLandType[cell] = k;
        if (k === 1) bits |= TERRAIN_BITS.highland;
        if (k === 2) bits |= TERRAIN_BITS.mountain;
      } else {
        if (shoreWater || land > 0) bits |= TERRAIN_BITS.shoreWater;
        if (ocean * 2 >= water) bits |= TERRAIN_BITS.ocean;
      }
      this.mmTerrain[cell] = bits;
      // Owner sample points for this cell.
      const sxn = Math.min(SUB, tw);
      const syn = Math.min(SUB, th);
      const pts: number[] = [];
      for (let sy = 0; sy < syn; sy++) {
        for (let sx = 0; sx < sxn; sx++) {
          pts.push((r.y0 + Math.floor(((sy + 0.5) * th) / syn)) * w + r.x0 + Math.floor(((sx + 0.5) * tw) / sxn));
        }
      }
      this.sampleOffsets.push(pts);
    }
  }

  /** Recompute everything that changes over time. Call once per decision step. */
  update(): void {
    const game = this.g.game;
    const map = game.map();
    this.ticks = game.ticks();

    // Players.
    this.players = game.allPlayers();
    this.bySmallId.clear();
    this.densityF32.fill(0);
    for (const p of this.players) {
      const id = p.smallID();
      this.bySmallId.set(id, p);
      const tiles = p.numTilesOwned();
      if (tiles > 0 && id < MAX_SMALL_ID) this.densityF32[id] = Math.min(1, Math.log1p(p.troops() / tiles) / 8);
    }
    for (let i = 0; i < MAX_SMALL_ID; i++) this.density[i] = toF16(this.densityF32[i]);

    // Owner raster (see SUB).
    const counts = new Map<number, number>();
    this.ownerCells = new Map();
    const sumX = new Map<number, number>();
    const sumY = new Map<number, number>();
    for (let cell = 0; cell < MM * MM; cell++) {
      const pts = this.sampleOffsets[cell];
      if (pts.length === 0) continue; // letterbox stays 0xFFFF
      if (!(this.mmTerrain[cell] & TERRAIN_BITS.land)) {
        this.mmOwner[cell] = 0;
        continue;
      }
      counts.clear();
      for (const t of pts) {
        const o = map.ownerID(t);
        if (o === 0 || !map.isLand(t)) continue;
        counts.set(o, (counts.get(o) ?? 0) + 1);
      }
      let best = 0;
      let bestN = -1;
      for (const [o, n] of counts) {
        if (n > bestN || (n === bestN && o < best)) {
          best = o;
          bestN = n;
        }
      }
      const owner = best === 0 ? 0 : clampSmallId(best);
      this.mmOwner[cell] = owner;
      if (owner !== 0) {
        let list = this.ownerCells.get(owner);
        if (!list) this.ownerCells.set(owner, (list = []));
        list.push(cell);
        sumX.set(owner, (sumX.get(owner) ?? 0) + (cell % MM));
        sumY.set(owner, (sumY.get(owner) ?? 0) + Math.floor(cell / MM));
      }
    }
    this.centroids = new Map();
    const geo = this.geometry;
    for (const [owner, cells] of this.ownerCells) {
      const cx = sumX.get(owner)! / cells.length - geo.offX + 0.5;
      const cy = sumY.get(owner)! / cells.length - geo.offY + 0.5;
      this.centroids.set(owner, { x: Math.min(game.width() - 1, cx * geo.tpc), y: Math.min(game.height() - 1, cy * geo.tpc) });
    }

    // Border adjacency between land cells of different owners (0 = unowned land).
    this.adjacency = new Map();
    const add = (a: number, b: number, cellA: number, cellB: number) => {
      for (const [x, y] of [[a, b], [b, a]]) {
        let m = this.adjacency.get(x);
        if (!m) this.adjacency.set(x, (m = new Map()));
        let e = m.get(y);
        if (!e) m.set(y, (e = { count: 0, terrain: [0, 0, 0] }));
        e.count++;
        for (const c of [cellA, cellB]) {
          const k = this.mmLandType[c];
          if (k < 3) e.terrain[k as 0 | 1 | 2]++;
        }
      }
    };
    for (let cell = 0; cell < MM * MM; cell++) {
      if (!(this.mmTerrain[cell] & TERRAIN_BITS.land)) continue;
      const a = this.mmOwner[cell];
      const col = cell % MM;
      if (col + 1 < MM) {
        const right = cell + 1;
        if (this.mmTerrain[right] & TERRAIN_BITS.land) {
          const b = this.mmOwner[right];
          if (a !== b && b !== LETTERBOX) add(a, b, cell, right);
        }
      }
      const down = cell + MM;
      if (down < MM * MM && this.mmTerrain[down] & TERRAIN_BITS.land) {
        const b = this.mmOwner[down];
        if (a !== b && b !== LETTERBOX) add(a, b, cell, down);
      }
    }

    this.buildUnitList();
  }

  private buildUnitList(): void {
    const game = this.g.game;
    const w = game.width();
    this.listedUnits = rankUnits(game.units(), MAX_UNITS);
    this.unitCount = this.listedUnits.length;
    this.units.fill(-1);
    for (let i = 0; i < this.unitCount; i++) {
      const u = this.listedUnits[i];
      const o = i * UNIT_COLS;
      const tile = u.tile();
      this.units[o] = UNIT_CODE.get(u.type()) ?? -1;
      this.units[o + 1] = clampSmallId(u.owner().smallID());
      this.units[o + 2] = tile % w;
      this.units[o + 3] = Math.floor(tile / w);
      this.units[o + 4] = STRUCTURES.has(u.type()) ? u.level() : 0;
      let flags = 0;
      if (STRUCTURES.has(u.type()) && u.isUnderConstruction()) flags |= UNIT_FLAGS.underConstruction;
      if ((u.type() === UnitType.SAMLauncher || u.type() === UnitType.MissileSilo) && u.isInCooldown()) flags |= UNIT_FLAGS.inCooldown;
      if (NUKES.has(u.type()) && u.targetedBySAM()) flags |= UNIT_FLAGS.targetedBySAM;
      this.units[o + 5] = flags;
      const target = u.targetTile();
      this.units[o + 6] = target === undefined ? -1 : target % w;
      this.units[o + 7] = target === undefined ? -1 : Math.floor(target / w);
    }
  }

  /** Per-game wire meta: map_w, map_h, mm_tpc, ticks, spawn_phase, off_x, off_y, sam_range. */
  gameMeta(out: Float32Array): void {
    const game = this.g.game;
    out[0] = game.width();
    out[1] = game.height();
    out[2] = this.geometry.tpc;
    out[3] = game.ticks();
    out[4] = game.inSpawnPhase() ? 1 : 0;
    out[5] = this.geometry.offX;
    out[6] = this.geometry.offY;
    out[7] = game.config().defaultSamRange();
  }
}

/** Units in overflow priority: nukes in flight, warships, transports, then structures; trains etc. excluded. */
export function rankUnits<U extends { type(): string; isActive(): boolean }>(units: readonly U[], max: number): U[] {
  const byType = new Map<string, U[]>();
  for (const u of units) {
    if (!u.isActive()) continue;
    let list = byType.get(u.type());
    if (!list) byType.set(u.type(), (list = []));
    list.push(u);
  }
  const out: U[] = [];
  for (const t of LISTED_TYPES) {
    for (const u of byType.get(t) ?? []) {
      if (out.length >= max) return out;
      out.push(u);
    }
  }
  return out;
}

export function isNukeType(t: string): boolean {
  return NUKES.has(t);
}

export function isStructureType(t: string): boolean {
  return STRUCTURES.has(t);
}
