import { DecodeStats, decodeStep, emptyAction, type DecodeContext } from "../../src/action/Codec";
import { type EngineGame } from "../../src/engine/Adapter";
import { GameObsCache } from "../../src/obs/GameObsCache";
import { A, BUILD_TYPES, F, K, LOC_FAR, LOC_NEAR, MM, T, TERRAIN_BITS, UNOWNED_TOKEN } from "../../src/obs/Layout";
import { ObsBuffer } from "../../src/obs/ObsBuffer";
import { encodeSeat } from "../../src/obs/SeatEncoder";
import { tileToWindowCell } from "../../src/obs/Geometry";
import type { RecordIntent } from "../../src/record/Record";
import { makeSyntheticRecord, seatClientID } from "../../src/record/SyntheticRecord";
import { ENGINE_ROOT, SMALL_MAP } from "../helpers";

let ctx: DecodeContext;
let g: EngineGame;
let cache: GameObsCache;

beforeAll(async () => {
  await makeSyntheticRecord({
    map: SMALL_MAP, seats: 2, turns: 3200, seed: "codec", engineRoot: ENGINE_ROOT,
    script: (eg, t) => {
      if (t !== 1550) return [];
      const p = eg.game.playerByClientID(seatClientID(0))!;
      const tiles = [...p.tiles()];
      return [{ type: "build_unit", clientID: seatClientID(0), unit: "City", tile: tiles[Math.floor(tiles.length / 2)] } as RecordIntent];
    },
    onTick: (eg, t) => {
      if (t !== 3199) return;
      g = eg;
      cache = new GameObsCache(eg);
      cache.update();
      const buf = new ObsBuffer(1);
      const tables = encodeSeat(cache, eg, seatClientID(0), buf, 0);
      ctx = { g: eg, clientID: seatClientID(0), tables, geometry: cache.geometry };
    },
  });
});

function step(...acts: Partial<Record<keyof typeof F, number>>[]): Int32Array {
  const out = new Int32Array(K * A).fill(-1);
  acts.forEach((a, k) => {
    const one = emptyAction();
    for (const [name, v] of Object.entries(a)) one[F[name as keyof typeof F]] = v!;
    out.set(one, k * A);
  });
  return out;
}

const self = () => g.game.playerByClientID(seatClientID(0))!;

test("attack decodes target and troops", () => {
  const other = g.game.playerByClientID(seatClientID(1))!;
  ctx.tables.players[5] = other.smallID();
  const [d] = decodeStep(step({ type: T.ATTACK, player: 5, troops: 9 }), ctx);
  expect("intent" in d && d.intent).toMatchObject({ type: "attack", targetID: other.id(), troops: Math.floor(self().troops() * 0.5) });
  ctx.tables.players[5] = null;
});

test("attack on unowned token sends null target", () => {
  const [d] = decodeStep(step({ type: T.ATTACK, player: UNOWNED_TOKEN, troops: 3 }), ctx);
  expect("intent" in d && d.intent).toMatchObject({ type: "attack", targetID: null, troops: Math.floor(self().troops() * 0.2) });
});

test("build snaps to nearest buildable tile", () => {
  // The cheapest structure we can afford now (costs grow with how many we own).
  const config = g.game.config();
  const b = [3, 2, 0].find((i) => self().gold() >= config.unitInfo(BUILD_TYPES[i] as never).cost(g.game, self()))!;
  expect(b).toBeDefined();
  const core = ctx.tables.windows[0];
  const w = g.game.width();
  const own = [...self().tiles()].find((t) => tileToWindowCell(core, t % w, Math.floor(t / w)) >= 0 && self().canBuild(BUILD_TYPES[b] as never, t) !== false)!;
  expect(own).toBeDefined();
  const cell = tileToWindowCell(core, own % w, Math.floor(own / w));
  const [d] = decodeStep(step({ type: T.BUILD, build: b, locMode: LOC_NEAR, window: 0, cell }), ctx);
  expect("intent" in d).toBe(true);
  if ("intent" in d) {
    expect(d.intent).toMatchObject({ type: "build_unit", unit: BUILD_TYPES[b] });
    expect(self().canBuild(BUILD_TYPES[b] as never, d.intent.tile as number)).not.toBe(false);
    const t = d.intent.tile as number;
    expect(Math.hypot(t % w - own % w, Math.floor(t / w) - Math.floor(own / w))).toBeLessThanOrEqual(2 * core.tpc + 1);
  }
});

test("unsnappable build is dropped and counted", () => {
  // A minimap cell with no land around it.
  let water = -1;
  for (let c = 0; c < MM * MM && water < 0; c++) {
    if (cache.geometry.isLetterbox(c)) continue;
    const r = cache.geometry.tileRectOfCell(c)!;
    let land = false;
    for (let y = Math.max(0, r.y0 - 4); y < Math.min(g.game.height(), r.y1 + 4) && !land; y++)
      for (let x = Math.max(0, r.x0 - 4); x < Math.min(g.game.width(), r.x1 + 4) && !land; x++) land = g.game.isLand(g.game.ref(x, y));
    if (!land && !(cache.mmTerrain[c] & TERRAIN_BITS.land)) water = c;
  }
  expect(water).toBeGreaterThanOrEqual(0);
  const stats = new DecodeStats();
  const [d] = decodeStep(step({ type: T.BUILD, build: 0, locMode: LOC_FAR, mmCell: water, fine: 0 }), ctx, stats);
  expect(d).toEqual({ dropped: "unsnappable", type: "BUILD" });
  expect(stats.dropped.BUILD.unsnappable).toBe(1);
});

test("stale entity is dropped", () => {
  expect(ctx.tables.structs.length).toBeGreaterThan(0);
  const unit = g.game.unit(ctx.tables.structs[0])!;
  const tables = { ...ctx.tables, structs: [999_999] };
  const [d] = decodeStep(step({ type: T.UPGRADE, entity: 0 }), { ...ctx, tables });
  expect(d).toEqual({ dropped: "stale_entity", type: "UPGRADE" });
  const [ok] = decodeStep(step({ type: T.UPGRADE, entity: 0 }), ctx);
  expect("intent" in ok && ok.intent).toMatchObject({ type: "upgrade_structure", unitId: unit.id(), unit: unit.type() });
});

test("actions after STOP are ignored", () => {
  const out = decodeStep(step({ type: T.ATTACK, player: UNOWNED_TOKEN, troops: 0 }, { type: T.STOP }, { type: T.ATTACK, player: UNOWNED_TOKEN, troops: 1 }), ctx);
  expect(out).toHaveLength(1);
});

test("never retargets to another player", () => {
  const stats = new DecodeStats();
  ctx.tables.players[7] = 4000; // a smallID nobody has
  const out = decodeStep(step({ type: T.ATTACK, player: 7, troops: 2 }, { type: T.ALLIANCE_REQUEST, player: 8 }), ctx, stats);
  expect(out).toEqual([{ dropped: "bad_pointer", type: "ATTACK" }, { dropped: "bad_pointer", type: "ALLIANCE_REQUEST" }]);
  ctx.tables.players[7] = null;
});
