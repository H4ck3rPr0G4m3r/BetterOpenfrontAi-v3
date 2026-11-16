import { decodeAction } from "../../src/action/Codec";
import { type EngineGame } from "../../src/engine/Adapter";
import { GameObsCache } from "../../src/obs/GameObsCache";
import { A, F, K, T, UNOWNED_TOKEN } from "../../src/obs/Layout";
import { ObsBuffer } from "../../src/obs/ObsBuffer";
import { encodeSeat } from "../../src/obs/SeatEncoder";
import type { RecordIntent } from "../../src/record/Record";
import { makeSyntheticRecord, seatClientID } from "../../src/record/SyntheticRecord";
import { compareRoundTrip, reportRecord } from "../../src/translate/Report";
import { StepLabeler, translateIntent, troopBucket, type TranslateContext } from "../../src/translate/Translator";
import { ENGINE_ROOT, SMALL_MAP } from "../helpers";

let g: EngineGame;
let ctx: TranslateContext;
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
      const cache = new GameObsCache(eg);
      cache.update();
      const tables = encodeSeat(cache, eg, seatClientID(0), new ObsBuffer(1), 0);
      const self = eg.game.playerByClientID(seatClientID(0))!;
      ctx = { g: eg, clientID: seatClientID(0), tables, geometry: cache.geometry, troopsAtObs: self.troops() };
    },
  });
});

const roundTrip = (intent: RecordIntent) => {
  const t = translateIntent(intent, ctx);
  const d = "action" in t ? decodeAction(t.action, 0, { g, clientID: ctx.clientID, tables: ctx.tables, geometry: ctx.geometry }) : null;
  return compareRoundTrip(intent, t, d, ctx);
};
const C = seatClientID(0);

test("each intent type round trips", () => {
  const self = g.game.playerByClientID(C)!;
  const other = g.game.playerByClientID(seatClientID(1))!;
  ctx.tables.tokenOf.set(other.smallID(), 3);
  ctx.tables.players[3] = other.smallID();
  const city = ctx.tables.structs[0];
  const ownTile = [...self.tiles()].find((t) => self.canBuild("Defense Post" as never, t) !== false) ?? null;
  const cases: RecordIntent[] = [
    { type: "attack", clientID: C, targetID: null, troops: Math.floor(self.troops() * 0.35) },
    { type: "attack", clientID: C, targetID: other.id(), troops: null },
    { type: "allianceRequest", clientID: C, recipient: other.id() },
    { type: "targetPlayer", clientID: C, target: other.id() },
    { type: "embargo", clientID: C, targetID: other.id(), action: "start" },
    { type: "upgrade_structure", clientID: C, unit: "City", unitId: city },
    { type: "delete_unit", clientID: C, unitId: city },
  ];
  if (ownTile !== null) cases.push({ type: "build_unit", clientID: C, unit: "Defense Post", tile: ownTile });
  for (const c of cases) expect([c.type, roundTrip(c).reason]).toEqual([c.type, null]);
});

test("null troops use default ratio bucket", () => {
  expect(troopBucket(null, 1000)).toBe(3);
  expect(troopBucket(200, 1000)).toBe(3);
  expect(troopBucket(1000, 1000)).toBe(19);
  expect(troopBucket(1, 1000)).toBe(0);
});

test("excluded types are counted, not failed", () => {
  const t = translateIntent({ type: "emoji", clientID: C, recipient: "x", emoji: 1 }, ctx);
  expect(t).toEqual({ unrepresentable: "excluded:emoji" });
});

test("overflow carries to next step", () => {
  const l = new StepLabeler();
  const atk = { type: "attack", clientID: C, targetID: null, troops: null } as RecordIntent;
  l.push([atk, atk, atk, atk, atk, atk]);
  const first = l.take(ctx);
  for (let k = 0; k < K; k++) expect(first[k * A + F.type]).toBe(T.ATTACK);
  expect(l.carried).toBe(2);
  const second = l.take(ctx);
  expect([second[F.type], second[A + F.type], second[2 * A + F.type]]).toEqual([T.ATTACK, T.ATTACK, T.STOP]);
  expect(first[F.player]).toBe(UNOWNED_TOKEN);
});

test("inside window rate counts near path", () => {
  const w = ctx.tables.windows[0];
  const inside = g.game.ref(w.x0 + 16 * w.tpc, w.y0 + 16 * w.tpc);
  const t = translateIntent({ type: "spawn", clientID: C, tile: inside }, ctx);
  expect("action" in t && t.near).toBe(true);
  const outside = translateIntent({ type: "spawn", clientID: C, tile: g.game.ref(w.x0 > 200 ? 0 : g.game.width() - 1, 0) }, ctx);
  expect("action" in outside && outside.near).toBe(false);
});

test("report on a synthetic record has rate 1.0", async () => {
  const rec = await makeSyntheticRecord({ map: SMALL_MAP, seats: 3, turns: 600, seed: "report", engineRoot: ENGINE_ROOT });
  const r = (await reportRecord(rec, ENGINE_ROOT)).build();
  expect(r.total).toBeGreaterThan(10);
  expect(r.byType.attack.total).toBeGreaterThan(0);
  expect(r.rate).toBe(1);
});
