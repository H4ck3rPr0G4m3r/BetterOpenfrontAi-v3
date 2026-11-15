import { type EngineGame } from "../../src/engine/Adapter";
import { fromF16 } from "../../src/obs/f16";
import { GameObsCache } from "../../src/obs/GameObsCache";
import { ENTITY_SLOTS, NUM_PLAYER_TOKENS, PF, PI, REST_TOKEN, rowBytes, T, UNOWNED_TOKEN } from "../../src/obs/Layout";
import { ObsBuffer } from "../../src/obs/ObsBuffer";
import { encodeSeat, type EntityTables } from "../../src/obs/SeatEncoder";
import type { RecordIntent } from "../../src/record/Record";
import { makeSyntheticRecord, seatClientID } from "../../src/record/SyntheticRecord";
import { ENGINE_ROOT, SMALL_MAP } from "../helpers";

interface Snap { g: EngineGame; buf: ObsBuffer; tables: EntityTables; cache: GameObsCache }

async function encodeAt(turns: number, opts: Partial<Parameters<typeof makeSyntheticRecord>[0]> = {}, seat = 0): Promise<Snap> {
  let snap: Snap | undefined;
  await makeSyntheticRecord({
    map: SMALL_MAP, seats: 3, turns, seed: "enc", engineRoot: ENGINE_ROOT, ...opts,
    onTick: (g, t) => {
      if (t !== turns - 1) return;
      const cache = new GameObsCache(g);
      cache.update();
      const buf = new ObsBuffer(2);
      const tables = encodeSeat(cache, g, seatClientID(seat), buf, 1);
      snap = { g, buf, tables, cache };
    },
  });
  return snap!;
}

const feat = (buf: ObsBuffer, tok: number, f: number) => fromF16(buf.view<Uint16Array>("players", 1)[tok * PF + f]);

test("self is token 0", async () => {
  const { buf, tables, g } = await encodeAt(250);
  expect(tables.players[0]).toBe(g.game.playerByClientID(seatClientID(0))!.smallID());
  expect(feat(buf, 0, PI.is_self)).toBe(1);
  expect(feat(buf, 0, PI.valid)).toBe(1);
  expect(feat(buf, UNOWNED_TOKEN, PI.is_unowned)).toBe(1);
});

test("attackers are always tokens", async () => {
  // Seat 1 attacks seat 0 once they border; seat 0 must see seat 1 as a token.
  const { tables, g } = await encodeAt(420, { map: SMALL_MAP, seats: 2, seed: "fight", attackEvery: 10,
    script: (eg, t) => {
      if (t < 300 || t % 10 !== 5) return [];
      const a = eg.game.playerByClientID(seatClientID(1));
      const b = eg.game.playerByClientID(seatClientID(0));
      if (!a || !b || !a.isAlive() || !b.isAlive()) return [];
      return [{ type: "attack", clientID: seatClientID(1), targetID: b.id(), troops: Math.floor(a.troops() / 3) } as RecordIntent];
    } });
  const self = g.game.playerByClientID(seatClientID(0))!;
  for (const a of self.incomingAttacks()) expect(tables.tokenOf.has(a.attacker().smallID())).toBe(true);
});

test("rest token pools the others", async () => {
  const { buf, tables, cache } = await encodeAt(60, { map: "World", seats: 1, nations: "default", bots: 0, seed: "rest" });
  const alive = cache.players.filter((p) => p.isAlive()).length;
  const real = tables.players.filter((p) => p !== null).length;
  expect(real).toBe(Math.min(32, alive));
  if (alive > 32) {
    expect(feat(buf, REST_TOKEN, PI.is_rest)).toBe(1);
    expect(Math.round(Math.expm1(feat(buf, REST_TOKEN, PI.rest_count) * 7))).toBe(alive - 32);
  }
});

test("entity tables map back to live units", async () => {
  const { tables, g, buf } = await encodeAt(1700, { seats: 2, seed: "build",
    script: (eg, t) => {
      if (t < 1500 || t % 40 !== 0) return [];
      const p = eg.game.playerByClientID(seatClientID(0));
      if (!p || !p.isAlive()) return [];
      const tiles = [...p.tiles()];
      return [{ type: "build_unit", clientID: seatClientID(0), unit: "City", tile: tiles[(t * 7919) % tiles.length] } as RecordIntent];
    } });
  for (const id of tables.structs) expect(g.game.unit(id)?.isActive()).toBe(true);
  for (const id of tables.mobile) expect(g.game.unit(id)?.isActive()).toBe(true);
  const self = g.game.playerByClientID(seatClientID(0))!;
  for (const id of tables.attacks) expect(self.outgoingAttacks().concat(self.incomingAttacks()).some((a) => a.id() === id)).toBe(true);
  // At least one city got built and its upgrade legality matches the engine.
  expect(tables.structs.length).toBeGreaterThan(0);
  const em = buf.view<Uint8Array>("entity_mask", 1);
  tables.structs.forEach((id, i) => expect(em[T.UPGRADE * ENTITY_SLOTS + i] === 1).toBe(self.canUpgradeUnit(g.game.unit(id)!)));
});

test("masks follow the engine", async () => {
  const early = await encodeAt(20);
  const tm = early.buf.view<Uint8Array>("type_mask", 1);
  expect(early.g.game.inSpawnPhase()).toBe(true);
  expect([...tm].map((v, i) => (v ? i : -1)).filter((i) => i >= 0)).toEqual([T.STOP, T.SPAWN]);

  const later = await encodeAt(260);
  const tm2 = later.buf.view<Uint8Array>("type_mask", 1);
  expect(tm2[T.SPAWN]).toBe(0);
  expect(tm2[T.STOP]).toBe(1);
  expect(tm2[T.ATTACK]).toBe(1); // borders unowned land with troops
  const pm = later.buf.view<Uint8Array>("player_mask", 1);
  expect(pm[T.ATTACK * NUM_PLAYER_TOKENS + UNOWNED_TOKEN]).toBe(1);
  expect(pm[T.ATTACK * NUM_PLAYER_TOKENS + 0]).toBe(0); // never ourselves
  // Gold decides builds.
  const self = later.g.game.playerByClientID(seatClientID(0))!;
  const bm = later.buf.view<Uint8Array>("build_mask", 1);
  const cityCost = later.g.game.config().unitInfo("City" as never).cost(later.g.game, self);
  expect(bm[0] === 1).toBe(self.gold() >= cityCost);
});

test("seat bytes match layout", () => {
  const buf = new ObsBuffer(3);
  expect(buf.bytes(1).length).toBe(rowBytes().total);
  expect(buf.bytes(3).length).toBe(3 * rowBytes().total);
});
