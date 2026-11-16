import fs from "fs";
import os from "os";
import path from "path";
import { A, F, K, T } from "../../src/obs/Layout";
import { ObsBuffer } from "../../src/obs/ObsBuffer";
import { GameEnv, type GameSpec, seatUsername } from "../../src/env/GameEnv";
import { Latency } from "../../src/env/Latency";
import { readRecord } from "../../src/record/Record";
import { replayRecord } from "../../src/record/Replayer";
import { freeLandNear, seatClientID } from "../../src/record/SyntheticRecord";
import { translateIntent } from "../../src/translate/Translator";
import { ENGINE_ROOT, SMALL_MAP } from "../helpers";

const spec = (over: Partial<GameSpec> = {}): GameSpec => ({
  map: SMALL_MAP, seed: "env", seats: 4, bots: 0, nations: "disabled", difficulty: "Medium", tickCap: 100,
  latency: [1, 3], record: false, ...over,
});

/** A scripted policy through the translator: spawn near a corner, then grab unowned land. */
function scripted(env: GameEnv, step: number, skip: Set<number> = new Set()): Int32Array {
  const acts = new Int32Array(env.maxSeats * K * A).fill(-1);
  const g = env.g!;
  env.seats.forEach((seat, i) => {
    acts[i * K * A + F.type] = T.STOP;
    if (!seat.active || skip.has(i)) return;
    const p = g.game.playerByClientID(seat.clientID)!;
    let intent = null;
    if (g.game.inSpawnPhase() && !p.hasSpawned() && step % 5 === i) {
      const land = [...Array(g.game.width() * g.game.height()).keys()].find((t) => g.game.isLand(t) && !g.game.map().isImpassable(t))!;
      const tile = freeLandNear(g, g.game.x(land) + 60 + (i % 2) * 60, g.game.y(land) + 60 + Math.floor(i / 2) * 60);
      if (tile !== null) intent = { type: "spawn", clientID: seat.clientID, tile };
    } else if (!g.game.inSpawnPhase() && step % 4 === 0) {
      intent = { type: "attack", clientID: seat.clientID, targetID: null, troops: Math.floor(p.troops() * 0.25) };
    }
    if (intent === null) return;
    const ctx = { g, clientID: seat.clientID, tables: env.tables[i], geometry: env.cache!.geometry, troopsAtObs: p.troops() };
    const tr = translateIntent(intent, ctx);
    if ("action" in tr) {
      acts.set(tr.action, i * K * A);
      acts[i * K * A + A + F.type] = T.STOP;
    }
  });
  return acts;
}

test("four seats reset and step on the smallest map", async () => {
  const env = new GameEnv(ENGINE_ROOT, 4);
  const buf = new ObsBuffer(4);
  const tables = await env.reset(spec(), buf, 0);
  expect(tables).toHaveLength(4);
  for (let step = 0; step < 80; step++) {
    const res = env.step(scripted(env, step), buf, 0);
    expect(res).toHaveLength(4);
    if (env.over) break;
  }
  const g = env.g!;
  expect(env.seats.filter((s) => g.game.playerByClientID(s.clientID)!.hasSpawned())).toHaveLength(4);
  expect(g.game.ticks()).toBeGreaterThan(300);
});

test("seat names carry the AI tag", async () => {
  const env = new GameEnv(ENGINE_ROOT, 2);
  await env.reset(spec({ seats: 2, names: ["Alpha", "[AI] Beta"] }), new ObsBuffer(2), 0);
  const names = env.seats.map((s) => env.g!.game.playerByClientID(s.clientID)!.displayName());
  expect(names).toEqual(["[AI] Alpha", "[AI] Beta"]);
  expect(seatUsername("[AI] x<y>", 0)).toBe("x_y_");
});

test("latency delays intents 1-3 turns deterministically", () => {
  const run = () => {
    const l = new Latency("seed-1", [1, 3]);
    const lands: number[] = [];
    for (let now = 0; now < 500; now += 5) lands.push(l.push(now, "S", [{ type: "x", clientID: "S" }]) - now);
    return lands;
  };
  const a = run();
  expect(a).toEqual(run());
  expect(new Set(a.map((d) => Math.min(d, 2)))).toEqual(new Set([0, 1, 2]));
  expect(Math.max(...a)).toBeLessThanOrEqual(2); // delay d lands in turn now + d - 1
  // Never reorders a seat's intents: a short delay can't overtake a longer one queued earlier.
  const l = new Latency("order", [1, 3]);
  let last = -1;
  for (let now = 0; now < 200; now++) {
    const at = l.push(now, "S", [{ type: "x", clientID: "S", n: now }]);
    expect(at).toBeGreaterThanOrEqual(last);
    last = at;
  }
  const order = [];
  for (let t = 0; t < 210; t++) order.push(...l.due(t).map((i) => i.n as number));
  expect(order).toEqual([...order].sort((x, y) => x - y));
});

test("dead seat goes done and inactive", async () => {
  // Seat 1 never spawns: when the spawn phase ends it is out (death), once.
  const env = new GameEnv(ENGINE_ROOT, 2);
  const buf = new ObsBuffer(2);
  await env.reset(spec({ seats: 2, seed: "dead" }), buf, 0);
  const seen: string[] = [];
  for (let step = 0; step < 90 && !env.over; step++) {
    const r = env.step(scripted(env, step, new Set([1])), buf, 0)[1];
    if (r.done) seen.push(r.terminal);
    if (seen.length > 0) expect(r.next).toBe(false);
  }
  expect(seen).toEqual(["death"]);
  expect(env.seats[1].active).toBe(false);
});

test("record of an env game replays with all hashes matching", async () => {
  const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "ofai-env-")), "game.json.gz");
  const env = new GameEnv(ENGINE_ROOT, 3);
  const buf = new ObsBuffer(3);
  await env.reset(spec({ seats: 3, seed: "rec", record: file, tickCap: 150 }), buf, 0);
  let steps = 0;
  while (!env.over && steps < 200) env.step(scripted(env, steps++), buf, 0);
  expect(env.over).toBe(true);
  expect(env.lastRecord).toBe(file);
  const rec = readRecord(file);
  expect(rec.info.players.map((p) => p.clanTag)).toEqual(["AI", "AI", "AI"]);
  expect(rec.turns.some((t) => t.intents.some((i) => i.type === "spawn"))).toBe(true);
  const r = await replayRecord(rec, ENGINE_ROOT);
  expect(r.error).toBeNull();
  expect(r.mismatchAt).toBeNull();
  expect(r.hashChecks).toBeGreaterThan(30);
  expect(seatClientID(0)).toBe(rec.info.players[0].clientID);
});
