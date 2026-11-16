/**
 * Step-cost table (M2 Task 12): what one decision step costs per map, for an
 * env game (6 seats, a random policy that spawns and grabs land) and for the
 * replay sampler on a real record of that map when one is given.
 */
import { GameEnv } from "../env/GameEnv";
import { PseudoRandom, simpleHash } from "../engine/Adapter";
import { A, F, K, rowBytes, T } from "../obs/Layout";
import { ObsBuffer } from "../obs/ObsBuffer";
import { FLAG, Sampler } from "../replay/Sampler";
import { freeLandNear } from "../record/SyntheticRecord";
import { translateIntent } from "../translate/Translator";

export interface StepCostRow {
  map: string;
  mapW: number;
  mapH: number;
  source: "env" | "replay";
  steps: number;
  ticksPerSec: number;
  cacheUpdateMs: number;
  encodeSeatMs: number;
  bytesPerRow: number;
  rowsPerSec: number;
  error?: string;
}

/** Random policy through the translator: spawn on random land, then attack unowned land now and then. */
function randomActions(env: GameEnv, rng: PseudoRandom): Int32Array {
  const g = env.g!;
  const game = g.game;
  const acts = new Int32Array(env.maxSeats * K * A).fill(-1);
  env.seats.forEach((seat, i) => {
    acts[i * K * A + F.type] = T.STOP;
    if (!seat.active) return;
    const p = game.playerByClientID(seat.clientID)!;
    let intent = null;
    if (game.inSpawnPhase() && !p.hasSpawned() && rng.nextInt(0, 4) === 0) {
      const tile = freeLandNear(g, rng.nextInt(0, game.width()), rng.nextInt(0, game.height()), 64);
      if (tile !== null) intent = { type: "spawn", clientID: seat.clientID, tile };
    } else if (!game.inSpawnPhase() && rng.nextInt(0, 4) === 0) {
      intent = { type: "attack", clientID: seat.clientID, targetID: null, troops: Math.floor(p.troops() * (rng.nextInt(1, 8) * 0.05)) };
    }
    if (intent === null) return;
    const tr = translateIntent(intent, { g, clientID: seat.clientID, tables: env.tables[i], geometry: env.cache!.geometry, troopsAtObs: p.troops() });
    if ("action" in tr) acts.set(tr.action, i * K * A);
  });
  return acts;
}

export async function envStepCost(engineRoot: string, map: string, seats = 6, minutesAfterSpawn = 5): Promise<StepCostRow> {
  const env = new GameEnv(engineRoot, seats);
  const buf = new ObsBuffer(seats);
  const tickCap = minutesAfterSpawn * 600;
  const row: StepCostRow = { map, mapW: 0, mapH: 0, source: "env", steps: 0, ticksPerSec: 0, cacheUpdateMs: 0, encodeSeatMs: 0, bytesPerRow: rowBytes().total, rowsPerSec: 0 };
  try {
    await env.reset({ map, seed: `step-cost:${map}`, seats, bots: 400, nations: "default", difficulty: "Medium", tickCap, latency: [1, 3], record: false }, buf, 0);
    row.mapW = env.g!.game.width();
    row.mapH = env.g!.game.height();
    const rng = new PseudoRandom(simpleHash(`step-cost:${map}`));
    let rows = 0;
    const t0 = performance.now();
    while (!env.over && row.steps < 10_000) {
      const res = env.step(randomActions(env, rng), buf, 0);
      rows += res.filter((r) => r.valid).length;
      row.steps++;
    }
    const s = (performance.now() - t0) / 1000;
    const t = env.timing;
    row.ticksPerSec = Math.round(t.ticks / (t.tickMs / 1000));
    row.cacheUpdateMs = round2(t.cacheMs / Math.max(1, row.steps + 1));
    row.encodeSeatMs = round2(t.encodeMs / Math.max(1, t.encodes));
    row.rowsPerSec = Math.round(rows / s);
    if (env.lastError) row.error = env.lastError;
  } catch (e) {
    row.error = e instanceof Error ? e.message : String(e);
  }
  return row;
}

export async function replayStepCost(engineRoot: string, recordPath: string, seats = 6, steps = 600): Promise<StepCostRow> {
  const s = new Sampler(engineRoot, 1, seats);
  const { readRecord } = await import("../record/Record");
  const rec = readRecord(recordPath);
  const counts = new Map<string, number>();
  for (const t of rec.turns) for (const i of t.intents) counts.set(i.clientID, (counts.get(i.clientID) ?? 0) + 1);
  const teachers = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, seats).map(([clientID]) => ({ clientID, weight: 1, value: 0.5 }));
  s.load([{ record: rec, teachers }]);
  const row: StepCostRow = { map: String(rec.info.config.gameMap), mapW: 0, mapH: 0, source: "replay", steps: 0, ticksPerSec: 0, cacheUpdateMs: 0, encodeSeatMs: 0, bytesPerRow: rowBytes().total, rowsPerSec: 0 };
  let rows = 0;
  const t0 = performance.now();
  for (; row.steps < steps; row.steps++) {
    const b = await s.next();
    const valid = [...b.flags].filter((f) => f & FLAG.VALID).length;
    if (valid === 0 && s.active() === 0) break;
    rows += valid;
  }
  row.rowsPerSec = Math.round(rows / ((performance.now() - t0) / 1000));
  if (s.stats.errors.length > 0) row.error = s.stats.errors[0];
  return row;
}

function round2(v: number): number {
  return Math.round(v * 100) / 100;
}
