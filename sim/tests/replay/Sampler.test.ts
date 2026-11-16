import { A, F, K, T } from "../../src/obs/Layout";
import { PlacementTracker, placementValue } from "../../src/replay/Placements";
import { FLAG, Sampler, type SampleItem } from "../../src/replay/Sampler";
import { cloneRecord, type GameRecordJson } from "../../src/record/Record";
import { replayRecord } from "../../src/record/Replayer";
import { freeLandNear, makeSyntheticRecord, seatClientID } from "../../src/record/SyntheticRecord";
import { ENGINE_ROOT, SMALL_MAP } from "../helpers";

const S0 = seatClientID(0);
const S1 = seatClientID(1);
const S2 = seatClientID(2);
const teacher = (clientID: string, weight = 1, value = 0.5) => ({ clientID, weight, value });

/** Seat 0 expands until it borders seat 1, then kills it (dead well before turn 400). Seat 2 never asks to spawn. */
let duel: GameRecordJson;
let plain: GameRecordJson;
beforeAll(async () => {
  duel = await makeSyntheticRecord({
    map: SMALL_MAP, seats: 3, turns: 600, seed: "duel", engineRoot: ENGINE_ROOT, attackEvery: 100000,
    spawnAt: (g, seat) => {
      if (seat === 2) return null;
      const land = [...Array(g.game.width() * g.game.height()).keys()].find((t) => g.game.isLand(t) && !g.game.map().isImpassable(t))!;
      return freeLandNear(g, g.game.x(land) + 80 + seat * 12, g.game.y(land) + 80);
    },
    script: (g, t) => {
      const a = g.game.playerByClientID(S0);
      const b = g.game.playerByClientID(S1);
      if (!a || !b || g.game.inSpawnPhase() || t % 10 !== 0 || !b.isAlive()) return [];
      const near = a.sharesBorderWith(b);
      return [{ type: "attack", clientID: S0, targetID: near ? b.id() : null, troops: Math.floor(a.troops() * (near ? 0.9 : 0.3)) }];
    },
  });
  plain = await makeSyntheticRecord({ map: SMALL_MAP, seats: 2, turns: 400, seed: "plain", engineRoot: ENGINE_ROOT });
}, 300_000);

async function steps(s: Sampler, n: number, row = 0) {
  const out: { flags: number; labels: Int32Array }[] = [];
  for (let i = 0; i < n; i++) {
    const b = await s.next();
    out.push({ flags: b.flags[row], labels: b.labels.slice(row * K * A, (row + 1) * K * A) });
  }
  return out;
}

test("labels come from the turns after the observation", async () => {
  const s = new Sampler(ENGINE_ROOT, 1, 1);
  s.load([{ record: plain, teachers: [teacher(S0)] }]);
  const turns = plain.turns;
  const got = await steps(s, 80);
  for (let step = 0; step < got.length; step++) {
    const t0 = step * 5;
    const mine = turns.filter((x) => x.turnNumber >= t0 && x.turnNumber < t0 + 5).flatMap((x) => x.intents).filter((i) => i.clientID === S0);
    const first = got[step].labels[F.type];
    if (mine.length === 0) expect(first).toBe(T.STOP);
    else expect(first).toBe(mine[0].type === "spawn" ? T.SPAWN : T.ATTACK);
  }
  // Seat 0 asks to spawn on turn 2: the first step's observation is pre-spawn, its label the spawn.
  expect(got[0].labels[F.type]).toBe(T.SPAWN);
  expect(got[0].flags).toBe(FLAG.VALID | FLAG.NEXT | FLAG.FRESH);
  expect(got[1].flags & FLAG.FRESH).toBe(0);
});

test("dead teacher row goes done then invalid", async () => {
  const s = new Sampler(ENGINE_ROOT, 1, 1);
  s.load([{ record: duel, teachers: [teacher(S1)] }]);
  const got = await steps(s, 120);
  const done = got.findIndex((x) => x.flags & FLAG.DONE);
  expect(done).toBeGreaterThan(10);
  expect(got[done].flags & FLAG.VALID).toBe(0);
  expect(got.slice(0, done).every((x) => x.flags & FLAG.VALID)).toBe(true);
  expect(got.slice(done + 1).every((x) => x.flags === 0)).toBe(true);
});

test("teacher who never spawns stays valid through spawn phase then done", async () => {
  const s = new Sampler(ENGINE_ROOT, 1, 1);
  s.load([{ record: duel, teachers: [teacher(S2)] }]);
  const got = await steps(s, 80);
  const done = got.findIndex((x) => x.flags & FLAG.DONE);
  expect(done).toBeGreaterThan(20); // the spawn phase is many steps long
  expect(got.slice(0, done).every((x) => x.flags & FLAG.VALID)).toBe(true);
  expect(got.slice(0, done).every((x) => x.labels[F.type] === T.STOP)).toBe(true);
  expect(got.slice(done + 1).every((x) => x.flags === 0)).toBe(true);
});

test("finished game is replaced and flagged fresh", async () => {
  const short = cloneRecord(plain);
  short.turns = short.turns.filter((t) => t.turnNumber < 50);
  short.info.num_turns = 50;
  const s = new Sampler(ENGINE_ROOT, 1, 2);
  s.load([{ record: short, teachers: [teacher(S0), teacher(S1)] }, { record: plain, teachers: [teacher(S1)] }]);
  const got = await steps(s, 12);
  expect(got[9].flags & FLAG.DONE).toBeTruthy(); // 50 turns = 10 steps; the last one is VALID|DONE
  expect(got[9].flags & FLAG.VALID).toBeTruthy();
  expect(got[10].flags).toBe(FLAG.VALID | FLAG.NEXT | FLAG.FRESH);
  expect(s.stats.gamesFinished).toBe(1);
  expect(s.stats.gamesStarted).toBe(2);
});

test("empty queue leaves rows invalid", async () => {
  const s = new Sampler(ENGINE_ROOT, 2, 2);
  const b = await s.next();
  expect([...b.flags].every((f) => f === 0)).toBe(true);
  expect(b.weight.every((w) => w === 0)).toBe(true);
});

test("hash mismatch aborts that game only", async () => {
  const bad = cloneRecord(plain);
  bad.info.gameID = plain.info.gameID; // same game, one wrong hash
  bad.turns.find((t) => t.turnNumber === 100)!.hash! += 1;
  const s = new Sampler(ENGINE_ROOT, 2, 1);
  s.load([{ record: bad, teachers: [teacher(S0)] }, { record: plain, teachers: [teacher(S0)] } as SampleItem]);
  const flags: number[][] = [];
  for (let i = 0; i < 30; i++) flags.push([...(await s.next()).flags]);
  expect(flags[20][0]).toBe(FLAG.DONE); // turns 100..104 hold the mismatch
  expect(flags.slice(21).every((f) => f[0] === 0)).toBe(true);
  expect(flags.every((f) => f[1] & FLAG.VALID)).toBe(true);
  expect(s.stats.gamesAborted).toBe(1);
});

test("placements rank the survivor first and skip players who never spawned", async () => {
  const r = await replayRecord(duel, ENGINE_ROOT);
  expect(r.mismatchAt).toBeNull();
  expect(r.placements).toEqual({ [S0]: 1, [S1]: 2 });
  expect(r.ranked).toBe(2);
  expect(placementValue(1, 2)).toBe(1);
  expect(placementValue(2, 2)).toBe(0);
  expect(placementValue(1, 1)).toBe(1);
  expect(new PlacementTracker()).toBeDefined();
});
