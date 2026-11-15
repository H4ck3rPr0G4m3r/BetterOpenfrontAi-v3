import { cloneRecord, type GameRecordJson } from "../../src/record/Record";
import { replayRecord } from "../../src/record/Replayer";
import { makeSyntheticRecord } from "../../src/record/SyntheticRecord";
import { ENGINE_ROOT, SMALL_MAP } from "../helpers";

let base: GameRecordJson;
beforeAll(async () => {
  base = await makeSyntheticRecord({ map: SMALL_MAP, seats: 3, turns: 600, seed: "replayer", engineRoot: ENGINE_ROOT });
});

test("synthetic record replays with every hash matching", async () => {
  const r = await replayRecord(base, ENGINE_ROOT);
  expect(r.error).toBeNull();
  expect(r.mismatchAt).toBeNull();
  expect(r.hashChecks).toBe(60);
  expect(r.turns).toBe(600);
  expect(r.gameId).toBe(base.info.gameID);
  expect(r.ticksPerSec).toBeGreaterThan(0);
});

test("tampered hash is caught at that turn", async () => {
  const rec = cloneRecord(base);
  const t = rec.turns.find((x) => x.turnNumber === 300)!;
  t.hash = (t.hash as number) + 1;
  const r = await replayRecord(rec, ENGINE_ROOT);
  expect(r.mismatchAt).toBe(300);
  expect(r.error).toBeNull();
});

test("tampered intent desyncs", async () => {
  const rec = cloneRecord(base);
  const turn = rec.turns.find((x) => x.turnNumber > 200 && x.intents.some((i) => i.type === "attack"))!;
  const attack = turn.intents.find((i) => i.type === "attack")!;
  attack.troops = Math.max(1, Math.floor((attack.troops as number) / 3));
  const r = await replayRecord(rec, ENGINE_ROOT);
  expect(r.mismatchAt).not.toBeNull();
  expect(r.mismatchAt!).toBeGreaterThanOrEqual(turn.turnNumber);
});

test("unknown intent type is an error result", async () => {
  const rec = cloneRecord(base);
  const t = rec.turns.find((x) => x.turnNumber >= 50)!;
  t.intents.push({ type: "warp_drive", clientID: "SEAT0000" });
  const r = await replayRecord(rec, ENGINE_ROOT);
  expect(r.error).toMatch(/intent/);
  expect(r.mismatchAt).toBeNull();
});

test("fills to num_turns", async () => {
  const rec = cloneRecord(base);
  const last = rec.turns[rec.turns.length - 1].turnNumber;
  rec.info.num_turns = last + 41;
  const r = await replayRecord(rec, ENGINE_ROOT);
  expect(r.turns).toBe(last + 41);
  expect(r.error).toBeNull();
  expect(r.mismatchAt).toBeNull();
});

test("onTick sees every turn in order", async () => {
  const seen: number[] = [];
  await replayRecord(base, ENGINE_ROOT, { onTick: (_g, t) => seen.push(t) });
  expect(seen).toHaveLength(600);
  expect(seen[0]).toBe(0);
  expect(seen[599]).toBe(599);
});
