import { fillTurns } from "../../src/record/Record";
import { makeSyntheticRecord } from "../../src/record/SyntheticRecord";
import { ENGINE_ROOT, SMALL_MAP } from "../helpers";

test("synthetic record round-trips through readRecord/fillTurns", async () => {
  const rec = await makeSyntheticRecord({ map: SMALL_MAP, seats: 3, turns: 600, seed: "s1", engineRoot: ENGINE_ROOT });
  expect(fillTurns(rec)).toHaveLength(600);
  expect(rec.turns.filter((t) => t.hash !== undefined).length).toBe(60);
  expect(rec.turns.some((t) => t.intents.some((i) => i.type === "spawn"))).toBe(true);
  expect(rec.turns.some((t) => t.intents.some((i) => i.type === "attack"))).toBe(true);
  expect(rec.info.players.every((p) => p.clanTag === "AI")).toBe(true);
});

test("same seed gives identical hashes", async () => {
  const a = await makeSyntheticRecord({ map: SMALL_MAP, seats: 2, turns: 300, seed: "same", engineRoot: ENGINE_ROOT });
  const b = await makeSyntheticRecord({ map: SMALL_MAP, seats: 2, turns: 300, seed: "same", engineRoot: ENGINE_ROOT });
  const hashes = (r: typeof a) => r.turns.filter((t) => t.hash !== undefined).map((t) => [t.turnNumber, t.hash]);
  expect(hashes(a)).toEqual(hashes(b));
  expect(hashes(a).length).toBe(30);
});

test("synthetic seats really spawn and grow", async () => {
  const tiles: number[] = [];
  await makeSyntheticRecord({
    map: SMALL_MAP, seats: 3, turns: 600, seed: "grow", engineRoot: ENGINE_ROOT,
    onTick: (g, t) => {
      if (t !== 599) return;
      for (const p of g.game.players()) if (p.clientID()?.startsWith("SEAT")) tiles.push(p.numTilesOwned());
    },
  });
  expect(tiles).toHaveLength(3);
  for (const n of tiles) expect(n).toBeGreaterThan(50);
});
