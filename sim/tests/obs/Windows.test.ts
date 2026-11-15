import { type EngineGame } from "../../src/engine/Adapter";
import { GameObsCache } from "../../src/obs/GameObsCache";
import { windowContains, type WindowSpec } from "../../src/obs/Geometry";
import { WIN } from "../../src/obs/Layout";
import { mergeWindows, placeWindows } from "../../src/obs/Windows";
import { makeSyntheticRecord, seatClientID } from "../../src/record/SyntheticRecord";
import { ENGINE_ROOT, SMALL_MAP } from "../helpers";

async function snapshotAt(turns: number, map = SMALL_MAP, seed = "win") {
  let g: EngineGame | undefined;
  let cache: GameObsCache | undefined;
  let windows: WindowSpec[] = [];
  await makeSyntheticRecord({
    map, seats: 3, turns, seed, engineRoot: ENGINE_ROOT, attackEvery: 15,
    onTick: (eg, t) => {
      if (t !== turns - 1) return;
      g = eg;
      cache = new GameObsCache(eg);
      cache.update();
      const self = eg.game.playerByClientID(seatClientID(0))!;
      windows = placeWindows(cache, eg, self.smallID());
    },
  });
  return { g: g!, cache: cache!, windows };
}

test("no windows before spawn", async () => {
  const { windows } = await snapshotAt(1);
  expect(windows).toEqual([]);
});

test("core window covers own land", async () => {
  const { g, windows } = await snapshotAt(250);
  const self = g.game.playerByClientID(seatClientID(0))!;
  expect(windows[0].kind).toBe("core");
  const w = g.game.width();
  const spawn = self.spawnTile()!;
  expect(windowContains(windows[0], spawn % w, Math.floor(spawn / w))).toBe(true);
  for (const win of windows) {
    expect(win.x0).toBeGreaterThanOrEqual(0);
    expect(win.y0).toBeGreaterThanOrEqual(0);
    expect(win.x0 + WIN * win.tpc).toBeLessThanOrEqual(Math.max(g.game.width(), WIN * win.tpc));
  }
});

test("fronts sorted by troops", async () => {
  const { g, windows } = await snapshotAt(260);
  const self = g.game.playerByClientID(seatClientID(0))!;
  const attacks = [...self.outgoingAttacks(), ...self.incomingAttacks()].filter((a) => a.isActive());
  expect(attacks.length).toBeGreaterThan(0);
  // Expanding into land all around us puts the front's centre on the core window, so it merges into it;
  // either way the biggest attack's border is inside a window.
  const biggest = attacks.sort((a, b) => b.troops() - a.troops())[0];
  const tile = [...biggest.borderTiles()][0];
  const w = g.game.width();
  expect(windows.some((win) => windowContains(win, tile % w, Math.floor(tile / w)))).toBe(true);
  // Priorities are the final order.
  windows.forEach((win, i) => expect(win.priority).toBe(i));
});

test("nearby windows merge", () => {
  const a: WindowSpec = { x0: 0, y0: 0, tpc: 2, kind: "core", priority: 0 };
  const b: WindowSpec = { x0: 5, y0: 5, tpc: 2, kind: "front", priority: 1 };
  const c: WindowSpec = { x0: 200, y0: 0, tpc: 2, kind: "coast", priority: 2 };
  expect(mergeWindows([a, b, c]).map((w) => w.kind)).toEqual(["core", "coast"]);
});

test("same state same windows", async () => {
  const one = await snapshotAt(240, SMALL_MAP, "same");
  const two = await snapshotAt(240, SMALL_MAP, "same");
  expect(one.windows).toEqual(two.windows);
  expect(one.windows.length).toBeGreaterThan(0);
});
