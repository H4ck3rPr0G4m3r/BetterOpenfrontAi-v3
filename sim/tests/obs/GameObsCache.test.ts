import fs from "fs";
import path from "path";
import { type EngineGame } from "../../src/engine/Adapter";
import { GameObsCache, LETTERBOX, rankUnits } from "../../src/obs/GameObsCache";
import { MM, PLANES } from "../../src/obs/Layout";
import { expandView, minimapView } from "../../src/obs/reference/expandPlanes";
import { makeSyntheticRecord } from "../../src/record/SyntheticRecord";
import { ENGINE_ROOT, SMALL_MAP } from "../helpers";

async function gameAt(map: string, turns: number, seats = 3, extra = {}): Promise<EngineGame> {
  let g: EngineGame | undefined;
  await makeSyntheticRecord({ map, seats, turns, seed: `cache-${map}`, engineRoot: ENGINE_ROOT, onTick: (eg) => (g = eg), ...extra });
  return g!;
}

const maps = [SMALL_MAP, "Sol", "Las Vegas Strip"].filter((m) =>
  fs.existsSync(path.join(ENGINE_ROOT, "resources/maps", m.toLowerCase().replace(/ /g, ""), "map.bin")),
);

test.each(maps)("encodes on extreme maps: %s", async (map) => {
  const g = await gameAt(map, 260);
  const c = new GameObsCache(g);
  c.update();
  let letterbox = 0;
  for (let i = 0; i < MM * MM; i++) if (c.mmOwner[i] === LETTERBOX) letterbox++;
  expect(letterbox).toBe(MM * MM - c.geometry.cellsW * c.geometry.cellsH);
  // The seats spawned: their land shows up in the raster.
  const seatIds = c.players.filter((p) => p.clientID()?.startsWith("SEAT")).map((p) => p.smallID());
  expect(seatIds.length).toBe(3);
  for (const id of seatIds) expect(c.ownerCells.get(id)?.length ?? 0).toBeGreaterThan(0);
  // Seats border unowned land.
  for (const id of seatIds) expect(c.adjacency.get(id)?.get(0)?.count ?? 0).toBeGreaterThan(0);
  // Reference expansion runs and its valid plane matches the letterbox.
  const rel = new Uint8Array(1024);
  const planes = expandView(
    minimapView(c.mmOwner, c.mmTerrain, MM, c.geometry.tpc, c.geometry.offX, c.geometry.offY),
    rel, c.density, c.units, 100,
  );
  const valid = planes.subarray((PLANES.length - 1) * MM * MM, PLANES.length * MM * MM);
  expect(valid.reduce((a, b) => a + b, 0)).toBe(MM * MM - letterbox);
});

test("owner majority is deterministic", async () => {
  const a = new GameObsCache(await gameAt(SMALL_MAP, 300));
  const b = new GameObsCache(await gameAt(SMALL_MAP, 300));
  a.update();
  b.update();
  expect(Buffer.from(a.mmOwner.buffer).equals(Buffer.from(b.mmOwner.buffer))).toBe(true);
  expect(Buffer.from(a.units.buffer).equals(Buffer.from(b.units.buffer))).toBe(true);
});

test("unit overflow keeps nukes and warships", () => {
  const mk = (type: string) => ({ type: () => type, isActive: () => true });
  const units = [
    ...Array.from({ length: 10 }, () => mk("City")),
    mk("Warship"), mk("Atom Bomb"), mk("Transport"), mk("Train"),
    { type: () => "Warship", isActive: () => false },
  ];
  const ranked = rankUnits(units, 3);
  expect(ranked.map((u) => u.type())).toEqual(["Atom Bomb", "Warship", "Transport"]);
  expect(rankUnits(units, 100).some((u) => u.type() === "Train")).toBe(false);
});

test("update cost", async () => {
  const g = await gameAt("World", 1800, 6, { bots: 200, nations: "default" });
  const c = new GameObsCache(g);
  c.update();
  const t0 = performance.now();
  for (let i = 0; i < 5; i++) c.update();
  const ms = (performance.now() - t0) / 5;
  console.log(`GameObsCache.update on World after 3 min: ${ms.toFixed(1)} ms, ${c.unitCount} units, ${c.players.length} players`);
  expect(ms).toBeLessThan(20);
});
