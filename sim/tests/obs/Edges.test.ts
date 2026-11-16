import { fromF16 } from "../../src/obs/f16";
import { GameObsCache } from "../../src/obs/GameObsCache";
import { EF, EI, NUM_PLAYER_TOKENS } from "../../src/obs/Layout";
import { ObsBuffer } from "../../src/obs/ObsBuffer";
import { encodeSeat, type EntityTables } from "../../src/obs/SeatEncoder";
import { freeLandNear, makeSyntheticRecord, seatClientID } from "../../src/record/SyntheticRecord";
import { ENGINE_ROOT, SMALL_MAP } from "../helpers";

let buf: ObsBuffer;
let tables: EntityTables;
let cache: GameObsCache;
beforeAll(async () => {
  await makeSyntheticRecord({
    map: SMALL_MAP, seats: 4, turns: 900, seed: "edges", engineRoot: ENGINE_ROOT, attackEvery: 10,
    // Four seats in a 2x2 cluster so every pair ends up neighbours.
    spawnAt: (g, seat) => {
      const land = [...Array(g.game.width() * g.game.height()).keys()].find((t) => g.game.isLand(t) && !g.game.map().isImpassable(t))!;
      const x = g.game.x(land) + 60 + (seat % 2) * 40;
      const y = g.game.y(land) + 60 + Math.floor(seat / 2) * 40;
      return freeLandNear(g, x, y);
    },
    onTick: (g, t) => {
      if (t !== 899) return;
      cache = new GameObsCache(g);
      cache.update();
      buf = new ObsBuffer(1);
      tables = encodeSeat(cache, g, seatClientID(0), buf, 0);
    },
  });
});

const edge = (i: number, j: number, f: number) => fromF16(buf.view<Uint16Array>("edges", 0)[(i * NUM_PLAYER_TOKENS + j) * EF + f]);
const tokens = () => tables.players.map((p, i) => [i, p] as const).filter(([, p]) => p !== null).map(([i]) => i);

test("neighbour-neighbour border appears", () => {
  // Some pair of tokens that are not self (token 0) shares a land border, as the cache says.
  const toks = tokens().filter((t) => t !== 0);
  let found = false;
  for (const a of toks) for (const b of toks) {
    if (a === b) continue;
    const border = cache.adjacency.get(tables.players[a]!)?.get(tables.players[b]!);
    expect(edge(a, b, EI.land_border) === 1).toBe((border?.count ?? 0) > 0);
    if (border) found = true;
  }
  expect(found).toBe(true);
});

test("symmetric border length", () => {
  for (const a of tokens()) for (const b of tokens()) {
    if (a === b) continue;
    expect(edge(a, b, EI.border_log_tiles)).toBe(edge(b, a, EI.border_log_tiles));
    expect(edge(a, b, EI.density_i)).toBe(edge(b, a, EI.density_j));
  }
});

test("no border means sea distance flag", () => {
  for (const a of tokens()) for (const b of tokens()) {
    if (a === b) continue;
    expect(edge(a, b, EI.valid)).toBe(1);
    expect(edge(a, b, EI.sea_only) === 1).toBe(edge(a, b, EI.land_border) === 0);
  }
});
