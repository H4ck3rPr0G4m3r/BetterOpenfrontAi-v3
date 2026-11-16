import { createGame, type GameStartInfo } from "../../src/engine/Adapter";
import { makeSyntheticRecord, syntheticConfig } from "../../src/record/SyntheticRecord";
import { ENGINE_ROOT, SMALL_MAP } from "../helpers";

function emptyStart(gameID: string): GameStartInfo {
  return {
    gameID,
    lobbyCreatedAt: 0,
    config: syntheticConfig({ map: SMALL_MAP }),
    players: [{ clientID: "SEAT0000", username: "seat0", clanTag: "AI" }],
  } as unknown as GameStartInfo;
}

async function hashAtTick10(gameID: string): Promise<number> {
  const g = await createGame(emptyStart(gameID), ENGINE_ROOT);
  let hash: number | undefined;
  for (let t = 0; t <= 10; t++) {
    const out = g.tick({ turnNumber: t, intents: [] });
    expect(out.ok).toBe(true);
    for (const h of out.hashes) if (h.tick === 10) hash = h.hash;
  }
  expect(hash).toBeDefined();
  return hash!;
}

test("fresh terrain per game", async () => {
  // Game A plays 200 turns with spawns and attacks on the same map first.
  await makeSyntheticRecord({ map: SMALL_MAP, seats: 3, turns: 200, seed: "dirty", engineRoot: ENGINE_ROOT });
  const b = await createGame(emptyStart("freshBBB"), ENGINE_ROOT);
  const game = b.game;
  let owned = 0;
  for (let ref = 0; ref < game.width() * game.height(); ref++) if (game.hasOwner(ref)) owned++;
  expect(owned).toBe(0);
  expect(await hashAtTick10("freshCCC")).toBe(await hashAtTick10("freshCCC"));
});
