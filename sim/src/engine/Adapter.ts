/**
 * The engine adapter: the only place in sim/ that reaches the OpenFront
 * engine at runtime (spec §7 "Narrow engine adapter"). Upstream API drift is
 * fixed here, once.
 *
 * Game construction mirrors the engine's own `createGameRunner`
 * (packages/engine/src/GameRunner.ts) and its replay harness
 * (tests/replay/ReplayGame.ts): PRNG from simpleHash(gameID), humans drawn
 * with random.nextID() in start order, then nations, then createGame and
 * GameRunner.init(). Map files are read fresh for every game.
 */
import { GameMapSize, GameMapType, PlayerInfo, PlayerType } from "@openfront/engine-api/game/GameTypes";
import { GameUpdateType, type HashUpdate } from "@openfront/engine-api/game/GameUpdates";
import { type GameStartInfo, StampedIntentSchema, type Turn } from "@openfront/engine-api/Schemas";
import { loadTerrainMap } from "@openfront/engine-lib/game/TerrainMapLoader";
import { PseudoRandom } from "@openfront/engine-lib/PseudoRandom";
import { simpleHash } from "@openfront/engine-lib/Util";
import { EngineConfig } from "@openfront/engine/configuration/EngineConfig";
import { Executor } from "@openfront/engine/execution/ExecutionManager";
import type { Game } from "@openfront/engine/game/Game";
import { createGame as engineCreateGame } from "@openfront/engine/game/GameImpl";
import { createNationsForGame } from "@openfront/engine/game/NationCreation";
import { GameRunner } from "@openfront/engine/GameRunner";
import { loadMapFiles } from "@openfront/shared/GameMapLoader";
import { toWireGameStartInfo } from "@openfront/shared/SharedUtil";
import path from "path";
import type { GameRecordJson, RecordTurn } from "../record/Record";
import { FsMapLoader } from "./FsMapLoader";

export type { Game, GameStartInfo, Turn };

export interface TickOutcome {
  ok: boolean;
  /** Hash updates the engine emitted during this tick (every 10 ticks). */
  hashes: { tick: number; hash: number }[];
  error?: string;
}

export interface EngineGame {
  readonly game: Game;
  readonly runner: GameRunner;
  readonly gameID: string;
  /** Feed one turn and execute it as the next tick. */
  tick(turn: RecordTurn | Turn): TickOutcome;
  ticks(): number;
}

let quieted = false;
/** The engine logs per-tick debug lines; keep stdout for our JSON. */
export function quietEngineLogs(): void {
  if (quieted) return;
  quieted = true;
  console.debug = () => {};
  console.log = (...args: unknown[]) => process.stderr.write(args.map(String).join(" ") + "\n");
  console.info = console.log;
}

export function mapsDir(engineRoot: string): string {
  return path.join(engineRoot, "resources", "maps");
}

export async function createGame(start: GameStartInfo, engineRoot: string): Promise<EngineGame> {
  const config = new EngineConfig(start.config, false, start.listed ?? false);
  const files = await loadMapFiles(
    new FsMapLoader(mapsDir(engineRoot)),
    start.config.gameMap,
    start.config.gameMapSize,
  );
  const terrain = await loadTerrainMap(files);
  const random = new PseudoRandom(simpleHash(start.gameID));
  const humans = start.players.map(
    (p) =>
      new PlayerInfo(
        p.username,
        PlayerType.Human,
        p.clientID,
        random.nextID(),
        p.isLobbyCreator ?? false,
        p.clanTag,
        p.friends ?? [],
        p.teamIndex ?? null,
      ),
  );
  const nations = createNationsForGame(start, terrain.nations, terrain.additionalNations, humans.length, random);
  const game = engineCreateGame(
    humans,
    nations,
    terrain.gameMap,
    terrain.miniGameMap,
    config,
    terrain.teamGameSpawnAreas,
  );

  let tickHashes: { tick: number; hash: number }[] = [];
  let fatal: string | undefined;
  const runner = new GameRunner(
    game,
    new Executor(
      game,
      start.gameID,
      undefined,
      start.tribes?.map((t) => t.name),
    ),
    (gu) => {
      if ("errMsg" in gu) {
        fatal = `${gu.errMsg}${gu.stack ? `\n${gu.stack}` : ""}`;
        return;
      }
      const hashes = gu.updates[GameUpdateType.Hash] as HashUpdate[] | undefined;
      if (hashes) for (const h of hashes) tickHashes.push({ tick: h.tick, hash: h.hash });
    },
  );
  runner.init();

  return {
    game,
    runner,
    gameID: start.gameID,
    tick(turn) {
      tickHashes = [];
      fatal = undefined;
      runner.addTurn(turn as Turn);
      const ok = runner.executeNextTick();
      if (!ok) return { ok: false, hashes: tickHashes, error: fatal ?? "tick did not execute" };
      return { ok: true, hashes: tickHashes };
    },
    ticks: () => game.ticks(),
  };
}

/** GameStartInfo for replaying a record, with the client replay path's own blanking. */
export function startInfoFromRecord(rec: GameRecordJson): GameStartInfo {
  const info = rec.info;
  return toWireGameStartInfo({
    gameID: info.gameID,
    lobbyCreatedAt: info.lobbyCreatedAt ?? 0,
    config: info.config as unknown as GameStartInfo["config"],
    players: info.players.map((p) => ({
      clientID: p.clientID,
      username: p.username,
      clanTag: p.clanTag ?? null,
      ...(p.isLobbyCreator !== undefined ? { isLobbyCreator: p.isLobbyCreator as boolean } : {}),
      ...(p.friends !== undefined ? { friends: p.friends as string[] } : {}),
      ...(p.teamIndex !== undefined ? { teamIndex: p.teamIndex as number } : {}),
    })),
    tribes: info.tribes,
  } as GameStartInfo) as GameStartInfo;
}

/**
 * Validates a turn's intents with the engine's own schema. Returns the parsed
 * turn (what the client would execute) or the first problem.
 */
export function parseTurn(turn: RecordTurn): { turn: Turn } | { error: string } {
  const intents = [];
  for (const raw of turn.intents) {
    const parsed = StampedIntentSchema.safeParse(raw);
    if (!parsed.success) {
      const issue = parsed.error.issues[0];
      const where = issue?.path?.length ? ` at ${issue.path.join(".")}` : "";
      return { error: `bad intent at turn ${turn.turnNumber}: ${issue?.message ?? "invalid"}${where} (type ${String(raw.type)})` };
    }
    intents.push(parsed.data);
  }
  return { turn: { turnNumber: turn.turnNumber, intents } as Turn };
}

/** Engine-side enums the rest of sim/ may name without importing the engine. */
export function mapTypeFromName(name: string): GameMapType {
  const value = Object.values(GameMapType).find((v) => v === name);
  if (value !== undefined) return value as GameMapType;
  const byKey = (GameMapType as Record<string, string>)[name];
  if (byKey !== undefined) return byKey as GameMapType;
  throw new Error(`unknown map ${name}`);
}

export { GameMapSize, GameMapType, PlayerType, PseudoRandom, simpleHash };
