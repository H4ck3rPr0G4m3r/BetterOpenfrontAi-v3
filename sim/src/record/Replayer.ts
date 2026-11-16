/**
 * Replays a game record through the engine turn by turn and checks the
 * record's state hashes (spec §7 "Determinism check"): after turn t executes,
 * every hash update for tick t is compared with turns[t].hash when the record
 * has one. The first mismatch stops the replay; the game is rejected.
 */
import { createGame, type EngineGame, parseTurn, startInfoFromRecord } from "../engine/Adapter";
import { PlacementTracker } from "../replay/Placements";
import { fillTurns, type GameRecordJson } from "./Record";

export interface ReplayResult {
  gameId: string;
  commit: string;
  map: string;
  turns: number;
  hashChecks: number;
  mismatchAt: number | null;
  error: string | null;
  ms: number;
  ticksPerSec: number;
  /** clientID -> final place among spawned humans and nations (only for a clean full replay). */
  placements: Record<string, number>;
  /** Number of ranked players the places are out of. */
  ranked: number;
}

export interface ReplayOptions {
  /** Called after each turn executes (the M2 sampler hooks in here). Return false to stop. */
  onTick?: (g: EngineGame, turn: number) => void | boolean;
  /** Called once the game exists, before turn 0. */
  onStart?: (g: EngineGame) => void;
}

export async function replayRecord(
  rec: GameRecordJson,
  engineRoot: string,
  opts: ReplayOptions = {},
): Promise<ReplayResult> {
  const t0 = performance.now();
  const result: ReplayResult = {
    gameId: rec.info?.gameID ?? "unknown",
    commit: rec.gitCommit ?? "unknown",
    map: String(rec.info?.config?.gameMap ?? "unknown"),
    turns: 0,
    hashChecks: 0,
    mismatchAt: null,
    error: null,
    ms: 0,
    ticksPerSec: 0,
    placements: {},
    ranked: 0,
  };
  const finish = () => {
    result.ms = Math.round(performance.now() - t0);
    result.ticksPerSec = result.ms > 0 ? Math.round((result.turns * 1000) / result.ms) : 0;
    return result;
  };

  let turns;
  let g: EngineGame;
  try {
    turns = fillTurns(rec);
    g = await createGame(startInfoFromRecord(rec), engineRoot);
  } catch (e) {
    result.error = `setup failed: ${errorText(e)}`;
    return finish();
  }
  opts.onStart?.(g);
  const placements = new PlacementTracker();

  for (const turn of turns) {
    const parsed = parseTurn(turn);
    if ("error" in parsed) {
      result.error = parsed.error;
      return finish();
    }
    let out;
    try {
      out = g.tick(parsed.turn);
    } catch (e) {
      result.error = `engine threw at turn ${turn.turnNumber}: ${errorText(e)}`;
      return finish();
    }
    if (!out.ok) {
      result.error = `engine error at turn ${turn.turnNumber}: ${out.error}`;
      return finish();
    }
    result.turns++;
    const expected = turn.hash;
    if (expected !== undefined && expected !== null) {
      for (const h of out.hashes) {
        if (h.tick !== turn.turnNumber) continue;
        result.hashChecks++;
        if (h.hash !== expected) {
          result.mismatchAt = turn.turnNumber;
          return finish();
        }
      }
    }
    if (turn.turnNumber % 10 === 0) placements.observe(g);
    if (opts.onTick?.(g, turn.turnNumber) === false) break;
  }
  const p = placements.result(g);
  result.placements = p.byClient;
  result.ranked = p.n;
  return finish();
}

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
