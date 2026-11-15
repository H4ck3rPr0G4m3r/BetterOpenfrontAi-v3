/**
 * Synthetic game records from our own engine runs. Tests use them instead of
 * real records (real records hold usernames and are never committed, roadmap
 * §2). Seats spawn on random land during the spawn phase, then attack unowned
 * land on a fixed cadence; callers can script more intents per turn.
 */
import type { GameStartInfo } from "../engine/Adapter";
import { createGame, type EngineGame, GameMapSize, PseudoRandom, simpleHash } from "../engine/Adapter";
import type { GameRecordJson, RecordIntent, RecordTurn } from "./Record";

/** Record-valid client ID for seat i (GAME_ID_REGEX: 8-10 alphanumerics). */
export function seatClientID(seat: number): string {
  if (!Number.isInteger(seat) || seat < 0 || seat > 9999) throw new Error(`bad seat ${seat}`);
  return `SEAT${String(seat).padStart(4, "0")}`;
}

/** Deterministic 8-char game ID from a seed. */
export function syntheticGameID(seed: string): string {
  return ((simpleHash(seed) >>> 0).toString(36) + "00000000").slice(0, 8).replace(/[^A-Za-z0-9]/g, "0");
}

/** The tag every AI seat carries; the game renders it as "[AI] name". */
export const AI_CLAN_TAG = "AI";

export interface SyntheticOptions {
  map: string;
  seats: number;
  turns: number;
  seed: string;
  engineRoot: string;
  bots?: number;
  nations?: "default" | "disabled" | number;
  difficulty?: "Easy" | "Medium" | "Hard" | "Impossible";
  /** Turns between each seat's attacks on unowned land (default 20). */
  attackEvery?: number;
  /** Extra intents for a turn, decided from the state before it executes. */
  script?: (g: EngineGame, turnNumber: number, rng: PseudoRandom) => RecordIntent[];
  /** Called after each tick (tests use it to inspect state). */
  onTick?: (g: EngineGame, turnNumber: number) => void;
}

export function syntheticConfig(opts: Pick<SyntheticOptions, "map" | "bots" | "nations" | "difficulty">) {
  return {
    gameMap: opts.map,
    gameMapSize: GameMapSize.Normal,
    gameMode: "Free For All",
    gameType: "Private",
    difficulty: opts.difficulty ?? "Medium",
    nations: opts.nations ?? "disabled",
    bots: opts.bots ?? 0,
    donateGold: false,
    donateTroops: false,
    infiniteGold: false,
    infiniteTroops: false,
    instantBuild: false,
    randomSpawn: false,
  };
}

export async function makeSyntheticRecord(opts: SyntheticOptions): Promise<GameRecordJson> {
  const gameID = syntheticGameID(opts.seed);
  const players = Array.from({ length: opts.seats }, (_, i) => ({
    clientID: seatClientID(i),
    username: `seat${i}`,
    clanTag: AI_CLAN_TAG,
  }));
  const config = syntheticConfig(opts);
  const start = { gameID, lobbyCreatedAt: 0, config, players } as unknown as GameStartInfo;
  const g = await createGame(start, opts.engineRoot);
  const rng = new PseudoRandom(simpleHash(`${opts.seed}:intents`));
  const attackEvery = opts.attackEvery ?? 20;

  const turns: RecordTurn[] = [];
  for (let t = 0; t < opts.turns; t++) {
    const intents: RecordIntent[] = [];
    if (g.game.inSpawnPhase()) {
      // Seat i asks to spawn on turn 2 + i, and again every 10 turns until the engine accepts a tile.
      for (let i = 0; i < opts.seats; i++) {
        if (t < 2 + i || (t - 2 - i) % 10 !== 0) continue;
        const p = g.game.playerByClientID(seatClientID(i));
        if (p !== null && p.hasSpawned()) continue;
        const tile = randomFreeLand(g, rng);
        if (tile !== null) intents.push({ type: "spawn", clientID: seatClientID(i), tile });
      }
    } else if (t % attackEvery === 0) {
      for (let i = 0; i < opts.seats; i++) {
        const p = g.game.playerByClientID(seatClientID(i));
        if (p === null || !p.isAlive()) continue;
        intents.push({
          type: "attack",
          clientID: seatClientID(i),
          targetID: null,
          troops: Math.floor(p.troops() * 0.25),
        });
      }
    }
    if (opts.script) intents.push(...opts.script(g, t, rng));
    const turn: RecordTurn = { turnNumber: t, intents };
    const out = g.tick(turn);
    if (!out.ok) throw new Error(`synthetic game ${gameID} failed at turn ${t}: ${out.error}`);
    for (const h of out.hashes) if (h.tick === t) turn.hash = h.hash;
    opts.onTick?.(g, t);
    turns.push(turn);
  }

  return {
    version: "v0.0.2",
    gitCommit: "SYNTHETIC",
    info: {
      gameID,
      lobbyCreatedAt: 0,
      config,
      players: players.map((p) => ({ ...p, persistentID: null })),
      start: 0,
      end: opts.turns * 100,
      duration: opts.turns / 10,
      num_turns: opts.turns,
      winner: undefined,
      lobbyFillTime: 0,
    },
    turns: turns.filter((t) => t.intents.length > 0 || t.hash !== undefined),
  };
}

function randomFreeLand(g: EngineGame, rng: PseudoRandom): number | null {
  const game = g.game;
  const w = game.width();
  const h = game.height();
  for (let tries = 0; tries < 10_000; tries++) {
    const ref = game.ref(rng.nextInt(0, w), rng.nextInt(0, h));
    if (game.isLand(ref) && !game.map().isImpassable(ref) && !game.hasOwner(ref)) return ref;
  }
  return null;
}
