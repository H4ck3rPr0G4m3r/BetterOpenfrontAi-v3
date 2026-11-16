/**
 * Multi-seat game driven by policy actions (M2 Task 11, spec §12). Each
 * decision step: decode every active seat's K actions against the tables of
 * the observation they saw, queue the intents with latency, run
 * DECISION_TURNS turns, then judge terminals and encode the next
 * observation. Bots and nations play through the engine's own executions.
 *
 * Terminals: `win` (the engine named this seat the winner), `death` (out of
 * the game after the spawn phase, including never having spawned),
 * `tick_cap` (time is up, or someone else won: placement decides),
 * `sim_error` (engine failure), `aborted` (the client reset it mid-game).
 * Reward is 0 until M6; stats carry [tiles, troops, kills, placement, ticks].
 */
import fs from "fs";
import path from "path";
import { DecodeStats, decodeStep } from "../action/Codec";
import { createGame, type EngineGame, type GameStartInfo, parseTurn } from "../engine/Adapter";
import { GameObsCache } from "../obs/GameObsCache";
import { A, DECISION_TURNS, K } from "../obs/Layout";
import type { ObsBuffer } from "../obs/ObsBuffer";
import { encodeSeat, type EntityTables } from "../obs/SeatEncoder";
import { PlacementTracker } from "../replay/Placements";
import { type RecordIntent, type RecordPlayer, writeRecord } from "../record/Record";
import { RecordWriter } from "../record/RecordWriter";
import { AI_CLAN_TAG, seatClientID, syntheticConfig, syntheticGameID } from "../record/SyntheticRecord";
import type { Terminal } from "../bridge/Protocol";
import { Latency } from "./Latency";

export interface GameSpec {
  map: string;
  seed: string;
  seats: number;
  names?: string[];
  bots: number;
  nations: "default" | "disabled" | number;
  difficulty: "Easy" | "Medium" | "Hard" | "Impossible";
  /** Ticks after the spawn phase before the game is cut off. */
  tickCap: number;
  latency: [number, number];
  /** false, or a path to write the engine-format record to when the game ends. */
  record: false | string;
}

export interface StepResult {
  reward: number;
  /** The seat acted this step. */
  valid: boolean;
  /** The seat's episode ended this step. */
  done: boolean;
  /** The seat acts next step. */
  next: boolean;
  terminal: Terminal;
  stats: [number, number, number, number, number];
  decode: DecodeStats;
}

export const AI_PREFIX = "[AI] ";

/** Username the engine accepts ([a-zA-Z0-9_ üÜ.]); the AI tag travels as the clan tag. */
export function seatUsername(name: string | undefined, seat: number): string {
  const bare = (name ?? `seat${seat}`).replace(/^\[AI\]\s*/, "");
  const clean = bare.replace(/[^a-zA-Z0-9_ üÜ.]/g, "_").trim().slice(0, 27);
  return clean.length > 0 ? clean : `seat${seat}`;
}

interface Seat {
  clientID: string;
  active: boolean;
  kills: number;
  terminal: Terminal;
}

export class GameEnv {
  g: EngineGame | null = null;
  cache: GameObsCache | null = null;
  spec: GameSpec | null = null;
  tables: EntityTables[] = [];
  seats: Seat[] = [];
  private latency: Latency | null = null;
  private writer: RecordWriter | null = null;
  private tracker = new PlacementTracker();
  private turn = 0;
  private spawnEndTick: number | null = null;
  private winner: unknown[] | undefined;
  /** Path of the record written when the last game ended (if any). */
  lastRecord: string | null = null;
  lastError: string | null = null;

  constructor(
    readonly engineRoot: string,
    readonly maxSeats: number,
    readonly gitCommit = "SYNTHETIC",
  ) {}

  get over(): boolean {
    return this.seats.every((s) => !s.active);
  }

  /** Starts a new game and writes each seat's first observation to rows [rowBase, rowBase + maxSeats). */
  async reset(spec: GameSpec, buf: ObsBuffer, rowBase: number): Promise<EntityTables[]> {
    if (spec.seats < 1 || spec.seats > this.maxSeats) throw new Error(`game wants ${spec.seats} seats, env has ${this.maxSeats}`);
    this.spec = spec;
    const gameID = syntheticGameID(spec.seed);
    const config = syntheticConfig(spec);
    const players: RecordPlayer[] = Array.from({ length: spec.seats }, (_, i) => ({
      clientID: seatClientID(i),
      username: seatUsername(spec.names?.[i], i),
      clanTag: AI_CLAN_TAG,
    }));
    const start = { gameID, lobbyCreatedAt: 0, config, players } as unknown as GameStartInfo;
    this.g = await createGame(start, this.engineRoot);
    this.cache = new GameObsCache(this.g);
    this.latency = new Latency(spec.seed, spec.latency);
    this.writer = spec.record ? new RecordWriter(gameID, config, players, this.gitCommit) : null;
    this.tracker = new PlacementTracker();
    this.turn = 0;
    this.spawnEndTick = null;
    this.winner = undefined;
    this.lastRecord = null;
    this.lastError = null;
    this.seats = players.map((p) => ({ clientID: p.clientID, active: true, kills: 0, terminal: "none" }));
    this.encode(buf, rowBase);
    return this.tables;
  }

  /** One decision step. `actions` holds maxSeats * K * A values for this game. */
  step(actions: Int32Array, buf: ObsBuffer, rowBase: number): StepResult[] {
    const g = this.g;
    if (g === null || this.over) return this.seats.map(() => this.idleResult());
    const game = g.game;
    const decodes = this.seats.map(() => new DecodeStats());
    const actedBefore = this.seats.map((s) => s.active);

    // Decode against the observation each seat saw, then queue with latency.
    this.seats.forEach((seat, i) => {
      if (!seat.active) return;
      const ctx = { g, clientID: seat.clientID, tables: this.tables[i], geometry: this.cache!.geometry };
      const decoded = decodeStep(actions.subarray(i * K * A, (i + 1) * K * A), ctx, decodes[i]);
      const intents = decoded.filter((d): d is { intent: RecordIntent } => "intent" in d).map((d) => d.intent);
      this.latency!.push(this.turn, seat.clientID, intents);
    });

    // Run the turns.
    let error: string | null = null;
    for (let n = 0; n < DECISION_TURNS && error === null; n++) {
      const intents = this.latency!.due(this.turn);
      const parsed = parseTurn({ turnNumber: this.turn, intents });
      if ("error" in parsed) {
        error = parsed.error;
        break;
      }
      let out;
      try {
        out = g.tick(parsed.turn);
      } catch (e) {
        error = `engine threw: ${e instanceof Error ? e.message : String(e)}`;
        break;
      }
      if (!out.ok) {
        error = out.error ?? "tick failed";
        break;
      }
      this.writer?.add(this.turn, intents, out.hashes.find((h) => h.tick === this.turn)?.hash);
      for (const c of out.conquests) {
        const seat = this.seats.find((s) => game.playerByClientID(s.clientID)?.id() === c.conqueror);
        if (seat) seat.kills++;
      }
      if (out.winner) this.winner = out.winner;
      this.turn++;
      if (this.spawnEndTick === null && !game.inSpawnPhase()) this.spawnEndTick = game.ticks();
    }
    this.tracker.observe(g);

    // Terminals.
    const capped = this.spawnEndTick !== null && game.ticks() - this.spawnEndTick >= this.spec!.tickCap;
    for (const seat of this.seats) {
      if (!seat.active) continue;
      let t: Terminal = "none";
      const p = game.playerByClientID(seat.clientID);
      if (error !== null) t = "sim_error";
      else if (this.winner && this.winner[0] === "player" && this.winner[1] === seat.clientID) t = "win";
      else if (p === null || (!game.inSpawnPhase() && (!p.hasSpawned() || !p.isAlive()))) t = "death";
      else if (this.winner || capped) t = "tick_cap";
      if (t !== "none") {
        seat.active = false;
        seat.terminal = t;
      }
    }
    if (error !== null) this.lastError = error;

    const anyDone = this.seats.some((s, i) => actedBefore[i] && !s.active);
    const places = anyDone ? this.tracker.result(g).byClient : {};
    const results: StepResult[] = this.seats.map((seat, i) => {
      const p = game.playerByClientID(seat.clientID);
      const done = actedBefore[i] && !seat.active;
      return {
        reward: 0,
        valid: actedBefore[i],
        done,
        next: seat.active,
        terminal: done ? seat.terminal : "none",
        stats: [
          p?.isAlive() ? p.numTilesOwned() : 0,
          p?.isAlive() ? p.troops() : 0,
          seat.kills,
          done ? (places[seat.clientID] ?? 0) : 0,
          game.ticks(),
        ],
        decode: decodes[i],
      };
    });
    if (error === null) this.encode(buf, rowBase);
    if (this.over) this.finish();
    return results;
  }

  /** Ends a running game early (client reset mid-game): active seats get `aborted`. */
  abort(): StepResult[] {
    const g = this.g;
    const results = this.seats.map((seat) => {
      const was = seat.active;
      seat.active = false;
      if (was) seat.terminal = "aborted";
      return { ...this.idleResult(), done: was, valid: was, terminal: was ? ("aborted" as Terminal) : ("none" as Terminal), stats: [0, 0, seat.kills, 0, g?.ticks() ?? 0] as StepResult["stats"] };
    });
    return results;
  }

  private idleResult(): StepResult {
    return { reward: 0, valid: false, done: false, next: false, terminal: "none", stats: [0, 0, 0, 0, this.g?.ticks() ?? 0], decode: new DecodeStats() };
  }

  private encode(buf: ObsBuffer, rowBase: number): void {
    const g = this.g!;
    this.cache!.update();
    this.tables = [];
    for (let i = 0; i < this.maxSeats; i++) {
      const seat = this.seats[i];
      if (seat === undefined || !seat.active) {
        buf.clearRow(rowBase + i);
        if (seat !== undefined) this.tables.push(emptyTables());
        continue;
      }
      this.tables.push(encodeSeat(this.cache!, g, seat.clientID, buf, rowBase + i));
    }
  }

  private finish(): void {
    if (this.writer === null || !this.spec?.record) return;
    fs.mkdirSync(path.dirname(path.resolve(this.spec.record)), { recursive: true });
    writeRecord(this.spec.record, this.writer.build(this.turn, this.winner));
    this.lastRecord = this.spec.record;
    this.writer = null;
  }
}

function emptyTables(): EntityTables {
  return { players: [], tokenOf: new Map(), structs: [], mobile: [], attacks: [], attackOutgoing: [], windows: [], selfSmallId: -1 };
}
