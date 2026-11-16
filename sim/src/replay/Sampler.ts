/**
 * Streaming teacher samples from game records (M2 Task 9, roadmap §3.4 replay
 * role). Each of `gamesInFlight` slots replays one record; every `next()`
 * advances every slot by one decision step. A row's observation is the state
 * before the step's DECISION_TURNS turns, its label the teacher's intents in
 * those turns.
 *
 * Row lifecycle (flags): FRESH|VALID on a game's first step; VALID while the
 * teacher plays; DONE when the teacher dies, disconnects, never spawned by the
 * end of the spawn phase, or the record ends / desyncs; 0 (invalid) after
 * that until the slot gets a new game. A finished game is replaced from the
 * queue in the same `next()`.
 */
import { createGame, type EngineGame, parseTurn, startInfoFromRecord } from "../engine/Adapter";
import { GameObsCache } from "../obs/GameObsCache";
import { A, DECISION_TURNS, K } from "../obs/Layout";
import { ObsBuffer } from "../obs/ObsBuffer";
import { encodeSeat } from "../obs/SeatEncoder";
import { fillTurns, type GameRecordJson, readRecord, type RecordTurn } from "../record/Record";
import { StepLabeler, type TranslateContext } from "../translate/Translator";

export const FLAG = { VALID: 1, DONE: 2, NEXT: 4, FRESH: 8 } as const;

export interface TeacherSpec {
  clientID: string;
  weight: number;
  value: number;
}

export interface SampleItem {
  /** Record path, or the record itself (tests). */
  path?: string;
  record?: GameRecordJson;
  teachers: TeacherSpec[];
}

export interface SampleBatch {
  buf: ObsBuffer;
  labels: Int32Array; // rows * K * A
  weight: Float32Array;
  value: Float32Array;
  flags: Uint8Array;
}

export interface SamplerStats {
  gamesStarted: number;
  gamesFinished: number;
  gamesAborted: number;
  samples: number;
  carried: number;
  errors: string[];
}

type SeatState = "fresh" | "active" | "finished" | "idle";

interface Seat {
  teacher: TeacherSpec;
  labeler: StepLabeler;
  state: SeatState;
}

interface Slot {
  g: EngineGame;
  cache: GameObsCache;
  turns: RecordTurn[];
  t: number;
  seats: Seat[];
  gameId: string;
}

export class Sampler {
  readonly rows: number;
  readonly buf: ObsBuffer;
  private readonly labels: Int32Array;
  private readonly weight: Float32Array;
  private readonly value: Float32Array;
  private readonly flags: Uint8Array;
  private readonly slots: (Slot | null)[];
  /** Rows that were active and still owe a DONE flag (game ended with no replacement). */
  private readonly owesDone: boolean[];
  private queue: SampleItem[] = [];
  readonly stats: SamplerStats = { gamesStarted: 0, gamesFinished: 0, gamesAborted: 0, samples: 0, carried: 0, errors: [] };

  constructor(
    readonly engineRoot: string,
    readonly gamesInFlight: number,
    readonly seatsPerGame: number,
  ) {
    this.rows = gamesInFlight * seatsPerGame;
    this.buf = new ObsBuffer(this.rows);
    this.labels = new Int32Array(this.rows * K * A);
    this.weight = new Float32Array(this.rows);
    this.value = new Float32Array(this.rows);
    this.flags = new Uint8Array(this.rows);
    this.slots = new Array(gamesInFlight).fill(null);
    this.owesDone = new Array(this.rows).fill(false);
  }

  load(items: SampleItem[]): void {
    this.queue.push(...items);
  }

  queued(): number {
    return this.queue.length;
  }

  active(): number {
    return this.slots.filter((s) => s !== null).length;
  }

  async next(): Promise<SampleBatch> {
    this.labels.fill(-1);
    this.weight.fill(0);
    this.value.fill(0);
    this.flags.fill(0);
    for (let s = 0; s < this.gamesInFlight; s++) {
      if (this.slots[s] === null) await this.fill(s);
      const slot = this.slots[s];
      if (slot === null) {
        this.flushDone(s);
        continue;
      }
      const finished = this.stepSlot(s, slot);
      if (finished) {
        this.slots[s] = null;
        for (let j = 0; j < this.seatsPerGame; j++) {
          const r = s * this.seatsPerGame + j;
          // Rows that already got DONE this step don't owe another.
          this.owesDone[r] = j < slot.seats.length && slot.seats[j].state === "active" && !(this.flags[r] & FLAG.DONE);
        }
      }
    }
    return { buf: this.buf, labels: this.labels, weight: this.weight, value: this.value, flags: this.flags };
  }

  private flushDone(s: number): void {
    for (let j = 0; j < this.seatsPerGame; j++) {
      const r = s * this.seatsPerGame + j;
      if (this.owesDone[r]) {
        this.flags[r] = FLAG.DONE;
        this.owesDone[r] = false;
      }
    }
  }

  /** Starts the next loadable game from the queue in slot s. */
  private async fill(s: number): Promise<void> {
    while (this.queue.length > 0) {
      const item = this.queue.shift()!;
      try {
        const rec = item.record ?? readRecord(item.path!);
        const ids = new Set(rec.info.players.map((p) => p.clientID));
        const teachers = item.teachers
          .filter((t) => ids.has(t.clientID))
          .sort((a, b) => b.weight - a.weight)
          .slice(0, this.seatsPerGame);
        if (teachers.length === 0) continue;
        const g = await createGame(startInfoFromRecord(rec), this.engineRoot);
        this.slots[s] = {
          g,
          cache: new GameObsCache(g),
          turns: fillTurns(rec),
          t: 0,
          seats: teachers.map((teacher) => ({ teacher, labeler: new StepLabeler(), state: "fresh" })),
          gameId: rec.info.gameID,
        };
        for (let j = 0; j < this.seatsPerGame; j++) this.owesDone[s * this.seatsPerGame + j] = false;
        this.stats.gamesStarted++;
        return;
      } catch (e) {
        this.note(`load ${item.path ?? item.record?.info.gameID}: ${e instanceof Error ? e.message : String(e)}`);
      }
    }
  }

  /** One decision step for a slot; returns true when the game is over. */
  private stepSlot(s: number, slot: Slot): boolean {
    const { g, cache } = slot;
    const game = g.game;
    cache.update();
    const upcoming = slot.turns.slice(slot.t, slot.t + DECISION_TURNS).flatMap((x) => x.intents);
    const rowsThisStep: number[] = [];
    for (let j = 0; j < slot.seats.length; j++) {
      const seat = slot.seats[j];
      const r = s * this.seatsPerGame + j;
      if (seat.state === "finished") continue;
      const self = game.playerByClientID(seat.teacher.clientID);
      const gone =
        self === null ||
        self.isDisconnected() ||
        (self.hasSpawned() && !self.isAlive() && !game.inSpawnPhase()) ||
        (!self.hasSpawned() && !game.inSpawnPhase());
      if (gone) {
        this.flags[r] = seat.state === "fresh" ? 0 : FLAG.DONE;
        seat.state = "finished";
        seat.labeler.clear();
        continue;
      }
      const tables = encodeSeat(cache, g, seat.teacher.clientID, this.buf, r);
      seat.labeler.push(upcoming.filter((i) => i.clientID === seat.teacher.clientID));
      const ctx: TranslateContext = { g, clientID: seat.teacher.clientID, tables, geometry: cache.geometry, troopsAtObs: self.troops() };
      this.labels.set(seat.labeler.take(ctx), r * K * A);
      this.weight[r] = seat.teacher.weight;
      this.value[r] = seat.teacher.value;
      this.flags[r] = FLAG.VALID | FLAG.NEXT | (seat.state === "fresh" ? FLAG.FRESH : 0);
      seat.state = "active";
      rowsThisStep.push(r);
    }

    // Advance the game, checking the record's hashes.
    const end = Math.min(slot.t + DECISION_TURNS, slot.turns.length);
    for (; slot.t < end; slot.t++) {
      const turn = slot.turns[slot.t];
      const problem = this.tick(g, turn);
      if (problem !== null) {
        // Desync or engine error: these labels can't be trusted; the rows end here.
        this.note(`${slot.gameId}: ${problem}`);
        for (const r of rowsThisStep) this.flags[r] = FLAG.DONE;
        for (const seat of slot.seats) seat.state = "finished";
        this.stats.gamesAborted++;
        this.finish(slot);
        return true;
      }
    }
    for (const r of rowsThisStep) if (this.flags[r] & FLAG.VALID) this.stats.samples++;

    const recordEnded = slot.t >= slot.turns.length;
    const everyoneDone = slot.seats.every((x) => x.state === "finished");
    if (recordEnded) {
      for (let j = 0; j < slot.seats.length; j++) {
        if (slot.seats[j].state !== "active") continue;
        this.flags[s * this.seatsPerGame + j] |= FLAG.DONE;
        this.flags[s * this.seatsPerGame + j] &= ~FLAG.NEXT;
      }
    }
    if (recordEnded || everyoneDone) {
      this.stats.gamesFinished++;
      this.finish(slot);
      return true;
    }
    return false;
  }

  private finish(slot: Slot): void {
    for (const seat of slot.seats) this.stats.carried += seat.labeler.carried;
  }

  private tick(g: EngineGame, turn: RecordTurn): string | null {
    const parsed = parseTurn(turn);
    if ("error" in parsed) return parsed.error;
    let out;
    try {
      out = g.tick(parsed.turn);
    } catch (e) {
      return `engine threw at turn ${turn.turnNumber}: ${e instanceof Error ? e.message : String(e)}`;
    }
    if (!out.ok) return `engine error at turn ${turn.turnNumber}: ${out.error}`;
    if (turn.hash !== undefined && turn.hash !== null) {
      for (const h of out.hashes) if (h.tick === turn.turnNumber && h.hash !== turn.hash) return `hash mismatch at turn ${turn.turnNumber}`;
    }
    return null;
  }

  private note(msg: string): void {
    this.stats.errors.push(msg);
    if (this.stats.errors.length > 50) this.stats.errors.shift();
  }
}
