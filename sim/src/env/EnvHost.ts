/**
 * The env role's game pool: `games` GameEnvs of up to `seats` seats each,
 * rows game-major (row = game * seats + seat). RESET takes one spec per game
 * (null keeps that game as it is); a finished game's rows stay invalid until
 * the client resets it.
 */
import { A, K } from "../obs/Layout";
import { ObsBuffer } from "../obs/ObsBuffer";
import { FLAG } from "../replay/Sampler";
import { type EnvHost, type EnvRows } from "../bridge/Server";
import { STATS_PER_ROW, TERMINAL } from "../bridge/Protocol";
import { GameEnv, type GameSpec, type StepResult } from "./GameEnv";

export class GameEnvHost implements EnvHost {
  readonly rows: number;
  readonly buf: ObsBuffer;
  readonly envs: GameEnv[];
  private readonly reward: Float32Array;
  private readonly flags: Uint8Array;
  private readonly stats: Float32Array;
  private readonly terminal: Uint8Array;

  constructor(engineRoot: string, readonly games: number, readonly seats: number, gitCommit = "SYNTHETIC") {
    this.rows = games * seats;
    this.buf = new ObsBuffer(this.rows);
    this.envs = Array.from({ length: games }, () => new GameEnv(engineRoot, seats, gitCommit));
    this.reward = new Float32Array(this.rows);
    this.flags = new Uint8Array(this.rows);
    this.stats = new Float32Array(this.rows * STATS_PER_ROW);
    this.terminal = new Uint8Array(this.rows);
    for (let r = 0; r < this.rows; r++) this.buf.clearRow(r);
  }

  async reset(specs: unknown[]): Promise<EnvRows> {
    if (specs.length !== this.games) throw new Error(`RESET needs ${this.games} entries (null keeps a game), got ${specs.length}`);
    this.clear();
    const errors: Record<number, string> = {};
    for (let gi = 0; gi < this.games; gi++) {
      const spec = specs[gi] as GameSpec | null;
      if (spec === null) continue;
      const env = this.envs[gi];
      if (env.g !== null && !env.over) this.put(gi, env.abort());
      try {
        await env.reset(spec, this.buf, gi * this.seats);
        for (let j = 0; j < spec.seats; j++) this.flags[gi * this.seats + j] |= FLAG.NEXT | FLAG.FRESH;
      } catch (e) {
        errors[gi] = e instanceof Error ? e.message : String(e);
      }
    }
    return this.out({ resetErrors: errors });
  }

  async step(actions: Int32Array): Promise<EnvRows> {
    this.clear();
    const records: { game: number; path: string }[] = [];
    const simErrors: { game: number; error: string; seed: string }[] = [];
    const decode: Record<string, number> = {};
    for (let gi = 0; gi < this.games; gi++) {
      const env = this.envs[gi];
      if (env.g === null || env.over) continue;
      const results = env.step(actions.subarray(gi * this.seats * K * A, (gi + 1) * this.seats * K * A), this.buf, gi * this.seats);
      this.put(gi, results);
      for (const r of results) {
        for (const [k, v] of Object.entries(r.decode.ok)) decode[`ok.${k}`] = (decode[`ok.${k}`] ?? 0) + v;
        for (const [k, m] of Object.entries(r.decode.dropped)) for (const [why, v] of Object.entries(m)) if (v) decode[`drop.${k}.${why}`] = (decode[`drop.${k}.${why}`] ?? 0) + v;
      }
      if (env.over && env.lastRecord) records.push({ game: gi, path: env.lastRecord });
      if (env.lastError && results.some((r) => r.terminal === "sim_error")) simErrors.push({ game: gi, error: env.lastError, seed: env.spec!.seed });
    }
    const land = this.envs.map((e) => (e.g === null ? 0 : e.g.game.totalLandTiles()));
    return this.out({ records, simErrors, decode, land, over: this.envs.map((e) => e.g === null || e.over) });
  }

  private put(gi: number, results: StepResult[]): void {
    results.forEach((r, j) => {
      const row = gi * this.seats + j;
      this.reward[row] = r.reward;
      this.flags[row] = (r.valid ? FLAG.VALID : 0) | (r.done ? FLAG.DONE : 0) | (r.next ? FLAG.NEXT : 0);
      this.terminal[row] = TERMINAL[r.terminal];
      this.stats.set(r.stats, row * STATS_PER_ROW);
    });
  }

  private clear(): void {
    this.reward.fill(0);
    this.flags.fill(0);
    this.stats.fill(0);
    this.terminal.fill(0);
  }

  private out(extra: unknown): EnvRows {
    return { buf: this.buf, reward: this.reward, flags: this.flags, stats: this.stats, terminal: this.terminal, extra };
  }
}

export async function makeEnvHost(engineRoot: string, games: number, seats: number, gitCommit?: string): Promise<GameEnvHost> {
  return new GameEnvHost(engineRoot, games, seats, gitCommit);
}
