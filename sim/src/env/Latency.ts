/**
 * Self-play latency (roadmap §2 Timing): a step's intents reach the engine
 * after a delay of lo..hi turns drawn from a PRNG seeded by the game seed.
 * Delay d puts them in the d-th turn executed after the observation. A seat's
 * intents are never reordered: a later step never overtakes an earlier one.
 */
import { PseudoRandom, simpleHash } from "../engine/Adapter";
import type { RecordIntent } from "../record/Record";

export class Latency {
  private readonly rng: PseudoRandom;
  private queue: { at: number; intent: RecordIntent }[] = [];
  private lastAt = new Map<string, number>();

  constructor(seed: string, readonly range: readonly [number, number]) {
    if (!(range[0] >= 1 && range[1] >= range[0])) throw new Error(`bad latency range ${range}`);
    this.rng = new PseudoRandom(simpleHash(`${seed}:latency`));
  }

  /** Queues one seat's intents decided before executing turn `now`; returns the turn they land in. */
  push(now: number, seat: string, intents: RecordIntent[]): number {
    const d = this.rng.nextInt(this.range[0], this.range[1] + 1);
    const at = Math.max(now + d - 1, this.lastAt.get(seat) ?? 0);
    if (intents.length === 0) return at;
    this.lastAt.set(seat, at);
    for (const intent of intents) this.queue.push({ at, intent });
    return at;
  }

  /** Intents for turn `turn`, in queue order. */
  due(turn: number): RecordIntent[] {
    const out = this.queue.filter((q) => q.at <= turn).map((q) => q.intent);
    if (out.length > 0) this.queue = this.queue.filter((q) => q.at > turn);
    return out;
  }

  pending(): number {
    return this.queue.length;
  }
}
