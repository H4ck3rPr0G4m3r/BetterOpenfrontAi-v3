/**
 * Final placement of a game (spec §8.3 value target): humans and nations
 * (bots excluded) ranked by when they died, the last to die ranking best;
 * survivors rank above every dead player, by final tiles. Players who never
 * spawned are not ranked (real lobbies hold many idle humans).
 */
import type { EngineGame, Player } from "../engine/Adapter";
import { PlayerType } from "../engine/Adapter";

export interface Placements {
  /** clientID -> place (1 = best), humans only. */
  byClient: Record<string, number>;
  /** Number of ranked players (humans + nations that spawned). */
  n: number;
}

/** Value target for a place among n: 1 for the winner, 0 for the first out. */
export function placementValue(place: number, n: number): number {
  if (n <= 1) return 1;
  return 1 - (Math.min(Math.max(place, 1), n) - 1) / (n - 1);
}

export class PlacementTracker {
  private deathTick = new Map<number, number>();
  private spawned = new Set<number>();

  /** Call after ticks (every tick or every few; a death is dated to the first call that sees it). */
  observe(g: EngineGame): void {
    const tick = g.ticks();
    for (const p of g.game.allPlayers()) {
      if (p.type() === PlayerType.Bot) continue;
      const id = p.smallID();
      if (this.deathTick.has(id)) continue;
      if (p.isAlive()) {
        if (p.hasSpawned()) this.spawned.add(id);
      } else if (this.spawned.has(id) || (p.hasSpawned() && !g.game.inSpawnPhase())) {
        this.spawned.add(id);
        this.deathTick.set(id, tick);
      }
    }
  }

  /** Death tick of a ranked player (null while alive). */
  diedAt(p: Player): number | null {
    return this.deathTick.get(p.smallID()) ?? null;
  }

  result(g: EngineGame): Placements {
    this.observe(g);
    const ranked = g.game
      .allPlayers()
      .filter((p) => p.type() !== PlayerType.Bot && this.spawned.has(p.smallID()))
      .map((p) => ({ p, died: this.deathTick.get(p.smallID()) ?? Infinity, tiles: p.isAlive() ? p.numTilesOwned() : 0 }));
    ranked.sort((a, b) => b.died - a.died || b.tiles - a.tiles || a.p.smallID() - b.p.smallID());
    const byClient: Record<string, number> = {};
    ranked.forEach((r, i) => {
      const id = r.p.clientID();
      if (r.p.type() === PlayerType.Human && id !== null) byClient[id] = i + 1;
    });
    return { byClient, n: ranked.length };
  }
}
