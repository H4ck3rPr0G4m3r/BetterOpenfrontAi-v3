/**
 * Engine-format game records of games we run (env, bench), so AI games open
 * in the replay viewer and replay with every hash checked. Turns with no
 * intents and no hash are omitted, as the server does.
 */
import type { GameRecordJson, RecordPlayer, RecordTurn } from "./Record";

export class RecordWriter {
  private turns: RecordTurn[] = [];
  private readonly startedAt = Date.now();

  constructor(
    private readonly gameID: string,
    private readonly config: Record<string, unknown>,
    private readonly players: RecordPlayer[],
    private readonly gitCommit: string,
  ) {}

  add(turnNumber: number, intents: RecordTurn["intents"], hash: number | undefined): void {
    if (intents.length === 0 && hash === undefined) return;
    this.turns.push(hash === undefined ? { turnNumber, intents } : { turnNumber, intents, hash });
  }

  build(numTurns: number, winner: unknown[] | undefined): GameRecordJson {
    const end = Date.now();
    return {
      version: "v0.0.2",
      gitCommit: this.gitCommit,
      info: {
        gameID: this.gameID,
        lobbyCreatedAt: this.startedAt,
        config: this.config as GameRecordJson["info"]["config"],
        players: this.players.map((p) => ({ ...p, persistentID: null })),
        start: this.startedAt,
        end,
        duration: numTurns / 10,
        num_turns: numTurns,
        winner,
        lobbyFillTime: 0,
      },
      turns: this.turns,
    };
  }
}
