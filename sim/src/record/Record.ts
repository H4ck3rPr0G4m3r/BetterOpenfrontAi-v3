/**
 * Game records as the OpenFront API serves them (`GET /game/{id}`) and as our
 * own engine runs write them. Plain JSON types: nothing here imports the
 * engine, so record handling never depends on which engine commit is bundled.
 */
import fs from "fs";
import zlib from "zlib";

export interface RecordIntent {
  type: string;
  clientID: string;
  [field: string]: unknown;
}

export interface RecordTurn {
  turnNumber: number;
  intents: RecordIntent[];
  hash?: number | null;
}

export interface RecordPlayer {
  clientID: string;
  username: string;
  clanTag?: string | null;
  publicID?: string | null;
  [field: string]: unknown;
}

export interface GameRecordJson {
  version?: string;
  gitCommit: string;
  info: {
    gameID: string;
    lobbyCreatedAt?: number;
    config: Record<string, unknown> & { gameMap: string; gameMapSize: string };
    players: RecordPlayer[];
    tribes?: { name: string }[];
    winner?: unknown;
    start?: number;
    end?: number;
    duration?: number;
    num_turns: number;
    [field: string]: unknown;
  };
  turns: RecordTurn[];
  [field: string]: unknown;
}

export function readRecord(path: string): GameRecordJson {
  const raw = fs.readFileSync(path);
  const text = path.endsWith(".gz") ? zlib.gunzipSync(raw).toString("utf8") : raw.toString("utf8");
  return JSON.parse(text) as GameRecordJson;
}

export function writeRecord(path: string, rec: GameRecordJson): void {
  const text = JSON.stringify(rec);
  const tmp = `${path}.tmp`;
  fs.writeFileSync(tmp, path.endsWith(".gz") ? zlib.gzipSync(text) : text);
  fs.renameSync(tmp, path);
}

/**
 * Dense turns 0..n-1, where n = max(num_turns, last stored turn + 1): the
 * record omits turns with no intents and no hash, and the engine's
 * decompressGameRecord fills them back the same way.
 */
export function fillTurns(rec: GameRecordJson): RecordTurn[] {
  const out: RecordTurn[] = [];
  for (const turn of rec.turns) {
    while (out.length < turn.turnNumber) out.push({ turnNumber: out.length, intents: [] });
    if (turn.turnNumber < out.length) {
      throw new Error(`record turns out of order at ${turn.turnNumber}`);
    }
    out.push(turn);
  }
  while (out.length < rec.info.num_turns) out.push({ turnNumber: out.length, intents: [] });
  return out;
}

/** Deep clone for tests that tamper with a record. */
export function cloneRecord(rec: GameRecordJson): GameRecordJson {
  return JSON.parse(JSON.stringify(rec)) as GameRecordJson;
}
