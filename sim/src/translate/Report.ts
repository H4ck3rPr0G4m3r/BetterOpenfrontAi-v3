/**
 * Round-trip report (spec §10.4 exit gate): translate each teacher intent,
 * decode it back against the same state, compare. Representable means the
 * decoded intent matches in type, target, troops (within one bucket) and
 * location (near <= 3 tiles, far <= the fine-spot span, at least 3 tiles).
 */
import { decodeAction, type DecodeContext } from "../action/Codec";
import { createGame, parseTurn, startInfoFromRecord } from "../engine/Adapter";
import { GameObsCache } from "../obs/GameObsCache";
import { DECISION_TURNS, F } from "../obs/Layout";
import { ObsBuffer } from "../obs/ObsBuffer";
import { encodeSeat } from "../obs/SeatEncoder";
import { fillTurns, type GameRecordJson, type RecordIntent } from "../record/Record";
import { StepLabeler, type Translated, translateIntent, troopBucket, type TranslateContext } from "./Translator";

export interface TranslatorReport {
  games: number;
  total: number;
  excluded: number;
  representable: number;
  rate: number;
  byType: Record<string, { total: number; ok: number; reasons: Record<string, number> }>;
  tileErr: { near: { p50: number; p95: number }; far: { p50: number; p95: number } };
  insideWindowRate: number;
  carried: number;
}

const TARGET_FIELD: Record<string, string> = {
  attack: "targetID", allianceRequest: "recipient", allianceReject: "requestor", allianceExtension: "recipient",
  breakAlliance: "recipient", targetPlayer: "target", embargo: "targetID",
};
const TILE_FIELD: Record<string, string> = { spawn: "tile", boat: "dst", build_unit: "tile", move_warship: "tile" };
const ENTITY_FIELD: Record<string, string> = { upgrade_structure: "unitId", cancel_boat: "unitID", delete_unit: "unitId", cancel_attack: "attackID" };

/** Compares a decoded intent with the original; returns null when they agree, else the reason. */
export function compareRoundTrip(
  orig: RecordIntent,
  t: Translated,
  decoded: ReturnType<typeof decodeAction>,
  ctx: TranslateContext,
): { reason: string | null; tileErr?: number } {
  if ("unrepresentable" in t) return { reason: t.unrepresentable };
  if (decoded === null) return { reason: "decoded_stop" };
  if ("dropped" in decoded) return { reason: `decode:${decoded.dropped}` };
  const d = decoded.intent;
  if (d.type !== orig.type) return { reason: "type_mismatch" };
  if (orig.type === "embargo" && d.action !== orig.action) return { reason: "embargo_action" };
  const tf = TARGET_FIELD[orig.type];
  if (tf && (d[tf] ?? null) !== (orig[tf] ?? null)) return { reason: "target_mismatch" };
  if (orig.type === "attack" || orig.type === "boat") {
    const want = troopBucket(orig.troops as number | null, ctx.troopsAtObs);
    const got = troopBucket(d.troops as number, ctx.troopsAtObs);
    if (Math.abs(want - got) > 1) return { reason: "troops_bucket" };
  }
  const ef = ENTITY_FIELD[orig.type];
  if (ef && d[ef] !== orig[ef]) return { reason: "entity_mismatch" };
  if (orig.type === "build_unit" && d.unit !== orig.unit) return { reason: "build_type" };
  const tile = TILE_FIELD[orig.type];
  if (tile) {
    const w = ctx.g.game.width();
    const a = orig[tile] as number;
    const b = d[tile] as number;
    const err = Math.hypot((a % w) - (b % w), Math.floor(a / w) - Math.floor(b / w));
    const near = "action" in t && t.action[F.locMode] === 0;
    const limit = near ? 3 : Math.max(3, 2 * ctx.geometry.fineSpan());
    if (err > limit) return { reason: near ? "tile_far_near" : "tile_far_far", tileErr: err };
    return { reason: null, tileErr: err };
  }
  return { reason: null };
}

function pct(xs: number[], p: number): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(p * (s.length - 1)))];
}

export class ReportBuilder {
  games = 0;
  total = 0;
  excluded = 0;
  ok = 0;
  carried = 0;
  nearCount = 0;
  locCount = 0;
  byType: TranslatorReport["byType"] = {};
  nearErr: number[] = [];
  farErr: number[] = [];

  add(orig: RecordIntent, t: Translated, cmp: { reason: string | null; tileErr?: number }): void {
    this.total++;
    const e = (this.byType[orig.type] ??= { total: 0, ok: 0, reasons: {} });
    e.total++;
    if ("unrepresentable" in t && t.unrepresentable.startsWith("excluded:")) {
      this.excluded++;
      e.reasons.excluded = (e.reasons.excluded ?? 0) + 1;
      return;
    }
    if ("action" in t && t.near !== undefined) {
      this.locCount++;
      if (t.near) this.nearCount++;
      if (cmp.tileErr !== undefined) (t.near ? this.nearErr : this.farErr).push(cmp.tileErr);
    }
    if (cmp.reason === null) {
      this.ok++;
      e.ok++;
    } else e.reasons[cmp.reason] = (e.reasons[cmp.reason] ?? 0) + 1;
  }

  build(): TranslatorReport {
    const representable = this.total - this.excluded;
    return {
      games: this.games,
      total: this.total,
      excluded: this.excluded,
      representable,
      rate: representable > 0 ? this.ok / representable : 1,
      byType: this.byType,
      tileErr: {
        near: { p50: pct(this.nearErr, 0.5), p95: pct(this.nearErr, 0.95) },
        far: { p50: pct(this.farErr, 0.5), p95: pct(this.farErr, 0.95) },
      },
      insideWindowRate: this.locCount > 0 ? this.nearCount / this.locCount : 0,
      carried: this.carried,
    };
  }
}

/**
 * Replays a record and, at every decision step, translates and round-trips the
 * intents each human seat sends in the next DECISION_TURNS turns, against the
 * state before those turns (the observation the label is paired with).
 */
export async function reportRecord(rec: GameRecordJson, engineRoot: string, builder = new ReportBuilder()): Promise<ReportBuilder> {
  const turns = fillTurns(rec);
  const g = await createGame(startInfoFromRecord(rec), engineRoot);
  const cache = new GameObsCache(g);
  const buf = new ObsBuffer(1);
  const seats = rec.info.players.map((p) => p.clientID);
  const labelers = new Map(seats.map((s) => [s, new StepLabeler()]));
  builder.games++;
  for (let t = 0; t < turns.length; t++) {
    if (t % DECISION_TURNS === 0) {
      cache.update();
      const upcoming = turns.slice(t, t + DECISION_TURNS).flatMap((x) => x.intents);
      for (const clientID of seats) {
        const mine = upcoming.filter((i) => i.clientID === clientID);
        const labeler = labelers.get(clientID)!;
        labeler.push(mine);
        const self = g.game.playerByClientID(clientID);
        if (self === null) {
          labeler.clear();
          continue;
        }
        const tables = encodeSeat(cache, g, clientID, buf, 0);
        const tctx: TranslateContext = { g, clientID, tables, geometry: cache.geometry, troopsAtObs: self.troops() };
        const dctx: DecodeContext = { g, clientID, tables, geometry: cache.geometry };
        labeler.take(tctx, (intent, tr) => {
          const decoded = "action" in tr ? decodeAction(tr.action, 0, dctx) : null;
          builder.add(intent, tr, compareRoundTrip(intent, tr, decoded, tctx));
        });
      }
    }
    const parsed = parseTurn(turns[t]);
    if ("error" in parsed) break;
    if (!g.tick(parsed.turn).ok) break;
  }
  for (const l of labelers.values()) builder.carried += l.carried;
  return builder;
}

export { translateIntent };
