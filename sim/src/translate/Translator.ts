/**
 * Human intents -> v3 action labels (spec §10.4), using the same entity
 * tables and geometry as the observation the labels are paired with.
 * Locations use the near pointer when the tile lies in a window (the
 * highest-priority one containing it) and the far pointer otherwise.
 */
import type { EngineGame } from "../engine/Adapter";
import type { MinimapGeometry } from "../obs/Geometry";
import { tileToWindowCell } from "../obs/Geometry";
import { A, BUILD_TYPES, F, K, LOC_FAR, LOC_NEAR, NUM_STRUCTS, T, TROOP_BUCKETS, UNOWNED_TOKEN } from "../obs/Layout";
import type { EntityTables } from "../obs/SeatEncoder";
import type { RecordIntent } from "../record/Record";

export const EXCLUDED_TYPES = new Set([
  "emoji", "quick_chat", "donate_gold", "donate_troops", "embargo_all", "mark_disconnected", "kick_player",
  "toggle_pause", "update_game_config", "toggle_game_start_timer",
]);

export interface TranslateContext {
  g: EngineGame;
  clientID: string;
  tables: EntityTables;
  geometry: MinimapGeometry;
  /** The seat's troops at observation time (bucket base). */
  troopsAtObs: number;
}

export type Translated = { action: Int32Array; near?: boolean } | { unrepresentable: string };

/** Troop bucket for a sent amount; null troops = the client's default ratio (1/5 -> bucket 3). */
export function troopBucket(troops: number | null | undefined, base: number): number {
  if (troops === null || troops === undefined) return 3;
  if (base <= 0) return 0;
  return Math.max(0, Math.min(TROOP_BUCKETS - 1, Math.round(troops / base / 0.05) - 1));
}

function setLoc(a: Int32Array, tile: number, ctx: TranslateContext): boolean {
  const w = ctx.g.game.width();
  const x = tile % w;
  const y = Math.floor(tile / w);
  for (let i = 0; i < ctx.tables.windows.length; i++) {
    const cell = tileToWindowCell(ctx.tables.windows[i], x, y);
    if (cell >= 0) {
      a[F.locMode] = LOC_NEAR;
      a[F.window] = i;
      a[F.cell] = cell;
      return true;
    }
  }
  const { cell, fine } = ctx.geometry.tileToFine(x, y);
  a[F.locMode] = LOC_FAR;
  a[F.mmCell] = cell;
  a[F.fine] = fine;
  return false;
}

function playerToken(ctx: TranslateContext, playerId: unknown): number | null {
  if (typeof playerId !== "string") return null;
  const p = ctx.g.game.players().find((q) => q.id() === playerId);
  if (!p) return null;
  return ctx.tables.tokenOf.get(p.smallID()) ?? null;
}

export function translateIntent(intent: RecordIntent, ctx: TranslateContext): Translated {
  const type = intent.type;
  if (EXCLUDED_TYPES.has(type)) return { unrepresentable: `excluded:${type}` };
  const a = new Int32Array(A).fill(-1);
  const pointPlayer = (t: number, id: unknown): Translated => {
    const tok = playerToken(ctx, id);
    if (tok === null) return { unrepresentable: "player_not_token" };
    a[F.type] = t;
    a[F.player] = tok;
    return { action: a };
  };
  const loc = (t: number, tile: unknown): Translated => {
    if (typeof tile !== "number") return { unrepresentable: "bad_tile" };
    a[F.type] = t;
    const near = setLoc(a, tile, ctx);
    return { action: a, near };
  };
  switch (type) {
    case "spawn":
      return loc(T.SPAWN, intent.tile);
    case "attack": {
      a[F.troops] = troopBucket(intent.troops as number | null, ctx.troopsAtObs);
      if (intent.targetID === null || intent.targetID === undefined) {
        a[F.type] = T.ATTACK;
        a[F.player] = UNOWNED_TOKEN;
        return { action: a };
      }
      return pointPlayer(T.ATTACK, intent.targetID);
    }
    case "boat":
      a[F.troops] = troopBucket(intent.troops as number, ctx.troopsAtObs);
      return loc(T.BOAT, intent.dst);
    case "build_unit": {
      const b = BUILD_TYPES.indexOf(intent.unit as (typeof BUILD_TYPES)[number]);
      if (b < 0) return { unrepresentable: `build:${String(intent.unit)}` };
      a[F.build] = b;
      return loc(T.BUILD, intent.tile);
    }
    case "upgrade_structure": {
      const i = ctx.tables.structs.indexOf(intent.unitId as number);
      if (i < 0) return { unrepresentable: "struct_not_listed" };
      a[F.type] = T.UPGRADE;
      a[F.entity] = i;
      return { action: a };
    }
    case "move_warship": {
      const ids = intent.unitIds as number[];
      const i = ctx.tables.mobile.indexOf(ids?.[0]);
      if (i < 0) return { unrepresentable: "warship_not_listed" };
      a[F.entity] = i;
      return loc(T.MOVE_WARSHIP, intent.tile);
    }
    case "cancel_attack": {
      const i = ctx.tables.attacks.indexOf(intent.attackID as string);
      if (i < 0 || !ctx.tables.attackOutgoing[i]) return { unrepresentable: "attack_not_listed" };
      a[F.type] = T.CANCEL_ATTACK;
      a[F.entity] = i;
      return { action: a };
    }
    case "cancel_boat": {
      const i = ctx.tables.mobile.indexOf(intent.unitID as number);
      if (i < 0) return { unrepresentable: "boat_not_listed" };
      a[F.type] = T.CANCEL_BOAT;
      a[F.entity] = i;
      return { action: a };
    }
    case "delete_unit": {
      const s = ctx.tables.structs.indexOf(intent.unitId as number);
      const m = ctx.tables.mobile.indexOf(intent.unitId as number);
      if (s < 0 && m < 0) return { unrepresentable: "unit_not_listed" };
      a[F.type] = T.DELETE_UNIT;
      a[F.entity] = s >= 0 ? s : NUM_STRUCTS + m;
      return { action: a };
    }
    case "allianceRequest":
      return pointPlayer(T.ALLIANCE_REQUEST, intent.recipient);
    case "allianceReject":
      return pointPlayer(T.ALLIANCE_REJECT, intent.requestor);
    case "allianceExtension":
      return pointPlayer(T.ALLIANCE_EXTEND, intent.recipient);
    case "breakAlliance":
      return pointPlayer(T.BREAK_ALLIANCE, intent.recipient);
    case "targetPlayer":
      return pointPlayer(T.TARGET_PLAYER, intent.target);
    case "embargo":
      return pointPlayer(intent.action === "stop" ? T.EMBARGO_STOP : T.EMBARGO_START, intent.targetID);
  }
  return { unrepresentable: `unknown:${type}` };
}

/**
 * Collects one seat's intents per decision step into at most K actions.
 * Intents beyond K wait for the next step (`carried`), keeping their order.
 */
export class StepLabeler {
  private pending: RecordIntent[] = [];
  carried = 0;

  push(intents: RecordIntent[]): void {
    this.pending.push(...intents);
  }

  /** Translates up to K pending intents into a K*A label row; the rest carry over. */
  take(ctx: TranslateContext, onResult?: (intent: RecordIntent, r: Translated) => void): Int32Array {
    const out = new Int32Array(K * A).fill(-1);
    let k = 0;
    while (this.pending.length > 0 && k < K) {
      const intent = this.pending.shift()!;
      const r = translateIntent(intent, ctx);
      onResult?.(intent, r);
      if ("action" in r) out.set(r.action, (k++) * A);
    }
    this.carried += this.pending.length;
    if (k < K) out[k * A + F.type] = T.STOP;
    return out;
  }

  clear(): void {
    this.pending = [];
  }
}
