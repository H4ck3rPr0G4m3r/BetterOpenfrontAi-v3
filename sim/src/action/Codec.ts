/**
 * Action codec: policy actions (roadmap §3.6) -> engine intents.
 *
 * Pointers resolve through the EntityTables of the observation the actions
 * were chosen from. Locations snap to the nearest tile the engine accepts;
 * an action whose pointer is stale or whose location can't be snapped is
 * dropped and counted (DecodeStats), never retargeted (v1's retarget bug).
 */
import type { EngineGame, Player } from "../engine/Adapter";
import { UnitType } from "../engine/Adapter";
import type { MinimapGeometry, WindowSpec } from "../obs/Geometry";
import { windowCellToTile } from "../obs/Geometry";
import {
  A, ACTION_TYPES, BUILD_TYPES, F, FINE, K, LOC_FAR, LOC_NEAR, MM, NUM_BUILD, NUM_MOBILE, NUM_PLAYER_TOKENS,
  NUM_STRUCTS, NUM_TYPES, NUM_WINDOWS, T, TROOP_BUCKETS, UNOWNED_TOKEN, WIN,
} from "../obs/Layout";
import type { EntityTables } from "../obs/SeatEncoder";
import type { RecordIntent } from "../record/Record";
import { snapTile } from "./Snap";

export type DropReason = "illegal_type" | "bad_pointer" | "unsnappable" | "stale_entity";
export type Decoded = { intent: RecordIntent; tile?: number } | { dropped: DropReason; type: string };

export interface DecodeContext {
  g: EngineGame;
  clientID: string;
  tables: EntityTables;
  geometry: MinimapGeometry;
}

export class DecodeStats {
  ok: Record<string, number> = {};
  dropped: Record<string, Record<DropReason, number>> = {};
  add(d: Decoded): void {
    if ("intent" in d) {
      this.ok[d.intent.type] = (this.ok[d.intent.type] ?? 0) + 1;
    } else {
      const m = (this.dropped[d.type] ??= { illegal_type: 0, bad_pointer: 0, unsnappable: 0, stale_entity: 0 });
      m[d.dropped]++;
    }
  }
  merge(o: DecodeStats): void {
    for (const [k, v] of Object.entries(o.ok)) this.ok[k] = (this.ok[k] ?? 0) + v;
    for (const [k, m] of Object.entries(o.dropped)) {
      const mine = (this.dropped[k] ??= { illegal_type: 0, bad_pointer: 0, unsnappable: 0, stale_entity: 0 });
      for (const r of Object.keys(m) as DropReason[]) mine[r] += m[r];
    }
  }
  toJSON() {
    return { ok: this.ok, dropped: this.dropped };
  }
}

/** Troops sent for bucket b: (b + 1) * 5 % of current troops, floored. */
export function bucketTroops(current: number, bucket: number): number {
  return Math.floor(current * (bucket + 1) * 0.05);
}

export interface LocPoint {
  x: number;
  y: number;
  /** Snap radius in tiles for this pointer. */
  radius: number;
}

/** The point a location pointer names (before snapping), or null for a bad pointer. */
export function decodeLoc(a: Int32Array | number[], off: number, tables: EntityTables, geometry: MinimapGeometry, mapW: number, mapH: number): LocPoint | null {
  const mode = a[off + F.locMode];
  if (mode === LOC_NEAR) {
    const wi = a[off + F.window];
    const cell = a[off + F.cell];
    if (wi < 0 || wi >= NUM_WINDOWS || cell < 0 || cell >= WIN * WIN) return null;
    const w: WindowSpec | undefined = tables.windows[wi];
    if (!w) return null;
    const t = windowCellToTile(w, cell, mapW, mapH);
    return t === null ? null : { ...t, radius: 2 * w.tpc };
  }
  if (mode === LOC_FAR) {
    const mm = a[off + F.mmCell];
    const fine = a[off + F.fine];
    if (mm < 0 || mm >= MM * MM || fine < 0 || fine >= FINE * FINE) return null;
    const t = geometry.fineToTile(mm, fine);
    return t === null ? null : { ...t, radius: 2 * geometry.fineSpan() };
  }
  return null;
}

function tokenPlayer(ctx: DecodeContext, tok: number): Player | null | "unowned" {
  if (tok === UNOWNED_TOKEN) return "unowned";
  if (tok < 0 || tok >= NUM_PLAYER_TOKENS) return null;
  const id = ctx.tables.players[tok];
  if (id === null || id === undefined) return null;
  let p;
  try {
    p = ctx.g.game.playerBySmallID(id);
  } catch {
    return null;
  }
  return p && p.isPlayer() && p.isAlive() ? p : null;
}

export function decodeAction(a: Int32Array | number[], off: number, ctx: DecodeContext): Decoded | null {
  const typeIdx = a[off + F.type];
  if (typeIdx === T.STOP || typeIdx < 0) return null;
  const typeName = ACTION_TYPES[typeIdx] ?? `type${typeIdx}`;
  const drop = (r: DropReason): Decoded => ({ dropped: r, type: typeName });
  if (typeIdx >= NUM_TYPES) return drop("illegal_type");
  const game = ctx.g.game;
  const self = game.playerByClientID(ctx.clientID);
  if (self === null) return drop("stale_entity");
  const mapW = game.width();
  const mapH = game.height();
  const map = game.map();
  const clientID = ctx.clientID;
  const intent = (i: Record<string, unknown>, tile?: number): Decoded => ({ intent: { ...i, clientID } as RecordIntent, tile });

  const playerArg = (): Player | null | "unowned" => tokenPlayer(ctx, a[off + F.player]);
  const troopsArg = (): number | null => {
    const b = a[off + F.troops];
    return b < 0 || b >= TROOP_BUCKETS ? null : bucketTroops(self.troops(), b);
  };
  const loc = (): LocPoint | null => decodeLoc(a, off, ctx.tables, ctx.geometry, mapW, mapH);
  const unitAt = (id: number | undefined) => (id === undefined ? undefined : game.unit(id));

  switch (typeIdx) {
    case T.SPAWN: {
      const p = loc();
      if (!p) return drop("bad_pointer");
      const t = snapTile(p, (t) => map.isLand(t) && !map.isImpassable(t) && !map.hasOwner(t), p.radius, mapW, mapH);
      return t === null ? drop("unsnappable") : intent({ type: "spawn", tile: t }, t);
    }
    case T.ATTACK: {
      const target = playerArg();
      const troops = troopsArg();
      if (target === null || troops === null) return drop("bad_pointer");
      return intent({ type: "attack", targetID: target === "unowned" ? null : target.id(), troops });
    }
    case T.BOAT: {
      const p = loc();
      const troops = troopsArg();
      if (!p || troops === null) return drop("bad_pointer");
      const selfId = self.smallID();
      const cheap = (t: number) => {
        if (!map.isLand(t) || map.isImpassable(t)) return false;
        const o = map.ownerID(t);
        if (o === selfId) return false;
        if (o === 0) return true;
        const owner = game.playerBySmallID(o);
        return !!owner && owner.isPlayer() && self.canAttackPlayer(owner);
      };
      let tries = 0;
      const t = snapTile(p, (t) => cheap(t) && tries++ < 12 && self.canBuild(UnitType.TransportShip, t) !== false, p.radius, mapW, mapH);
      // (tries counts only tiles that passed the cheap checks: the engine's boat check pathfinds.)
      return t === null ? drop("unsnappable") : intent({ type: "boat", troops, dst: t }, t);
    }
    case T.BUILD: {
      const b = a[off + F.build];
      const p = loc();
      if (b < 0 || b >= NUM_BUILD || !p) return drop("bad_pointer");
      const unit = BUILD_TYPES[b] as UnitType;
      // canBuild itself searches the engine's valid spots around a tile; probe the nearest 16 points.
      const t = snapTile(p, (tile) => self.canBuild(unit, tile) !== false, p.radius, mapW, mapH, 16);
      return t === null ? drop("unsnappable") : intent({ type: "build_unit", unit, tile: t }, t);
    }
    case T.UPGRADE: {
      if (a[off + F.entity] < 0 || a[off + F.entity] >= NUM_STRUCTS) return drop("bad_pointer");
      const u = unitAt(ctx.tables.structs[a[off + F.entity]]);
      if (!u || !u.isActive() || u.owner() !== self) return drop("stale_entity");
      return intent({ type: "upgrade_structure", unit: u.type(), unitId: u.id() });
    }
    case T.MOVE_WARSHIP: {
      const e = a[off + F.entity];
      if (e < 0 || e >= NUM_MOBILE) return drop("bad_pointer");
      const u = unitAt(ctx.tables.mobile[e]);
      if (!u || !u.isActive() || u.owner() !== self || u.type() !== UnitType.Warship) return drop("stale_entity");
      const p = loc();
      if (!p) return drop("bad_pointer");
      const comp = game.getWaterComponent(u.tile());
      const t = snapTile(p, (t) => !map.isLand(t) && (comp === null || game.hasWaterComponent(t, comp)), p.radius, mapW, mapH);
      return t === null ? drop("unsnappable") : intent({ type: "move_warship", unitIds: [u.id()], tile: t }, t);
    }
    case T.CANCEL_ATTACK: {
      const e = a[off + F.entity];
      if (e < 0 || e >= ctx.tables.attacks.length) return drop(e < 0 || e >= 32 ? "bad_pointer" : "stale_entity");
      const id = ctx.tables.attacks[e];
      if (!ctx.tables.attackOutgoing[e] || !self.outgoingAttacks().some((x) => x.id() === id && x.isActive())) return drop("stale_entity");
      return intent({ type: "cancel_attack", attackID: id });
    }
    case T.CANCEL_BOAT: {
      const e = a[off + F.entity];
      if (e < 0 || e >= NUM_MOBILE) return drop("bad_pointer");
      const u = unitAt(ctx.tables.mobile[e]);
      if (!u || !u.isActive() || u.owner() !== self || u.type() !== UnitType.TransportShip) return drop("stale_entity");
      return intent({ type: "cancel_boat", unitID: u.id() });
    }
    case T.ALLIANCE_REQUEST:
    case T.ALLIANCE_REJECT:
    case T.ALLIANCE_EXTEND:
    case T.BREAK_ALLIANCE:
    case T.TARGET_PLAYER:
    case T.EMBARGO_START:
    case T.EMBARGO_STOP: {
      const p = playerArg();
      if (p === null || p === "unowned" || p === self) return drop("bad_pointer");
      const id = p.id();
      switch (typeIdx) {
        case T.ALLIANCE_REQUEST:
          return intent({ type: "allianceRequest", recipient: id });
        case T.ALLIANCE_REJECT:
          return intent({ type: "allianceReject", requestor: id });
        case T.ALLIANCE_EXTEND:
          return intent({ type: "allianceExtension", recipient: id });
        case T.BREAK_ALLIANCE:
          return intent({ type: "breakAlliance", recipient: id });
        case T.TARGET_PLAYER:
          return intent({ type: "targetPlayer", target: id });
        case T.EMBARGO_START:
          return intent({ type: "embargo", targetID: id, action: "start" });
        default:
          return intent({ type: "embargo", targetID: id, action: "stop" });
      }
    }
    case T.DELETE_UNIT: {
      const e = a[off + F.entity];
      if (e < 0 || e >= NUM_STRUCTS + NUM_MOBILE) return drop("bad_pointer");
      const u = unitAt(e < NUM_STRUCTS ? ctx.tables.structs[e] : ctx.tables.mobile[e - NUM_STRUCTS]);
      if (!u || !u.isActive() || u.owner() !== self) return drop("stale_entity");
      return intent({ type: "delete_unit", unitId: u.id() });
    }
  }
  return drop("illegal_type");
}

/** Decodes one step (K actions); stops at the first STOP. */
export function decodeStep(actions: Int32Array | number[], ctx: DecodeContext, stats?: DecodeStats): Decoded[] {
  const out: Decoded[] = [];
  for (let k = 0; k < K; k++) {
    const d = decodeAction(actions, k * A, ctx);
    if (d === null) break;
    stats?.add(d);
    out.push(d);
  }
  return out;
}

export function emptyAction(): Int32Array {
  return new Int32Array(A).fill(-1);
}
