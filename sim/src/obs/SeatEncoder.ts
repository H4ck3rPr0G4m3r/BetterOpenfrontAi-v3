/**
 * One seat's view (spec §9): perspective on the shared per-game cache,
 * player tokens with border-link edges, zoom windows, entity tables and the
 * engine's legality masks. Writes one ObsBuffer row and returns the tables
 * the action codec and the translator need to point at the same things.
 */
import type { Attack, EngineGame, Player, Unit } from "../engine/Adapter";
import { PlayerType, UnitType } from "../engine/Adapter";
import { toF16 } from "./f16";
import { clampSmallId, type GameObsCache, LETTERBOX, safe } from "./GameObsCache";
import { WINDOW_KIND_CODE, type WindowSpec } from "./Geometry";
import {
  AI, ATTACK_FEATURES, BUILD_TYPES, EF, EI, GAME_META, GI, MAX_SMALL_ID, MF, MI, MM, NUM_ATTACKS,
  NUM_BUILD, NUM_MOBILE, NUM_PLAYER_TOKENS, NUM_REAL_TOKENS, NUM_STRUCTS, NUM_WINDOWS, PF, PI, RELATION,
  REST_TOKEN, SF, SI, T, TERRAIN_BITS, UNOWNED_TOKEN, WIN, WIN_META, ENTITY_SLOTS, DECISION_TURNS,
} from "./Layout";
import type { ObsBuffer } from "./ObsBuffer";
import { placeWindows } from "./Windows";

export interface EntityTables {
  /** token -> smallID (REST and UNOWNED and empty slots are null). */
  players: (number | null)[];
  tokenOf: Map<number, number>;
  /** struct slot -> unit id */
  structs: number[];
  /** mobile slot -> unit id */
  mobile: number[];
  /** attack slot -> attack id */
  attacks: string[];
  /** attack slot -> true when outgoing */
  attackOutgoing: boolean[];
  windows: WindowSpec[];
  selfSmallId: number;
}

const STRUCT_ORDER = [UnitType.City, UnitType.Port, UnitType.Factory, UnitType.SAMLauncher, UnitType.MissileSilo, UnitType.DefensePost];
const STRUCT_ONEHOT: Record<string, number> = {
  [UnitType.City]: SI.t_city,
  [UnitType.Port]: SI.t_port,
  [UnitType.Factory]: SI.t_factory,
  [UnitType.DefensePost]: SI.t_defense,
  [UnitType.SAMLauncher]: SI.t_sam,
  [UnitType.MissileSilo]: SI.t_silo,
};
const BUILD_UNIT: UnitType[] = BUILD_TYPES.map((b) => b as UnitType);
const COUNT_FEATURES: [string, UnitType][] = [
  ["n_city", UnitType.City], ["n_port", UnitType.Port], ["n_factory", UnitType.Factory],
  ["n_defense", UnitType.DefensePost], ["n_sam", UnitType.SAMLauncher], ["n_silo", UnitType.MissileSilo],
  ["n_warship", UnitType.Warship], ["n_transport", UnitType.TransportShip],
];
const AFFORD_FEATURES = [
  "afford_city", "afford_port", "afford_factory", "afford_defense", "afford_sam", "afford_silo", "afford_warship",
  "afford_atom", "afford_hydrogen", "afford_mirv",
] as const;

const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);
const lg = (v: number, scale: number) => clamp(Math.log1p(Math.max(0, v)) / scale, 0, 1);

/** Relation of `p` seen from `self` (roadmap §3.5 codes). */
export function relationOf(self: Player, p: Player, attackers: Set<Player>, targets: Set<Player>): number {
  if (p === self) return RELATION.self;
  if (self.isFriendly(p)) return RELATION.ally;
  const them = attackers.has(p);
  const us = targets.has(p);
  if (them && us) return RELATION.both;
  if (them) return RELATION.attacksUs;
  if (us) return RELATION.weAttack;
  return RELATION.neutral;
}

function tileTerrainBits(g: EngineGame, t: number): number {
  const map = g.game.map();
  if (map.isLand(t)) {
    let b = TERRAIN_BITS.land;
    if (map.isShoreline(t)) b |= TERRAIN_BITS.coast;
    const tt = map.terrainType(t);
    if (tt === 1) b |= TERRAIN_BITS.highland;
    else if (tt === 2) b |= TERRAIN_BITS.mountain;
    return b;
  }
  let b = 0;
  if (map.isShoreline(t)) b |= TERRAIN_BITS.shoreWater;
  if (map.isOcean(t)) b |= TERRAIN_BITS.ocean;
  return b;
}

function attackTroopsBetween(from: Player, to: Player | null): number {
  let sum = 0;
  for (const a of from.outgoingAttacks()) {
    if (!a.isActive()) continue;
    const t = a.target();
    if (to === null ? !t.isPlayer() : t === to) sum += a.troops();
  }
  return sum;
}

/** Token order: self; anyone attacking us or attacked by us; land neighbours; then the biggest. */
export function selectTokens(cache: GameObsCache, self: Player, attackers: Set<Player>, targets: Set<Player>): Player[] {
  const chosen: Player[] = [self];
  const seen = new Set<Player>([self]);
  const add = (p: Player) => {
    if (chosen.length >= NUM_REAL_TOKENS || seen.has(p) || !p.isAlive()) return;
    if (p.smallID() >= MAX_SMALL_ID) return;
    seen.add(p);
    chosen.push(p);
  };
  const byTroops = (a: Player, b: Player) => b.troops() - a.troops() || a.smallID() - b.smallID();
  for (const p of [...new Set([...attackers, ...targets])].sort(byTroops)) add(p);
  const adj = cache.adjacency.get(self.smallID());
  if (adj) {
    const neighbours = [...adj.entries()]
      .filter(([id]) => id !== 0 && id !== LETTERBOX)
      .sort((a, b) => b[1].count - a[1].count || a[0] - b[0]);
    for (const [id] of neighbours) {
      const p = cache.bySmallId.get(id);
      if (p) add(p);
    }
  }
  const rest = cache.players.filter((p) => p.isAlive() && !seen.has(p));
  rest.sort((a, b) => b.numTilesOwned() - a.numTilesOwned() || a.smallID() - b.smallID());
  for (const p of rest) add(p);
  return chosen;
}

export function encodeSeat(cache: GameObsCache, g: EngineGame, clientID: string, buf: ObsBuffer, row: number): EntityTables {
  const game = g.game;
  const config = game.config();
  const mapW = game.width();
  const mapH = game.height();
  buf.clearRow(row);

  // Shared per-game fields.
  buf.view<Uint16Array>("mm_owner", row).set(cache.mmOwner);
  buf.view<Uint8Array>("mm_terrain", row).set(cache.mmTerrain);
  buf.view<Uint16Array>("density", row).set(cache.density);
  buf.view<Int16Array>("units", row).set(cache.units);
  const meta = new Float32Array(GAME_META);
  cache.gameMeta(meta);
  buf.view<Float32Array>("game_meta", row).set(meta);

  const tables: EntityTables = {
    players: new Array(NUM_PLAYER_TOKENS).fill(null),
    tokenOf: new Map(),
    structs: [],
    mobile: [],
    attacks: [],
    attackOutgoing: [],
    windows: [],
    selfSmallId: -1,
  };
  const typeMask = buf.view<Uint8Array>("type_mask", row);
  typeMask[T.STOP] = 1;
  const self = game.playerByClientID(clientID);
  if (self === null) return tables;
  tables.selfSmallId = self.smallID();
  const selfId = self.smallID();

  // Relations.
  const attackers = new Set<Player>();
  const targets = new Set<Player>();
  for (const a of self.incomingAttacks()) if (a.isActive()) attackers.add(a.attacker());
  for (const a of self.outgoingAttacks()) {
    const t = a.target();
    if (a.isActive() && t.isPlayer()) targets.add(t);
  }
  const relation = buf.view<Uint8Array>("relation", row);
  const relOf = new Map<Player, number>();
  for (const p of cache.players) {
    const r = relationOf(self, p, attackers, targets);
    relOf.set(p, r);
    const id = p.smallID();
    if (id < MAX_SMALL_ID) relation[id] = r;
  }

  // Windows.
  const windows = placeWindows(cache, g, selfId);
  tables.windows = windows;
  const winMeta = buf.view<Float32Array>("win_meta", row);
  const winOwner = buf.view<Uint16Array>("win_owner", row);
  const winTerrain = buf.view<Uint8Array>("win_terrain", row);
  const map = game.map();
  for (let wi = 0; wi < NUM_WINDOWS; wi++) {
    const w = windows[wi];
    if (!w) continue;
    const m = wi * WIN_META;
    winMeta[m] = w.x0;
    winMeta[m + 1] = w.y0;
    winMeta[m + 2] = w.tpc;
    winMeta[m + 3] = WINDOW_KIND_CODE[w.kind];
    winMeta[m + 4] = 1;
    winMeta[m + 5] = w.priority;
    const base = wi * WIN * WIN;
    const half = Math.floor(w.tpc / 2);
    for (let cy = 0; cy < WIN; cy++) {
      const y = w.y0 + cy * w.tpc + half;
      for (let cx = 0; cx < WIN; cx++) {
        const x = w.x0 + cx * w.tpc + half;
        const c = base + cy * WIN + cx;
        if (x >= mapW || y >= mapH) continue; // stays 0xFFFF (outside a map smaller than the window)
        const t = y * mapW + x;
        const bits = tileTerrainBits(g, t);
        winTerrain[c] = bits;
        winOwner[c] = bits & TERRAIN_BITS.land ? clampSmallId(map.ownerID(t)) : 0;
      }
    }
  }

  // Player tokens.
  const tokens = selectTokens(cache, self, attackers, targets);
  tokens.forEach((p, i) => {
    tables.players[i] = p.smallID();
    tables.tokenOf.set(p.smallID(), i);
  });
  const players = buf.view<Uint16Array>("players", row);
  const totalLand = cache.totalLand;
  const ticks = game.ticks();
  const allianceDuration = Math.max(1, config.allianceDuration());
  const selfTroops = self.troops();
  const incomingReq = new Set(self.incomingAllianceRequests().map((r) => r.requestor()));
  const outgoingReq = new Set(self.outgoingAllianceRequests().map((r) => r.recipient()));
  const selfAdj = cache.adjacency.get(selfId);
  const put = (tok: number, f: number, v: number) => (players[tok * PF + f] = toF16(v));
  const playerFeatures = (tok: number, p: Player) => {
    const tiles = p.numTilesOwned();
    const troops = p.troops();
    put(tok, PI.valid, 1);
    put(tok, PI.alive, p.isAlive() ? 1 : 0);
    put(tok, PI.is_self, p === self ? 1 : 0);
    const kind = p.type();
    put(tok, kind === PlayerType.Human ? PI.kind_human : kind === PlayerType.Nation ? PI.kind_nation : PI.kind_bot, 1);
    put(tok, PI.tiles_share, tiles / totalLand);
    put(tok, PI.log_tiles, lg(tiles, 15));
    put(tok, PI.troops_of_max, clamp(troops / Math.max(1, config.maxTroops(p)), 0, 2));
    put(tok, PI.log_troops, lg(troops, 16));
    put(tok, PI.log_gold, lg(Number(p.gold()), 22));
    put(tok, PI.strength_vs_self, clamp(Math.log((troops + 1) / (selfTroops + 1)) / 5, -1, 1));
    const rel = relOf.get(p) ?? RELATION.neutral;
    const relFeature = [-1, PI.rel_self, PI.rel_ally, PI.rel_neutral, PI.rel_attacks_us, PI.rel_we_attack, PI.rel_both][rel];
    if (relFeature >= 0) put(tok, relFeature, 1);
    if (p !== self) {
      if (incomingReq.has(p)) put(tok, PI.request_in, 1);
      if (outgoingReq.has(p)) put(tok, PI.request_out, 1);
      if (self.hasEmbargoAgainst(p)) put(tok, PI.embargo_we, 1);
      if (p.hasEmbargoAgainst(self)) put(tok, PI.embargo_they, 1);
      const alliance = self.allianceWith(p);
      if (alliance) put(tok, PI.alliance_left, clamp((alliance.expiresAt() - ticks) / allianceDuration, 0, 1));
      put(tok, PI.attacking_us, clamp(attackTroopsBetween(p, self) / (selfTroops + 1), 0, 4));
      put(tok, PI.we_attack, clamp(attackTroopsBetween(self, p) / (selfTroops + 1), 0, 4));
      if ((selfAdj?.get(p.smallID())?.count ?? 0) > 0) put(tok, PI.shares_border, 1);
    }
    if (p.isTraitor()) put(tok, PI.traitor, 1);
    if (p.isDisconnected()) put(tok, PI.disconnected, 1);
    if (p.hasSpawned()) put(tok, PI.spawned, 1);
  };
  tokens.forEach((p, i) => playerFeatures(i, p));

  // Rest token: everyone alive not in a token.
  const inToken = new Set(tokens);
  let restCount = 0;
  let restTiles = 0;
  let restMaxTroops = 0;
  for (const p of cache.players) {
    if (!p.isAlive() || inToken.has(p)) continue;
    restCount++;
    restTiles += p.numTilesOwned();
    restMaxTroops = Math.max(restMaxTroops, p.troops());
  }
  if (restCount > 0) {
    put(REST_TOKEN, PI.valid, 1);
    put(REST_TOKEN, PI.alive, 1);
    put(REST_TOKEN, PI.is_rest, 1);
    put(REST_TOKEN, PI.rest_count, lg(restCount, 7));
    put(REST_TOKEN, PI.tiles_share, restTiles / totalLand);
    put(REST_TOKEN, PI.log_tiles, lg(restTiles, 15));
    put(REST_TOKEN, PI.log_troops, lg(restMaxTroops, 16));
    put(REST_TOKEN, PI.strength_vs_self, clamp(Math.log((restMaxTroops + 1) / (selfTroops + 1)) / 5, -1, 1));
  }
  // Unowned land token.
  let ownedTiles = 0;
  for (const p of cache.players) ownedTiles += p.numTilesOwned();
  put(UNOWNED_TOKEN, PI.valid, 1);
  put(UNOWNED_TOKEN, PI.is_unowned, 1);
  put(UNOWNED_TOKEN, PI.tiles_share, clamp((totalLand - ownedTiles) / totalLand, 0, 1));
  if ((selfAdj?.get(0)?.count ?? 0) > 0) put(UNOWNED_TOKEN, PI.shares_border, 1);
  put(UNOWNED_TOKEN, PI.we_attack, clamp(attackTroopsBetween(self, null) / (selfTroops + 1), 0, 4));

  fillEdges(cache, tokens, buf.view<Uint16Array>("edges", row));

  // Structures.
  const structs = buf.view<Uint16Array>("structs", row);
  const own = self.units();
  const structUnits = STRUCT_ORDER.flatMap((t) => own.filter((u) => u.type() === t && u.isActive()).sort((a, b) => a.id() - b.id()));
  const geo = cache.geometry;
  const hostileCell = (cell: number) => {
    const o = cache.mmOwner[cell];
    if (o === 0 || o === LETTERBOX || o === selfId) return false;
    const p = cache.bySmallId.get(o);
    return p !== undefined && !self.isFriendly(p);
  };
  const entityMask = buf.view<Uint8Array>("entity_mask", row);
  structUnits.slice(0, NUM_STRUCTS).forEach((u, i) => {
    tables.structs.push(u.id());
    const s = (f: number, v: number) => (structs[i * SF + f] = toF16(v));
    const tile = u.tile();
    const x = tile % mapW;
    const y = Math.floor(tile / mapW);
    s(SI.valid, 1);
    s(STRUCT_ONEHOT[u.type()], 1);
    s(SI.level, clamp(u.level() / 10, 0, 1));
    if (u.isUnderConstruction()) s(SI.under_construction, 1);
    const wi = windows.findIndex((w) => x >= w.x0 && y >= w.y0 && x < w.x0 + WIN * w.tpc && y < w.y0 + WIN * w.tpc);
    if (wi >= 0) {
      s(SI.in_window, 1);
      s(SI.window_norm, (wi + 1) / NUM_WINDOWS);
    }
    s(SI.x, x / mapW);
    s(SI.y, y / mapH);
    const cell = geo.cellOfTile(x, y);
    const col = cell % MM;
    if ([col > 0 ? cell - 1 : -1, col < MM - 1 ? cell + 1 : -1, cell - MM, cell + MM, cell].some((c) => c >= 0 && c < MM * MM && hostileCell(c))) {
      s(SI.at_front, 1);
    }
    if ((u.type() === UnitType.SAMLauncher || u.type() === UnitType.MissileSilo) && safe(() => u.isInCooldown())) s(SI.in_cooldown, 1);
    const canUp = self.canUpgradeUnit(u);
    if (canUp) {
      s(SI.can_upgrade, 1);
      entityMask[T.UPGRADE * ENTITY_SLOTS + i] = 1;
    }
    if (self.canDeleteUnit() && !u.isUnderConstruction()) entityMask[T.DELETE_UNIT * ENTITY_SLOTS + i] = 1;
  });

  // Mobile: warships, then transports.
  const mobile = buf.view<Uint16Array>("mobile", row);
  const mobileUnits = [UnitType.Warship, UnitType.TransportShip].flatMap((t) =>
    own.filter((u) => u.type() === t && u.isActive()).sort((a, b) => a.id() - b.id()),
  );
  mobileUnits.slice(0, NUM_MOBILE).forEach((u, i) => {
    tables.mobile.push(u.id());
    const m = (f: number, v: number) => (mobile[i * MF + f] = toF16(v));
    const tile = u.tile();
    m(MI.valid, 1);
    const warship = u.type() === UnitType.Warship;
    m(warship ? MI.is_warship : MI.is_transport, 1);
    m(MI.log_troops, lg(u.troops(), 16));
    m(MI.x, (tile % mapW) / mapW);
    m(MI.y, Math.floor(tile / mapW) / mapH);
    const target = u.targetTile();
    if (target !== undefined) {
      m(MI.has_target, 1);
      m(MI.target_x, (target % mapW) / mapW);
      m(MI.target_y, Math.floor(target / mapW) / mapH);
    }
    if (u.hasHealth()) m(MI.health, clamp(u.health() / Math.max(1, u.maxHealth()), 0, 1));
    entityMask[(warship ? T.MOVE_WARSHIP : T.CANCEL_BOAT) * ENTITY_SLOTS + i] = 1;
    if (self.canDeleteUnit()) entityMask[T.DELETE_UNIT * ENTITY_SLOTS + NUM_STRUCTS + i] = 1;
  });

  // Attacks: outgoing (most troops first), then incoming.
  const attacksView = buf.view<Uint16Array>("attacks", row);
  const byTroops = (a: Attack, b: Attack) => b.troops() - a.troops() || (a.id() < b.id() ? -1 : 1);
  const outgoing = self.outgoingAttacks().filter((a) => a.isActive()).sort(byTroops);
  const incoming = self.incomingAttacks().filter((a) => a.isActive()).sort(byTroops);
  [...outgoing.map((a) => [a, true] as const), ...incoming.map((a) => [a, false] as const)]
    .slice(0, NUM_ATTACKS)
    .forEach(([a, out], i) => {
      tables.attacks.push(a.id());
      tables.attackOutgoing.push(out);
      const f = (k: number, v: number) => (attacksView[i * ATTACK_FEATURES.length + k] = toF16(v));
      f(AI.valid, 1);
      f(out ? AI.outgoing : AI.incoming, 1);
      f(AI.log_troops, lg(a.troops(), 16));
      f(AI.troops_of_ours, clamp(a.troops() / (selfTroops + 1), 0, 4));
      const other = out ? a.target() : a.attacker();
      if (other.isPlayer()) {
        const tok = tables.tokenOf.get(other.smallID());
        f(AI.other_token, tok === undefined ? (REST_TOKEN + 1) / NUM_PLAYER_TOKENS : (tok + 1) / NUM_PLAYER_TOKENS);
      } else f(AI.other_unowned, 1);
      if (a.sourceTile() !== null) f(AI.is_boat, 1);
      if (a.retreating()) f(AI.retreating, 1);
      if (out && !a.retreating()) entityMask[T.CANCEL_ATTACK * ENTITY_SLOTS + i] = 1;
    });

  // Globals.
  const globals = buf.view<Float32Array>("globals", row);
  const spawnPhase = game.inSpawnPhase();
  const gold = Number(self.gold());
  const maxTroops = Math.max(1, config.maxTroops(self));
  globals[GI.minutes] = ticks / 600;
  globals[GI.spawn_phase] = spawnPhase ? 1 : 0;
  globals[GI.spawn_left] = spawnPhase ? clamp((config.numSpawnPhaseTurns() - ticks) / Math.max(1, config.numSpawnPhaseTurns()), 0, 1) : 0;
  globals[GI.log_gold] = lg(gold, 22);
  globals[GI.log_troops] = lg(selfTroops, 16);
  globals[GI.troops_of_max] = clamp(selfTroops / maxTroops, 0, 2);
  globals[GI.log_max_troops] = lg(maxTroops, 16);
  globals[GI.growth] = lg(config.troopIncreaseRate(self), 10);
  globals[GI.tiles_share] = self.numTilesOwned() / totalLand;
  globals[GI.log_tiles] = lg(self.numTilesOwned(), 15);
  globals[GI.players_alive] = lg(cache.players.filter((p) => p.isAlive()).length, 7);
  for (const [name, type] of COUNT_FEATURES) globals[GI[name as keyof typeof GI]] = lg(self.unitCount(type), 5);
  const affordable = BUILD_UNIT.map((t) => !config.isUnitDisabled(t) && BigInt(Math.floor(gold)) >= config.unitInfo(t).cost(game, self));
  AFFORD_FEATURES.forEach((n, i) => (globals[GI[n]] = affordable[i] ? 1 : 0));
  globals[GI.nukes_enabled] = config.isUnitDisabled(UnitType.AtomBomb) ? 0 : 1;
  globals[GI.sams_enabled] = config.isUnitDisabled(UnitType.SAMLauncher) ? 0 : 1;
  globals[GI.alliances_enabled] = config.disableAlliances() ? 0 : 1;
  globals[GI.water_nukes] = config.gameConfig().waterNukes ? 1 : 0;
  globals[GI.log_map_w] = Math.log(mapW) / 9;
  globals[GI.log_map_h] = Math.log(mapH) / 9;
  globals[GI.mm_tpc] = cache.geometry.tpc / 40;
  globals[GI.decision_turns] = DECISION_TURNS / 10;
  globals[GI.humans_alive] = lg(cache.players.filter((p) => p.isAlive() && p.type() === PlayerType.Human).length, 6);
  globals[GI.outgoing_attacks] = lg(outgoing.length, 3);
  globals[GI.incoming_attacks] = lg(incoming.length, 3);

  // Masks.
  if (!self.isAlive() && !spawnPhase) return tables;
  if (spawnPhase) {
    typeMask[T.SPAWN] = 1;
    return tables;
  }
  const playerMask = buf.view<Uint8Array>("player_mask", row);
  const pm = (type: number, tok: number) => (playerMask[type * NUM_PLAYER_TOKENS + tok] = 1);
  const buildMask = buf.view<Uint8Array>("build_mask", row);
  const troops = self.troops();
  tokens.forEach((p, tok) => {
    if (p === self) return;
    const bordering = (selfAdj?.get(p.smallID())?.count ?? 0) > 0 || self.sharesBorderWith(p);
    if (troops > 0 && bordering && self.canAttackPlayer(p)) pm(T.ATTACK, tok);
    if (self.canSendAllianceRequest(p)) pm(T.ALLIANCE_REQUEST, tok);
    if (incomingReq.has(p)) pm(T.ALLIANCE_REJECT, tok);
    if (self.isAlliedWith(p)) {
      pm(T.ALLIANCE_EXTEND, tok);
      pm(T.BREAK_ALLIANCE, tok);
    }
    if (self.canTarget(p)) pm(T.TARGET_PLAYER, tok);
    if (self.hasEmbargoAgainst(p)) pm(T.EMBARGO_STOP, tok);
    else pm(T.EMBARGO_START, tok);
  });
  if (troops > 0 && ((selfAdj?.get(0)?.count ?? 0) > 0 || self.sharesBorderWith(game.terraNullius()))) pm(T.ATTACK, UNOWNED_TOKEN);
  for (const t of [T.ATTACK, T.ALLIANCE_REQUEST, T.ALLIANCE_REJECT, T.ALLIANCE_EXTEND, T.BREAK_ALLIANCE, T.TARGET_PLAYER, T.EMBARGO_START, T.EMBARGO_STOP]) {
    const rowMask = playerMask.subarray(t * NUM_PLAYER_TOKENS, (t + 1) * NUM_PLAYER_TOKENS);
    typeMask[t] = rowMask.some((v) => v === 1) ? 1 : 0;
  }
  const ownCoast = (cache.ownerCells.get(selfId) ?? []).some((c) => cache.mmTerrain[c] & TERRAIN_BITS.coast);
  if (troops > 0 && ownCoast && self.unitCount(UnitType.TransportShip) < config.boatMaxNumber()) typeMask[T.BOAT] = 1;
  const hasPort = self.unitCount(UnitType.Port) > 0;
  const readySilo = self.units(UnitType.MissileSilo).some((u) => u.isActive() && !u.isUnderConstruction() && !safe(() => u.isInCooldown()));
  for (let b = 0; b < NUM_BUILD; b++) {
    if (!affordable[b]) continue;
    const t = BUILD_UNIT[b];
    if (t === UnitType.Warship && !hasPort) continue;
    if (t === UnitType.Port && !ownCoast) continue;
    if ((t === UnitType.AtomBomb || t === UnitType.HydrogenBomb || t === UnitType.MIRV) && !readySilo) continue;
    buildMask[b] = 1;
  }
  typeMask[T.BUILD] = buildMask.some((v) => v === 1) ? 1 : 0;
  const anyEntity = (t: number) => entityMask.subarray(t * ENTITY_SLOTS, (t + 1) * ENTITY_SLOTS).some((v) => v === 1);
  for (const t of [T.UPGRADE, T.MOVE_WARSHIP, T.CANCEL_ATTACK, T.CANCEL_BOAT, T.DELETE_UNIT]) typeMask[t] = anyEntity(t) ? 1 : 0;
  return tables;
}

/** Border-link edge features between every pair of tokens (spec §9.1). */
export function fillEdges(cache: GameObsCache, tokens: Player[], out: Uint16Array): void {
  const tpc = cache.geometry.tpc;
  const diag = Math.hypot(cache.g.game.width(), cache.g.game.height());
  const outgoingBy = new Map<Player, Map<Player | null, number>>();
  const attackMap = (p: Player) => {
    let m = outgoingBy.get(p);
    if (m) return m;
    m = new Map();
    for (const a of p.outgoingAttacks()) {
      if (!a.isActive()) continue;
      const t = a.target();
      const key = t.isPlayer() ? t : null;
      m.set(key, (m.get(key) ?? 0) + a.troops());
    }
    outgoingBy.set(p, m);
    return m;
  };
  const e = (i: number, j: number, f: number, v: number) => (out[(i * NUM_PLAYER_TOKENS + j) * EF + f] = toF16(v));
  const slots: { tok: number; id: number; p: Player | null }[] = tokens.map((p, i) => ({ tok: i, id: p.smallID(), p }));
  slots.push({ tok: UNOWNED_TOKEN, id: 0, p: null });
  for (let a = 0; a < slots.length; a++) {
    for (let b = 0; b < slots.length; b++) {
      if (a === b) continue;
      const A = slots[a];
      const B = slots[b];
      if (A.p === null && B.p === null) continue;
      const border = cache.adjacency.get(A.id)?.get(B.id);
      e(A.tok, B.tok, EI.valid, 1);
      if (border && border.count > 0) {
        e(A.tok, B.tok, EI.land_border, 1);
        e(A.tok, B.tok, EI.border_log_tiles, lg(border.count * tpc, 10));
        const total = Math.max(1, border.terrain[0] + border.terrain[1] + border.terrain[2]);
        e(A.tok, B.tok, EI.plains_share, border.terrain[0] / total);
        e(A.tok, B.tok, EI.highland_share, border.terrain[1] / total);
        e(A.tok, B.tok, EI.mountain_share, border.terrain[2] / total);
      } else {
        e(A.tok, B.tok, EI.sea_only, 1);
        const ca = cache.centroids.get(A.id);
        const cb = cache.centroids.get(B.id);
        if (ca && cb) e(A.tok, B.tok, EI.distance, Math.hypot(ca.x - cb.x, ca.y - cb.y) / diag);
      }
      e(A.tok, B.tok, EI.density_i, A.p ? cache.densityF32[clampSmallId(A.id)] : 0);
      e(A.tok, B.tok, EI.density_j, B.p ? cache.densityF32[clampSmallId(B.id)] : 0);
      if (A.p) e(A.tok, B.tok, EI.attack_ij, lg(attackMap(A.p).get(B.p) ?? 0, 16));
      if (B.p) e(A.tok, B.tok, EI.attack_ji, lg(attackMap(B.p).get(A.p) ?? 0, 16));
    }
  }
}

export type { Unit };
