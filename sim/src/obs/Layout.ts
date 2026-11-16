/**
 * Observation and action layout v1: the single source of truth for the
 * encoder, the action codec, the translator, the bridge and (through HELLO /
 * `sim.mjs layout`) the Python model. Python never hard-codes any of this.
 *
 * Wire format (roadmap §3.5): compact per-game rasters plus per-seat
 * perspective; the GPU expands them into PLANES. Any change here is
 * deliberate: the snapshot test fails until LAYOUT_VERSION is bumped.
 */

export const LAYOUT_VERSION = 1;

// -- sizes -------------------------------------------------------------------
export const MM = 128; // minimap is MM x MM cells
export const WIN = 32; // zoom window is WIN x WIN cells
export const NUM_WINDOWS = 8;
export const WINDOW_TPC = [2, 3, 4] as const; // tiles per window cell
export const NUM_PLAYER_TOKENS = 34;
export const NUM_REAL_TOKENS = 32; // self + 31 others
export const REST_TOKEN = 32;
export const UNOWNED_TOKEN = 33;
export const MAX_SMALL_ID = 1024; // smallIDs at or above this are folded into the rest token
export const MAX_UNITS = 1024;
export const UNIT_COLS = 8;
export const NUM_STRUCTS = 64;
export const NUM_MOBILE = 32;
export const NUM_ATTACKS = 32;
export const K = 4; // actions per decision step
export const A = 10; // fields per action
export const TROOP_BUCKETS = 20; // bucket b = (b + 1) * 5 % of current troops
export const FINE = 16; // far pointer: FINE x FINE grid over one minimap cell
export const DECISION_TURNS = 5;
export const GAME_META = 8;
export const WIN_META = 6;

// -- actions -----------------------------------------------------------------
export const ACTION_FIELDS = ["type", "player", "entity", "build", "troops", "locMode", "window", "cell", "mmCell", "fine"] as const;
export const F = Object.fromEntries(ACTION_FIELDS.map((n, i) => [n, i])) as Record<(typeof ACTION_FIELDS)[number], number>;

export const ACTION_TYPES = [
  "STOP",
  "SPAWN",
  "ATTACK",
  "BOAT",
  "BUILD",
  "UPGRADE",
  "MOVE_WARSHIP",
  "CANCEL_ATTACK",
  "CANCEL_BOAT",
  "ALLIANCE_REQUEST",
  "ALLIANCE_REJECT",
  "ALLIANCE_EXTEND",
  "BREAK_ALLIANCE",
  "TARGET_PLAYER",
  "EMBARGO_START",
  "EMBARGO_STOP",
  "DELETE_UNIT",
] as const;
export const NUM_TYPES = ACTION_TYPES.length;
export const T = Object.fromEntries(ACTION_TYPES.map((n, i) => [n, i])) as Record<(typeof ACTION_TYPES)[number], number>;

/** Which argument fields each action type uses (the rest stay -1). */
export const TYPE_ARGS: Record<(typeof ACTION_TYPES)[number], readonly (typeof ACTION_FIELDS)[number][]> = {
  STOP: [],
  SPAWN: ["locMode", "window", "cell", "mmCell", "fine"],
  ATTACK: ["player", "troops"],
  BOAT: ["troops", "locMode", "window", "cell", "mmCell", "fine"],
  BUILD: ["build", "locMode", "window", "cell", "mmCell", "fine"],
  UPGRADE: ["entity"],
  MOVE_WARSHIP: ["entity", "locMode", "window", "cell", "mmCell", "fine"],
  CANCEL_ATTACK: ["entity"],
  CANCEL_BOAT: ["entity"],
  ALLIANCE_REQUEST: ["player"],
  ALLIANCE_REJECT: ["player"],
  ALLIANCE_EXTEND: ["player"],
  BREAK_ALLIANCE: ["player"],
  TARGET_PLAYER: ["player"],
  EMBARGO_START: ["player"],
  EMBARGO_STOP: ["player"],
  DELETE_UNIT: ["entity"],
};

/** The entity table each entity-pointing type points into. DELETE_UNIT: structs 0..63, mobile at 64.. */
export const ENTITY_TABLE: Partial<Record<(typeof ACTION_TYPES)[number], "structs" | "mobile" | "attacks" | "structs+mobile">> = {
  UPGRADE: "structs",
  MOVE_WARSHIP: "mobile",
  CANCEL_ATTACK: "attacks",
  CANCEL_BOAT: "mobile",
  DELETE_UNIT: "structs+mobile",
};
export const ENTITY_SLOTS = NUM_STRUCTS + NUM_MOBILE; // widest entity pointer (DELETE_UNIT)

export const BUILD_TYPES = [
  "City",
  "Port",
  "Factory",
  "Defense Post",
  "SAM Launcher",
  "Missile Silo",
  "Warship",
  "Atom Bomb",
  "Hydrogen Bomb",
  "MIRV",
] as const; // engine UnitType values
export const NUM_BUILD = BUILD_TYPES.length;
export const NUKE_BUILDS = new Set([7, 8, 9]);

export const LOC_NEAR = 0;
export const LOC_FAR = 1;

// -- relations (per smallID, per seat) -----------------------------------------
export const RELATION = {
  none: 0, // no player with this smallID (or unowned)
  self: 1,
  ally: 2,
  neutral: 3,
  attacksUs: 4,
  weAttack: 5,
  both: 6,
} as const;

// -- terrain bits (mm_terrain, win_terrain) ------------------------------------
export const TERRAIN_BITS = { land: 1, coast: 2, shoreWater: 4, highland: 8, mountain: 16, ocean: 32 } as const;

// -- unit list (per game) ------------------------------------------------------
/** Unit list type codes. Trains, trade ships, shells and SAM missiles are not listed. */
export const UNIT_TYPES = [
  "City",
  "Port",
  "Factory",
  "Defense Post",
  "SAM Launcher",
  "Missile Silo",
  "Transport",
  "Warship",
  "Atom Bomb",
  "Hydrogen Bomb",
  "MIRV",
  "MIRV Warhead",
] as const;
export const STRUCTURE_TYPES = UNIT_TYPES.slice(0, 6);
export const UNIT_COLUMNS = ["type", "owner", "x", "y", "level", "flags", "targetX", "targetY"] as const;
export const UNIT_FLAGS = { underConstruction: 1, inCooldown: 2, targetedBySAM: 4, retreating: 8 } as const;

// -- feature lists (index = position) -------------------------------------------
export const PLAYER_FEATURES = [
  "valid",
  "alive",
  "is_self",
  "kind_human",
  "kind_nation",
  "kind_bot",
  "tiles_share",
  "log_tiles",
  "troops_of_max",
  "log_troops",
  "log_gold",
  "strength_vs_self",
  "rel_self",
  "rel_ally",
  "rel_neutral",
  "rel_attacks_us",
  "rel_we_attack",
  "rel_both",
  "request_in",
  "request_out",
  "traitor",
  "embargo_we",
  "embargo_they",
  "alliance_left",
  "attacking_us",
  "we_attack",
  "disconnected",
  "spawned",
  "is_rest",
  "rest_count",
  "is_unowned",
  "shares_border",
] as const;

export const EDGE_FEATURES = [
  "valid",
  "land_border",
  "border_log_tiles",
  "density_i",
  "density_j",
  "plains_share",
  "highland_share",
  "mountain_share",
  "attack_ij",
  "attack_ji",
  "sea_only",
  "distance",
] as const;

export const STRUCT_FEATURES = [
  "valid",
  "t_city",
  "t_port",
  "t_factory",
  "t_defense",
  "t_sam",
  "t_silo",
  "level",
  "under_construction",
  "in_window",
  "window_norm",
  "x",
  "y",
  "at_front",
  "in_cooldown",
  "can_upgrade",
] as const;

export const MOBILE_FEATURES = [
  "valid",
  "is_warship",
  "is_transport",
  "log_troops",
  "x",
  "y",
  "has_target",
  "target_x",
  "target_y",
  "health",
] as const;

export const ATTACK_FEATURES = [
  "valid",
  "outgoing",
  "incoming",
  "log_troops",
  "troops_of_ours",
  "other_token",
  "other_unowned",
  "is_boat",
  "retreating",
] as const;

export const GLOBAL_FEATURES = [
  "minutes",
  "spawn_phase",
  "spawn_left",
  "log_gold",
  "log_troops",
  "troops_of_max",
  "log_max_troops",
  "growth",
  "tiles_share",
  "log_tiles",
  "players_alive",
  "n_city",
  "n_port",
  "n_factory",
  "n_defense",
  "n_sam",
  "n_silo",
  "n_warship",
  "n_transport",
  "afford_city",
  "afford_port",
  "afford_factory",
  "afford_defense",
  "afford_sam",
  "afford_silo",
  "afford_warship",
  "afford_atom",
  "afford_hydrogen",
  "afford_mirv",
  "nukes_enabled",
  "sams_enabled",
  "alliances_enabled",
  "water_nukes",
  "log_map_w",
  "log_map_h",
  "mm_tpc",
  "decision_turns",
  "humans_alive",
  "outgoing_attacks",
  "incoming_attacks",
] as const;

export const PF = PLAYER_FEATURES.length;
export const EF = EDGE_FEATURES.length;
export const SF = STRUCT_FEATURES.length;
export const MF = MOBILE_FEATURES.length;
export const AF = ATTACK_FEATURES.length;
export const GF = GLOBAL_FEATURES.length;

export const idx = <L extends readonly string[]>(list: L) =>
  Object.fromEntries(list.map((n, i) => [n, i])) as Record<L[number], number>;
export const PI = idx(PLAYER_FEATURES);
export const EI = idx(EDGE_FEATURES);
export const SI = idx(STRUCT_FEATURES);
export const MI = idx(MOBILE_FEATURES);
export const AI = idx(ATTACK_FEATURES);
export const GI = idx(GLOBAL_FEATURES);

// -- planes (built on the GPU from the compact wire fields) ----------------------
export interface PlaneSpec {
  name: string;
  /** terrain: a terrain bit; owner: relation of the owner; density: owner relation x density; units: unit cells; valid. */
  from: "terrain" | "owner" | "density" | "units" | "sam_cover" | "valid";
  bit?: number;
  rel?: number[];
  unit?: string[];
  ownerUnowned?: boolean;
}

const HOSTILE = [RELATION.neutral, RELATION.attacksUs, RELATION.weAttack, RELATION.both];
const OWN = [RELATION.self];
export const PLANES: PlaneSpec[] = [
  { name: "land", from: "terrain", bit: TERRAIN_BITS.land },
  { name: "water", from: "terrain", bit: 0 }, // not land and not letterbox
  { name: "coast", from: "terrain", bit: TERRAIN_BITS.coast },
  { name: "own", from: "owner", rel: OWN },
  { name: "allied", from: "owner", rel: [RELATION.ally] },
  { name: "hostile", from: "owner", rel: HOSTILE },
  { name: "unowned", from: "owner", ownerUnowned: true },
  { name: "density_own", from: "density", rel: OWN },
  { name: "density_hostile", from: "density", rel: HOSTILE },
  ...STRUCTURE_TYPES.flatMap((u) => [
    { name: `struct_${slug(u)}_own`, from: "units" as const, unit: [u], rel: OWN },
    { name: `struct_${slug(u)}_hostile`, from: "units" as const, unit: [u], rel: HOSTILE },
  ]),
  { name: "boats_own", from: "units", unit: ["Transport"], rel: OWN },
  { name: "boats_hostile", from: "units", unit: ["Transport"], rel: HOSTILE },
  { name: "warships_own", from: "units", unit: ["Warship"], rel: OWN },
  { name: "warships_hostile", from: "units", unit: ["Warship"], rel: HOSTILE },
  { name: "nukes_own", from: "units", unit: ["Atom Bomb", "Hydrogen Bomb", "MIRV", "MIRV Warhead"], rel: OWN },
  { name: "nukes_hostile", from: "units", unit: ["Atom Bomb", "Hydrogen Bomb", "MIRV", "MIRV Warhead"], rel: HOSTILE },
  { name: "sam_cover_hostile", from: "sam_cover", unit: ["SAM Launcher"], rel: HOSTILE },
  { name: "land_attacking_us", from: "owner", rel: [RELATION.attacksUs, RELATION.both] },
  { name: "land_we_attack", from: "owner", rel: [RELATION.weAttack, RELATION.both] },
  { name: "valid", from: "valid" },
];

function slug(s: string): string {
  return s.toLowerCase().replace(/[^a-z]+/g, "_").replace(/^_|_$/g, "");
}

// -- wire fields ----------------------------------------------------------------
export type DType = "u8" | "u16" | "i16" | "i32" | "f16" | "f32";
export interface FieldSpec {
  name: string;
  dtype: DType;
  shape: number[];
  scope: "game" | "seat";
}

export const DTYPE_BYTES: Record<DType, number> = { u8: 1, u16: 2, i16: 2, i32: 4, f16: 2, f32: 4 };

export function obsFieldSpecs(): FieldSpec[] {
  return [
    // per game (shared by the game's seats; repeated per row on the wire)
    { name: "mm_owner", dtype: "u16", shape: [MM, MM], scope: "game" },
    { name: "mm_terrain", dtype: "u8", shape: [MM, MM], scope: "game" },
    { name: "density", dtype: "f16", shape: [MAX_SMALL_ID], scope: "game" },
    { name: "units", dtype: "i16", shape: [MAX_UNITS, UNIT_COLS], scope: "game" }, // tile coords <= 4432 fit
    { name: "game_meta", dtype: "f32", shape: [GAME_META], scope: "game" },
    // per seat
    { name: "relation", dtype: "u8", shape: [MAX_SMALL_ID], scope: "seat" },
    { name: "win_meta", dtype: "f32", shape: [NUM_WINDOWS, WIN_META], scope: "seat" },
    { name: "win_owner", dtype: "u16", shape: [NUM_WINDOWS, WIN, WIN], scope: "seat" },
    { name: "win_terrain", dtype: "u8", shape: [NUM_WINDOWS, WIN, WIN], scope: "seat" },
    { name: "players", dtype: "f16", shape: [NUM_PLAYER_TOKENS, PF], scope: "seat" },
    { name: "edges", dtype: "f16", shape: [NUM_PLAYER_TOKENS, NUM_PLAYER_TOKENS, EF], scope: "seat" },
    { name: "structs", dtype: "f16", shape: [NUM_STRUCTS, SF], scope: "seat" },
    { name: "mobile", dtype: "f16", shape: [NUM_MOBILE, MF], scope: "seat" },
    { name: "attacks", dtype: "f16", shape: [NUM_ATTACKS, AF], scope: "seat" },
    { name: "globals", dtype: "f32", shape: [GF], scope: "seat" },
    { name: "type_mask", dtype: "u8", shape: [NUM_TYPES], scope: "seat" },
    { name: "build_mask", dtype: "u8", shape: [NUM_BUILD], scope: "seat" },
    { name: "player_mask", dtype: "u8", shape: [NUM_TYPES, NUM_PLAYER_TOKENS], scope: "seat" },
    { name: "entity_mask", dtype: "u8", shape: [NUM_TYPES, ENTITY_SLOTS], scope: "seat" },
  ];
}

export const GAME_META_FIELDS = ["map_w", "map_h", "mm_tpc", "ticks", "spawn_phase", "off_x", "off_y", "sam_range"] as const;
export const WIN_META_FIELDS = ["x0", "y0", "tpc", "kind", "valid", "priority"] as const;
export const WINDOW_KINDS = ["core", "front", "coast", "nuke"] as const;

export function fieldBytes(f: FieldSpec): number {
  return f.shape.reduce((a, b) => a * b, 1) * DTYPE_BYTES[f.dtype];
}

export function rowBytes(): { game: number; seat: number; total: number } {
  let game = 0;
  let seat = 0;
  for (const f of obsFieldSpecs()) {
    if (f.scope === "game") game += fieldBytes(f);
    else seat += fieldBytes(f);
  }
  return { game, seat, total: game + seat };
}

export function layoutJson(): object {
  return {
    layoutVersion: LAYOUT_VERSION,
    sizes: {
      MM, WIN, NUM_WINDOWS, WINDOW_TPC, NUM_PLAYER_TOKENS, NUM_REAL_TOKENS, REST_TOKEN, UNOWNED_TOKEN, MAX_SMALL_ID,
      MAX_UNITS, UNIT_COLS, NUM_STRUCTS, NUM_MOBILE, NUM_ATTACKS, ENTITY_SLOTS, K, A, NUM_TYPES, NUM_BUILD,
      TROOP_BUCKETS, FINE, DECISION_TURNS, GAME_META, WIN_META,
    },
    actionFields: ACTION_FIELDS,
    actionTypes: ACTION_TYPES,
    typeArgs: TYPE_ARGS,
    entityTable: ENTITY_TABLE,
    buildTypes: BUILD_TYPES,
    nukeBuilds: [...NUKE_BUILDS],
    relation: RELATION,
    terrainBits: TERRAIN_BITS,
    unitTypes: UNIT_TYPES,
    unitColumns: UNIT_COLUMNS,
    unitFlags: UNIT_FLAGS,
    gameMetaFields: GAME_META_FIELDS,
    winMetaFields: WIN_META_FIELDS,
    windowKinds: WINDOW_KINDS,
    features: {
      players: PLAYER_FEATURES,
      edges: EDGE_FEATURES,
      structs: STRUCT_FEATURES,
      mobile: MOBILE_FEATURES,
      attacks: ATTACK_FEATURES,
      globals: GLOBAL_FEATURES,
    },
    planes: PLANES,
    fields: obsFieldSpecs(),
    rowBytes: rowBytes(),
  };
}
