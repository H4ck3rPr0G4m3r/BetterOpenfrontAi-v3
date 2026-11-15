# M2: observation, actions, translator, sampler, multi-seat env — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** turn any game state on any map into the v3 observation, turn human intents into v3 action labels (and back) for ≥ 98 % of teacher intents, and serve both replay samples and live multi-seat games to Python over the bridge.

**Architecture:** everything in `sim/` behind the engine adapter. Per decision step a `GameObsCache` computes the seat-independent part once (minimap rasters, unit list, per-player density, adjacency counts); per seat it adds perspective, zoom windows, entity tables and masks. The `ActionCodec` decodes actions to engine intents with snapping; the `Translator` maps human intents to actions using the same tables. The bridge server runs in `replay` role (teacher samples from records) or `env` role (policy-controlled seats).

**Tech Stack:** TypeScript, vitest, Node `net`; Python `numpy` for the bridge client.

**Spec:** §8.2–8.3, §9, §10, §18 (M2). Roadmap §3.3–3.6.

## Global Constraints

All of roadmap §2. Most relevant: decision every 5 turns; K = 4; 20 troop buckets; minimap 128×128; ≤ 8 windows of 32×32 at 2–4 tiles/cell; 32 player tokens + rest + unowned; self-play latency 1–3 turns; every AI name in games we construct starts with `[AI] ` (roadmap §2, AI identity); nothing depends on map identity.

## Review Focus

1. **Extreme maps** (smallest 512×512, largest 4432×2528, portrait 2036×4428): no out-of-bounds reads, letterbox cells are `0xFFFF` (Task 2 tests `extreme maps keep aspect and letterbox`, Task 3 test `encodes on extreme maps`).
2. **Windows at map edges**: a window centred near a corner is shifted inside the map, never clipped (Task 4 test `edge windows are shifted inside`).
3. **Teacher who dies, disconnects or never spawns**: rows go DONE then invalid; the game continues for other teachers (Task 9 test `dead teacher row goes done then invalid`).
4. **More than K=4 intents from one seat in a step**: the extra intents carry to the next step, counted `carried` (Task 8 test `overflow carries to next step`).
5. **A decoded location that cannot be snapped** (all water around a build cell): the action is dropped and counted `unsnappable`, never retargeted (Task 7 test `unsnappable build is dropped and counted`).

---

### Task 1: Layout

**Files:**
- Create: `sim/src/obs/Layout.ts`, `sim/tests/obs/Layout.test.ts`
- Modify: `sim/src/cli.ts` (`layout` command)

**Interfaces:**
- Produces:
  - Constants: `LAYOUT_VERSION = 1`, `MM = 128`, `WIN = 32`, `NUM_WINDOWS = 8`, `NUM_PLAYER_TOKENS = 34` (`REST_TOKEN = 32`, `UNOWNED_TOKEN = 33`), `MAX_SMALL_ID = 1024`, `MAX_UNITS = 1024`, `NUM_STRUCTS = 64`, `NUM_MOBILE = 32`, `NUM_ATTACKS = 32`, `K = 4`, `A = 10`, `NUM_TYPES = 17`, `NUM_BUILD = 10`, `TROOP_BUCKETS = 20`, `FINE = 16`, `DECISION_TURNS = 5`.
  - Feature name lists (each feature's index is its position): `PLAYER_FEATURES`, `EDGE_FEATURES`, `STRUCT_FEATURES`, `MOBILE_FEATURES`, `ATTACK_FEATURES`, `GLOBAL_FEATURES`, `UNIT_TYPES` (unit-list type codes), `RELATION` codes (roadmap §3.5), `TERRAIN_BITS`, `ACTION_FIELDS`, `ACTION_TYPES`, `BUILD_TYPES` (roadmap §3.6 order).
  - `PLANES`: the 31 GPU planes with their derivation, e.g. `{name: "own", from: "owner", rel: [1]}`, `{name: "density_hostile", from: "owner*density", rel: [3,4,5,6]}`, `{name: "struct_city_own", from: "units", unit: "City", rel: [1]}`, `{name: "sam_cover_hostile", from: "units", unit: "SAMLauncher", rel: [3,4,5,6], radius: "samRange"}`, `{name: "valid", from: "owner", notValue: 0xFFFF}`. Planes (spec §9.2): land, water, coast; own, allied, hostile, unowned; density own, density hostile; 6 structure types × own/hostile; boats own/hostile; warships own/hostile; nukes own/hostile; hostile SAM cover; land attacking us; land we attack; valid.
  - `obsFieldSpecs(): {name: string; dtype: "u8"|"u16"|"i32"|"f16"|"f32"; shape: number[]; scope: "game"|"seat"}[]` in wire order (roadmap §3.5).
  - `layoutJson(): object` = everything above; `sim.mjs layout` prints it.

Feature lists are spec §9.1, §9.4, §9.5 made concrete. Player: alive, isSelf, kind one-hot (human/nation/bot), tiles share, log tiles, troops ÷ max troops, log troops, gold (log), strength vs self (log ratio of troops), relation one-hot, pending request in / out, traitor, embargo either way, alliance expiry (ticks left ÷ duration), attacking us (troops ÷ our troops), we attack (same), is in rest pool count (rest token only). Edge: shares land border, border length (log tiles), density each side, plains/highland/mountain share along border, attacking i→j troops (log), sea-only distance (normalized). Struct: type one-hot, level, window index or −1, x, y normalized, under attack. Mobile: kind (warship/transport), troops, x, y, has target. Attack: outgoing/incoming, troops (log), target/source token index, is boat. Globals: ticks, spawn phase, spawn phase ticks left, gold (log), troops, max troops, growth rate, tiles share, counts per structure type, affordability per build type, config flags (nukes, SAMs, alliances, waterNukes), map w and h (log), minimap tiles-per-cell, decision interval.

- [ ] **Step 1: Failing tests**: `field specs sum to the expected byte sizes` (per game and per seat bytes printed and asserted < 64 KiB per seat), `plane list has 31 unique names ending with valid`, `layout json snapshot` (vitest `toMatchSnapshot`, so any change is deliberate and bumps `LAYOUT_VERSION`).
- [ ] **Step 2: Run → FAIL.** **Step 3: Implement.** **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `"sim: observation and action layout v1"`.

---

### Task 2: Geometry

**Files:**
- Create: `sim/src/obs/Geometry.ts`, `sim/tests/obs/Geometry.test.ts`

**Interfaces:**
- Produces:
  - `class MinimapGeometry(width: number, height: number)` with `tpc` (= `ceil(max(w,h)/MM)`), `offX`, `offY` (centring letterbox, in cells), `cellOfTile(x, y): number`, `tileRectOfCell(cell): {x0, y0, x1, y1} | null` (null for letterbox), `fineToTile(cell, fine): {x, y}` (16×16 grid over the cell's tile rect, centre of the fine cell).
  - `interface WindowSpec { x0: number; y0: number; tpc: 2 | 3 | 4; kind: WindowKind; priority: number }`; `windowCellToTile(w, cell): {x, y}` (centre tile), `tileToWindowCell(w, x, y): number | -1`, `clampWindow(w, mapW, mapH): WindowSpec` (shift inside the map).

- [ ] **Step 1: Failing tests**

```ts
test.each([[512,512],[2000,1000],[4432,2528],[2036,4428]])("extreme maps keep aspect and letterbox %ix%i", (w,h) => {
  const g = new MinimapGeometry(w, h);
  // every tile maps to a cell inside 0..MM*MM; letterbox cells return null rects; corner tiles map to non-letterbox cells
});
test("cellOfTile and tileRectOfCell agree", ...)    // for 1000 random tiles: tile inside tileRectOfCell(cellOfTile(tile))
test("fineToTile stays inside its cell", ...)
test("edge windows are shifted inside", ...)        // clampWindow at x0=-10 -> 0; at x0 near w -> w - WIN*tpc
test("window round trip", ...)                      // tileToWindowCell(windowCellToTile(c)) === c for all 1024 cells
```
- [ ] **Step 2: Run → FAIL.** **Step 3: Implement.** **Step 4: Run → PASS.** **Step 5: Commit** — `"sim: minimap and window geometry for any map shape"`.

---

### Task 3: Shared per-game rasters and the unit list

**Files:**
- Create: `sim/src/obs/GameObsCache.ts`, `sim/src/obs/reference/expandPlanes.ts` (test-only reference), `sim/tests/obs/GameObsCache.test.ts`
- Modify: `sim/src/engine/Adapter.ts` (adapter accessors this needs: `terrainAt`, `ownerSmallIdAt`, `allUnits`, `playersBySmallId`)

**Interfaces:**
- Consumes: `MinimapGeometry` (Task 2), adapter.
- Produces:
  - `class GameObsCache(game: EngineGame)`: `update(): void` (call once per decision step), fields `mmOwner: Uint16Array(MM*MM)`, `mmTerrain: Uint8Array(MM*MM)` (computed once per game), `density: Uint16Array(MAX_SMALL_ID)` (f16 bits), `units: Int32Array(MAX_UNITS*6)`, `unitCount`, `adjacency: Map<number, Map<number, number>>` (mm-cell border counts between owners), `ownerCells: Map<smallId, number[]>`, `geometry`.
  - Owner per minimap cell = majority of a 2×2 sub-sample of the cell's tiles (ties → the lower smallID; deterministic). Unit list priority when over `MAX_UNITS`: nukes in flight, warships, transports, then structures; trains are excluded.
  - `expandPlanesReference(cache, relation: Uint8Array): Uint8Array` — CPU reference of the 31 planes (used by M3's Python `planes.expand` equivalence test via a fixture written by `sim.mjs dump-obs`).

- [ ] **Step 1: Failing tests**: `encodes on extreme maps` (synthetic games on the smallest and largest map present in `engine/resources/maps`, plus the portrait one if present; asserts letterbox `0xFFFF` count equals geometry's), `owner majority is deterministic`, `unit overflow keeps nukes and warships`, `update cost` (logs ms; asserts < 20 ms on World after 3 minutes of a synthetic game — a guard, not a benchmark).
- [ ] **Step 2: Run → FAIL.** **Step 3: Implement.** **Step 4: Run → PASS.** **Step 5: Commit** — `"sim: shared per-game observation cache"`.

---

### Task 4: Zoom window placement

**Files:**
- Create: `sim/src/obs/Windows.ts`, `sim/tests/obs/Windows.test.ts`

**Interfaces:**
- Consumes: `GameObsCache`, `clampWindow`, `WindowSpec` (Task 2).
- Produces: `placeWindows(cache: GameObsCache, game: EngineGame, selfSmallId: number): WindowSpec[]` (length ≤ 8), deterministic in the game state.

Rule (spec §9.3), all from the minimap so it is cheap:
1. **core**: centroid of the seat's minimap cells, snapped to the nearest own cell; `tpc` 2 (3 if own cells span > 64 tiles, 4 if > 96).
2. **fronts**: for each active attack (outgoing or incoming), the centroid of own cells adjacent to the other player's cells; sorted by attack troops desc; `tpc` 2 if that adjacent set spans ≤ 64 tiles, else 4.
3. **coast**: the largest 4-connected cluster of own coastal cells; centre of the cluster; `tpc` 3.
4. **nukes**: each hostile nuke in flight targeting within 2 cells of own land, at its target tile; `tpc` 2.
Then: merge any two windows whose centres are closer than 16 tiles (keep the higher-priority one), clamp, cap at 8. Before spawning (spawn phase, no own land): no windows (spawn uses the far path).

- [ ] **Step 1: Failing tests** (constructed situations from synthetic games): `core window covers own land`, `fronts sorted by troops`, `nearby windows merge`, `edge windows are shifted inside`, `no windows before spawn`, `same state same windows`.
- [ ] **Step 2: Run → FAIL.** **Step 3: Implement.** **Step 4: Run → PASS.** **Step 5: Commit** — `"sim: deterministic zoom window placement"`.

---

### Task 5: Seat encoder (perspective, tokens, entity tables, masks)

**Files:**
- Create: `sim/src/obs/SeatEncoder.ts`, `sim/src/obs/EntityTables.ts`, `sim/src/obs/ObsBuffer.ts`, `sim/tests/obs/SeatEncoder.test.ts`

**Interfaces:**
- Consumes: `GameObsCache`, `placeWindows`, Layout field specs.
- Produces:
  - `class ObsBuffer(games: number, seatsPerGame: number)` — preallocated typed arrays for every field in `obsFieldSpecs()` order; `bytes(): Uint8Array` views for the wire.
  - `interface EntityTables { players: (number | null)[] /* token -> smallID; REST/UNOWNED null */; tokenOf: Map<number, number>; structs: number[] /* unit ids */; mobile: number[]; attacks: string[] /* attack ids */; windows: WindowSpec[] }`
  - `encodeSeat(cache: GameObsCache, game: EngineGame, clientID: string, buf: ObsBuffer, row: number): EntityTables` — writes the seat's fields into row `row`.
  - Player token selection order: self; anyone attacking us or attacked by us; land neighbours by adjacency count; then the largest by tiles; up to 32 (self included). The rest are pooled into token 32 (counts, total tiles share, max strength). Token 33 = unowned land.
  - `type_mask`/`build_mask` from the engine: `buildableUnits` (affordability, silo readiness), `outgoingAttacks().length > 0` for CANCEL_ATTACK, own transports for CANCEL_BOAT, own warships for MOVE_WARSHIP, incoming requests for ALLIANCE_REJECT, alliances near expiry for ALLIANCE_EXTEND, spawn phase for SPAWN (and SPAWN only then). STOP is always legal.

- [ ] **Step 1: Failing tests**: `self is token 0`, `attackers are always tokens`, `rest token pools the others` (game with 60 nations → 31 + rest counts 29), `entity tables map back to live units`, `masks follow the engine` (no gold → build_mask all 0 except free builds; spawn phase → only SPAWN and STOP), `seat bytes match layout`.
- [ ] **Step 2: Run → FAIL.** **Step 3: Implement** (port the relevant feature code from v2 `env/src/ObservationEncoder.ts`, `TargetSlots.ts`, `Combat.ts`; drop v2's raster). **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `"sim: seat encoder with player tokens, border edges, entity tables, engine masks"`.

---

### Task 6: Border edges

**Files:**
- Create: `sim/src/obs/Edges.ts`, `sim/tests/obs/Edges.test.ts`
- Modify: `sim/src/obs/SeatEncoder.ts` (fill `edges`)

**Interfaces:**
- Consumes: `GameObsCache.adjacency`, `ownerCells`, `mmTerrain`; `EntityTables.players`.
- Produces: `fillEdges(cache, game, tables, out: Uint16Array /* f16 [34,34,EF] */): void` — for every token pair (i, j) incl. neighbour↔neighbour: land border (adjacency count × tpc as tiles), density both sides, terrain mix along the shared cells, attack troops i→j and j→i, and for pairs without a land border the centroid distance in tiles ÷ map diagonal, flagged sea-only.

- [ ] **Step 1: Failing tests**: `neighbour-neighbour border appears` (A–B fighting, both neighbours of self → edge (A,B) has border and attack troops), `symmetric border length`, `no border means sea distance flag`.
- [ ] **Step 2: Run → FAIL.** **Step 3: Implement.** **Step 4: Run → PASS.** **Step 5: Commit** — `"sim: border link edge features"`.

---

### Task 7: Action codec (decode with snapping)

**Files:**
- Create: `sim/src/action/Codec.ts`, `sim/src/action/Snap.ts`, `sim/tests/action/Codec.test.ts`

**Interfaces:**
- Consumes: `EntityTables`, `MinimapGeometry`, `windowCellToTile`, `fineToTile`, adapter predicates (`canBuild`, `canSpawnAt`, `isValidBoatDestination`, `isWater`).
- Produces:
  - `type Decoded = { intent: StampedIntent } | { dropped: DropReason }`, `DropReason = "illegal_type" | "bad_pointer" | "unsnappable" | "stale_entity"`.
  - `decodeStep(actions: Int32Array /* K*A */, ctx: { game; clientID; tables: EntityTables; geometry }): Decoded[]` — stops at the first STOP.
  - `snapTile(start: {x,y}, ok: (tile) => boolean, radius: number): TileRef | null` — BFS ring search, nearest first, ties broken by row-major order. Radius: near = 2 × window tpc; far = 2 × fine-cell span.
  - Troops: `floor(currentTroops × (bucket + 1) × 0.05)`. ATTACK on token 33 → `targetID: null`.
  - `DecodeStats` counters per type and reason, read by the env and the translator report.

- [ ] **Step 1: Failing tests**: `attack decodes target and troops` (bucket 9 of 1000 troops → 500), `attack on unowned token sends null target`, `build snaps to nearest buildable tile`, `unsnappable build is dropped and counted`, `stale entity is dropped` (structure destroyed between encode and decode), `actions after STOP are ignored`, `never retargets to another player` (pointer to a dead player → dropped `bad_pointer`).
- [ ] **Step 2: Run → FAIL.** **Step 3: Implement** (port intent shapes from v2 `ActionCodec.ts`; verify each intent's field names against the pinned commit's `Schemas.ts`). **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `"sim: action codec with engine snapping"`.

---

### Task 8: Human-intent translator and round-trip report

**Files:**
- Create: `sim/src/translate/Translator.ts`, `sim/src/translate/Report.ts`, `sim/tests/translate/Translator.test.ts`
- Modify: `sim/src/cli.ts` (`translate-report <records...>` command)

**Interfaces:**
- Consumes: `EntityTables`, geometry (Tasks 2, 5), `decodeStep` (Task 7).
- Produces:
  - `translateIntent(intent: StampedIntent, ctx: { game; clientID; tables; geometry; troopsAtIntent: number }): { action: Int32Array } | { unrepresentable: string }` — `attack` → ATTACK (target token, bucket = `clamp(round(troops / troopsAtIntent / 0.05) − 1, 0, 19)`, `troops: null` → default ratio 0.2 → bucket 3); `boat` → BOAT (dst location, troops); `build_unit` → BUILD (type, location); `upgrade_structure` → UPGRADE (struct index); `move_warship`, `cancel_attack`, `cancel_boat`, `allianceRequest`, `allianceReject`, `allianceExtension`, `breakAlliance`, `targetPlayer`, `embargo` (start/stop by its `action` field), `delete_unit`, `spawn`. Location: near when the tile lies in a window (the highest-priority window containing it), far otherwise. Excluded types (emoji, quick_chat, donations, embargo_all, mark_disconnected, admin) → `unrepresentable: "excluded:<type>"`, which the report counts separately and does not count against the 98 %.
  - `class StepLabeler` — collects a seat's intents for the next `DECISION_TURNS` turns into ≤ K actions + STOP; extra intents carry to the next step (`carried` counter).
  - `TranslatorReport = { total, excluded, representable, rate, byType: Record<string, {total, ok, reasons: Record<string, number>}>, tileErr: {near: {p50, p95}, far: {p50, p95}}, insideWindowRate, carried }`; round trip = translate → `decodeStep` → compare type, target, troops (within one bucket) and tile distance (near ≤ 3 tiles, far ≤ fine-cell span).

- [ ] **Step 1: Failing tests** (synthetic records with scripted intents of every type): `each intent type round trips`, `null troops use default ratio bucket`, `excluded types are counted, not failed`, `overflow carries to next step`, `inside window rate counts near path`, `report on a synthetic record has rate 1.0`.
- [ ] **Step 2: Run → FAIL.** **Step 3: Implement.** **Step 4: Run → PASS.**
- [ ] **Step 5: Real report**: `node engines/<pin>/sim.mjs translate-report data/records/<pin>/*.json.gz --limit 200 --engine-root engines/<pin>/src`. Expected: `rate ≥ 0.98`. If lower, the report's `byType.reasons` says why; fix the cause (e.g. widen token selection, entity table sizes) and rerun — do not lower the gate.
- [ ] **Step 6: Commit** — `"sim: human-intent translator with round-trip report"`.

---

### Task 9: Placements and the replay sampler

**Files:**
- Create: `sim/src/replay/Placements.ts`, `sim/src/replay/Sampler.ts`, `sim/tests/replay/Sampler.test.ts`
- Modify: `sim/src/record/Replayer.ts` (`ReplayResult.placements: Record<clientID, number>`), `learn/ofai/data/replay_check.py` + `index.py` (store `seats.placement`)

**Interfaces:**
- Consumes: `replayRecord(onTick)` (M0), `GameObsCache`, `encodeSeat`, `StepLabeler`.
- Produces:
  - Placement = rank among humans and nations (bots excluded) by death tick, survivors ranked by final tiles; value target `v = 1 − (place − 1) / (n − 1)`.
  - `class Sampler(engineRoot: string, gamesInFlight: number, seatsPerGame: number)`:
    - `load(items: {path: string; teachers: {clientID: string; weight: number; value: number}[]}[]): void` (queue);
    - `next(): {buf: ObsBuffer; labels: Int32Array /* rows*K*A */; weight: Float32Array; value: Float32Array; flags: Uint8Array}` — advances every game in flight by one decision step (`DECISION_TURNS` turns), observation = state **before** those turns, labels = the teacher's intents **in** those turns. Up to `seatsPerGame` teachers per game (highest weight first).
  - Row lifecycle: FRESH on the first step of a new game; DONE on the step the teacher dies or disconnects (or the record ends); invalid afterwards until the slot gets a new game. A finished game is replaced from the queue in the same `next()`.

- [ ] **Step 1: Failing tests**: `labels come from the turns after the observation`, `dead teacher row goes done then invalid`, `teacher who never spawns stays valid through spawn phase then done`, `finished game is replaced and flagged fresh`, `empty queue leaves rows invalid`, `hash mismatch aborts that game only` (tampered record → its rows DONE with no further samples, others continue).
- [ ] **Step 2: Run → FAIL.** **Step 3: Implement.** **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `"sim: placements and the streaming replay sampler"`.

---

### Task 10: Bridge v3 (TypeScript server, Python client)

**Files:**
- Create: `sim/src/bridge/Protocol.ts`, `sim/src/bridge/Server.ts`, `sim/tests/bridge/Bridge.test.ts`
- Create: `learn/ofai/layout.py`, `learn/ofai/bridge/__init__.py`, `learn/ofai/bridge/protocol.py`, `learn/ofai/bridge/client.py`, `learn/ofai/bridge/pool.py`, `learn/tests/test_bridge.py`
- Modify: `sim/src/cli.ts` (`serve`)

**Interfaces:**
- Consumes: roadmap §3.4; `Sampler` (Task 9); `GameEnv` (Task 11, `env` role wired in Task 11).
- Produces:
  - TS: `serve(port: number, role: "env" | "replay", opts)`; message codecs per roadmap §3.4.
  - Python: `Layout.from_json(d) -> Layout` (field specs → numpy dtypes/shapes; `Layout.obs_nbytes(rows)`); `BridgeClient(host, port)` with `hello() -> dict`, `load(items)`, `next() -> Batch`, `reset(games)`, `step(actions: np.ndarray) -> Batch`, `close()`; `Batch` = dict of numpy arrays keyed by field name plus `labels`, `weight`, `value`, `reward`, `flags`, `stats`, `terminal` as present.
  - `SimPool(commit: str, role: str, workers: int, **opts)` — starts `workers` `sim.mjs serve` processes for that commit's bundle, one `BridgeClient` each; `pool.next()` returns the concatenated batch; worker death raises `SimWorkerDied` with the worker's stderr tail.

- [ ] **Step 1: Failing tests**: TS `frames round trip`, `seq mismatch is fatal`; Python (spawns a real `npx tsx src/cli.ts serve --role replay` on a synthetic record) `hello carries layout`, `next returns arrays shaped by layout`, `labels shape rows x K x A`, `worker death raises SimWorkerDied`.
- [ ] **Step 2: Run → FAIL.** **Step 3: Implement** (port framing from v2 `bridge/Protocol.ts`, `WorkerServer.ts`, `train/openfront/protocol.py`, `vec.py`). **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `"bridge v3: replay and env roles, Python client and pool"`.

---

### Task 11: Multi-seat GameEnv and AI game records

**Files:**
- Create: `sim/src/env/GameEnv.ts`, `sim/src/env/Latency.ts`, `sim/src/record/RecordWriter.ts`, `sim/tests/env/GameEnv.test.ts`
- Modify: `sim/src/bridge/Server.ts` (`env` role)

**Interfaces:**
- Consumes: adapter `createGame`, `GameObsCache`, `encodeSeat`, `decodeStep`.
- Produces:
  - `interface GameSpec { map: string; seed: string; seats: number; names?: string[]; bots: number; nations: "default" | "disabled" | number; difficulty: "Easy" | "Medium" | "Hard" | "Impossible"; tickCap: number; latency: [number, number]; record: boolean }` (`names` default `"[AI] seat<i>"`; any name without the `[AI] ` prefix gets it prepended).
  - `class GameEnv` with `reset(spec: GameSpec, buf: ObsBuffer, game: number): EntityTables[]`, `step(actions: Int32Array, buf, game): StepResult[]` where `StepResult = { reward: number; done: boolean; terminal: "none"|"win"|"death"|"tick_cap"|"sim_error"|"aborted"; stats: [tiles, troops, kills, placement, ticks]; decode: DecodeStats }`. Reward is 0 until M6 except `placement` in stats at the end. Seats choose their own spawn with SPAWN actions during the spawn phase (a seat that hasn't spawned when the phase ends gets the engine's default handling).
  - `Latency`: each step's intents are queued with a delay drawn uniformly from `latency` turns using a PRNG seeded from `spec.seed`; the queue never reorders a seat's intents.
  - `RecordWriter` builds an engine-format `GameRecord` (as v2 `Replay.ts`) with `gitCommit` = the bundle's commit, so AI games open in the M1 viewer.

- [ ] **Step 1: Failing tests**: `four seats reset and step on the smallest map`, `seat names carry the AI tag`, `latency delays intents 1-3 turns deterministically`, `dead seat goes done and inactive`, `record of an env game replays with all hashes matching` (write → `replayRecord` → no mismatch).
- [ ] **Step 2: Run → FAIL.** **Step 3: Implement** (port the seat lifecycle from v2 `GameEnv.ts`, drop its reward). **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `"sim: multi-seat GameEnv with latency and AI game records"`.

---

### Task 12: Step-cost table and M2 exit check

**Files:**
- Create: `sim/src/bench/StepCost.ts`, `sim/src/cli.ts` (`step-cost` command)
- Modify: `docs/measurements.md`, `README.md`

- [ ] **Step 1: Implement `step-cost`**: for the smallest, largest, portrait and World maps, run a 6-seat env game with random legal actions for 5 minutes of game time and a replay sampler on one real record per map where available; print per map: engine ticks/s, `GameObsCache.update` ms, `encodeSeat` ms, bytes per row, rows/s per worker.
- [ ] **Step 2: Run it** — `node engines/<pin>/sim.mjs step-cost --engine-root engines/<pin>/src`.
- [ ] **Step 3: Translator report on 200 real games** (Task 8 Step 5) and an encoder run on the smallest, largest and portrait maps without errors.
- [ ] **Step 4: Record** both tables in `docs/measurements.md`, plus the estimate: rows/s with 14 workers vs. what the 3060 Ti needs (spec §8.2). If replay sampling is below ~1,200 rows/s total, note it: M3 Task 6 then adds the hot-subset cache.
- [ ] **Step 5: PR** — `m2-observation-actions`, with both tables in the description.
