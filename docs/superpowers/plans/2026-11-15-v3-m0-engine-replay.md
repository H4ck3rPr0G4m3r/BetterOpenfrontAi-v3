# M0: engine, version cache, record replayer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** replay real public FFA game records through the exact engine commit that played them, with every state hash matching.

**Architecture:** a rate-limited Python API client fetches a sample of records; the engine is a git submodule; per record commit, `ofai.engines` makes a `git worktree`, bundles `sim/src/cli.ts` against it with esbuild, and runs `sim.mjs replay`, which steps the real engine turn by turn and compares hashes.

**Tech Stack:** Python 3.11 (`httpx`, `pytest`), Node 24, TypeScript, vitest, esbuild.

**Spec:** [`docs/superpowers/specs/2026-11-14-v3-design.md`](../specs/2026-11-14-v3-design.md) §7, §8.1, §18 (M0). Roadmap: [`2026-11-15-v3-00-roadmap.md`](2026-11-15-v3-00-roadmap.md).

## Global Constraints

All of roadmap §2 apply. Most relevant here: ≤ 1 request/s, backoff 2 s → 120 s honouring `Retry-After`, User-Agent `ofai-crawler/0.1`, stop on 403; no real records committed; hash checked every 10 turns and any mismatch rejects the game; data filter `type == Public`, FFA, ≥ 300 s, no modifiers.

## Review Focus

1. **A record whose commit can't be fetched or built**: counted under that commit with the failing stage; the batch continues (Task 5 test `build failure is recorded, not raised`, Task 6 test `failed bundle counts and continues`).
2. **A record with an intent type the bundled engine rejects**: `ReplayResult.error` is set, the process exits 0 for that game and the batch continues (Task 4 test `unknown intent type is an error result`).
3. **Two records on the same map replayed in one process**: the second must not inherit the first's terrain (v2's `loadTerrainMap` cache bug) (Task 3 test `fresh terrain per game`).
4. **Records with trailing empty turns** (`num_turns` > last stored turn): replay runs to `num_turns` (Task 4 test `fills to num_turns`).
5. **API returns a history entry missing a field** (schema drift): the entry is skipped and counted, never a crash (Task 1 test `malformed history entry is skipped`).

---

### Task 1: Python package, API client, `fetch-sample`

**Files:**
- Create: `learn/pyproject.toml`, `learn/ofai/__init__.py`, `learn/ofai/__main__.py`, `learn/ofai/cli.py`, `learn/ofai/paths.py`, `learn/ofai/data/__init__.py`, `learn/ofai/data/api.py`, `learn/ofai/data/filters.py`
- Create: `learn/tests/test_api.py`, `learn/tests/test_filters.py`
- Modify: `.gitignore` (add `learn/.pytest_cache`, `*.egg-info`)

**Interfaces:**
- Produces:
  - `ofai.paths`: `REPO, ENGINE, ENGINES, DATA, RECORDS, RUNS, CHECKPOINTS: Path` (`RECORDS = DATA / "records"`).
  - `class OpenFrontApi(base: str = "https://api.openfront.io", min_interval: float = 1.0, transport: httpx.BaseTransport | None = None, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep)`
    - `leaderboard(page: int) -> dict` (`GET /leaderboard/ranked?page=N`)
    - `player_games(public_id: str, cursor: str | None = None) -> tuple[list[dict], str | None]` (`GET /public/player/{id}/games`, returns `results`, `nextCursor`)
    - `game(game_id: str) -> dict` (`GET /game/{id}`, the GameRecord)
    - raises `ApiForbidden` on 403, `ApiError` after 6 failed attempts.
  - `ofai.data.filters`: `wanted_summary(entry: dict) -> bool` (history entry: `type == "Public"`, `mode == "Free For All"`, `durationSeconds >= 300`); `modifiers(config: dict) -> list[str]` (names of active modifiers, empty = clean); `MIN_DURATION_S = 300`.
  - CLI `python -m ofai fetch-sample --players 20 --games 30` → writes `data/records/<commit>/<gameId>.json.gz` and prints a JSON summary `{"fetched", "skipped": {reason: n}, "commits": {commit: n}, "modifiers": {name: n}}`.

- [ ] **Step 1: Write the failing tests**

`learn/tests/test_api.py` uses `httpx.MockTransport` and a fake clock/sleep that records sleeps:

```python
def test_requests_are_spaced_one_second():      # 3 calls -> sleeps sum >= 2.0, no gap < 1.0
def test_user_agent_is_generic():               # request.headers["user-agent"] == "ofai-crawler/0.1"
def test_429_backs_off_and_honours_retry_after():  # 429 with Retry-After: 7 then 200 -> a sleep of 7 recorded, result returned
def test_5xx_backoff_doubles_from_2s_capped_120():  # five 503s then 200 -> sleeps 2,4,8,16,32 (plus spacing)
def test_gives_up_after_6_attempts():           # always 503 -> ApiError
def test_403_raises_forbidden_immediately():    # 403 -> ApiForbidden, exactly 1 request made
def test_player_games_returns_results_and_cursor():  # {"results":[...],"nextCursor":"abc"} -> (list, "abc")
```

`learn/tests/test_filters.py`:

```python
def test_wanted_summary_public_ffa_long():  assert wanted_summary({"type":"Public","mode":"Free For All","durationSeconds":300})
def test_rejects_private(): ...             # type "Private" -> False
def test_rejects_short(): ...               # durationSeconds 299 -> False
def test_malformed_history_entry_is_skipped():  # {} -> False (no KeyError)
def test_clean_config_has_no_modifiers():   # config with gameMapSize "Normal", randomSpawn False, disabledUnits [], doomsdayClock {"enabled": False} -> []
def test_detects_modifiers():
    cfg = {"gameMapSize": "Compact", "randomSpawn": True, "infiniteGold": True, "instantBuild": True,
           "disabledUnits": ["SAM Launcher"], "doomsdayClock": {"enabled": True}, "goldMultiplier": 2,
           "publicGameModifiers": {"isCrowded": True, "startingGold": 0}}
    assert set(modifiers(cfg)) == {"gameMapSize", "randomSpawn", "infiniteGold", "instantBuild", "disabledUnits",
                                   "doomsdayClock", "goldMultiplier", "publicGameModifiers.isCrowded"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd learn && python -m pytest tests/test_api.py tests/test_filters.py -q`
Expected: FAIL / errors with `ModuleNotFoundError: No module named 'ofai'` (install first with `pip install -e .[dev]`).

- [ ] **Step 3: Implement `pyproject.toml`, `paths.py`, `api.py`, `filters.py`**

`pyproject.toml`: package `ofai`, `requires-python = ">=3.11"`, deps `httpx>=0.27`, `pyyaml>=6`, `numpy>=1.26`; extra `dev = ["pytest>=8"]`; torch is installed separately (CUDA index, see README). `REPO` = parent of `learn/`. Backoff: wait `Retry-After` if present else `min(2 * 2**attempt, 120)`; spacing is enforced before every request, retries included. `modifiers()` flags: `gameMapSize != "Normal"`, truthy `randomSpawn/infiniteGold/infiniteTroops/instantBuild`, non-empty `disabledUnits`, `doomsdayClock.enabled`, `goldMultiplier` not in `(None, 1)`, and every truthy entry of `publicGameModifiers` as `publicGameModifiers.<key>` (`startingGold: 0` is not truthy).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd learn && python -m pytest tests/test_api.py tests/test_filters.py -q`
Expected: all pass.

- [ ] **Step 5: Implement `fetch-sample` in `cli.py`**

Walk `leaderboard(1)["ffa"]` top `--players` public IDs; per player page history until `--games` wanted games are collected overall (dedupe by gameId, skip already-stored files); fetch each record; drop it if `modifiers(record["info"]["config"])` is non-empty (count by name); store gzip JSON at `RECORDS / gitCommit / f"{gameId}.json.gz"`. Unknown-shape entries are counted under `skipped["malformed"]`.

- [ ] **Step 6: Run it for real**

Run: `cd learn && python -m ofai fetch-sample --players 20 --games 30`
Expected: ≥ 20 records stored under `data/records/`, a summary listing ≥ 1 commit. Copy the summary into `docs/measurements.md` under "M0 — sample fetch (date)".

- [ ] **Step 7: Commit**

```bash
git add learn .gitignore docs/measurements.md
git commit -m "learn: rate-limited OpenFront API client and fetch-sample"
```

---

### Task 2: Engine submodule and the `sim` package

**Files:**
- Create: `engine` (submodule), `sim/package.json`, `sim/tsconfig.json`, `sim/vitest.config.ts`, `sim/src/ambient.d.ts`, `sim/tests/engine.smoke.test.ts`
- Create: `docs/engine.md` (how the pin is chosen and bumped)

**Interfaces:**
- Produces: import alias `@engine/*` → `engine/src/*` and `resources/*` → `engine/resources/*` in `tsconfig.json` and `vitest.config.ts`; npm scripts `test`, `typecheck`, `bundle`.

- [ ] **Step 1: Pick the pin**

The pin is the most common `gitCommit` among the records fetched in Task 1 (`ls data/records`, largest directory). Record it in `docs/engine.md` with the date.

- [ ] **Step 2: Add the submodule at that commit and install**

```bash
git submodule add https://github.com/openfrontio/OpenFrontIO.git engine
git -C engine fetch origin <commit> && git -C engine checkout <commit>
cd engine && npm ci --ignore-scripts
```
Expected: `git -C engine rev-parse HEAD` prints the pinned commit. If the fetch fails ("not our ref"), stop: record commits don't resolve upstream (spec §7 open risk). Report to Peter before continuing.

- [ ] **Step 3: Write the failing smoke test**

```ts
// sim/tests/engine.smoke.test.ts
import { simpleHash } from "@engine/core/Util";
test("engine imports through the alias", () => {
  expect(typeof simpleHash("abc")).toBe("number");
});
```

- [ ] **Step 4: Run to verify it fails**

Run: `cd sim && npm install && npx vitest run tests/engine.smoke.test.ts`
Expected: FAIL, cannot resolve `@engine/core/Util` (no alias yet).

- [ ] **Step 5: Add `tsconfig.json` and `vitest.config.ts`**

Mirror the engine's own compiler flags (v2's `env/tsconfig.json` lists the exact combination that type-checks the engine: `experimentalDecorators`, `useDefineForClassFields: false`, `strictNullChecks`, `moduleResolution: "bundler"`). Add `paths` and vitest `resolve.alias` for `@engine` and `resources`, `testTimeout: 60_000`. `ambient.d.ts` declares the client globals the engine's `NameBoxCalculator` import chain needs (copy v2's `env/src/ambient.d.ts` and trim to what tsc asks for).

- [ ] **Step 6: Run to verify it passes, and typecheck**

Run: `cd sim && npx vitest run && npx tsc --noEmit`
Expected: 1 test passes; no type errors.

- [ ] **Step 7: Commit**

```bash
git add .gitmodules engine sim docs/engine.md
git commit -m "sim: engine submodule pinned to the live commit, sim package skeleton"
```

---

### Task 3: Engine adapter and synthetic records

**Files:**
- Create: `sim/src/engine/Adapter.ts`, `sim/src/engine/FsMapLoader.ts`, `sim/src/record/Record.ts`, `sim/src/record/SyntheticRecord.ts`
- Test: `sim/tests/engine/Adapter.test.ts`, `sim/tests/record/SyntheticRecord.test.ts`

**Interfaces:**
- Consumes: `@engine/*` alias (Task 2).
- Produces:
  - `Record.ts`: `type GameRecordJson` (the API shape: `version`, `gitCommit`, `info: {gameID, config, players[{clientID, username, publicID?}], winner, duration, num_turns}`, `turns: {turnNumber, intents, hash?}[]`); `readRecord(path: string): GameRecordJson` (gunzips `.gz`); `fillTurns(rec): Turn[]` (dense turns 0..num_turns−1, same rule as engine `decompressGameRecord`).
  - `Adapter.ts` (the only file importing `@engine/*` at runtime in M0):
    - `interface EngineGame { readonly game: Game; tick(turn: Turn): TickOutcome; ticks(): number; }`
    - `interface TickOutcome { ok: boolean; hashes: { tick: number; hash: number }[]; error?: string }`
    - `createGame(start: GameStartInfo, engineRoot: string): Promise<EngineGame>` — same construction order as the engine's `createGameRunner` (PRNG from `simpleHash(gameID)`, humans via `random.nextID()`, then `createNationsForGame`, then `createGame`, `GameRunner.init()`), but terrain loaded fresh every call through `FsMapLoader` (never the engine's cached `loadTerrainMap`).
    - `startInfoFromRecord(rec: GameRecordJson): GameStartInfo`.
  - `SyntheticRecord.ts`: `makeSyntheticRecord(opts: { map: string; seats: number; turns: number; seed: string; engineRoot: string }): Promise<GameRecordJson>` — runs a real game with `seats` human players that spawn on random land and then attack unowned land every 20 turns, collects hashes into the turns, omits empty turns, sets `gitCommit: "SYNTHETIC"`.

- [ ] **Step 1: Write the failing tests**

```ts
test("fresh terrain per game", async () => {
  // two createGame() calls on the same map in one process; after 200 ticks of game A,
  // game B's tile 0..N ownership is all unowned and B's hash at tick 10 equals a third fresh game's hash at tick 10
});
test("synthetic record round-trips through readRecord/fillTurns", async () => {
  const rec = await makeSyntheticRecord({ map: SMALL_MAP, seats: 3, turns: 600, seed: "s1", engineRoot });
  expect(fillTurns(rec)).toHaveLength(600);
  expect(rec.turns.filter(t => t.hash !== undefined).length).toBe(60);
  expect(rec.turns.some(t => t.intents.some(i => i.type === "spawn"))).toBe(true);
});
test("same seed gives identical hashes", ...);   // two makeSyntheticRecord with seed "s1" -> equal hash lists
```

Use the smallest map in `engine/resources/maps` by `map.bin` size for speed (look it up once; write its enum name into a `SMALL_MAP` constant in `sim/tests/helpers.ts`).

- [ ] **Step 2: Run to verify they fail**

Run: `cd sim && npx vitest run tests/engine tests/record`
Expected: FAIL, modules not found.

- [ ] **Step 3: Implement `Record.ts`, `FsMapLoader.ts`, `Adapter.ts`, `SyntheticRecord.ts`**

`FsMapLoader implements GameMapLoader` reading `<engineRoot>/resources/maps/<dir>/` (dir name = lowercased enum key, as the engine's own loaders do — check the commit's `FetchGameMapLoader` for the exact rule). Port the fresh-terrain logic from v2 `env/src/Terrain.ts` (`loadFreshTerrainMap`, including Compact scaling). Hash collection: subscribe to the runner callback and take `GameUpdateType.Hash` updates.

- [ ] **Step 4: Run to verify they pass**

Run: `cd sim && npx vitest run tests/engine tests/record`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add sim
git commit -m "sim: engine adapter with fresh terrain, record IO, synthetic records"
```

---

### Task 4: Record replayer and the sim CLI

**Files:**
- Create: `sim/src/record/Replayer.ts`, `sim/src/cli.ts`, `sim/scripts/bundle.mjs`
- Test: `sim/tests/record/Replayer.test.ts`, `sim/tests/cli.test.ts`

**Interfaces:**
- Consumes: `createGame`, `startInfoFromRecord`, `fillTurns`, `readRecord`, `makeSyntheticRecord` (Task 3).
- Produces:
  - `interface ReplayResult { gameId: string; commit: string; turns: number; hashChecks: number; mismatchAt: number | null; error: string | null; ms: number; ticksPerSec: number }`
  - `replayRecord(rec: GameRecordJson, engineRoot: string, opts?: { onTick?: (g: EngineGame, turn: number) => void }): Promise<ReplayResult>` — stops at the first mismatch; `onTick` is the hook M2's sampler uses.
  - `cli.ts` commands `selfcheck` and `replay` per roadmap §3.3.
  - `node sim/scripts/bundle.mjs --engine-root <dir> --out <file>`: esbuild bundle of `src/cli.ts`, `platform: node`, `format: esm`, aliases `@engine` → `<dir>/src`, `resources` → `<dir>/resources`, packages resolved from `<dir>/node_modules` (roadmap §3.2 dependency rule).

- [ ] **Step 1: Write the failing tests**

```ts
test("synthetic record replays with every hash matching", ...)      // mismatchAt null, hashChecks 60, error null
test("tampered hash is caught at that turn", ...)                  // set turn 300's hash += 1 -> mismatchAt 300
test("tampered intent desyncs", ...)                               // change one attack's troops -> mismatchAt !== null
test("unknown intent type is an error result", ...)                // push {type:"warp_drive"} at turn 50 -> error matches /intent/, no throw
test("fills to num_turns", ...)                                    // set info.num_turns = last turn + 40 -> result.turns === num_turns
test("cli replay prints one ReplayResult line", ...)               // spawn `npx tsx src/cli.ts replay <tmp.json.gz> --engine-root ../engine`
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd sim && npx vitest run tests/record/Replayer.test.ts tests/cli.test.ts`
Expected: FAIL, modules not found.

- [ ] **Step 3: Implement `Replayer.ts`, `cli.ts`, `bundle.mjs`**

Hash rule: after executing turn `t`, every Hash update with `tick == t` is compared with `fillTurns(rec)[t].hash` when that is defined. Validate intents with the engine's `StampedIntentSchema` before adding a turn; a failure becomes `error: "bad intent at turn t: <zod message>"`.

- [ ] **Step 4: Run to verify they pass**

Run: `cd sim && npx vitest run`
Expected: all pass.

- [ ] **Step 5: Bundle and replay one real record against the pin**

Run: `node sim/scripts/bundle.mjs --engine-root engine --out engines/_pin/sim.mjs && node engines/_pin/sim.mjs replay data/records/<pin>/<any>.json.gz --engine-root engine`
Expected: one JSON line, `mismatchAt: null`, `error: null`. If it mismatches, use superpowers:systematic-debugging before going on: everything later depends on this.

- [ ] **Step 6: Commit**

```bash
git add sim
git commit -m "sim: record replayer with hash checks, CLI and bundler"
```

---

### Task 5: Version cache (`ofai.engines`)

**Files:**
- Create: `learn/ofai/engines.py`, `learn/tests/test_engines.py`

**Interfaces:**
- Consumes: `sim/scripts/bundle.mjs`, `sim.mjs selfcheck` (Task 4).
- Produces:
  - `@dataclass BundleMeta` with the fields of roadmap §3.2; `BundleMeta.load(commit) -> BundleMeta | None`.
  - `ensure_bundle(commit: str, *, run: Runner = subprocess_runner) -> BundleMeta` — idempotent; never raises for build failures (returns `status="failed"` with `stage` and `error`); rebuilds when `simVersion` or the bundle script hash changed.
  - `run_sim(commit: str, *args: str) -> list[dict]` — runs `node engines/<commit>/sim.mjs <args> --engine-root engines/<commit>/src`, parses JSON lines.
  - `Runner = Callable[[list[str], Path], CompletedProcess]` (injectable for tests).

- [ ] **Step 1: Write the failing tests** (fake `Runner` recording commands)

```python
def test_ok_build_runs_stages_in_order(tmp_path): ...    # fetch, worktree add, bundle, selfcheck; meta.status == "ok"
def test_deps_junction_when_lockfile_matches(tmp_path): ...  # same lock hash -> node_modules junction to engine/node_modules, no "npm ci"; different -> "npm ci --ignore-scripts" in the worktree
def test_build_failure_is_recorded_not_raised(tmp_path): ...  # bundle step exit 1 -> status "failed", stage "bundle", error has stderr tail
def test_unresolvable_commit_fails_at_fetch(tmp_path): ...   # fetch exit 128 -> stage "fetch"
def test_second_call_is_cached(tmp_path): ...           # ok meta present -> no commands run
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd learn && python -m pytest tests/test_engines.py -q`
Expected: FAIL, `ofai.engines` missing.

- [ ] **Step 3: Implement `engines.py`**

Stages: `git -C engine fetch --depth 1 origin <commit>`; `git -C engine worktree add --detach <ENGINES>/<commit>/src <commit>` (skip if present); deps rule (junction via `mklink /J` on Windows, `os.symlink` elsewhere); `node sim/scripts/bundle.mjs --engine-root ... --out ...`; `node sim.mjs selfcheck`. Write `meta.json` atomically (tmp + `os.replace`).

- [ ] **Step 4: Run to verify they pass**

Run: `cd learn && python -m pytest tests/test_engines.py -q`
Expected: all pass.

- [ ] **Step 5: Build the pin for real and time it**

Run: `cd learn && python -m ofai build-engine <pinned commit>` (add this thin CLI command), then a second commit from `data/records` if one exists.
Expected: `status: ok`. Record build time and `engines/<commit>` disk size in `docs/measurements.md`.

- [ ] **Step 6: Commit**

```bash
git add learn
git commit -m "learn: per-commit engine bundles with recorded failure stages"
```

---

### Task 6: `replay-check` and the M0 exit check

**Files:**
- Modify: `learn/ofai/cli.py` (add `replay-check`)
- Create: `learn/ofai/data/replay_check.py`, `learn/tests/test_replay_check.py`
- Modify: `docs/measurements.md`, `README.md` (status line: M0 done)

**Interfaces:**
- Consumes: `ensure_bundle`, `run_sim` (Task 5); `RECORDS`.
- Produces: `check_records(paths: list[Path], *, jobs: int) -> CheckReport`; `CheckReport` = `{"games": int, "ok": int, "mismatch": int, "error": int, "perCommit": {commit: {"bundle": "ok"|"failed:<stage>", "ok": n, "mismatch": n, "error": n}}, "ticksPerSec": {map: median}}`; results appended to `data/replay_check.jsonl` (one `ReplayResult` per line, plus `map`).

- [ ] **Step 1: Write the failing tests** (fake `ensure_bundle`/`run_sim`)

```python
def test_groups_by_commit_and_counts(): ...            # 3 records, 2 commits -> perCommit has both with right counts
def test_failed_bundle_counts_and_continues(): ...     # commit B bundle failed -> its games counted under "failed:bundle", commit A still checked
def test_runs_in_parallel_up_to_jobs(): ...            # jobs=4 -> never more than 4 concurrent run_sim calls
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd learn && python -m pytest tests/test_replay_check.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement `replay_check.py` and the CLI command**

`python -m ofai replay-check [--jobs 14] [--since <ISO date>]`, default `jobs = os.cpu_count() - 2`. Prints the `CheckReport` as JSON.

- [ ] **Step 4: Run to verify they pass**

Run: `cd learn && python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Exit check**

Run: `cd learn && python -m ofai fetch-sample --players 30 --games 25 && python -m ofai replay-check`
Expected: ≥ 20 freshly fetched public FFA records with `ok == games` (zero mismatches, zero errors) among records whose bundle built. Record in `docs/measurements.md`: games per commit, how many commits resolved/built, replay ticks/s per map (median), bundle build time and size. If more than half of the fetched games fall under failed bundles, stop and report to Peter (spec §7).

- [ ] **Step 6: Commit and open the PR**

```bash
git add learn docs README.md
git commit -m "M0: replay-check; 20 fresh public FFA records replay with every hash matching"
git push -u origin m0-engine-replay
gh pr create --title "M0: engine, version cache, record replayer" --body "<problem, solution, tradeoff, verification incl. the measurements table>"
```
