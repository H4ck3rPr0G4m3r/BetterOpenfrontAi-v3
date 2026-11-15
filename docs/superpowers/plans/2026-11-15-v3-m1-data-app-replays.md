# M1: crawler, data index, app shell, replay viewer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** crawl ~2,000 teacher games into a resumable index, and watch any of them in the real game client inside one desktop app.

**Architecture:** `ofai.data.crawl` walks leaderboard teachers' histories into `data/records/` and a SQLite index. `ofai.service` (FastAPI on `127.0.0.1:47800`) exposes data, crawl and engine jobs. The Electron app spawns the service, and for a replay starts the engine's own dev server **at the record's commit** (the client only replays records of its own engine; a `GIT_COMMIT=DEV` build skips the version check), then injects the record into the page with the client's `join-lobby` event, as v2's `watch_replay.py` did, through `webContents.executeJavaScript` instead of Playwright.

**Tech Stack:** Python (`sqlite3`, FastAPI, uvicorn, httpx), Electron + electron-vite + React + TypeScript, vitest, `@playwright/test` (Electron smoke test only).

**Spec:** §6, §8.1, §16, §18 (M1). Roadmap §3.1, §3.7, §3.8.

## Global Constraints

All of roadmap §2. Most relevant: ≤ 1 request/s for the whole crawler; `ofai-crawler/0.1`; stop on 403; version window 21 days / 10 commits; data filter; UI work loads the `frontend` skill first.

## Review Focus

1. **Crawl killed mid-run** (power cut, app closed): restart resumes from stored cursors without refetching stored records (Task 2 test `resume skips stored and continues cursor`).
2. **App closed while a dev server or the service runs**: next start kills recorded orphans before binding ports (Task 5 test `kills stale pids from pids.json`).
3. **Port 9000 already taken by something we didn't start**: error state with the hint "port 9000 is in use by another program", no kill (Task 5 test `foreign process on port is reported, not killed`).
4. **Opening a replay whose commit failed to build**: Replays shows the failing stage and keeps the rest of the library usable (Task 7 test `failed bundle shows stage`).
5. **Switching replays between two commits quickly**: only one dev server runs; the second request waits for the first to stop (Task 5 test `switching commits stops the old server first`).

---

### Task 1: SQLite index

**Files:**
- Create: `learn/ofai/data/index.py`, `learn/tests/test_index.py`

**Interfaces:**
- Consumes: `ofai.paths.DATA`; record dicts (API shape).
- Produces: `class Index(path: Path = DATA / "index.sqlite")` with
  - `add_record(rec: dict, path: Path, *, teachers: set[str]) -> None` — upserts `games(id PK, commit, map, start, duration_s, players, winner_client, in_window, path, replay_ok NULL)`, `seats(game, client_id, public_id, username, is_teacher, placement NULL)`;
  - `upsert_player(public_id: str, ffa_elo: float | None, source: str)`; `teachers(limit: int) -> list[str]` (by elo desc);
  - `get_cursor(public_id) -> str | None`; `set_cursor(public_id, cursor: str | None, done_before: str | None)`;
  - `has_game(game_id) -> bool`; `mark_replay(game_id, ok: bool)`;
  - `window_commits(now: datetime, days: int = 21, max_commits: int = 10) -> list[str]` — commits ordered by their first-seen game start, newest first, among games started within `days`;
  - `refresh_window(now) -> None` (sets `in_window`);
  - `stats() -> dict` `{"games", "inWindow", "teachers", "perMap": {map: n}, "perCommit": {commit: n}, "uniqueTeacherGames"}`;
  - `games(*, map: str | None, commit: str | None, offset: int, limit: int) -> list[dict]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_add_record_and_stats(tmp_path): ...        # 2 records on 2 maps -> stats perMap counts, seats rows with is_teacher set from the teachers set
def test_add_is_idempotent(tmp_path): ...           # same record twice -> 1 game row
def test_window_commits_last_21_days_cap_10(tmp_path): ...  # 12 commits first seen on 12 days -> newest 10; a commit only seen 30 days ago excluded
def test_cursor_roundtrip(tmp_path): ...
def test_teachers_ordered_by_elo(tmp_path): ...
```

- [ ] **Step 2: Run to verify they fail** — `cd learn && python -m pytest tests/test_index.py -q` → FAIL (no module).
- [ ] **Step 3: Implement `index.py`** (stdlib `sqlite3`, WAL mode, schema version table `meta(key, value)` with `schema=1`).
- [ ] **Step 4: Run to verify they pass** — same command → PASS.
- [ ] **Step 5: Commit** — `git add learn && git commit -m "learn: SQLite data index with version window"`

---

### Task 2: Crawler

**Files:**
- Create: `learn/ofai/data/crawl.py`, `learn/tests/test_crawl.py`
- Modify: `learn/ofai/cli.py` (`crawl` command; `fetch-sample` becomes a thin call into `Crawler` with small limits)

**Interfaces:**
- Consumes: `OpenFrontApi`, `wanted_summary`, `modifiers` (M0 Task 1); `Index` (Task 1).
- Produces:
  - `@dataclass CrawlConfig(teachers: int = 200, days: int = 21, target_games: int = 2000, max_history_pages: int = 50)`
  - `class Crawler(api: OpenFrontApi, index: Index, cfg: CrawlConfig, progress: Callable[[dict], None] = noop, stop: threading.Event | None = None)` with `run() -> dict` (final progress dict).
  - Progress dict (also written to `data/crawl_progress.json`, atomically, every 10 requests): `{"phase": "teachers"|"histories"|"records"|"done"|"stopped"|"error", "teachers": n, "historiesDone": n, "queued": n, "fetched": n, "skipped": {reason: n}, "targetGames": n, "error": str|None, "updatedAt": iso}`.

Order: (1) teachers = union of `ffa` and `ffaSeason` leaderboard pages until an empty page, top `cfg.teachers` by elo, stored with `upsert_player`; (2) for each teacher, page history from the stored cursor, stop at entries older than `now − days`; queue `wanted_summary` game IDs not in the index; (3) fetch queued records until `target_games`, drop records with `modifiers()`, `add_record(... teachers=teacher set)`; `refresh_window`.

- [ ] **Step 1: Write the failing tests** (a fake API object with canned pages; no network)

```python
def test_teachers_from_both_leaderboards_dedup_top_n(): ...
def test_history_stops_at_window_start(): ...
def test_modifier_games_are_skipped_and_counted(): ...     # skipped["modifiers"] == 1
def test_resume_skips_stored_and_continues_cursor(): ...   # run, stop event after 3 fetches, new Crawler.run() -> no refetch of the 3, continues
def test_stop_event_ends_with_phase_stopped(): ...
def test_forbidden_ends_with_phase_error(): ...            # api raises ApiForbidden -> phase "error", error message mentions 403
```

- [ ] **Step 2: Run to verify they fail** — `python -m pytest tests/test_crawl.py -q` → FAIL.
- [ ] **Step 3: Implement `crawl.py` and `python -m ofai crawl [--teachers 200] [--days 21] [--target 2000]`.**
- [ ] **Step 4: Run to verify they pass** — `python -m pytest -q` → all pass.
- [ ] **Step 5: Commit** — `git commit -am "learn: resumable teacher crawler"` (add new files first).

---

### Task 3: Batch replay-check against the index

**Files:**
- Modify: `learn/ofai/data/replay_check.py` (read targets from `Index`, write back `mark_replay`)
- Test: `learn/tests/test_replay_check.py` (extend)

**Interfaces:**
- Produces: `check_index(index: Index, *, jobs: int, only_unchecked: bool = True) -> CheckReport` (M0's `CheckReport`).

- [ ] **Step 1: Failing tests**: `test_marks_replay_ok_in_index`, `test_unknown_intent_record_is_dropped_and_counted` (fake `run_sim` returns `error`) → index `replay_ok = 0`, report `error == 1`, others still checked.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement.** `python -m ofai replay-check` now defaults to the index.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit.**

---

### Task 4: Python service

**Files:**
- Create: `learn/ofai/service/__init__.py`, `learn/ofai/service/app.py`, `learn/ofai/service/jobs.py`, `learn/tests/test_service.py`
- Modify: `learn/pyproject.toml` (add `fastapi>=0.115`, `uvicorn>=0.30`)

**Interfaces:**
- Consumes: `Index`, `Crawler`, `ensure_bundle`, `check_index`.
- Produces: `python -m ofai serve [--port 47800]` binds `127.0.0.1` only. Endpoints (JSON):

| Method, path | Returns |
|---|---|
| `GET /status` | `{"ok": true, "version", "jobs": [Job]}` |
| `GET /data/stats` | `Index.stats()` + `{"window": [commits]}` |
| `GET /data/games?map&commit&offset&limit` | `Index.games(...)` |
| `GET /data/games/{id}/record` | the record JSON (gunzipped) |
| `POST /jobs/crawl` body `CrawlConfig` fields | `Job` |
| `POST /jobs/replay-check` | `Job` |
| `POST /jobs/engine/{commit}` | `Job` (ensure_bundle) |
| `POST /jobs/{id}/stop` | `Job` |
| `GET /engines` | `[BundleMeta]` |
| `GET /events` | Server-Sent Events: `{"type": "job", "job": Job}` on every progress change |

`Job = {"id", "kind", "state": "running"|"done"|"failed"|"stopped", "progress": dict, "error": str|None, "startedAt"}`. One job per kind at a time (a second POST returns the running one). Jobs run in threads; job state is persisted to `data/jobs.json` so the app can show the last result after a restart.

- [ ] **Step 1: Failing tests** (FastAPI `TestClient`, fake Crawler/Index): `test_status_ok`, `test_crawl_job_singleton`, `test_stop_job`, `test_events_stream_emits_progress`, `test_binds_localhost_only` (the uvicorn config host is `127.0.0.1`).
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement `app.py`, `jobs.py`.**
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `"learn: local service with data, crawl and engine jobs"`.

---

### Task 5: Electron shell and process manager

**Files:**
- Create: `app/package.json`, `app/electron.vite.config.ts`, `app/tsconfig.json`, `app/src/main/index.ts`, `app/src/main/processes.ts`, `app/src/main/ports.ts`, `app/src/preload/index.ts`, `app/src/renderer/index.html`, `app/src/renderer/main.tsx`, `app/src/renderer/App.tsx`, `app/src/renderer/screens/Placeholder.tsx`
- Test: `app/tests/processes.test.ts`, `app/tests/e2e/smoke.spec.ts`

**Interfaces:**
- Produces:
  - `class ProcessManager(stateFile: string = <repo>/data/app/pids.json)`:
    - `cleanupOrphans(): Promise<string[]>` — for each recorded pid that is still alive and whose command line matches what we recorded, kill its tree (`taskkill /T /F` on Windows, process group kill elsewhere); returns what was killed.
    - `start(name: "service" | "devserver", spec: { cmd: string; args: string[]; cwd: string; env?: Record<string,string>; port: number; readyUrl: string; readyTimeoutMs: number }): Promise<void>` — refuses with `PortInUseError` if `port` is taken by a pid we didn't record; records pid; resolves when `readyUrl` answers 2xx/3xx.
    - `stop(name): Promise<void>`; `stopAll(): Promise<void>` (on app quit); `status(): Record<name, "stopped"|"starting"|"running"|"failed">`.
  - `ports.ts`: `pidOnPort(port: number): Promise<number | null>` (`netstat -ano` on Windows, `lsof` elsewhere).
  - Preload bridge `window.ofai`: `{ service: (path: string, init?: RequestInit) => Promise<any>; events: (cb) => unsubscribe; replay: { open(gameId: string): Promise<void>; status(cb): unsubscribe } }`.
  - Screens routed by a left rail: Home, Train, Data, Replays, Play, Models. Not-yet-built screens render `Placeholder` with the milestone that brings them.

- [ ] **Step 1: Load the `frontend` skill**, then write a short visual direction into `app/DESIGN.md` (palette tokens, type scale, density, how running/empty/error states look). The app is a tool Peter uses daily: dark, dense, map-first; no generic card grids.
- [ ] **Step 2: Failing tests** (`vitest`, fake spawn/ports):

```ts
test("kills stale pids from pids.json", ...)                 // recorded pid alive with matching cmdline -> killed, file cleared
test("foreign process on port is reported, not killed", ...) // pidOnPort -> 999 not recorded -> PortInUseError("port 9000 is in use by another program (pid 999)")
test("switching commits stops the old server first", ...)    // start(devserver A) then start(devserver B) -> stop(A) resolved before spawn(B)
test("start resolves only when readyUrl answers", ...)
test("window close with a running job hides to tray", ...)
```
- [ ] **Step 3: Run → FAIL** (`cd app && npx vitest run`).
- [ ] **Step 4: Implement main process, preload, renderer shell.** On `ready`: `cleanupOrphans()`, then start the service (`python -m ofai serve`, cwd `learn/`, readyUrl `http://127.0.0.1:47800/status`). Closing the window while a job runs (crawl, replay-check, engine build) hides the app to the tray with a "still crawling…" tooltip; **Quit** from the tray asks first, then `stopAll()` (spec §16: running jobs survive a window close). Training sessions are detached processes (M4) and survive even Quit.
- [ ] **Step 5: Run → PASS**, then the e2e smoke: `npx playwright test tests/e2e/smoke.spec.ts` launches the app via `_electron.launch`, sees the six rail entries and Home's "service: running".
- [ ] **Step 6: Look at it** — run `npm run dev`, screenshot each screen, check against `DESIGN.md`.
- [ ] **Step 7: Commit** — `"app: Electron shell, process manager, service lifecycle"`.

---

### Task 6: Data screen

**Files:**
- Create: `app/src/renderer/screens/Data.tsx`, `app/src/renderer/lib/useService.ts`, `app/src/renderer/lib/useJobs.ts`
- Test: `app/tests/Data.test.tsx` (vitest + Testing Library, mocked `window.ofai`)

**Interfaces:**
- Consumes: service endpoints (Task 4), preload bridge (Task 5).
- Produces: `useService<T>(path): {data, error, loading, reload}`; `useJobs(): Record<kind, Job>` (live from `/events`).

Contents: crawl progress (phase, fetched/target, skipped by reason, last error with fix hint), Start/Stop crawl, games per map (sorted bar list), games per commit with bundle status (ok / failed:<stage> / not built) and a "Build" action, teacher count, current version window, replay-check summary and "Run replay-check".

- [ ] **Step 1: Failing tests**: `renders empty state when no games` ("No games yet — start a crawl"), `shows crawl progress from events`, `shows failed bundle stage per commit`, `start button posts /jobs/crawl`.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** following `app/DESIGN.md`.
- [ ] **Step 4: Run → PASS**; screenshot empty, running and error states.
- [ ] **Step 5: Commit** — `"app: Data screen"`.

---

### Task 7: Replays screen and the in-app viewer

**Files:**
- Create: `app/src/main/replay.ts`, `app/src/renderer/screens/Replays.tsx`, `app/src/renderer/screens/ReplayView.tsx`
- Test: `app/tests/replay.test.ts`, `app/tests/Replays.test.tsx`

**Interfaces:**
- Consumes: `ProcessManager` (Task 5); `GET /data/games`, `GET /data/games/{id}/record`, `POST /jobs/engine/{commit}`, `GET /engines`.
- Produces:
  - `openReplay(gameId: string, view: WebContentsView, deps): AsyncIterable<ReplayStatus>` where `ReplayStatus = {stage: "bundle"|"devserver"|"loading"|"playing"|"failed", detail?: string, hint?: string}`.
  - `joinLobbyScript(record: object): string` — returns the JS that dispatches `new CustomEvent("join-lobby", {detail: {gameID: record.info.gameID, gameRecord: record, source: "private"}})` (v2 `watch_replay.py` contract; re-verify the event name and detail shape against the pinned commit's `src/client/Main.ts` and note it in the code).

Flow: ensure bundle for the record's commit → `ProcessManager.start("devserver", {cmd: "npm", args: ["run", "dev"], cwd: engines/<commit>/src, env: {GIT_COMMIT: "DEV"}, port: 9000, readyUrl: "http://localhost:9000", readyTimeoutMs: 240_000})` unless the running dev server already serves that commit → load `http://localhost:9000` in a `WebContentsView` → wait for `did-finish-load` → `executeJavaScript(joinLobbyScript(record))`.

- [ ] **Step 1: Failing tests**: `joinLobbyScript embeds the record and gameID`, `reuses the running dev server for the same commit`, `failed bundle shows stage` (status stream ends with `{stage: "failed", detail: "bundle", hint: ...}`), `library filters by map and commit`.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** the library (map, players, duration, teacher names, commit, replay-check badge) and the view (full-bleed client, thin top bar with back, game info, stage progress while starting; first start of a commit shows "starting the game client for <short commit> (first time can take a few minutes)").
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Watch one by hand**: open a crawled game in the app, confirm it plays with the client's own speed controls, and that the client console shows no desync. Screenshot it.
- [ ] **Step 6: Commit** — `"app: replay library and in-app viewer at the record's commit"`.

---

### Task 8: Engine update action

**Files:**
- Create: `learn/ofai/engine_update.py`, `learn/tests/test_engine_update.py`
- Modify: `learn/ofai/service/app.py` (`POST /jobs/engine-update`), `app/src/renderer/screens/Data.tsx` ("Update engine" button with the job's steps)

**Interfaces:**
- Produces: `update_engine(index: Index, *, run: Runner) -> dict` — target = the most common commit among in-window games of the last 3 days; steps `fetch → checkout submodule → npm ci → sim tests (npx vitest run) → ensure_bundle → replay 5 newest records of that commit`; result `{"from", "to", "steps": [{name, ok, detail}]}`. On any failed step the submodule is checked back out to `from`. Never commits; the app tells Peter to commit the bump (`git add engine && git commit`).

- [ ] **Step 1: Failing tests**: `already on target is a no-op`, `failed sim tests roll back to previous commit`, `success reports all steps ok`.
- [ ] **Step 2: Run → FAIL.** **Step 3: Implement.** **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `"engine update action with rollback"`.

---

### Task 9: M1 exit check

- [ ] **Step 1: Crawl**: in the app, Data → Start crawl (target 2,000). Leave it (≈ 1 h + history pages).
- [ ] **Step 2: Replay-check** all in-window games from the app.
- [ ] **Step 3: Verify**: Data shows ≥ 2,000 games with teacher seats; replay-check pass rate per commit; any crawled game opens and plays in Replays.
- [ ] **Step 4: Record** in `docs/measurements.md`: unique games per window and teacher threshold (spec §8.1 volume check), crawl duration, records on disk, replay-check pass rate, dev-server first start time per commit.
- [ ] **Step 5: PR** — push `m1-data-app-replays`, `gh pr create` with problem/solution/tradeoff/verification and screenshots.
