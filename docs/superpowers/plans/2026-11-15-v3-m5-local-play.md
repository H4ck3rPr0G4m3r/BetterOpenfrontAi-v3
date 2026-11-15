# M5: headless client and local play — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Before starting, apply the roadmap's refresh rule.**

**Goal:** from the app, Peter hosts a local lobby, adds several AIs (a checkpoint each), and plays them in the real game client, or spectates an all-AI game.

**Architecture:** the app runs the engine's dev server at the pinned commit (no captcha in Dev). A Node `HeadlessClient` process joins that server over the game's own WebSocket protocol with N AI seats, keeps the game state with the matching engine bundle, encodes once per decision step for all its seats (shared `GameObsCache`), asks the Python inference service for actions in one batched call, and sends intents back. Transport is an interface; this repo ships only the local implementation.

**Tech Stack:** Node `ws`, the M2 encoder/codec, Python inference service (bridge framing), React.

**Spec:** §11 (inference), §14.5 (real metric), §15.1–15.2, §16 (Play), §17, §18 (M5). Roadmap §3.4 (`ACT`/`ACTIONS`).

## Global Constraints

All of roadmap §2. Most relevant: AI joins with clan tag `AI`; only the local transport lives here; a decision still running when the next step is due skips that step (never queued); starting a local game pauses or shrinks the running session.

## Review Focus

1. **Inference slower than 0.5 s** (session hogging the GPU): steps are skipped and counted, the AI never sends a burst of stale intents (Task 2 test `slow inference skips steps`).
2. **Dev server restarts mid-game**: clients report "disconnected", the app shows it, no orphan Node process remains (Task 2 test `server close ends client cleanly`).
3. **Checkpoint layout version differs from the bundle's layout** (old checkpoint): refused at lobby setup with a readable error (Task 1 test `layout mismatch refused`).
4. **Peter leaves the lobby before start**: AI clients leave too and the session resumes (Task 4 test `abandoned lobby cleans up`).
5. **Game ends**: record + mind sidecar saved to `replays/local/`, a line appended to `data/games_vs_people.jsonl` (Task 3 test `game end writes record, sidecar, log`).

---

### Task 1: Inference service

**Files:** Create `learn/ofai/service/inference.py`, `learn/tests/test_inference.py`; Modify `learn/ofai/bridge/protocol.py`, `sim/src/bridge/Protocol.ts` (ACT/ACTIONS).

**Interfaces:**
- Produces: `InferenceServer(port: int = 47801, max_models: int = 4, device="cuda")` — bridge-framed TCP; on `ACT`, groups rows by checkpoint, loads checkpoints on demand (LRU on GPU), keeps LSTM state per `seatKey` (reset when `fresh`), runs `Policy.core` + `sample` (temperature from the seat's settings, default 1.0), replies `ACTIONS` with actions and mind lines (M3 Task 9 format). Started by the service on demand (`POST /inference/start`).
- [ ] Failing tests: `act returns K x A per row`, `state persists per seat and resets on fresh`, `two checkpoints in one call`, `layout mismatch refused`, `p99 latency under 100 ms for 8 rows on cuda` (skipped without CUDA; recorded in measurements).
- [ ] Run → FAIL. Implement. Run → PASS. Commit `"learn: batched inference service"`.

### Task 2: Headless client core

**Files:** Create `sim/src/client/HeadlessClient.ts`, `sim/src/client/Transport.ts`, `sim/src/client/LocalWsTransport.ts`, `sim/src/client/InferenceClient.ts`, `sim/tests/client/HeadlessClient.test.ts`; Modify `sim/src/cli.ts` (`client` command), `sim/src/engine/Adapter.ts` (live game construction from the server's `start` message).

**Interfaces:**
- Produces:
  - `interface ClientTransport { connect(): Promise<void>; send(msg: ClientMessage): void; onMessage(cb: (msg: ServerMessage) => void): void; onClose(cb: (reason: string) => void): void; close(): void }` — the seam the private M7 repo implements.
  - `LocalWsTransport(url: string)` — refuses any host other than `localhost`/`127.0.0.1` (throws `"local transport only"`).
  - `class HeadlessClient(opts: { transports: ClientTransport[] /* one per seat */; seats: { username: string; checkpoint: string; temperature: number }[]; engineRoot: string; inference: InferenceClient; gameID: string })` with `run(): Promise<ClientGameResult>`. Each seat joins with `clanTag: "AI"` and a dev token built the way the pinned commit's dev client builds it (read `src/client` join code at that commit). The seats share one engine instance and one `GameObsCache`: turns from seat 0's connection drive the engine; the others only send intents and hashes.
  - Every `DECISION_TURNS` turns: encode all live seats → one `ACT` → decode → send `intent` messages; if the previous `ACT` hasn't returned, skip this step (`skipped` counter). Hash messages are sent like the browser client so the server sees no desync.
  - `ClientGameResult = { record: GameRecordJson; mind: MindLine[]; placements: Record<clientID, number>; skipped: number; dropped: DecodeStats }`.
  - `node sim.mjs client --server ws://localhost:3000 --game <id> --seats <json> --inference 127.0.0.1:47801`.
- [ ] Failing tests (fake transport + fake inference): `joins with AI clan tag`, `one ACT per decision step for all seats`, `slow inference skips steps`, `server close ends client cleanly`, `non-local url refused`; integration (marked slow, real dev server at the pin): `two AI seats play 3 minutes without desync`.
- [ ] Run → FAIL. Implement. Run → PASS. Commit `"sim: headless client with local transport"`.

### Task 3: Local lobby control

**Files:** Create `app/src/main/localGame.ts`, `app/tests/localGame.test.ts`; Modify `learn/ofai/service/app.py` (`POST /games/vs-people` to append to `data/games_vs_people.jsonl`).

**Interfaces:**
- Produces: `hostLocalGame(cfg: { map; nations; bots; difficulty; ais: { name; checkpoint; temperature }[]; peterPlays: boolean }): AsyncIterable<LocalGameStatus>` — ensures dev server (pinned commit) and inference service; creates a private lobby through the dev server's `POST /api/create_game` (check route and body at the pin); starts the `HeadlessClient`; opens the embedded client at the lobby join URL (or as spectator); pauses or shrinks the session (M4 `CONTROL`, default: shrink to 4 workers); on game end saves `replays/local/<gameId>.json.gz` + `.mind.jsonl.gz`, posts a `games_vs_people` line `{gameId, date, map, ais: [{checkpoint, placement}], humans: [{name, placement}]}`, restores the session.
- [ ] Failing tests: `session shrinks during game and restores after`, `game end writes record, sidecar, log`, `abandoned lobby cleans up`.
- [ ] Run → FAIL. Implement. Run → PASS. Commit `"app: local lobby control"`.

### Task 4: Play screen (Local)

**Files:** Create `app/src/renderer/screens/Play.tsx`, `app/tests/Play.test.tsx`.

Contents: a **Games vs people** chart at the top (spec §14.5: every logged game's AI placement over time, with the humans' names and ratings where known, from `data/games_vs_people.jsonl`; load the `dataviz` skill); host form (map with thumbnail, nations, bots, difficulty), AI seat list (name, checkpoint from Models, temperature), **Play** (embedded client joined) and **Spectate**; while running: status, skipped-step counter per AI, a stop button; after: placements and "Watch replay". The **Online** tab appears only when the private package is present (M7); otherwise it isn't shown at all.
- [ ] Load the `frontend` skill. Failing tests: `host posts config`, `online tab hidden without plugin`, `shows skipped steps`, `games vs people chart reads the log`. Run → FAIL. Implement. Run → PASS. Screenshot. Commit `"app: Play screen (local)"`.

### Task 5: M5 exit check

- [ ] Peter hosts a World lobby with 4 AIs (two checkpoints), plays it to the end from the app; then spectates an all-AI game on a portrait map.
- [ ] Record skipped steps, inference latency and any desync in `docs/measurements.md`; the game is in `games_vs_people.jsonl`.
- [ ] PR `m5-local-play` with screenshots.
