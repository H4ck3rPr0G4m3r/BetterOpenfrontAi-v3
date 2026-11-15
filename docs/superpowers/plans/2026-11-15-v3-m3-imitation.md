# M3: imitation, `bench-v1`, AI mind overlay — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a recurrent policy cloned from top FFA players, measured on a frozen benchmark, whose games can be watched with its "mind" beside the replay.

**Architecture:** replay workers (M2 `SimPool`, role `replay`) stream teacher samples into a lane-based sequence buffer; the GPU expands compact rasters into planes, runs the ~10–15M-parameter encoder → fusion transformer → LSTM → autoregressive heads, and trains with weighted cross-entropy plus value regression. `bench-v1` runs policy seats in `env`-role games on fixed maps and seeds. AI games write the engine record plus a per-step "mind" sidecar the app shows next to the replay.

**Tech Stack:** PyTorch 2.8 (CUDA 12.8, bf16 autocast on Ampere), numpy, pytest; app: React (M1 shell).

**Spec:** §8.2–8.3, §11, §12, §14, §16 (Replays), §18 (M3). Roadmap §3.4–3.8.

## Global Constraints

All of roadmap §2. Most relevant: LSTM 512; 10–15M parameters; plain PyTorch checkpoints written atomically; `bench-v1` never edited after freezing; split by game, never by step; held-out set frozen at milestone start; checkpoints never committed.

## Review Focus

1. **Power loss mid-checkpoint**: `latest_checkpoint` never returns a truncated file (Task 5 test `truncated checkpoint is skipped`).
2. **A sim worker dies mid-session**: the trainer restarts that worker, its lanes restart FRESH, training continues; three deaths in 10 minutes stop the run with the stderr tail in `events.jsonl` (Task 6 test `worker death restarts lanes`).
3. **All lanes of a batch invalid** (queue drained at the end of the dataset): no NaN loss; the step is skipped (Task 3 test `all-invalid batch gives zero loss and no grad`).
4. **A bench game hits a sim error**: recorded as `sim_error` with the seed, excluded from rates, and the suite still completes (Task 8 test `sim error is recorded and excluded`).
5. **Opening a replay that has no mind sidecar** (human games): the overlay panel shows "no AI in this game" and the replay works normally (Task 9 test `no sidecar hides overlay`).

---

### Task 1: GPU plane expansion

**Files:**
- Create: `learn/ofai/model/__init__.py`, `learn/ofai/model/planes.py`, `learn/tests/test_planes.py`
- Modify: `sim/src/cli.ts` (`dump-obs <record> --steps 3 --out <dir>`: writes the raw `ObsBuffer` bytes plus `expandPlanesReference` output and a JSON header)

**Interfaces:**
- Consumes: `Layout` (M2 Task 10) including `PLANES`.
- Produces: `expand_minimap(obs: dict[str, Tensor], layout: Layout) -> Tensor  # [B, P, 128, 128] float` and `expand_windows(obs, layout) -> Tensor  # [B, 8, P, 32, 32]`, both pure tensor ops (gather `relation[owner]`, `density[owner]`, scatter units, SAM disks by a precomputed kernel).

- [ ] **Step 1: Failing test**: `test_matches_ts_reference` loads the `dump-obs` fixture produced in a session-scoped pytest fixture (runs `npx tsx sim/src/cli.ts dump-obs` on a synthetic record) and asserts `expand_minimap(...)` equals the TS reference exactly for every plane (and the same for windows).
- [ ] **Step 2: Run → FAIL.** **Step 3: Implement.** **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `"learn: GPU plane expansion matching the TS reference"`.

---

### Task 2: Policy network

**Files:**
- Create: `learn/ofai/model/encoders.py`, `learn/ofai/model/heads.py`, `learn/ofai/model/policy.py`, `learn/ofai/model/config.py`, `learn/tests/test_policy.py`

**Interfaces:**
- Consumes: `expand_minimap`, `expand_windows` (Task 1); `Layout`.
- Produces:
  - `@dataclass ModelConfig(width: int = 256, lstm: int = 512, fusion_layers: int = 3, player_layers: int = 3, heads: int = 8, cnn_channels: tuple = (64, 96, 128), win_channels: tuple = (48, 96))`.
  - `class Policy(nn.Module)(layout: Layout, cfg: ModelConfig)`:
    - `initial_state(batch: int) -> tuple[Tensor, Tensor]`
    - `core(obs: dict, state, fresh: Tensor[B]) -> tuple[Tensor[B, lstm], CoreCtx, state]` — encoders (spec §11): minimap residual CNN → feature map (kept in `CoreCtx` for the far pointer) + summary token; shared window CNN → per-window token + feature maps, with position/scale embeddings; player transformer whose attention logits get a learned bias from `edges`; MLP tokens for structs/mobile/attacks; globals MLP; fusion transformer over all tokens; LSTM (state zeroed where `fresh`).
    - `log_prob(core, ctx, obs, actions: Tensor[B, K, A]) -> tuple[Tensor[B], Tensor[B], dict]` (sum of log-probs of the fields each action type uses, entropy, per-head logits for metrics). Autoregressive: action k's query = core + embedding of actions 0..k−1 of the step; within an action: type → (player | entity | build) → troops → locMode → (window → cell) | (mmCell → fine).
    - `sample(core, ctx, obs, temperature: float = 1.0, generator=None) -> tuple[Tensor[B, K, A], Tensor[B], dict]` — Gumbel-max over masked logits (v2's NaN-proof sampling), stops each row at STOP; returns actions, log-prob, and `top` (top-5 per head for the mind sidecar).
    - `value(core) -> Tensor[B]` (sigmoid, predicts the placement value).
  - Masks: `type_mask`, `build_mask`; pointer masks from token/entity validity; window-cell and minimap-cell masks from the expanded planes (own land for BUILD, water/coast for BOAT and MOVE_WARSHIP, land for nukes and SPAWN, unowned land for SPAWN). Heads and value run in fp32; encoders under bf16 autocast.

- [ ] **Step 1: Failing tests**: `param count between 10M and 15M` (default config), `masked fields never sampled` (10k samples with a mask allowing one type → only that type), `log_prob of sampled actions equals returned log_prob`, `fields unused by a type do not enter log_prob`, `fresh resets lstm state`, `no NaN with all-but-STOP masked`, `cpu and cuda give the same log_prob within 1e-4` (skipped without CUDA).
- [ ] **Step 2: Run → FAIL.** **Step 3: Implement** (port pointer heads and Gumbel sampling from v2 `train/openfront/model.py`). **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `"learn: recurrent policy with map-agnostic encoders and autoregressive heads"`.

---

### Task 3: BC loss and sample weights

**Files:**
- Create: `learn/ofai/bc/__init__.py`, `learn/ofai/bc/loss.py`, `learn/ofai/bc/weights.py`, `learn/tests/test_bc_loss.py`

**Interfaces:**
- Consumes: `Policy.log_prob`, `Policy.value`.
- Produces:
  - `teacher_weight(elo: float, value: float, elo_p50: float, elo_p90: float) -> float` = `(0.5 + 0.5 * clip((elo - elo_p50) / (elo_p90 - elo_p50), 0, 1)) * (0.5 + 0.5 * value)` (spec §8.3: rating and outcome).
  - `bc_loss(policy, core, ctx, obs, labels, weight, value_target, valid, stop_weight: float = 0.3, value_coef: float = 0.5) -> tuple[Tensor, dict]` — per-row `−log_prob(labels)`; rows whose step is only STOP get `stop_weight`; mean over valid rows weighted by `weight`; plus `value_coef × MSE(value, value_target)`. Metrics: per-head top-1/top-5 accuracy, STOP precision/recall, location error in tiles (decoded from argmax via the same geometry formulas as the sim, implemented in `ofai.model.geometry`), value MAE.

- [ ] **Step 1: Failing tests**: `weights are in [0.25, 1]`, `stop-only rows are down-weighted`, `all-invalid batch gives zero loss and no grad`, `perfect logits give near-zero loss`.
- [ ] **Step 2: Run → FAIL.** **Step 3: Implement.** **Step 4: Run → PASS.** **Step 5: Commit** — `"learn: weighted BC loss and metrics"`.

---

### Task 4: Teacher stream and frozen split

**Files:**
- Create: `learn/ofai/bc/split.py`, `learn/ofai/bc/stream.py`, `learn/tests/test_stream.py`

**Interfaces:**
- Consumes: `Index` (M1), `SimPool` (M2), `teacher_weight`.
- Produces:
  - `make_split(index: Index, name: str = "bc-v1", heldout: float = 0.05) -> Path` — writes `data/splits/<name>.json` `{"created", "train": [gameIds], "heldout": [gameIds]}` from replay-ok, in-window games, by a stable hash of the game id; refuses to overwrite an existing split.
  - `class TeacherStream(index, split: str, part: "train"|"heldout", pools: dict[str, SimPool], lanes: int, seq_len: int = 64, seed: int = 0)` — keeps every pool's queue filled (`LOAD` with per-teacher weight and value), groups pools by commit (one `SimPool` per commit in the window, workers split by game share), and yields `SeqBatch` = dict of tensors `[T, lanes, ...]` + `fresh [T, lanes]` + `valid [T, lanes]`. Epoch = one pass over the part's games in a shuffled order.

- [ ] **Step 1: Failing tests** (fake pool returning scripted batches): `split is stable and disjoint`, `split refuses overwrite`, `sequences keep lane order across chunks`, `fresh marks reset points`, `queue is refilled before it drains`.
- [ ] **Step 2: Run → FAIL.** **Step 3: Implement.** **Step 4: Run → PASS.** **Step 5: Commit** — `"learn: frozen split and streamed teacher sequences"`.

---

### Task 5: Checkpoints

**Files:**
- Create: `learn/ofai/checkpoints.py`, `learn/tests/test_checkpoints.py`

**Interfaces:**
- Produces: `save_checkpoint(path: Path, payload: dict) -> None` (write `path.tmp`, fsync, `os.replace`, then write `path.sha256`); `load_checkpoint(path) -> dict` (verifies sha256); `latest_checkpoint(run_dir: Path) -> Path | None` (newest whose sha256 verifies); `register(name, path, phase, step, extra) -> None` (updates `checkpoints/index.json` atomically).

- [ ] **Step 1: Failing tests**: `roundtrip`, `truncated checkpoint is skipped` (truncate newest → latest returns the previous), `missing sha is skipped`, `register is atomic` (simulate crash between write and replace → old index intact).
- [ ] **Step 2: Run → FAIL.** **Step 3: Implement.** **Step 4: Run → PASS.** **Step 5: Commit** — `"learn: atomic checkpoints"`.

---

### Task 6: BC trainer

**Files:**
- Create: `learn/ofai/bc/train.py`, `learn/ofai/runlog.py`, `learn/configs/bc.yaml`, `learn/tests/test_bc_train.py`
- Modify: `learn/ofai/cli.py` (`bc` command)

**Interfaces:**
- Consumes: Tasks 2–5.
- Produces:
  - `train_bc(cfg: dict, run_dir: Path, *, until: datetime | None = None, stop: threading.Event | None = None) -> dict` — resumes from `latest_checkpoint(run_dir)` if present (optimizer, scheduler, stream position, RNG); AdamW lr 3e-4, warmup 1k steps, cosine to 10 % over `total_samples`; grad clip 1.0; bf16 autocast; truncated BPTT over `seq_len`; eval on held-out every `eval_every` samples; checkpoint every 30 min and at stop; stops at `until`, on `stop`, or when `run_dir/STOP` exists.
  - `RunLog(run_dir)`: `metric(**kv)` → `metrics.jsonl`; `event(kind, **kv)` → `events.jsonl` (roadmap §3.8).
  - `python -m ofai bc --config configs/bc.yaml [--until 15:30] [--resume <run_dir>]`.
  - `bc.yaml` defaults: `lanes: 64`, `seq_len: 64`, `workers: 14`, `eval_every: 2_000_000`, `total_samples: 40_000_000`, `model: ModelConfig defaults`.

- [ ] **Step 1: Failing tests** (tiny model, fake stream): `loss decreases on a fixed batch`, `resume continues step count and lr`, `STOP file stops and checkpoints`, `until stops at time`, `worker death restarts lanes` (fake pool raises `SimWorkerDied` once → lanes fresh, run continues; three times in 10 min → run stops with an `error` event).
- [ ] **Step 2: Run → FAIL.** **Step 3: Implement.** **Step 4: Run → PASS.**
- [ ] **Step 5: Throughput measurement**: a 20-minute real run; record samples/s, GPU utilisation (`nvidia-smi`), GPU memory, CPU load, and whether the GPU waits on the stream, in `docs/measurements.md`. If the model can't reach ~1,000 samples/s, shrink `ModelConfig` (record the new param count) rather than shorten sequences.
- [ ] **Step 6: Commit** — `"learn: resumable BC trainer"`.

---

### Task 7: Hot-subset cache (only if Task 6 shows the GPU starving)

Skip this task, and note it in `docs/measurements.md`, if the GPU waits on the stream less than 10 % of the time.

**Files:**
- Create: `learn/ofai/bc/cache.py`, `learn/tests/test_cache.py`

**Interfaces:**
- Produces: `ShardWriter(dir, max_gb: float = 60)` writes `SeqBatch`es as compressed `.npz` shards; `ShardReader(dir, lanes, seq_len)` yields `SeqBatch`es; `TeacherStream(mix_cached: float)` mixes cached and live sequences.

- [ ] Steps: failing tests (`shard roundtrip`, `cap respected`, `mix ratio`), run → FAIL, implement, run → PASS, commit `"learn: optional hot-subset cache"`.

---

### Task 8: `bench-v1`

**Files:**
- Create: `learn/ofai/bench/__init__.py`, `learn/ofai/bench/suites.py`, `learn/ofai/bench/runner.py`, `learn/ofai/bench/agent.py`, `learn/tests/test_bench.py`
- Create: `sim/src/report/DeathReport.ts` (port from v2), `sim/src/cli.ts` (`land-curves <records>`)
- Create: `data/bench/bench-v1.json` is generated, not committed; its content hash is committed in `docs/bench-v1.md`

**Interfaces:**
- Consumes: `SimPool` role `env` (M2), `Policy.sample`, `RecordWriter` (M2), checkpoints.
- Produces:
  - `freeze_bench_v1(index: Index, proxy_checkpoint: Path) -> Path` — writes `data/bench/bench-v1.json`: 12 maps (World, the smallest, the largest, a portrait map, and 8 more by rotation weight from the pinned commit's `MapPlaylist`), 5 fixed seeds per map, bots and nation counts = medians of in-window public FFA configs per map, `commit` = the pinned engine commit, and copies the proxy checkpoint to `checkpoints/bench-v1-proxy.pt` with its sha256. Refuses if `bench-v1.json` exists.
  - Suites: `solo_medium` (1 AI seat, Medium nations; metric: survived 20 min), `solo_hard` (Hard nations; metrics: placement, survival at 20/30 min, land share curve), `human_proxy` (1 AI seat + 20 seats of the frozen proxy at temperature 1.0; placement).
  - `PolicyAgent(checkpoint, temperature=1.0, device)` — batched `core` + `sample` for any number of seats, keeps LSTM state per seat.
  - `run_bench(checkpoint: Path, suites: list[str] = ["solo_medium", "solo_hard", "human_proxy"], *, workers: int, replays: bool = True) -> dict` — writes `runs/<id>-bench/results.json` `{"bench": "bench-v1", "checkpoint", "suites": {name: {"games", "simErrors", <metrics>}}}`, death reports for every AI death, replays + mind sidecars to `replays/bench/`; updates `checkpoints/index.json`.
  - Teacher land curves: `sim.mjs land-curves <records> [--client <id>]` emits one seat's land share per minute with `--client`, otherwise per-map median teacher land share per minute; `run_bench` adds `landCurveGap` (mean absolute gap to the teacher curve over minutes 1–20) to `solo_hard`.

- [ ] **Step 1: Failing tests**: `bench file is frozen` (second freeze raises), `seeds and maps are fixed`, `sim error is recorded and excluded`, `results aggregate rates`, `ai seats carry the tag`.
- [ ] **Step 2: Run → FAIL.** **Step 3: Implement.** **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `"learn: bench-v1 suites and runner with death reports"`.

---

### Task 9: AI mind sidecar and overlay

**Files:**
- Create: `learn/ofai/bench/mind.py`, `learn/tests/test_mind.py`
- Create: `app/src/renderer/screens/MindPanel.tsx`, `app/src/main/tickProbe.ts`, `app/tests/MindPanel.test.tsx`
- Modify: `app/src/renderer/screens/ReplayView.tsx`, `app/src/renderer/screens/Replays.tsx` (library includes `replays/bench`, `replays/local`, `replays/selfplay`)

**Interfaces:**
- Consumes: `Policy.sample(... )["top"]`, `Policy.value`, `EntityTables` via env stats.
- Produces:
  - Sidecar line (one per AI seat per decision step): `{"turn", "clientID", "value", "actions": [{"text": "Attack Bob 35%", "tile": [x, y] | null, "p": 0.41}], "top": [{"head": "type", "options": [["Attack", 0.41], ["Build", 0.22], ...]}]}`; `MindWriter(path).write(...)`, gzip JSONL.
  - Overlay (spec §16): a right-hand panel beside the embedded client — value curve over the game with a playhead, the current step's top-k per head and the chosen actions in words, and a map thumbnail (`resources/maps/<map>/thumbnail.webp`) with the chosen actions' tiles marked. Synced by `tickProbe`: an injected script that wraps the page's game `Worker` `onmessage` to read each game update's `tick` and posts it to the app. If the client changes and the probe fails, the panel says "can't follow this client version" and the replay still plays.
  - **Compare** (spec §16): from an AI game, pick a teacher game on the same map; the panel overlays both players' land-share curves (teacher curve from `sim.mjs land-curves <record> --client <id>`) and the two replays open one after the other.
  - Drawing on the client's own canvas is out of scope for M3 (needs the client's camera transform); revisit if the probe exposes it cheaply.

- [ ] **Step 1: Load the `frontend` skill** (panel design follows `app/DESIGN.md`).
- [ ] **Step 2: Failing tests**: Python `sidecar lines carry value and top-k`; app `no sidecar hides overlay`, `playhead follows probe tick`, `probe failure shows notice`, `compare overlays teacher curve`.
- [ ] **Step 3: Run → FAIL.** **Step 4: Implement.** **Step 5: Run → PASS.**
- [ ] **Step 6: Look at it**: open a bench replay, screenshot the panel during an attack and a build.
- [ ] **Step 7: Commit** — `"AI mind sidecar and replay overlay"`.

---

### Task 10: Train, freeze, measure (M3 exit check)

- [ ] **Step 1**: `python -m ofai crawl` top-up and `replay-check` so the split has every replay-ok in-window game; `make_split` (`bc-v1`).
- [ ] **Step 2**: train in 8 h sessions (`python -m ofai bc --until <time>`) until held-out loss flattens for two consecutive evals.
- [ ] **Step 3**: `freeze_bench_v1(index, <best BC checkpoint>)`; commit `docs/bench-v1.md` (maps, seeds, counts, file sha256, proxy sha256).
- [ ] **Step 4**: `python -m ofai bench <best checkpoint>`.
- [ ] **Step 5: Exit check**: `solo_medium` survival at 20 min ≥ 70 %; `solo_hard` top-3 ≥ 50 %; watch at least 5 bench replays with the overlay and write one paragraph per replay in `docs/measurements.md` on whether it plays like a person (spawning, expansion, defence, boats, diplomacy). If a bar fails, diagnose with superpowers:systematic-debugging (translator gaps, weights, model size, data volume) before M4.
- [ ] **Step 6: PR** — `m3-imitation`, with bench results, held-out metrics, throughput, and screenshots.
