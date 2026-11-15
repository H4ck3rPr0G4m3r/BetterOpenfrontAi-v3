# M8: scale to 20k games and a cloud recipe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Before starting, apply the roadmap's refresh rule** — this plan's numbers depend most on M0–M3 measurements.

**Goal:** a 20k-game dataset and a one-command Linux setup, so a rented GPU trains a bigger model that still runs live on the 3060 Ti.

**Architecture:** the crawler widens either the version window or the teacher threshold (chosen from measured numbers); a headless Linux recipe (`scripts/setup-linux.sh`) installs Node, Python, CUDA PyTorch, builds the window's bundles, and runs `python -m ofai bc|rl` without the app; data and checkpoints move with `scripts/sync.py`. A latency check gates any cloud model for live play.

**Spec:** §4 (cloud), §8.1 (volume), §11 (cloud model size), §18 (M8).

## Global Constraints

All of roadmap §2. Most relevant: any model must run live on the 3060 Ti at 2 decisions/s per seat; crawl ≤ 1 request/s; checkpoints never published; training code runs headless on Linux + CUDA.

## Review Focus

1. **Cloud instance dies mid-run**: checkpoints were synced home at every checkpoint, so at most 30 min is lost (Task 3 test `sync after checkpoint`).
2. **Windows path assumptions** in `learn/` or `sim/` break on Linux: the full test suites run in CI on `ubuntu-latest` (Task 2).
3. **Wider window adds commits whose bundles fail**: counted and excluded as in M0, never blocking the crawl (covered by M0/M1 tests; re-verified in Task 1 Step 4).

---

### Task 1: Crawl to 20k

**Files:** Modify `learn/ofai/data/crawl.py` (`CrawlConfig.min_elo: float | None` — any seat whose `ffaRating` in the history entry is ≥ `min_elo` becomes a teacher, its own history is walked), `learn/tests/test_crawl.py`.
- [ ] Failing test `rating cutoff adds teachers from histories`. Run → FAIL. Implement. Run → PASS.
- [ ] Decide window vs threshold from `docs/measurements.md` (adapter churn per commit from M0/M1 replay-check; unique games per threshold from the M1 volume report); write the choice and numbers there.
- [ ] Crawl in the background during sessions until 20k games; replay-check; record disk use. Commit `"crawl: rating cutoff teachers; 20k dataset"`.

### Task 2: Linux CI and setup script

**Files:** Create `.github/workflows/ci.yml` (ubuntu: `sim` vitest, `learn` pytest on CPU), `scripts/setup-linux.sh`, `docs/cloud.md`.
- [ ] CI green on Linux. `setup-linux.sh` on a fresh Ubuntu 22.04 CUDA image: installs Node 24, Python 3.11 venv, torch cu128, `npm ci` in `engine/` and `sim/`, builds bundles for the window, runs `pytest -q`. `docs/cloud.md`: Vast.ai and RunPod steps (pick a single 4090-class GPU for BC; many vCPUs for RL), prices to check at rent time, expected hours. Commit `"Linux CI and cloud setup"`.

### Task 3: Sync and remote sessions

**Files:** Create `scripts/sync.py`, `learn/tests/test_sync.py`; Modify `learn/ofai/bc/train.py`, `learn/ofai/rl/train.py` (`--sync-cmd` run after each checkpoint).
- [ ] `sync.py up|down --host <ssh> --what records|checkpoints|runs` (rsync over ssh; checkpoints down after every save). Failing test `sync after checkpoint` (fake command recorded). Implement. Commit `"sync data and checkpoints with a remote box"`.

### Task 4: Bigger model and the live latency gate

**Files:** Create `learn/configs/model-large.yaml` (30–60M params), `learn/ofai/latency.py`; Modify `learn/ofai/cli.py` (`latency`).
- [ ] `python -m ofai latency <checkpoint> --seats 8` on the 3060 Ti: p99 of one batched decision for 8 seats < 250 ms (half of the 0.5 s step), measured with a session running at 4 workers. Record. Commit `"large model config and latency gate"`.

### Task 5: M8 exit check

- [ ] One rented-GPU BC run on the 20k set with `model-large`; checkpoint synced home; latency gate passes; bench-v1 beats the local BC model; one local game (M5) with it.
- [ ] Record costs (hours × price) and results in `docs/measurements.md`; PR `m8-scale-cloud`.
