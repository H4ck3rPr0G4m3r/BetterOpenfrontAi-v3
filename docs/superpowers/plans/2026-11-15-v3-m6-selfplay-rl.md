# M6: self-play RL with a league — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Before starting, apply the roadmap's refresh rule.**

**Goal:** improve on the BC policy through self-play in public-FFA-like games, without drifting away from human-like play.

**Architecture:** PPO with recurrent state over `env`-role `SimPool` games. Each game mixes learner seats, frozen league snapshots and BC "human stand-ins" with public-lobby bots and nations. Loss = PPO clip + value + small entropy + a KL penalty toward the frozen BC policy, annealed. Reward = final placement + win bonus (+ at most one potential-based land-share term, decided by an A/B). A multi-player rating ladder (openskill, Plackett–Luce) ranks every checkpoint.

**Tech Stack:** PyTorch, `openskill`, the M2 env, the M4 sessions.

**Spec:** §13, §14 (ladder), §18 (M6). Roadmap §2 (reward).

## Global Constraints

All of roadmap §2. Most relevant: reward is placement + win bonus + at most one potential-based land-share term; a new term needs a benchmark A/B; `bench-v1` unchanged; self-play latency 1–3 turns; sessions use all cores but 2.

## Review Focus

1. **A game with no learner seat left alive**: aborted and replaced, its rows marked `aborted`, no wasted simulation (Task 2 test `learner-less game aborted`).
2. **NaN or inf in advantages or KL** (empty masks, extreme logits): the update is skipped and logged, never written into weights (Task 2 test `nan update skipped`).
3. **League snapshot file deleted or corrupt**: that opponent is removed from the league and logged; games continue (Task 3 test `missing snapshot dropped`).
4. **Imitation collapse**: held-out top-1 type accuracy below 80 % of BC's stops the session with an `error` event (Task 4 test `imitation guard stops run`).
5. **Resume mid-rollout**: resuming starts fresh rollouts (no half rollouts mixed with a new policy) (Task 4 test `resume discards partial rollout`).

---

### Task 1: Reward in the env

**Files:** Modify `sim/src/env/GameEnv.ts`; Create `sim/src/env/Reward.ts`, `sim/tests/env/Reward.test.ts`.

**Interfaces:**
- Produces: `GameSpec.reward = { winBonus: number /* 1.0 */; landShare: number /* coefficient, 0 = off */; gamma: number /* 0.999 */ }`; per step `r = landShare × (γ·Φ(s′) − Φ(s))` with `Φ = own tiles ÷ land tiles`; at a seat's terminal step add placement value `1 − (place − 1)/(n − 1)` (humans + nations, bots excluded; M2 Task 9 rule) and `winBonus` on a win.
- [ ] Failing tests: `terminal reward is placement value`, `win adds bonus`, `shaping sums to gamma-discounted potential difference`, `shaping off gives zero until terminal`. Run → FAIL. Implement. Run → PASS. Commit `"sim: placement reward with optional potential shaping"`.

### Task 2: Recurrent PPO with KL to BC

**Files:** Create `learn/ofai/rl/__init__.py`, `learn/ofai/rl/rollout.py`, `learn/ofai/rl/ppo.py`, `learn/tests/test_ppo.py`.

**Interfaces:**
- Consumes: `Policy` (M3), `SimPool` env role (M2), `load_checkpoint`.
- Produces:
  - `Rollout(T: int = 128, rows: int)` storing obs, actions, log-probs, values, rewards, flags, LSTM states at chunk starts.
  - `gae(rewards, values, dones, valid, gamma=0.999, lam=0.95) -> (adv, ret)` per row (v2's per-row GAE).
  - `ppo_update(policy, bc_policy, rollout, *, clip=0.2, vf_coef=0.5, ent_coef=0.001, kl_coef: float, epochs=2, minibatch_rows=32) -> dict` — KL = sum over the heads each sampled action used of `KL(π‖π_BC)` on the full categorical (both policies' logits on the same obs); skips and logs any minibatch with non-finite loss; fp32 heads; grad clip 1.0.
  - `kl_schedule(samples: int, start=0.5, end=0.05, over=200_000_000) -> float` (linear).
- [ ] Failing tests: `gae matches reference`, `ppo improves a toy bandit`, `kl zero when policy equals bc`, `nan update skipped`, `learner-less game aborted` (fake pool). Run → FAIL. Implement (port v2 `ppo.py`, `selfplay.py` pieces). Run → PASS. Commit `"learn: recurrent PPO with KL to BC"`.

### Task 3: League and game composition

**Files:** Create `learn/ofai/rl/league.py`, `learn/ofai/rl/compose.py`, `learn/tests/test_league.py`.

**Interfaces:**
- Produces:
  - `League(dir: Path)`: `add_snapshot(checkpoint, step)`, `sample_opponents(n: int, learner_rating) -> list[Opponent]` with probability ∝ `p(1−p)` where `p` = expected learner win probability from ratings (keeps games near 50 %), `stand_ins(n) -> list[Opponent]` = the BC checkpoint at temperatures {0.7, 1.0, 1.3}.
  - `compose_game(rng, league, maps: MapTable) -> GameSpec + seat assignment`: 4 learner seats, 4 league seats, 4 stand-in seats by default; bots and nations from public-lobby medians per map; map drawn with weight = rotation weight ÷ measured cost (M2 step-cost table).
  - Snapshot every 10M samples.
- [ ] Failing tests: `opponent sampling favours near-even ratings`, `stand-ins use BC at three temperatures`, `missing snapshot dropped`, `map weights use cost`. Run → FAIL. Implement. Run → PASS. Commit `"learn: league with human stand-ins"`.

### Task 4: RL trainer, ladder and guards

**Files:** Create `learn/ofai/rl/train.py`, `learn/ofai/rl/ladder.py`, `learn/configs/rl.yaml`, `learn/tests/test_rl_train.py`; Modify `learn/ofai/cli.py` (`rl`, `ladder`), `learn/ofai/bench/runner.py` (rating in results).

**Interfaces:**
- Produces: `train_rl(cfg, run_dir, *, until, stop)` with the same session contract as `train_bc` (CONTROL file, checkpoints, end-of-session bench, summary); `Ladder(path=DATA/"ladder.json")` with `play(checkpoints: list[str], games: int)` (multi-seat FFA games between checkpoints, openskill Plackett–Luce update from placements) and `rating(name) -> (mu, sigma)`; imitation guard every eval: held-out top-1 type accuracy vs the BC checkpoint's.
- [ ] Failing tests: `ladder ranks a stronger fake above a weaker one`, `imitation guard stops run`, `resume discards partial rollout`, `session control honoured`. Run → FAIL. Implement. Run → PASS. Commit `"learn: RL trainer, rating ladder, imitation guard"`.

### Task 5: A/B harness and the land-share decision

**Files:** Create `learn/ofai/rl/ab.py`, `learn/tests/test_ab.py`, `docs/experiments/rl-landshare.md`.

**Interfaces:**
- Produces: `python -m ofai ab --a configs/rl.yaml --b configs/rl-landshare.yaml --samples 50_000_000` → two runs from the same BC checkpoint and seed set, then `bench-v1` and a ladder match between their finals; prints a table and a verdict (B adopted only if `human_proxy` placement and ladder rating are both better beyond their 95 % intervals).
- [ ] Failing tests: `verdict requires both metrics`, `runs share seeds`. Run → FAIL. Implement. Run → PASS.
- [ ] Run the A/B (land-share off vs `landShare: 1.0`) over two sessions; write the result into `docs/experiments/rl-landshare.md`; set `rl.yaml` accordingly. Commit `"rl: land-share A/B and decision"`.

### Task 6: M6 exit check

- [ ] RL sessions until the ladder rating plateaus over two sessions.
- [ ] Exit: the best RL checkpoint beats the BC checkpoint on ladder rating and `human_proxy` placement; held-out imitation top-1 stays ≥ 80 % of BC's; at least one local game against Peter.
- [ ] Record everything in `docs/measurements.md`; PR `m6-selfplay-rl`.
