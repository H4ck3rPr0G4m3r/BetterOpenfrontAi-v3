"""The "Update engine" action (spec §7): move the `engine/` submodule to the
commit live players use now, prove it works, or roll back.

Target: the most common commit among in-window games that started in the
last 3 days (or an explicit commit). Steps: fetch -> checkout -> npm ci ->
sim tests -> bundle -> replay up to 5 newest stored records of that commit.
Any failed step checks the submodule back out at the previous commit and
reinstalls. Never commits: the app tells the user to commit the bump.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ofai import paths
from ofai.engines import Runner, ensure_bundle, run_sim, subprocess_runner


def recent_target(index, now: datetime, days: int = 3) -> str | None:
    cutoff = int((now - timedelta(days=days)).timestamp() * 1000)
    counts = Counter(g["commit"] for g in index.games(only_in_window=True, limit=100_000) if (g["start"] or 0) >= cutoff)
    return counts.most_common(1)[0][0] if counts else None


def update_engine(
    index,
    *,
    target: str | None = None,
    run: Runner = subprocess_runner,
    ensure: Callable = ensure_bundle,
    replay: Callable = run_sim,
    engine: Path | None = None,
    sim: Path | None = None,
    now: datetime | None = None,
    progress: Callable[[dict], None] = lambda p: None,
) -> dict:
    engine = engine or paths.ENGINE
    sim = sim or paths.SIM
    now = now or datetime.now(timezone.utc)
    head = run(["git", "-C", str(engine), "rev-parse", "HEAD"], engine)
    current = head.stdout.strip()
    target = target or recent_target(index, now)
    result: dict = {"from": current, "to": target, "steps": [], "ok": False}
    if target is None:
        result["steps"].append({"name": "target", "ok": False, "detail": "no in-window games from the last 3 days"})
        return result
    if target == current:
        result["ok"] = True
        result["steps"].append({"name": "target", "ok": True, "detail": "already on the target commit"})
        return result

    def step(name: str, cmd: list[str], cwd: Path) -> bool:
        progress({"step": name, "steps": result["steps"]})
        r = run(cmd, cwd)
        detail = "\n".join(((r.stderr or "") + (r.stdout or "")).strip().splitlines()[-12:])
        result["steps"].append({"name": name, "ok": r.returncode == 0, "detail": detail})
        return r.returncode == 0

    ok = (
        step("fetch", ["git", "-C", str(engine), "fetch", "--depth", "1", "origin", target], engine)
        and step("checkout", ["git", "-C", str(engine), "checkout", "--detach", target], engine)
        and step("npm ci", ["npm", "ci", "--ignore-scripts", "--engine-strict=false", "--no-audit", "--no-fund"], engine)
        and step("sim tests", ["npx", "vitest", "run"], sim)
    )
    if ok:
        progress({"step": "bundle", "steps": result["steps"]})
        meta = ensure(target)
        ok = meta.status == "ok"
        result["steps"].append({"name": "bundle", "ok": ok, "detail": meta.error or f"built in {meta.buildSeconds}s"})
    if ok:
        progress({"step": "replay", "steps": result["steps"]})
        games = [g for g in index.games(commit=target, limit=5) if Path(g["path"]).exists()]
        bad = []
        for g in games:
            lines = replay(target, "replay", g["path"])
            r = next((x for x in lines if "gameId" in x), {"error": "no result"})
            if r.get("error") or r.get("mismatchAt") is not None:
                bad.append(f"{g['id']}: {r.get('error') or 'hash mismatch at turn ' + str(r.get('mismatchAt'))}")
        ok = not bad
        detail = f"{len(games)} records replayed" + (f"; failed: {'; '.join(bad)}" if bad else "")
        if not games:
            detail = "no stored records of this commit to replay (crawl first for a stronger check)"
        result["steps"].append({"name": "replay", "ok": ok, "detail": detail})

    if not ok:
        progress({"step": "rollback", "steps": result["steps"]})
        rolled = run(["git", "-C", str(engine), "checkout", "--detach", current], engine).returncode == 0
        if rolled:
            rolled = run(["npm", "ci", "--ignore-scripts", "--engine-strict=false", "--no-audit", "--no-fund"], engine).returncode == 0
        result["steps"].append({"name": "rollback", "ok": rolled, "detail": f"back on {current[:10]}"})
        return result
    result["ok"] = True
    result["hint"] = "Engine updated. Commit the bump: git add engine && git commit -m 'engine: bump to <commit>'"
    return result
