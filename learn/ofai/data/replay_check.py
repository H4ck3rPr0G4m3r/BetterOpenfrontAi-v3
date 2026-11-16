"""Replays stored records with their own engine commit and checks every hash
(spec §7, M0 exit check). Records are grouped by commit; each commit's bundle
is built once; replays run in parallel processes. A commit whose bundle fails
is counted (`failed:<stage>`) and the batch goes on (Review Focus 1)."""

from __future__ import annotations

import json
import os
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from ofai import paths
from ofai.data.records import all_records
from ofai.engines import BundleMeta, ensure_bundle, run_sim

Ensure = Callable[[str], BundleMeta]
RunSim = Callable[..., list[dict]]


def default_jobs() -> int:
    return max(1, (os.cpu_count() or 4) - 2)


def _error_kind(error: str) -> str:
    for prefix in ("bad intent", "engine error", "engine threw", "setup failed", "sim crashed"):
        if error.startswith(prefix):
            return prefix
    return "other"


def check_records(
    record_paths: list[Path],
    *,
    jobs: int | None = None,
    ensure: Ensure = ensure_bundle,
    run: RunSim = run_sim,
    log: Path | None = None,
    progress: Callable[[dict], None] | None = None,
    on_result: Callable[[dict], None] | None = None,
    stop: Callable[[], bool] | None = None,
) -> dict:
    jobs = jobs or default_jobs()
    log = log or (paths.DATA / "replay_check.jsonl")
    by_commit: dict[str, list[Path]] = defaultdict(list)
    for p in record_paths:
        by_commit[p.parent.name].append(p)

    report: dict = {"games": len(record_paths), "ok": 0, "mismatch": 0, "error": 0, "unbuilt": 0,
                    "perCommit": {}, "errors": Counter(), "ticksPerSec": {}}
    speeds: dict[str, list[float]] = defaultdict(list)
    todo: list[tuple[str, Path]] = []
    for commit, recs in sorted(by_commit.items()):
        meta = ensure(commit)
        entry = {"bundle": "ok" if meta.status == "ok" else f"failed:{meta.stage}", "ok": 0, "mismatch": 0, "error": 0}
        if meta.status != "ok":
            entry["unbuilt"] = len(recs)
            report["unbuilt"] += len(recs)
        else:
            todo.extend((commit, p) for p in recs)
        report["perCommit"][commit] = entry

    def one(item: tuple[str, Path]) -> dict:
        commit, p = item
        if stop is not None and stop():
            return {"gameId": p.name.split(".")[0], "commit": commit, "path": str(p), "skipped": True}
        try:
            lines = run(commit, "replay", str(p))
            r = next((x for x in lines if "gameId" in x), None) or {"error": f"sim crashed: no result ({lines})"}
        except Exception as e:  # noqa: BLE001 - one bad game never stops the batch
            r = {"gameId": p.name.split(".")[0], "error": f"sim crashed: {e}"}
        return {**r, "commit": commit, "path": str(p)}

    log.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool, log.open("a", encoding="utf-8") as out:
        for done, r in enumerate(pool.map(one, todo), 1):
            if r.get("skipped"):
                continue
            entry = report["perCommit"][r["commit"]]
            if r.get("error"):
                kind, key = "error", _error_kind(str(r["error"]))
                report["errors"][key] += 1
            elif r.get("mismatchAt") is not None:
                kind = "mismatch"
            else:
                kind = "ok"
                if r.get("ticksPerSec"):
                    speeds[str(r.get("map", "unknown"))].append(float(r["ticksPerSec"]))
            report[kind] += 1
            entry[kind] += 1
            if on_result:
                on_result({**r, "outcome": kind})
            out.write(json.dumps({**r, "checkedAt": datetime.now().isoformat(timespec="seconds")}) + "\n")
            out.flush()
            if progress:
                progress({"done": done, "total": len(todo), "ok": report["ok"], "mismatch": report["mismatch"],
                          "error": report["error"]})
    report["ticksPerSec"] = {m: round(statistics.median(v)) for m, v in sorted(speeds.items())}
    report["errors"] = dict(report["errors"])
    return report


def check_index(
    index,
    *,
    jobs: int | None = None,
    only_unchecked: bool = True,
    only_in_window: bool = True,
    **kw,
) -> dict:
    """Replay-checks indexed games and writes the verdict back (`replay_ok`)."""
    rows = index.games(only_in_window=only_in_window, limit=10_000_000)
    if only_unchecked:
        rows = [g for g in rows if g["replay_ok"] is None]
    record_paths = [Path(g["path"]) for g in rows if Path(g["path"]).exists()]

    def record_verdict(r: dict) -> None:
        index.mark_replay(r["gameId"], r["outcome"] == "ok")
        if r.get("placements"):
            index.set_placements(r["gameId"], r["placements"], r.get("ranked"))
        if kw_on_result:
            kw_on_result(r)

    kw_on_result = kw.pop("on_result", None)
    report = check_records(record_paths, jobs=jobs, on_result=record_verdict, **kw)
    report["missingFiles"] = len(rows) - len(record_paths)
    return report


def replay_check_cli(*, jobs: int | None, since: str | None, limit: int | None) -> int:
    from ofai.data.index import Index
    from ofai.data.records import read_record

    index_path = paths.DATA / "index.sqlite"
    if index_path.exists() and not since:
        index = Index(index_path)
        if index.count_games():
            report = check_index(index, jobs=jobs)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["mismatch"] == 0 and report["error"] == 0 else 1

    record_paths = all_records()
    if since:
        cutoff = datetime.fromisoformat(since).timestamp() * 1000
        record_paths = [p for p in record_paths if (read_record(p)["info"].get("start") or 0) >= cutoff]
    if limit:
        record_paths = record_paths[:limit]
    if not record_paths:
        print(json.dumps({"games": 0, "error": f"no records under {paths.RECORDS}"}))
        return 1
    report = check_records(record_paths, jobs=jobs)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["mismatch"] == 0 and report["error"] == 0 else 1
