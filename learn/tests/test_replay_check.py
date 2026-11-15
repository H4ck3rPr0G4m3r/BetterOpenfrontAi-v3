import json
import threading
import time
from pathlib import Path

from ofai.data.replay_check import check_records
from ofai.engines import BundleMeta


def make_records(root: Path, spec: dict[str, list[str]]) -> list[Path]:
    out = []
    for commit, games in spec.items():
        (root / commit).mkdir(parents=True, exist_ok=True)
        for g in games:
            p = root / commit / f"{g}.json.gz"
            p.write_bytes(b"x")
            out.append(p)
    return out


def ok_bundle(commit: str) -> BundleMeta:
    return BundleMeta(commit=commit, status="ok")


def result(path: str, **kw) -> dict:
    gid = Path(path).name.split(".")[0]
    base = {"gameId": gid, "map": "World", "turns": 100, "hashChecks": 10, "mismatchAt": None, "error": None,
            "ms": 50, "ticksPerSec": 2000, "path": path}
    return {**base, **kw}


def test_groups_by_commit_and_counts(tmp_path):
    paths = make_records(tmp_path / "records", {"A" * 40: ["g1", "g2"], "B" * 40: ["g3"]})

    def run(commit, *args):
        p = args[1]
        if p.endswith("g2.json.gz"):
            return [result(p, mismatchAt=40)]
        return [result(p)]

    rep = check_records(paths, jobs=2, ensure=ok_bundle, run=run, log=tmp_path / "log.jsonl")
    assert rep["games"] == 3 and rep["ok"] == 2 and rep["mismatch"] == 1 and rep["error"] == 0
    assert rep["perCommit"]["A" * 40] == {"bundle": "ok", "ok": 1, "mismatch": 1, "error": 0}
    assert rep["perCommit"]["B" * 40]["ok"] == 1
    assert rep["ticksPerSec"] == {"World": 2000}
    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    assert len(lines) == 3 and json.loads(lines[0])["commit"] in ("A" * 40, "B" * 40)


def test_failed_bundle_counts_and_continues(tmp_path):
    paths = make_records(tmp_path / "records", {"A" * 40: ["g1"], "B" * 40: ["g2", "g3"]})

    def ensure(commit):
        if commit.startswith("B"):
            return BundleMeta(commit=commit, status="failed", stage="bundle", error="boom")
        return ok_bundle(commit)

    rep = check_records(paths, jobs=2, ensure=ensure, run=lambda c, *a: [result(a[1])], log=tmp_path / "l.jsonl")
    assert rep["perCommit"]["B" * 40] == {"bundle": "failed:bundle", "ok": 0, "mismatch": 0, "error": 0, "unbuilt": 2}
    assert rep["unbuilt"] == 2
    assert rep["ok"] == 1 and rep["games"] == 3


def test_runs_in_parallel_up_to_jobs(tmp_path):
    paths = make_records(tmp_path / "records", {"A" * 40: [f"g{i}" for i in range(12)]})
    lock = threading.Lock()
    state = {"now": 0, "max": 0}

    def run(commit, *args):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.02)
        with lock:
            state["now"] -= 1
        return [result(args[1])]

    rep = check_records(paths, jobs=4, ensure=ok_bundle, run=run, log=tmp_path / "l.jsonl")
    assert rep["ok"] == 12
    assert 2 <= state["max"] <= 4


def test_sim_crash_is_an_error_not_a_raise(tmp_path):
    paths = make_records(tmp_path / "records", {"A" * 40: ["g1", "g2"]})

    def run(commit, *args):
        if args[1].endswith("g1.json.gz"):
            raise RuntimeError("worker died")
        return [result(args[1], error="bad intent at turn 5: warp")]

    rep = check_records(paths, jobs=1, ensure=ok_bundle, run=run, log=tmp_path / "l.jsonl")
    assert rep["error"] == 2 and rep["ok"] == 0
    assert rep["errors"]["bad intent"] == 1 and rep["errors"]["sim crashed"] == 1
