import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ofai.engine_update import recent_target, update_engine
from ofai.engines import BundleMeta

NOW = datetime(2026, 11, 15, tzinfo=timezone.utc)
OLD, NEW = "0" * 40, "1" * 40


class FakeIndex:
    def __init__(self, games):
        self._games = games

    def games(self, commit=None, only_in_window=False, limit=100):
        return [g for g in self._games if commit is None or g["commit"] == commit][:limit]


def g(commit, days_ago, path="nope"):
    return {"id": f"{commit[:2]}{days_ago}", "commit": commit, "start": int((NOW - timedelta(days=days_ago)).timestamp() * 1000), "path": path}


class Runner:
    def __init__(self, fail=None):
        self.fail = fail
        self.calls = []

    def __call__(self, cmd, cwd):
        self.calls.append(cmd)
        if cmd[-2:] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(cmd, 0, OLD + "\n", "")
        failing = self.fail and (self.fail in cmd)
        return subprocess.CompletedProcess(cmd, 1 if failing else 0, "", "boom" if failing else "")


def test_target_is_most_common_recent_commit():
    ix = FakeIndex([g(NEW, 1), g(NEW, 2), g(OLD, 1), g(OLD, 10), g(OLD, 11)])
    assert recent_target(ix, NOW) == NEW


def test_already_on_target_is_a_noop():
    run = Runner()
    r = update_engine(FakeIndex([]), target=OLD, run=run, now=NOW)
    assert r["ok"] and len(run.calls) == 1


def test_failed_sim_tests_roll_back_to_previous_commit():
    run = Runner(fail="vitest")
    r = update_engine(FakeIndex([]), target=NEW, run=run, now=NOW, ensure=lambda c: BundleMeta(commit=c, status="ok"))
    assert not r["ok"]
    names = [s["name"] for s in r["steps"]]
    assert names == ["fetch", "checkout", "npm ci", "sim tests", "rollback"]
    assert ["git", "-C", str(Path(run.calls[-2][2])), "checkout", "--detach", OLD] == run.calls[-2]
    assert r["steps"][-1]["ok"]


def test_success_reports_all_steps_ok(tmp_path):
    rec = tmp_path / "r.json.gz"
    rec.write_bytes(b"x")
    ix = FakeIndex([g(NEW, 1, str(rec))])
    replayed = []

    def replay(commit, *args):
        replayed.append(args[1])
        return [{"gameId": "x", "mismatchAt": None, "error": None}]

    r = update_engine(ix, run=Runner(), now=NOW, ensure=lambda c: BundleMeta(commit=c, status="ok", buildSeconds=5),
                      replay=replay)
    assert r["ok"] and r["to"] == NEW
    assert all(s["ok"] for s in r["steps"])
    assert replayed == [str(rec)]
    assert "git add engine" in r["hint"]


def test_replay_mismatch_rolls_back(tmp_path):
    rec = tmp_path / "r.json.gz"
    rec.write_bytes(b"x")
    r = update_engine(FakeIndex([g(NEW, 1, str(rec))]), run=Runner(), now=NOW,
                      ensure=lambda c: BundleMeta(commit=c, status="ok"),
                      replay=lambda c, *a: [{"gameId": "x", "mismatchAt": 40, "error": None}])
    assert not r["ok"] and r["steps"][-2]["name"] == "replay" and "mismatch at turn 40" in r["steps"][-2]["detail"]
    assert r["steps"][-1]["name"] == "rollback"
