import json
import os
import subprocess
from pathlib import Path

import pytest

from ofai.engines import ENGINE_PACKAGES, BundleMeta, ensure_bundle, link_node_modules, remove_tree

COMMIT = "a" * 40


class FakeRunner:
    def __init__(self, root: Path, *, have_object=False, fetch_rc=0, bundle_rc=0, lock="lock-1", selfcheck_ok=True):
        self.root = root
        self.calls: list[list[str]] = []
        self.have_object = have_object
        self.fetch_rc = fetch_rc
        self.bundle_rc = bundle_rc
        self.lock = lock
        self.selfcheck_ok = selfcheck_ok

    def names(self) -> list[str]:
        out = []
        for c in self.calls:
            if c[0] == "git":
                out.append(c[3])
            elif c[0] == "npm":
                out.append("npm ci")
            elif c[0] == "node":
                out.append("bundle" if c[1].endswith("bundle.mjs") else c[2])
        return out

    def __call__(self, cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
        self.calls.append(cmd)
        ok = subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[0] == "git" and cmd[3] == "cat-file":
            return subprocess.CompletedProcess(cmd, 0 if self.have_object else 1, "", "")
        if cmd[0] == "git" and cmd[3] == "fetch":
            if self.fetch_rc:
                return subprocess.CompletedProcess(cmd, self.fetch_rc, "", "fatal: remote error: upload-pack: not our ref")
            return ok
        if cmd[0] == "git" and cmd[3] == "worktree" and cmd[4] == "add":
            wt = Path(cmd[-2])
            wt.mkdir(parents=True)
            (wt / "package.json").write_text("{}")
            (wt / "package-lock.json").write_text(self.lock)
            for p in ENGINE_PACKAGES:
                (wt / "packages" / p / "src").mkdir(parents=True)
                (wt / "packages" / p / "package.json").write_text("{}")
            return ok
        if cmd[0] == "node" and cmd[1].endswith("bundle.mjs"):
            if self.bundle_rc:
                return subprocess.CompletedProcess(cmd, 1, "", "error: cannot resolve @openfront/engine/Foo\n1 error")
            Path(cmd[-1]).write_text("// bundle")
            return ok
        if cmd[0] == "node" and cmd[2] == "selfcheck":
            body = {"ok": True, "layoutVersion": 0} if self.selfcheck_ok else {"ok": False, "error": "mismatch at 10"}
            return subprocess.CompletedProcess(cmd, 0 if self.selfcheck_ok else 1, json.dumps(body) + "\n", "")
        return ok


@pytest.fixture
def dirs(tmp_path: Path):
    engine = tmp_path / "engine"
    (engine / "node_modules" / "zod").mkdir(parents=True)
    (engine / "node_modules" / ".bin").mkdir()
    (engine / "node_modules" / "@openfront" / "engine").mkdir(parents=True)
    (engine / "node_modules" / ".package-lock.json").write_text("{}")
    (engine / "package-lock.json").write_text("lock-1")
    sim = tmp_path / "sim"
    (sim / "src").mkdir(parents=True)
    (sim / "src" / "cli.ts").write_text("export {}")
    (sim / "scripts").mkdir()
    (sim / "scripts" / "bundle.mjs").write_text("// bundler")
    (sim / "package.json").write_text(json.dumps({"version": "0.1.0"}))
    return {"engine": engine, "engines": tmp_path / "engines", "sim": sim}


def test_ok_build_runs_stages_in_order(dirs):
    run = FakeRunner(dirs["engines"])
    meta = ensure_bundle(COMMIT, run=run, **dirs)
    assert meta.status == "ok", meta.error
    assert run.names() == ["cat-file", "fetch", "worktree", "worktree", "bundle", "selfcheck"]
    assert run.calls[3][3:5] == ["worktree", "add"]
    saved = BundleMeta.load(COMMIT, dirs["engines"])
    assert saved is not None and saved.status == "ok" and saved.simVersion == "0.1.0" and saved.layoutVersion == 0


def test_deps_links_when_lockfile_matches(dirs):
    run = FakeRunner(dirs["engines"], lock="lock-1")
    ensure_bundle(COMMIT, run=run, **dirs)
    assert "npm ci" not in run.names()
    nm = dirs["engines"] / COMMIT / "src" / "node_modules"
    assert (nm / "zod").exists()
    # Workspace packages point at the commit's own worktree, never the pinned engine.
    assert Path(os.path.realpath(nm / "@openfront" / "engine")) == Path(
        os.path.realpath(dirs["engines"] / COMMIT / "src" / "packages" / "engine")
    )


def test_deps_npm_ci_when_lockfile_differs(dirs):
    run = FakeRunner(dirs["engines"], lock="lock-2")
    meta = ensure_bundle(COMMIT, run=run, **dirs)
    assert meta.status == "ok"
    npm = [c for c in run.calls if c[0] == "npm"]
    assert npm and npm[0][:3] == ["npm", "ci", "--ignore-scripts"]


def test_build_failure_is_recorded_not_raised(dirs):
    meta = ensure_bundle(COMMIT, run=FakeRunner(dirs["engines"], bundle_rc=1), **dirs)
    assert meta.status == "failed" and meta.stage == "bundle"
    assert "cannot resolve" in meta.error


def test_unresolvable_commit_fails_at_fetch(dirs):
    meta = ensure_bundle(COMMIT, run=FakeRunner(dirs["engines"], fetch_rc=128), **dirs)
    assert meta.status == "failed" and meta.stage == "fetch"
    assert "not our ref" in meta.error


def test_selfcheck_failure_is_recorded(dirs):
    meta = ensure_bundle(COMMIT, run=FakeRunner(dirs["engines"], selfcheck_ok=False), **dirs)
    assert meta.status == "failed" and meta.stage == "selfcheck" and "mismatch" in meta.error


def test_second_call_is_cached(dirs):
    ensure_bundle(COMMIT, run=FakeRunner(dirs["engines"]), **dirs)
    run2 = FakeRunner(dirs["engines"])
    meta = ensure_bundle(COMMIT, run=run2, **dirs)
    assert meta.status == "ok" and run2.calls == []


def test_sim_change_rebuilds_bundle_only(dirs):
    ensure_bundle(COMMIT, run=FakeRunner(dirs["engines"]), **dirs)
    (dirs["sim"] / "src" / "cli.ts").write_text("export const x = 1")
    run2 = FakeRunner(dirs["engines"], have_object=True)
    meta = ensure_bundle(COMMIT, run=run2, **dirs)
    assert meta.status == "ok"
    assert run2.names() == ["cat-file", "bundle", "selfcheck"]


def test_failed_fetch_is_not_retried_every_call(dirs):
    ensure_bundle(COMMIT, run=FakeRunner(dirs["engines"], fetch_rc=128), **dirs)
    run2 = FakeRunner(dirs["engines"])
    assert ensure_bundle(COMMIT, run=run2, **dirs).stage == "fetch"
    assert run2.calls == []
    assert ensure_bundle(COMMIT, run=run2, force=True, **dirs).status == "ok"


def test_remove_tree_never_follows_links(dirs, tmp_path):
    wt = tmp_path / "wt"
    (wt / "packages" / "engine").mkdir(parents=True)
    (wt / "packages" / "engine" / "package.json").write_text("{}")
    link_node_modules(wt, dirs["engine"] / "node_modules")
    remove_tree(wt / "node_modules")
    assert not (wt / "node_modules").exists()
    assert (dirs["engine"] / "node_modules" / "zod").exists()
    assert (wt / "packages" / "engine" / "package.json").exists()
