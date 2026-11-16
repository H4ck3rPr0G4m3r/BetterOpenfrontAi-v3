"""Per-commit engine bundles: `engines/<commit>/` (spec §7, roadmap §3.2).

For a record's `gitCommit`:

1. fetch     `git -C engine fetch --depth 1 origin <commit>` (skipped when the object exists)
2. worktree  `git -C engine worktree add --detach engines/<commit>/src <commit>`
3. deps      a `node_modules` for the worktree (see `link_node_modules`)
4. bundle    `node sim/scripts/bundle.mjs --engine-root engines/<commit>/src --out engines/<commit>/sim.mjs`
5. selfcheck `node engines/<commit>/sim.mjs selfcheck --engine-root engines/<commit>/src`

Build failures never raise: they are recorded in `meta.json` with the failing
stage, so a batch can count them and carry on (Review Focus 1).

Dependency rule, refined from the roadmap: when the worktree's
`package-lock.json` equals `engine/`'s, its `node_modules` is a real directory
whose entries are links to `engine/node_modules/*`, except `@openfront/*`,
which link to the worktree's own `packages/*`. A plain link of the whole
directory would resolve every commit's workspace packages to the pinned
checkout (they are links to absolute paths inside `engine/packages`). When the
lockfiles differ, `npm ci` runs in the worktree.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ofai import paths

Runner = Callable[[list[str], Path], subprocess.CompletedProcess]

STAGES = ("fetch", "worktree", "deps", "bundle", "selfcheck")
ENGINE_PACKAGES = ("engine", "engine-api", "engine-lib", "shared", "zbin")


def subprocess_runner(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    exe = shutil.which(cmd[0]) or cmd[0]
    return subprocess.run([exe, *cmd[1:]], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")


@dataclass
class BundleMeta:
    commit: str
    status: str  # "ok" | "failed"
    stage: str | None = None
    error: str | None = None
    builtAt: str | None = None
    lockHash: str | None = None
    simVersion: str | None = None
    simHash: str | None = None
    layoutVersion: int | None = None
    buildSeconds: float | None = None
    timings: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def path(commit: str, engines: Path | None = None) -> Path:
        return (engines or paths.ENGINES) / commit / "meta.json"

    @classmethod
    def load(cls, commit: str, engines: Path | None = None) -> BundleMeta | None:
        p = cls.path(commit, engines)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            return cls(**known)
        except (OSError, ValueError, TypeError):
            return None

    def save(self, engines: Path | None = None) -> None:
        p = self.path(self.commit, engines)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
        os.replace(tmp, p)


def file_sha256(p: Path) -> str | None:
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sim_version(sim: Path | None = None) -> str:
    pkg = (sim or paths.SIM) / "package.json"
    return json.loads(pkg.read_text(encoding="utf-8"))["version"]


def sim_hash(sim: Path | None = None) -> str:
    """Content hash of everything that goes into a bundle from our side."""
    sim = sim or paths.SIM
    h = hashlib.sha256()
    files = sorted([*(sim / "src").rglob("*.ts"), *(sim / "scripts").glob("*.mjs"), sim / "package.json"])
    for f in files:
        if f.is_file():
            h.update(f.relative_to(sim).as_posix().encode())
            h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def _link_dir(link: Path, target: Path) -> None:
    """Directory junction on Windows (no admin rights needed), symlink elsewhere."""
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        os.symlink(target, link, target_is_directory=True)


def link_node_modules(worktree: Path, shared: Path) -> int:
    """Builds `<worktree>/node_modules` from `shared` (engine/node_modules) with
    the worktree's own `packages/*` as `@openfront/*`. Returns links made."""
    nm = worktree / "node_modules"
    if nm.exists() or nm.is_symlink():
        remove_tree(nm)
    nm.mkdir()
    made = 0
    for entry in shared.iterdir():
        if entry.name == "@openfront" or not entry.is_dir():
            continue
        _link_dir(nm / entry.name, entry)
        made += 1
    own = nm / "@openfront"
    own.mkdir()
    for pkg_dir in sorted((worktree / "packages").iterdir()):
        if (pkg_dir / "package.json").exists():
            _link_dir(own / pkg_dir.name, pkg_dir)
            made += 1
    return made


def remove_tree(p: Path) -> None:
    """Removes a directory without following links into their targets."""
    if not (p.exists() or p.is_symlink()):
        return
    if p.is_symlink() or _is_junction(p):
        os.rmdir(p) if sys.platform == "win32" else p.unlink()
        return
    for child in p.iterdir():
        if child.is_symlink() or _is_junction(child):
            os.rmdir(child) if (sys.platform == "win32" and child.is_dir()) else child.unlink()
        elif child.is_dir():
            remove_tree(child)
        else:
            child.unlink()
    p.rmdir()


def _is_junction(p: Path) -> bool:
    """True for a Windows directory junction (a reparse point that isn't a symlink)."""
    if sys.platform != "win32":
        return False
    try:
        attrs = os.lstat(p).st_file_attributes  # type: ignore[attr-defined]
    except OSError:
        return False
    return bool(attrs & 0x400) and not p.is_symlink()  # FILE_ATTRIBUTE_REPARSE_POINT


def _tail(text: str | None, lines: int = 15) -> str:
    return "\n".join((text or "").strip().splitlines()[-lines:])


class _BuildFailed(Exception):
    def __init__(self, stage: str, error: str) -> None:
        super().__init__(error)
        self.stage = stage


def ensure_bundle(
    commit: str,
    *,
    run: Runner = subprocess_runner,
    engine: Path | None = None,
    engines: Path | None = None,
    sim: Path | None = None,
    force: bool = False,
) -> BundleMeta:
    """Idempotent: an ok bundle built from the current sim sources is reused."""
    engine = engine or paths.ENGINE
    engines = engines or paths.ENGINES
    sim = sim or paths.SIM
    want_hash = sim_hash(sim)
    existing = BundleMeta.load(commit, engines)
    out_dir = engines / commit
    if (
        not force
        and existing is not None
        and existing.status == "ok"
        and existing.simHash == want_hash
        and (out_dir / "sim.mjs").exists()
    ):
        return existing
    # A failure at fetch/worktree/deps is about the commit, not our code: don't retry it every call.
    if (
        not force
        and existing is not None
        and existing.status == "failed"
        and existing.stage in ("fetch", "worktree", "deps")
        and existing.simHash == want_hash
    ):
        return existing

    meta = BundleMeta(commit=commit, status="failed", simVersion=sim_version(sim), simHash=want_hash)
    t_start = time.monotonic()
    worktree = out_dir / "src"
    stage = "fetch"
    try:
        t = time.monotonic()
        if run(["git", "-C", str(engine), "cat-file", "-e", f"{commit}^{{commit}}"], engine).returncode != 0:
            r = run(["git", "-C", str(engine), "fetch", "--depth", "1", "origin", commit], engine)
            if r.returncode != 0:
                raise _BuildFailed("fetch", _tail(r.stderr) or f"git fetch exited {r.returncode}")
        meta.timings["fetch"] = round(time.monotonic() - t, 2)

        stage = "worktree"
        t = time.monotonic()
        if not (worktree / "package.json").exists():
            out_dir.mkdir(parents=True, exist_ok=True)
            run(["git", "-C", str(engine), "worktree", "prune"], engine)
            r = run(["git", "-C", str(engine), "worktree", "add", "--force", "--detach", str(worktree), commit], engine)
            if r.returncode != 0:
                raise _BuildFailed("worktree", _tail(r.stderr) or f"git worktree add exited {r.returncode}")
        meta.timings["worktree"] = round(time.monotonic() - t, 2)
        missing = [p for p in ENGINE_PACKAGES if not (worktree / "packages" / p / "src").is_dir()]
        if missing:
            raise _BuildFailed("worktree", f"commit predates the engine package split (no packages/{missing[0]})")

        stage = "deps"
        t = time.monotonic()
        meta.lockHash = file_sha256(worktree / "package-lock.json")
        if meta.lockHash is not None and meta.lockHash == file_sha256(engine / "package-lock.json") and (
            engine / "node_modules"
        ).is_dir():
            link_node_modules(worktree, engine / "node_modules")
        elif not (worktree / "node_modules" / ".package-lock.json").exists():
            r = run(["npm", "ci", "--ignore-scripts", "--engine-strict=false", "--no-audit", "--no-fund"], worktree)
            if r.returncode != 0:
                raise _BuildFailed("deps", _tail(r.stderr) or f"npm ci exited {r.returncode}")
        meta.timings["deps"] = round(time.monotonic() - t, 2)

        stage = "bundle"
        t = time.monotonic()
        r = run(
            ["node", str(sim / "scripts" / "bundle.mjs"), "--engine-root", str(worktree), "--out", str(out_dir / "sim.mjs")],
            sim,
        )
        if r.returncode != 0:
            raise _BuildFailed("bundle", _tail(r.stderr) or _tail(r.stdout) or f"bundle exited {r.returncode}")
        meta.timings["bundle"] = round(time.monotonic() - t, 2)

        stage = "selfcheck"
        t = time.monotonic()
        r = run(["node", str(out_dir / "sim.mjs"), "selfcheck", "--engine-root", str(worktree)], out_dir)
        lines = _json_lines(r.stdout)
        if r.returncode != 0 or not lines or not lines[-1].get("ok"):
            detail = (lines[-1].get("error") if lines else None) or _tail(r.stderr) or f"selfcheck exited {r.returncode}"
            raise _BuildFailed("selfcheck", str(detail))
        meta.layoutVersion = int(lines[-1].get("layoutVersion", 0))
        meta.timings["selfcheck"] = round(time.monotonic() - t, 2)

        meta.status = "ok"
    except _BuildFailed as e:
        meta.stage, meta.error = e.stage, str(e)
    except Exception as e:  # noqa: BLE001 - recorded, never raised (Review Focus 1)
        meta.stage, meta.error = stage, f"{type(e).__name__}: {e}"
    meta.builtAt = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta.buildSeconds = round(time.monotonic() - t_start, 2)
    meta.save(engines)
    return meta


def _json_lines(text: str | None) -> list[dict]:
    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


class SimError(RuntimeError):
    pass


def run_sim(commit: str, *args: str, engines: Path | None = None, timeout: float | None = None) -> list[dict]:
    """Runs one bundled sim command for `commit` and returns its JSON lines."""
    engines = engines or paths.ENGINES
    out_dir = engines / commit
    cmd = [shutil.which("node") or "node", str(out_dir / "sim.mjs"), *args, "--engine-root", str(out_dir / "src")]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    lines = _json_lines(r.stdout)
    if r.returncode != 0 and not lines:
        raise SimError(f"sim {args[0] if args else ''} for {commit[:10]} exited {r.returncode}: {_tail(r.stderr)}")
    return lines


def list_bundles(engines: Path | None = None) -> list[BundleMeta]:
    engines = engines or paths.ENGINES
    if not engines.exists():
        return []
    metas = [BundleMeta.load(d.name, engines) for d in sorted(engines.iterdir()) if d.is_dir()]
    return [m for m in metas if m is not None]
