"""`python -m ofai <command>` (roadmap §3.7)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

from ofai.data.api import ApiError, ApiForbidden, ApiNotFound, OpenFrontApi
from ofai.data.filters import is_wellformed_summary, modifiers, wanted_summary
from ofai.data.records import find_record, write_record


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def fetch_sample(api: OpenFrontApi, *, players: int, games: int, seed_players: Sequence[str] = ()) -> dict:
    """Small, polite sample: the top `players` FFA players' recent public FFA games.

    `seed_players` replaces the leaderboard lookup (public IDs given by hand)."""
    skipped: Counter[str] = Counter()
    mods: Counter[str] = Counter()
    commits: Counter[str] = Counter()
    summary: dict = {"fetched": 0, "skipped": skipped, "commits": commits, "modifiers": mods, "error": None}

    try:
        if seed_players:
            teacher_ids = list(seed_players)[:players]
        else:
            board = api.leaderboard(1)
            ffa = board.get("ffa") if isinstance(board, dict) else None
            if not isinstance(ffa, list):
                raise ApiError("leaderboard has no ffa list")
            teacher_ids = [e["public_id"] for e in ffa if isinstance(e, dict) and isinstance(e.get("public_id"), str)]
            teacher_ids = teacher_ids[:players]

        seen: set[str] = set()
        for public_id in teacher_ids:
            if summary["fetched"] >= games:
                break
            cursor: str | None = None
            for _page in range(5):
                entries, cursor = api.player_games(public_id, cursor)
                for entry in entries:
                    if summary["fetched"] >= games:
                        break
                    if not is_wellformed_summary(entry):
                        skipped["malformed"] += 1
                        continue
                    if not wanted_summary(entry):
                        skipped["filter"] += 1
                        continue
                    gid = entry["gameId"]
                    if gid in seen:
                        continue
                    seen.add(gid)
                    if find_record(gid) is not None:
                        skipped["stored"] += 1
                        continue
                    try:
                        record = api.game(gid)
                    except ApiNotFound:
                        skipped["not_found"] += 1
                        continue
                    found = modifiers(record.get("info", {}).get("config", {}))
                    if found:
                        skipped["modifiers"] += 1
                        mods.update(found)
                        continue
                    write_record(record)
                    commits[record.get("gitCommit") or "unknown"] += 1
                    summary["fetched"] += 1
                if cursor is None or summary["fetched"] >= games:
                    break
    except ApiForbidden as e:
        summary["error"] = str(e)
    except ApiError as e:
        summary["error"] = str(e)
    summary["requests"] = api.requests
    return summary


def _parse_until(text: str | None) -> datetime | None:
    if not text:
        return None
    if len(text) <= 5 and ":" in text:
        hh, mm = (int(x) for x in text.split(":"))
        now = datetime.now()
        at = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        return at if at > now else at + timedelta(days=1)
    return datetime.fromisoformat(text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ofai")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch-sample", help="fetch a small sample of public FFA records")
    p.add_argument("--players", type=int, default=20)
    p.add_argument("--games", type=int, default=30)
    p.add_argument("--player", action="append", default=[], help="public ID to use instead of the leaderboard")

    p = sub.add_parser("crawl", help="crawl teacher games into data/records and the index")
    p.add_argument("--teachers", type=int, default=200)
    p.add_argument("--days", type=int, default=21)
    p.add_argument("--target", type=int, default=2000)
    p.add_argument("--player", action="append", default=[], help="seed teacher public ID (skips the leaderboard)")

    p = sub.add_parser("serve", help="run the app's local service on 127.0.0.1")
    p.add_argument("--port", type=int, default=47800)

    p = sub.add_parser("build-engine", help="build the sim bundle for one engine commit")
    p.add_argument("commit")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("replay-check", help="replay stored records and check every hash")
    p.add_argument("--jobs", type=int, default=None)
    p.add_argument("--since", default=None, help="only records whose game started on/after this ISO date")
    p.add_argument("--limit", type=int, default=None)

    p = sub.add_parser("split", help="freeze the train / held-out split (refuses to overwrite)")
    p.add_argument("--name", default="bc-v1")
    p.add_argument("--heldout", type=float, default=0.05)

    p = sub.add_parser("bc", help="train (or resume) behaviour cloning")
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "bc.yaml"))
    p.add_argument("--until", default=None, help="stop at this local time (HH:MM, today or tomorrow) or ISO datetime")
    p.add_argument("--resume", default=None, help="run directory to continue")
    p.add_argument("--max-steps", type=int, default=None)

    p = sub.add_parser("freeze-bench", help="freeze bench-v1 with a proxy checkpoint (refuses if it exists)")
    p.add_argument("proxy")

    p = sub.add_parser("bench", help="run bench-v1 suites with a checkpoint")
    p.add_argument("checkpoint")
    p.add_argument("--suites", default="solo_medium,solo_hard,human_proxy")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument("--games-per-worker", type=int, default=2)
    p.add_argument("--no-replays", action="store_true")
    p.add_argument("--no-teacher-curves", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "freeze-bench":
        from ofai.bench.suites import bench_file_sha, freeze_bench_v1
        from ofai.data.index import Index

        out = freeze_bench_v1(Index(), Path(args.proxy))
        _print({"bench": str(out), "sha256": bench_file_sha()})
        return 0

    if args.command == "bench":
        from ofai.bench.runner import run_bench, teacher_curves
        from ofai.bench.suites import load_bench
        from ofai.data.index import Index

        curves = None
        if not args.no_teacher_curves:
            curves = teacher_curves(Index(), [m["map"] for m in load_bench()["maps"]])
        _print(run_bench(Path(args.checkpoint), args.suites.split(","), workers=args.workers,
                         games_per_worker=args.games_per_worker, replays=not args.no_replays, teacher_curves=curves))
        return 0

    if args.command == "split":
        from ofai.bc.split import make_split
        from ofai.data.index import Index

        _print({"split": str(make_split(Index(), args.name, args.heldout))})
        return 0

    if args.command == "bc":
        import yaml

        from ofai import paths
        from ofai.bc.train import train_bc

        cfg = yaml.safe_load(Path(args.config).read_text()) or {}
        run_dir = Path(args.resume) if args.resume else paths.RUNS / f"{datetime.now():%Y%m%d-%H%M}-bc"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
        _print(train_bc(cfg, run_dir, until=_parse_until(args.until), max_steps=args.max_steps))
        return 0

    if args.command == "fetch-sample":
        with OpenFrontApi() as api:
            summary = fetch_sample(api, players=args.players, games=args.games, seed_players=args.player)
        _print(summary)
        return 1 if summary["error"] else 0

    if args.command == "crawl":
        from ofai.data.crawl import CrawlConfig, Crawler
        from ofai.data.index import Index

        cfg = CrawlConfig(teachers=args.teachers, days=args.days, target_games=args.target, seed_players=args.player)
        with OpenFrontApi() as api:
            result = Crawler(api, Index(), cfg, progress=lambda p: print(json.dumps(p), file=sys.stderr)).run()
        _print(result)
        return 1 if result["phase"] == "error" else 0

    if args.command == "serve":
        from ofai.service.app import serve

        serve(args.port)
        return 0

    if args.command == "build-engine":
        from ofai.engines import ensure_bundle

        meta = ensure_bundle(args.commit, force=args.force)
        _print(meta.to_json())
        return 0 if meta.status == "ok" else 1

    if args.command == "replay-check":
        from ofai.data.replay_check import replay_check_cli

        return replay_check_cli(jobs=args.jobs, since=args.since, limit=args.limit)

    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
