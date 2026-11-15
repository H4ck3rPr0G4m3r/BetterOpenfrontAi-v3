# Measurements

Measured numbers, appended per milestone (roadmap §1 "Refresh rule"). Newest at the bottom of each milestone.

## M0 — sample fetch (2026-11-15)

**Blocked: the API challenges our honest client.** With `User-Agent: ofai-crawler/0.1`:

| Endpoint | Answer |
|---|---|
| `GET /leaderboard/ranked?page=1` | `403`, `Cf-Mitigated: challenge` (Cloudflare "Just a moment...") |
| `GET /public/player/{id}/games` | `200` (history works) |
| `GET /game/{id}` | `403`, `Cf-Mitigated: challenge` |

curl's default UA gets the same challenge on the leaderboard. The planning probes on 2026-11-14 got `200` only
because they sent a browser-like UA (`Mozilla/5.0 (research-client)`). Sending a browser-like UA to pass a bot
challenge is hiding from a human check, which roadmap §2 forbids, so the crawler stops on 403 as specified.

`python -m ofai fetch-sample --players 20 --games 30`:

```json
{"fetched": 0, "requests": 1, "error": "403 from /leaderboard/ranked (Cloudflare challenge). ..."}
```

`python -m ofai fetch-sample --player <id> --games 1` (leaderboard skipped): history `200`, then
`403 from /game/<id> (Cloudflare challenge)` after 2 requests.

**Unblock:** OpenFront allows `ofai-crawler/0.1` (≤ 1 req/s) on `/game/{id}` and `/leaderboard/ranked`, or provides
an official export. Until then every real-record step (M0 exit check, M1 crawl, M2 translator report on real games,
M3 training data) waits; everything else is built and tested on synthetic records from our own engine runs.

## M0 — engine bundles (2026-11-15)

`python -m ofai build-engine <commit>` on the 16-core / NVMe Windows box:

| Commit | Status | Total | worktree | deps | bundle | selfcheck | Size on disk |
|---|---|---|---|---|---|---|---|
| `8b5d071` (pin, v0.36.2) | ok | 5.3 s | 4.7 s | 0.25 s (links) | 0.16 s | 0.14 s | 1.2 GB (814 MB `resources/`) |
| `4457217` (upstream main) | ok | 5.4 s | 4.8 s | 0.26 s (links) | 0.16 s | 0.12 s | 1.2 GB |

The bundle (`sim.mjs`) is 1.7 MB. Disk is dominated by the worktree's `resources/` (maps); 10 window commits ≈ 12 GB.
If that becomes a problem, a sparse checkout without `resources/` plus a shared maps directory is the fix (maps rarely
change between commits; the selfcheck would catch a map that did).

Upstream's own sample record (`engine/records/demoGame1.json`, Pangaea, 30 bots, default nations) can't serve as a
real-record check: its `gitCommit` `4444c9c…` is a placeholder (`not our ref` upstream). On the pin it diverges at
turn 210, as expected for a record from another version.
