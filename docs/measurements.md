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
