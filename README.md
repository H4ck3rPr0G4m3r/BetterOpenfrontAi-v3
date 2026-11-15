# BetterOpenfrontAi v3

An AI for [OpenFront](https://openfront.io) that learns from strong human
players' public games first, then improves through self-play. The goal is
an AI that holds its own against skilled people in public FFA lobbies.

**Status:** M0 built (engine submodule, per-commit bundles, record replayer, replay-check); its exit check waits on
real game records, which the OpenFront API currently serves only behind a Cloudflare challenge (see
[docs/measurements.md](docs/measurements.md)). The design lives in
[docs/superpowers/specs/2026-11-14-v3-design.md](docs/superpowers/specs/2026-11-14-v3-design.md),
and the step-by-step build plan, one file per milestone, starts at
[docs/superpowers/plans/2026-11-15-v3-00-roadmap.md](docs/superpowers/plans/2026-11-15-v3-00-roadmap.md).

v1 and v2 live in [BetterOpenfrontAi](https://github.com/H4ck3rPr0G4m3r/BetterOpenfrontAi);
the design explains what carries over and what doesn't.

## License

AGPL-3.0, because the environment links the OpenFront engine (AGPL-3.0).
