# BetterOpenfrontAi v3

An AI for [OpenFront](https://openfront.io) that learns from strong human
players' public games first, then improves through self-play. The goal is
an AI that holds its own against skilled people in public FFA lobbies.

**Status:** M0 done: 25 fresh public FFA records replay with every hash matching through per-commit engine
bundles ([docs/measurements.md](docs/measurements.md)). M1's data backend and most of M2 (observation, action codec,
translator) are built; the app, the sampler/bridge/env and M3 are next.

v1 and v2 live in [BetterOpenfrontAi](https://github.com/H4ck3rPr0G4m3r/BetterOpenfrontAi);
the design explains what carries over and what doesn't.

## License

AGPL-3.0, because the environment links the OpenFront engine (AGPL-3.0).
