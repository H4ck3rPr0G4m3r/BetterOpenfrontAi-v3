# BetterOpenfrontAi v3

An AI for [OpenFront](https://openfront.io) that learns from strong human
players' public games first, then improves through self-play. The goal is
an AI that holds its own against skilled people in public FFA lobbies.

**Status:** M0 done (fresh public FFA records replay with every hash matching). M1's data backend, all of M2
(observation, action codec, translator, replay sampler, bridge, multi-seat env) and M3's code (GPU planes, 10M
policy, BC loss and trainer, checkpoints, `bench-v1` runner, AI mind sidecar) are built and tested. Still to do:
the M1 app (and with it M3's replay overlay), a real BC training run and freezing `bench-v1`
([docs/measurements.md](docs/measurements.md)).

## Training setup

```
pip install -e learn[dev]
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA build
python -m ofai crawl                     # teacher games via the public API (1 request/s)
python -m ofai replay-check --jobs 14    # every hash must match
python -m ofai split                     # freezes data/splits/bc-v1.json
python -m ofai bc --until 07:30          # resumable; STOP file or --until ends a session
python -m ofai freeze-bench <checkpoint> # once, at the end of M3
python -m ofai bench <checkpoint>
```

v1 and v2 live in [BetterOpenfrontAi](https://github.com/H4ck3rPr0G4m3r/BetterOpenfrontAi);
the design explains what carries over and what doesn't.

## License

AGPL-3.0, because the environment links the OpenFront engine (AGPL-3.0).
