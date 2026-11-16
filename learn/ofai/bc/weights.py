"""Per-teacher sample weights (spec §8.3: rating and outcome)."""

from __future__ import annotations


def teacher_weight(elo: float | None, value: float, elo_p50: float, elo_p90: float) -> float:
    """(0.5 + 0.5 * rating score) * (0.5 + 0.5 * placement value), in [0.25, 1].

    Rating score is the teacher's elo between the pool's median (0) and 90th
    percentile (1); an unknown elo counts as the median."""
    span = max(elo_p90 - elo_p50, 1e-9)
    r = 0.0 if elo is None else min(max((elo - elo_p50) / span, 0.0), 1.0)
    v = min(max(float(value), 0.0), 1.0)
    return (0.5 + 0.5 * r) * (0.5 + 0.5 * v)


def placement_value(place: int | None, ranked: int | None) -> float:
    """1 for the winner, 0 for the first out; 0.5 when unknown."""
    if place is None or ranked is None or ranked <= 0:
        return 0.5
    if ranked == 1:
        return 1.0
    return 1.0 - (min(max(place, 1), ranked) - 1) / (ranked - 1)
