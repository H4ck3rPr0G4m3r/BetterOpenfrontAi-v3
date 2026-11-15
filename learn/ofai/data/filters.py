"""What counts as a teacher game (roadmap §2 Data filter, spec §8.1).

Two stages: `wanted_summary` checks a player-history entry before the record is
fetched (type, mode, duration); `modifiers` checks the fetched record's
`info.config` for anything that makes the game unlike a normal public FFA.
Field names follow the engine's `GameConfigSchema` (packages/engine-api/src/Schemas.ts).
"""

from __future__ import annotations

from typing import Any

MIN_DURATION_S = 300
RANKED_DUELS = frozenset({"1v1", "2v2", "clanwars"})


def wanted_summary(entry: Any) -> bool:
    """A history entry worth fetching: public FFA lasting at least 5 minutes.
    Anything malformed is simply not wanted (schema drift never crashes the crawl)."""
    if not isinstance(entry, dict):
        return False
    duration = entry.get("durationSeconds")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool):
        return False
    if entry.get("rankedType") in RANKED_DUELS:
        return False  # ranked 1v1 / 2v2 are listed with mode "Free For All" but aren't FFA lobbies
    return entry.get("type") == "Public" and entry.get("mode") == "Free For All" and duration >= MIN_DURATION_S


def is_wellformed_summary(entry: Any) -> bool:
    return (
        isinstance(entry, dict)
        and isinstance(entry.get("gameId"), str)
        and isinstance(entry.get("type"), str)
        and isinstance(entry.get("mode"), str)
        and isinstance(entry.get("durationSeconds"), (int, float))
    )


def modifiers(config: dict) -> list[str]:
    """Names of the active modifiers in a record's config; empty means clean."""
    found: list[str] = []
    if config.get("gameMapSize", "Normal") != "Normal":
        found.append("gameMapSize")
    for key in ("randomSpawn", "infiniteGold", "infiniteTroops", "instantBuild", "disableAlliances"):
        if config.get(key):
            found.append(key)
    if config.get("disabledUnits"):
        found.append("disabledUnits")
    doomsday = config.get("doomsdayClock")
    if isinstance(doomsday, dict) and doomsday.get("enabled"):
        found.append("doomsdayClock")
    if config.get("goldMultiplier") not in (None, 1):
        found.append("goldMultiplier")
    if config.get("startingGold"):
        found.append("startingGold")
    cheats = config.get("hostCheats")
    if isinstance(cheats, dict) and any(v and not (k == "goldMultiplier" and v == 1) for k, v in cheats.items()):
        found.append("hostCheats")
    public = config.get("publicGameModifiers")
    if isinstance(public, dict):
        for key, value in public.items():
            if value:
                found.append(f"publicGameModifiers.{key}")
    return found
