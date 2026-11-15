from ofai.data.filters import MIN_DURATION_S, modifiers, wanted_summary


def test_wanted_summary_public_ffa_long():
    assert wanted_summary({"type": "Public", "mode": "Free For All", "durationSeconds": 300})
    assert MIN_DURATION_S == 300


def test_rejects_private():
    assert not wanted_summary({"type": "Private", "mode": "Free For All", "durationSeconds": 900})


def test_rejects_team():
    assert not wanted_summary({"type": "Public", "mode": "Team", "durationSeconds": 900})


def test_rejects_short():
    assert not wanted_summary({"type": "Public", "mode": "Free For All", "durationSeconds": 299})


def test_malformed_history_entry_is_skipped():
    assert not wanted_summary({})
    assert not wanted_summary({"type": "Public", "mode": "Free For All", "durationSeconds": "long"})
    assert not wanted_summary(None)  # type: ignore[arg-type]


CLEAN = {
    "gameMapSize": "Normal",
    "randomSpawn": False,
    "infiniteGold": False,
    "infiniteTroops": False,
    "instantBuild": False,
    "disabledUnits": [],
    "doomsdayClock": {"enabled": False},
}


def test_clean_config_has_no_modifiers():
    assert modifiers(CLEAN) == []
    assert modifiers({**CLEAN, "goldMultiplier": None, "startingGold": None}) == []
    assert modifiers({**CLEAN, "goldMultiplier": 1, "publicGameModifiers": {"startingGold": 0}}) == []


def test_detects_modifiers():
    cfg = {
        "gameMapSize": "Compact",
        "randomSpawn": True,
        "infiniteGold": True,
        "instantBuild": True,
        "disabledUnits": ["SAM Launcher"],
        "doomsdayClock": {"enabled": True},
        "goldMultiplier": 2,
        "publicGameModifiers": {"isCrowded": True, "startingGold": 0},
    }
    assert set(modifiers(cfg)) == {
        "gameMapSize",
        "randomSpawn",
        "infiniteGold",
        "instantBuild",
        "disabledUnits",
        "doomsdayClock",
        "goldMultiplier",
        "publicGameModifiers.isCrowded",
    }


def test_detects_cheats_and_starting_gold():
    cfg = {**CLEAN, "startingGold": 5_000_000, "hostCheats": {"infiniteTroops": True}, "disableAlliances": True}
    assert set(modifiers(cfg)) == {"startingGold", "hostCheats", "disableAlliances"}
