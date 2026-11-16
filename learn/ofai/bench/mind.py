"""AI mind sidecar (M3 Task 9, Python half): one gzip JSONL line per AI seat per
decision step, read by the app's overlay next to the replay.

Line: {"turn", "clientID", "value", "actions": [{"text", "tile": [x, y] | null, "p"}],
       "top": [{"head": "type", "options": [[label, p], ...]}]}
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from ofai.layout import Layout


def describe_action(lay: Layout, a: np.ndarray) -> str:
    """Words for one action: "Attack token 3 35%", "Build City", "Boat 20%"..."""
    f = lambda name: int(a[lay.af(name)])  # noqa: E731
    t = f("type")
    if t < 0:
        return ""
    name = lay.action_types[t]
    words = name.replace("_", " ").capitalize()
    parts = [words]
    if f("build") >= 0:
        parts.append(lay.build_types[f("build")])
    if f("player") >= 0:
        p = f("player")
        parts.append("unowned land" if p == lay.sizes["UNOWNED_TOKEN"] else "others" if p == lay.sizes["REST_TOKEN"]
                     else "self" if p == 0 else f"player {p}")
    if f("entity") >= 0:
        parts.append(f"#{f('entity')}")
    if f("troops") >= 0:
        parts.append(f"{(f('troops') + 1) * 5}%")
    return " ".join(parts)


def action_tile(lay: Layout, a: np.ndarray, game_meta: np.ndarray, win_meta: np.ndarray) -> list[int] | None:
    """Tile an action points at, through the sim's geometry (near: window cell; far: minimap cell + fine)."""
    f = lambda name: int(a[lay.af(name)])  # noqa: E731
    mode = f("locMode")
    gm = list(lay.raw["gameMetaFields"])
    wm = list(lay.raw["winMetaFields"])
    if mode == 0 and f("window") >= 0 and f("cell") >= 0:
        w = win_meta[f("window")]
        tpc = float(w[wm.index("tpc")])
        win = int(lay.sizes["WIN"])
        cell = f("cell")
        return [int(w[wm.index("x0")] + (cell % win) * tpc + tpc // 2), int(w[wm.index("y0")] + (cell // win) * tpc + tpc // 2)]
    if mode == 1 and f("mmCell") >= 0:
        mm, fine_n = int(lay.sizes["MM"]), int(lay.sizes["FINE"])
        tpc = float(game_meta[gm.index("mm_tpc")])
        cx = f("mmCell") % mm - game_meta[gm.index("off_x")]
        cy = f("mmCell") // mm - game_meta[gm.index("off_y")]
        x0, y0 = cx * tpc, cy * tpc
        x1 = min(float(game_meta[gm.index("map_w")]), (cx + 1) * tpc)
        y1 = min(float(game_meta[gm.index("map_h")]), (cy + 1) * tpc)
        fine = max(0, f("fine"))
        fx, fy = fine % fine_n, fine // fine_n
        return [int(min(x1 - 1, x0 + ((fx + 0.5) * (x1 - x0)) // fine_n)), int(min(y1 - 1, y0 + ((fy + 0.5) * (y1 - y0)) // fine_n))]
    return None


def head_labels(lay: Layout, head: str) -> list[str]:
    if head == "type":
        return [t.replace("_", " ").capitalize() for t in lay.action_types]
    if head == "build":
        return list(lay.build_types)
    return []


class MindWriter:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = self.path.with_name(self.path.name + ".tmp")
        self._f = gzip.open(self._tmp, "wt", encoding="utf-8")

    def write(self, turn: int, client_id: str, value: float, actions: list[dict], top: list[dict]) -> None:
        self._f.write(json.dumps({"turn": turn, "clientID": client_id, "value": round(float(value), 4),
                                  "actions": actions, "top": top}) + "\n")

    def write_step(self, lay: Layout, turn: int, client_id: str, value: float, acts: np.ndarray,
                   top: list[dict], game_meta: np.ndarray, win_meta: np.ndarray) -> None:
        actions = []
        for k, a in enumerate(acts):
            if int(a[lay.af("type")]) < 0 or int(a[lay.af("type")]) == lay.at("STOP"):
                break
            p = None
            if k < len(top) and "type" in top[k]:
                idx, probs = top[k]["type"]
                p = next((round(pp, 4) for ii, pp in zip(idx, probs) if ii == int(a[lay.af("type")])), None)
            actions.append({"text": describe_action(lay, a), "tile": action_tile(lay, a, game_meta, win_meta), "p": p})
        heads = []
        if top:
            for head, (idx, probs) in top[0].items():
                labels = head_labels(lay, head)
                heads.append({"head": head, "options": [[labels[i] if i < len(labels) else str(i), round(p, 4)]
                                                        for i, p in zip(idx, probs)]})
        self.write(turn, client_id, value, actions, heads)

    def close(self) -> None:
        self._f.close()
        self._tmp.replace(self.path)

    def discard(self) -> None:
        """Drops the sidecar of a game that never ran."""
        self._f.close()
        self._tmp.unlink(missing_ok=True)


def read_mind(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
