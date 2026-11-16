"""The observation/action layout as the sim exports it (HELLO or `sim.mjs layout`).

Python never hard-codes layout numbers: everything here comes from that JSON
(roadmap §3.5). Field arrays are numpy views over the wire bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

DTYPES = {
    "u8": np.dtype("u1"),
    "u16": np.dtype("<u2"),
    "i16": np.dtype("<i2"),
    "i32": np.dtype("<i4"),
    "f16": np.dtype("<f2"),
    "f32": np.dtype("<f4"),
}


@dataclass(frozen=True)
class FieldSpec:
    name: str
    dtype: np.dtype
    shape: tuple[int, ...]
    scope: str

    @property
    def size(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1

    @property
    def nbytes(self) -> int:
        return self.size * self.dtype.itemsize


@dataclass(frozen=True)
class Layout:
    version: int
    sizes: dict[str, Any]
    fields: tuple[FieldSpec, ...]
    action_fields: tuple[str, ...]
    action_types: tuple[str, ...]
    type_args: dict[str, tuple[str, ...]]
    entity_table: dict[str, str]
    build_types: tuple[str, ...]
    planes: tuple[dict, ...]
    features: dict[str, tuple[str, ...]]
    raw: dict = field(repr=False, compare=False, default_factory=dict)

    @staticmethod
    def from_json(d: dict) -> Layout:
        return Layout(
            version=int(d["layoutVersion"]),
            sizes=dict(d["sizes"]),
            fields=tuple(FieldSpec(f["name"], DTYPES[f["dtype"]], tuple(f["shape"]), f["scope"]) for f in d["fields"]),
            action_fields=tuple(d["actionFields"]),
            action_types=tuple(d["actionTypes"]),
            type_args={k: tuple(v) for k, v in d["typeArgs"].items()},
            entity_table=dict(d["entityTable"]),
            build_types=tuple(d["buildTypes"]),
            planes=tuple(d["planes"]),
            features={k: tuple(v) for k, v in d["features"].items()},
            raw=d,
        )

    @property
    def K(self) -> int:
        return int(self.sizes["K"])

    @property
    def A(self) -> int:
        return int(self.sizes["A"])

    def field(self, name: str) -> FieldSpec:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(name)

    def af(self, name: str) -> int:
        """Index of an action field (type, player, ...)."""
        return self.action_fields.index(name)

    def at(self, name: str) -> int:
        """Index of an action type (STOP, ATTACK, ...)."""
        return self.action_types.index(name)

    def feat(self, table: str, name: str) -> int:
        return self.features[table].index(name)

    def obs_nbytes(self, rows: int) -> int:
        return sum(f.nbytes for f in self.fields) * rows

    def parse_obs(self, buf: memoryview | bytes, offset: int, rows: int) -> tuple[dict[str, np.ndarray], int]:
        """Field arrays [rows, *shape] over `buf` starting at `offset`; returns them and the end offset."""
        out: dict[str, np.ndarray] = {}
        for f in self.fields:
            n = f.size * rows
            out[f.name] = np.frombuffer(buf, dtype=f.dtype, count=n, offset=offset).reshape((rows, *f.shape))
            offset += n * f.dtype.itemsize
        return out, offset
