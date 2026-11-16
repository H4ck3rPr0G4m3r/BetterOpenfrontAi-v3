"""Blocking TCP client for one sim worker (replay or env role)."""

from __future__ import annotations

import json
import socket
import struct
from typing import Any

import numpy as np

from ofai.bridge import protocol as P
from ofai.layout import Layout

Batch = dict[str, Any]


class BridgeError(RuntimeError):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"sim error {code}: {message}")
        self.code = code


class BridgeClient:
    def __init__(self, host: str, port: int, timeout: float | None = 600.0) -> None:
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.seq = 0
        self._pending: int | None = None
        seq, msg, body = self._read()
        if msg != P.HELLO or seq != 0:
            raise BridgeError(0, f"expected HELLO, got 0x{msg:x}")
        self.hello: dict = json.loads(bytes(body))
        if self.hello.get("protocol") != P.PROTOCOL:
            raise BridgeError(0, f"protocol {self.hello.get('protocol')} != {P.PROTOCOL}")
        self.layout = Layout.from_json(self.hello["layout"])
        self.rows: int = int(self.hello["rows"])
        self.games: int = int(self.hello["games"])
        self.seats: int = int(self.hello["seats"])
        self.role: str = self.hello["role"]

    # -- requests ---------------------------------------------------------------

    def load(self, items: list[dict]) -> dict:
        return json.loads(bytes(self._call(P.LOAD, json.dumps({"records": items}).encode(), P.OK)))

    def next(self) -> Batch:
        self.send_next()
        return self.recv()

    def reset(self, games: list[dict | None]) -> Batch:
        self.send_reset(games)
        return self.recv()

    def step(self, actions: np.ndarray) -> Batch:
        self.send_step(actions)
        return self.recv()

    def close(self) -> None:
        try:
            if self._pending is None:
                self._call(P.CLOSE, b"", P.OK)
        except (OSError, BridgeError):
            pass
        finally:
            self.sock.close()

    # Split send/recv so a pool can keep every worker busy at once.

    def send_next(self) -> None:
        self._send(P.NEXT, b"", P.SAMPLES)

    def send_reset(self, games: list[dict | None]) -> None:
        self._send(P.RESET, json.dumps({"games": games}).encode(), P.ROWS)

    def send_step(self, actions: np.ndarray) -> None:
        a = np.ascontiguousarray(actions, dtype="<i4").reshape(self.rows, self.layout.K, self.layout.A)
        self._send(P.STEP, struct.pack("<I", self.rows) + a.tobytes(), P.ROWS)

    def recv(self) -> Batch:
        kind = self._pending
        return self._parse(kind, self._recv_reply())

    # -- framing ----------------------------------------------------------------

    def _call(self, msg: int, body: bytes, want: int) -> memoryview:
        self._send(msg, body, want)
        return self._recv_reply()

    def _send(self, msg: int, body: bytes, want: int) -> None:
        if self._pending is not None:
            raise BridgeError(0, "a request is already in flight")
        self.seq += 1
        self.sock.sendall(P.frame(self.seq, msg, body))
        self._pending = want

    def _recv_reply(self) -> memoryview:
        want = self._pending
        self._pending = None
        seq, msg, body = self._read()
        if msg == P.ERROR:
            raise BridgeError(body[0], bytes(body[1:]).decode("utf-8", "replace"))
        if seq != self.seq:
            raise BridgeError(0, f"reply seq {seq} != request seq {self.seq}")
        if msg != want:
            raise BridgeError(0, f"expected 0x{want:x}, got 0x{msg:x}")
        return body

    def _read(self) -> tuple[int, int, memoryview]:
        length, seq, msg = P.HEADER.unpack(self._exact(9))
        return seq, msg, memoryview(self._exact(length - 5))

    def _exact(self, n: int) -> bytearray:
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            k = self.sock.recv_into(view[got:], n - got)
            if k == 0:
                raise ConnectionError("sim worker closed the connection")
            got += k
        return buf

    # -- bodies -----------------------------------------------------------------

    def _parse(self, kind: int | None, body: memoryview) -> Batch:
        rows = struct.unpack_from("<I", body, 0)[0]
        obs, o = self.layout.parse_obs(body, 4, rows)
        out: Batch = dict(obs)
        K, A = self.layout.K, self.layout.A

        def take(dtype: str, count: int, shape: tuple[int, ...]) -> np.ndarray:
            nonlocal o
            arr = np.frombuffer(body, dtype=dtype, count=count, offset=o).reshape(shape)
            o += count * np.dtype(dtype).itemsize
            return arr

        if kind == P.SAMPLES:
            out["labels"] = take("<i4", rows * K * A, (rows, K, A))
            out["weight"] = take("<f4", rows, (rows,))
            out["value"] = take("<f4", rows, (rows,))
            out["flags"] = take("u1", rows, (rows,))
        else:
            out["reward"] = take("<f4", rows, (rows,))
            out["flags"] = take("u1", rows, (rows,))
            out["stats"] = take("<f4", rows * len(P.STATS), (rows, len(P.STATS)))
            out["terminal"] = take("u1", rows, (rows,))
        n = struct.unpack_from("<I", body, o)[0]
        out["info"] = json.loads(bytes(body[o + 4 : o + 4 + n]))
        return out
