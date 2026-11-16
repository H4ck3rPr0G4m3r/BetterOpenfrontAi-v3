"""Bridge protocol v3 framing. Mirror of sim/src/bridge/Protocol.ts."""

from __future__ import annotations

import struct

PROTOCOL = 3

RESET, STEP, CLOSE, LOAD, NEXT = 0x01, 0x02, 0x03, 0x04, 0x05
HELLO, ROWS, SAMPLES, OK, ERROR = 0x10, 0x11, 0x12, 0x14, 0xFF

FLAG_VALID, FLAG_DONE, FLAG_NEXT, FLAG_FRESH = 1, 2, 4, 8
TERMINALS = ("none", "win", "death", "tick_cap", "sim_error", "aborted")
STATS = ("tiles", "troops", "kills", "placement", "ticks")

HEADER = struct.Struct("<IIB")  # payloadLen, seq, msgType


def frame(seq: int, msg_type: int, body: bytes = b"") -> bytes:
    return HEADER.pack(5 + len(body), seq, msg_type) + body
