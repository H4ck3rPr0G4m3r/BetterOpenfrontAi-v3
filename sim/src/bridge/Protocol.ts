/**
 * Bridge protocol v3 (roadmap §3.4). Mirror: learn/ofai/bridge/protocol.py.
 *
 * Framing, both directions over local TCP, little endian:
 *   [u32 payloadLen][u32 seq][u8 msgType][body]
 * The client numbers requests from 1; replies echo the seq; HELLO is seq 0.
 * A seq mismatch is fatal (ERROR, then disconnect).
 *
 * Bodies:
 *   HELLO   JSON {protocol: 3, layoutVersion, role, rows, games, seats, layout}
 *   LOAD    JSON {records: [{path, teachers: [{clientID, weight, value}]}]}  -> OK
 *   NEXT    empty                                                           -> SAMPLES
 *   RESET   JSON {games: GameSpec[]}                                        -> ROWS
 *   STEP    [u32 rows][i32 rows*K*A]                                        -> ROWS
 *   CLOSE   empty                                                           -> OK, then disconnect
 *   OK      JSON (small status object)
 *   SAMPLES [u32 rows][ObsBlock][i32 labels rows*K*A][f32 weight rows][f32 value rows][u8 flags rows][u32 n][JSON stats]
 *   ROWS    [u32 rows][ObsBlock][f32 reward rows][u8 flags rows][f32 stats rows*5][u8 terminal rows][u32 n][JSON extra]
 *   ERROR   [u8 code][utf8 message]
 * ObsBlock = every layout field, rows x shape, field after field in layout order.
 */

export const PROTOCOL = 3;

export const MSG = {
  RESET: 0x01,
  STEP: 0x02,
  CLOSE: 0x03,
  LOAD: 0x04,
  NEXT: 0x05,
  HELLO: 0x10,
  ROWS: 0x11,
  SAMPLES: 0x12,
  OK: 0x14,
  ERROR: 0xff,
} as const;

export const ERR = { BAD_SEQUENCE: 1, BAD_LENGTH: 2, BAD_MESSAGE: 3, WRONG_ROLE: 4, INTERNAL: 6 } as const;

export const TERMINAL = { none: 0, win: 1, death: 2, tick_cap: 3, sim_error: 4, aborted: 5 } as const;
export type Terminal = keyof typeof TERMINAL;

export const MAX_FRAME_BYTES = 1024 * 1024 * 1024;
export const STATS_PER_ROW = 5;

export class ProtocolError extends Error {
  constructor(readonly code: number, message: string) {
    super(`protocol: ${message}`);
  }
}

const enc = new TextEncoder();
const dec = new TextDecoder("utf-8", { fatal: true });

export function encodeFrame(seq: number, type: number, body: Uint8Array): Uint8Array {
  const out = new Uint8Array(9 + body.length);
  const v = new DataView(out.buffer);
  v.setUint32(0, 5 + body.length, true);
  v.setUint32(4, seq >>> 0, true);
  out[8] = type;
  out.set(body, 9);
  return out;
}

export function decodePayload(p: Uint8Array): { seq: number; type: number; body: Uint8Array } {
  if (p.length < 5) throw new ProtocolError(ERR.BAD_LENGTH, `payload too short: ${p.length}`);
  return { seq: new DataView(p.buffer, p.byteOffset, 4).getUint32(0, true), type: p[4], body: p.subarray(5) };
}

/** Splits a byte stream into whole payloads (everything after the length prefix). */
export class FrameReader {
  private buf = new Uint8Array(0);

  push(chunk: Uint8Array): Uint8Array[] {
    const merged = new Uint8Array(this.buf.length + chunk.length);
    merged.set(this.buf);
    merged.set(chunk, this.buf.length);
    this.buf = merged;
    const frames: Uint8Array[] = [];
    let off = 0;
    while (this.buf.length - off >= 4) {
      const len = new DataView(this.buf.buffer, this.buf.byteOffset + off, 4).getUint32(0, true);
      if (len < 5 || len > MAX_FRAME_BYTES) throw new ProtocolError(ERR.BAD_LENGTH, `bad frame length ${len}`);
      if (this.buf.length - off < 4 + len) break;
      frames.push(this.buf.slice(off + 4, off + 4 + len));
      off += 4 + len;
    }
    this.buf = this.buf.slice(off);
    return frames;
  }
}

export function encodeJson(v: unknown): Uint8Array {
  return enc.encode(JSON.stringify(v));
}

export function decodeJson(body: Uint8Array, what: string): any {
  try {
    return JSON.parse(dec.decode(body));
  } catch {
    throw new ProtocolError(ERR.BAD_MESSAGE, `${what} is not valid JSON`);
  }
}

export function encodeError(code: number, message: string): Uint8Array {
  const text = enc.encode(message.slice(0, 4000));
  const out = new Uint8Array(1 + text.length);
  out[0] = code;
  out.set(text, 1);
  return out;
}

/** Concatenates byte parts (typed arrays are copied as raw little-endian bytes). */
export function concat(parts: (Uint8Array | ArrayBufferView)[]): Uint8Array {
  const bytes = parts.map((p) => (p instanceof Uint8Array ? p : new Uint8Array(p.buffer, p.byteOffset, p.byteLength)));
  const out = new Uint8Array(bytes.reduce((a, b) => a + b.length, 0));
  let o = 0;
  for (const b of bytes) {
    out.set(b, o);
    o += b.length;
  }
  return out;
}

export function u32(n: number): Uint8Array {
  const out = new Uint8Array(4);
  new DataView(out.buffer).setUint32(0, n >>> 0, true);
  return out;
}

export function jsonTrailer(v: unknown): Uint8Array[] {
  const j = encodeJson(v);
  return [u32(j.length), j];
}

/** STEP body: [u32 rows][i32 rows*K*A]. */
export function decodeStepBody(body: Uint8Array, rows: number, perRow: number): Int32Array {
  if (body.length < 4) throw new ProtocolError(ERR.BAD_LENGTH, "STEP too short");
  const n = new DataView(body.buffer, body.byteOffset, 4).getUint32(0, true);
  if (n !== rows) throw new ProtocolError(ERR.BAD_MESSAGE, `STEP covers ${n} rows, server has ${rows}`);
  if (body.length !== 4 + rows * perRow * 4) throw new ProtocolError(ERR.BAD_LENGTH, "STEP length mismatch");
  // Copy: the body is not 4-byte aligned in general.
  return new Int32Array(body.slice(4).buffer);
}
