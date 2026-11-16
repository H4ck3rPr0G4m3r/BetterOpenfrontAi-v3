import net from "net";
import { decodePayload, encodeFrame, ERR, FrameReader, MSG } from "../../src/bridge/Protocol";
import { BridgeServer } from "../../src/bridge/Server";
import { Sampler } from "../../src/replay/Sampler";
import { ENGINE_ROOT } from "../helpers";

test("frames round trip", () => {
  const body = new Uint8Array([1, 2, 3, 250]);
  const frame = encodeFrame(7, MSG.NEXT, body);
  const reader = new FrameReader();
  // Split the frame across chunks, with a second frame glued to the end.
  const both = new Uint8Array([...frame, ...encodeFrame(8, MSG.CLOSE, new Uint8Array(0))]);
  const out = [...reader.push(both.subarray(0, 3)), ...reader.push(both.subarray(3, 11)), ...reader.push(both.subarray(11))];
  expect(out).toHaveLength(2);
  const a = decodePayload(out[0]);
  expect(a.seq).toBe(7);
  expect(a.type).toBe(MSG.NEXT);
  expect([...a.body]).toEqual([...body]);
  expect(decodePayload(out[1]).type).toBe(MSG.CLOSE);
});

async function talk(port: number, frames: Uint8Array[]): Promise<{ seq: number; type: number; body: Uint8Array }[]> {
  return new Promise((resolve, reject) => {
    const sock = net.connect(port, "127.0.0.1");
    const reader = new FrameReader();
    const got: { seq: number; type: number; body: Uint8Array }[] = [];
    sock.on("data", (c) => {
      for (const p of reader.push(new Uint8Array(c))) got.push(decodePayload(p));
      if (got.length === 1) for (const f of frames) sock.write(f);
    });
    sock.on("close", () => resolve(got));
    sock.on("error", reject);
    setTimeout(() => sock.destroy(), 3000);
  });
}

test("seq mismatch is fatal", async () => {
  const server = new BridgeServer({ role: "replay", sampler: new Sampler(ENGINE_ROOT, 1, 1) });
  const port = await server.listen();
  const got = await talk(port, [encodeFrame(2, MSG.NEXT, new Uint8Array(0))]);
  await server.close();
  expect(got[0].type).toBe(MSG.HELLO);
  const hello = JSON.parse(new TextDecoder().decode(got[0].body));
  expect(hello.protocol).toBe(3);
  expect(hello.layout.fields.length).toBeGreaterThan(10);
  expect(got[1].type).toBe(MSG.ERROR);
  expect(got[1].body[0]).toBe(ERR.BAD_SEQUENCE);
});

test("wrong role message is an error", async () => {
  const server = new BridgeServer({ role: "replay", sampler: new Sampler(ENGINE_ROOT, 1, 1) });
  const port = await server.listen();
  const got = await talk(port, [encodeFrame(1, MSG.STEP, new Uint8Array(4))]);
  await server.close();
  expect(got[1].type).toBe(MSG.ERROR);
  expect(got[1].body[0]).toBe(ERR.WRONG_ROLE);
});
