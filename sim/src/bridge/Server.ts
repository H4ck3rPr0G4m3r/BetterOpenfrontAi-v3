/**
 * Node half of the bridge (roadmap §3.4). One client at a time; requests are
 * handled strictly in order. Role `replay` streams teacher samples from a
 * Sampler; role `env` drives policy-controlled games (GameEnv host). A
 * protocol violation sends ERROR and drops the connection: a silent desync
 * would corrupt training data.
 */
import net from "net";
import { A, K, LAYOUT_VERSION, layoutJson } from "../obs/Layout";
import type { ObsBuffer } from "../obs/ObsBuffer";
import type { Sampler, SampleItem } from "../replay/Sampler";
import {
  concat, decodeJson, decodePayload, decodeStepBody, encodeError, encodeFrame, encodeJson, ERR, FrameReader,
  jsonTrailer, MSG, PROTOCOL, ProtocolError, u32,
} from "./Protocol";

export interface EnvRows {
  buf: ObsBuffer;
  reward: Float32Array;
  flags: Uint8Array;
  stats: Float32Array; // rows * 5
  terminal: Uint8Array;
  extra: unknown;
}

/** What the env role drives (GameEnv pool, M2 Task 11). */
export interface EnvHost {
  readonly rows: number;
  readonly games: number;
  readonly seats: number;
  reset(specs: unknown[]): Promise<EnvRows>;
  step(actions: Int32Array): Promise<EnvRows>;
}

export interface ServerOptions {
  role: "replay" | "env";
  sampler?: Sampler;
  env?: EnvHost;
  host?: string;
  port?: number;
  log?: (msg: string) => void;
}

export class BridgeServer {
  private server: net.Server | null = null;
  private client: net.Socket | null = null;
  private expectedSeq = 1;
  private chain: Promise<void> = Promise.resolve();
  readonly rows: number;
  readonly games: number;
  readonly seats: number;

  constructor(private readonly opts: ServerOptions) {
    if (opts.role === "replay") {
      if (!opts.sampler) throw new Error("replay role needs a sampler");
      this.rows = opts.sampler.rows;
      this.games = opts.sampler.gamesInFlight;
      this.seats = opts.sampler.seatsPerGame;
    } else {
      if (!opts.env) throw new Error("env role needs an env host");
      this.rows = opts.env.rows;
      this.games = opts.env.games;
      this.seats = opts.env.seats;
    }
  }

  async listen(): Promise<number> {
    this.server = net.createServer((s) => this.onConnection(s));
    await new Promise<void>((resolve, reject) => {
      this.server!.once("error", reject);
      this.server!.listen(this.opts.port ?? 0, this.opts.host ?? "127.0.0.1", () => resolve());
    });
    return (this.server.address() as net.AddressInfo).port;
  }

  async close(): Promise<void> {
    this.client?.destroy();
    this.client = null;
    const s = this.server;
    this.server = null;
    if (s) await new Promise<void>((r) => s.close(() => r()));
  }

  private onConnection(socket: net.Socket): void {
    if (this.client !== null) {
      socket.destroy();
      return;
    }
    socket.setNoDelay(true);
    this.client = socket;
    this.expectedSeq = 1;
    const reader = new FrameReader();
    socket.on("data", (chunk: Buffer) => {
      const bytes = new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength).slice();
      this.chain = this.chain.then(async () => {
        if (this.client !== socket) return;
        let payloads: Uint8Array[];
        try {
          payloads = reader.push(bytes);
        } catch (e) {
          this.fail(socket, this.expectedSeq, e);
          return;
        }
        for (const p of payloads) {
          await this.onFrame(socket, p);
          if (this.client !== socket) return;
        }
      });
    });
    const drop = () => {
      if (this.client === socket) this.client = null;
    };
    socket.on("close", drop);
    socket.on("error", drop);
    const hello = {
      protocol: PROTOCOL, layoutVersion: LAYOUT_VERSION, role: this.opts.role, rows: this.rows, games: this.games,
      seats: this.seats, layout: layoutJson(),
    };
    socket.write(encodeFrame(0, MSG.HELLO, encodeJson(hello)));
  }

  private async onFrame(socket: net.Socket, payload: Uint8Array): Promise<void> {
    let seq = this.expectedSeq;
    try {
      const f = decodePayload(payload);
      seq = f.seq;
      if (f.seq !== this.expectedSeq) throw new ProtocolError(ERR.BAD_SEQUENCE, `expected seq ${this.expectedSeq}, got ${f.seq}`);
      this.expectedSeq = (this.expectedSeq + 1) >>> 0 || 1;
      const reply = await this.handle(f.type, f.body);
      socket.write(encodeFrame(seq, reply.type, reply.body));
      if (f.type === MSG.CLOSE) {
        socket.end();
        this.client = null;
      }
    } catch (e) {
      this.fail(socket, seq, e);
    }
  }

  private async handle(type: number, body: Uint8Array): Promise<{ type: number; body: Uint8Array }> {
    const { role } = this.opts;
    const need = (r: "replay" | "env") => {
      if (role !== r) throw new ProtocolError(ERR.WRONG_ROLE, `message 0x${type.toString(16)} needs role ${r}, server is ${role}`);
    };
    switch (type) {
      case MSG.CLOSE:
        return { type: MSG.OK, body: encodeJson({ closed: true }) };
      case MSG.LOAD: {
        need("replay");
        const msg = decodeJson(body, "LOAD");
        if (!Array.isArray(msg?.records)) throw new ProtocolError(ERR.BAD_MESSAGE, "LOAD needs records[]");
        const items: SampleItem[] = msg.records.map((r: any, i: number) => {
          if (typeof r?.path !== "string" || !Array.isArray(r.teachers)) throw new ProtocolError(ERR.BAD_MESSAGE, `LOAD record ${i} needs path and teachers`);
          return {
            path: r.path,
            teachers: r.teachers.map((t: any) =>
              typeof t === "string" ? { clientID: t, weight: 1, value: 0.5 } : { clientID: String(t.clientID), weight: Number(t.weight ?? 1), value: Number(t.value ?? 0.5) },
            ),
          };
        });
        this.opts.sampler!.load(items);
        return { type: MSG.OK, body: encodeJson({ queued: this.opts.sampler!.queued() }) };
      }
      case MSG.NEXT: {
        need("replay");
        const s = this.opts.sampler!;
        const b = await s.next();
        const stats = { ...s.stats, queued: s.queued(), active: s.active() };
        return {
          type: MSG.SAMPLES,
          body: concat([u32(s.rows), b.buf.bytes(), b.labels, b.weight, b.value, b.flags, ...jsonTrailer(stats)]),
        };
      }
      case MSG.RESET: {
        need("env");
        const msg = decodeJson(body, "RESET");
        if (!Array.isArray(msg?.games)) throw new ProtocolError(ERR.BAD_MESSAGE, "RESET needs games[]");
        return { type: MSG.ROWS, body: rowsBody(await this.opts.env!.reset(msg.games), this.rows) };
      }
      case MSG.STEP: {
        need("env");
        const actions = decodeStepBody(body, this.rows, K * A);
        return { type: MSG.ROWS, body: rowsBody(await this.opts.env!.step(actions), this.rows) };
      }
      default:
        throw new ProtocolError(ERR.BAD_MESSAGE, `unknown message type 0x${type.toString(16)}`);
    }
  }

  private fail(socket: net.Socket, seq: number, e: unknown): void {
    const code = e instanceof ProtocolError ? e.code : ERR.INTERNAL;
    const msg = e instanceof Error ? (e instanceof ProtocolError ? e.message : `${e.message}\n${e.stack ?? ""}`) : String(e);
    this.opts.log?.(`bridge error: ${msg}`);
    socket.end(encodeFrame(seq, MSG.ERROR, encodeError(code, msg)));
    if (this.client === socket) this.client = null;
  }
}

function rowsBody(r: EnvRows, rows: number): Uint8Array {
  return concat([u32(rows), r.buf.bytes(), r.reward, r.flags, r.stats, r.terminal, ...jsonTrailer(r.extra ?? {})]);
}
