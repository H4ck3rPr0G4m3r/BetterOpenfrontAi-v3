/**
 * `dump-obs`: raw observation rows from a record plus the CPU reference
 * planes for each (M3 Task 1). The Python GPU expansion must reproduce
 * `mm_ref.bin` and `win_ref.bin` exactly from `obs.bin`.
 *
 * Files: obs.bin (ObsBuffer bytes, rows field after field), mm_ref.bin
 * (f32 [rows, P, MM, MM]), win_ref.bin (f32 [rows, NUM_WINDOWS, P, WIN, WIN]),
 * header.json ({rows, steps, layoutVersion}).
 */
import fs from "fs";
import path from "path";
import { GAME_META_FIELDS, LAYOUT_VERSION, MM, NUM_WINDOWS, PLANES, WIN, WIN_META, WIN_META_FIELDS } from "../Layout";
import { ObsBuffer } from "../ObsBuffer";
import { FLAG, Sampler } from "../../replay/Sampler";
import { readRecord } from "../../record/Record";
import { expandView, minimapView, windowView } from "./expandPlanes";

const GM = Object.fromEntries(GAME_META_FIELDS.map((n, i) => [n, i])) as Record<(typeof GAME_META_FIELDS)[number], number>;
const WM = Object.fromEntries(WIN_META_FIELDS.map((n, i) => [n, i])) as Record<(typeof WIN_META_FIELDS)[number], number>;

export async function dumpObs(recordPath: string, engineRoot: string, outDir: string, steps: number, every: number, seats = 4): Promise<{ rows: number }> {
  const rec = readRecord(recordPath);
  const counts = new Map<string, number>();
  for (const t of rec.turns) for (const i of t.intents) counts.set(i.clientID, (counts.get(i.clientID) ?? 0) + 1);
  const teachers = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, seats).map(([clientID]) => ({ clientID, weight: 1, value: 0.5 }));
  const sampler = new Sampler(engineRoot, 1, seats);
  sampler.load([{ record: rec, teachers }]);

  const picked: { step: number; row: number }[] = [];
  const out = new ObsBuffer(steps * seats);
  let n = 0;
  for (let step = 0; picked.length < steps * seats && step < steps * every * 4; step++) {
    const b = await sampler.next();
    if (sampler.active() === 0 && ![...b.flags].some((f) => f & FLAG.VALID)) break;
    if ((step + 1) % every !== 0) continue;
    for (let r = 0; r < seats; r++) {
      if (!(b.flags[r] & FLAG.VALID) || n >= out.rows) continue;
      out.copyRow(r, n++, b.buf);
      picked.push({ step, row: r });
    }
  }

  const P = PLANES.length;
  const mm = new Float32Array(n * P * MM * MM);
  const win = new Float32Array(n * NUM_WINDOWS * P * WIN * WIN);
  for (let r = 0; r < n; r++) {
    const meta = out.view<Float32Array>("game_meta", r);
    const relation = out.view<Uint8Array>("relation", r);
    const density = out.view<Uint16Array>("density", r);
    const units = out.view<Int16Array>("units", r);
    const samRange = meta[GM.sam_range];
    const mmView = minimapView(out.view<Uint16Array>("mm_owner", r), out.view<Uint8Array>("mm_terrain", r), MM, meta[GM.mm_tpc], meta[GM.off_x], meta[GM.off_y]);
    mm.set(expandView(mmView, relation, density, units, samRange), r * P * MM * MM);
    const wm = out.view<Float32Array>("win_meta", r);
    const wOwner = out.view<Uint16Array>("win_owner", r);
    const wTerrain = out.view<Uint8Array>("win_terrain", r);
    for (let w = 0; w < NUM_WINDOWS; w++) {
      const m = wm.subarray(w * WIN_META, (w + 1) * WIN_META);
      if (!m[WM.valid]) continue;
      const view = windowView(wOwner.subarray(w * WIN * WIN, (w + 1) * WIN * WIN), wTerrain.subarray(w * WIN * WIN, (w + 1) * WIN * WIN), WIN, m[WM.x0], m[WM.y0], m[WM.tpc]);
      win.set(expandView(view, relation, density, units, samRange), ((r * NUM_WINDOWS + w) * P) * WIN * WIN);
    }
  }

  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, "obs.bin"), out.bytes(n));
  fs.writeFileSync(path.join(outDir, "mm_ref.bin"), new Uint8Array(mm.buffer));
  fs.writeFileSync(path.join(outDir, "win_ref.bin"), new Uint8Array(win.buffer));
  fs.writeFileSync(path.join(outDir, "header.json"), JSON.stringify({ rows: n, picked, layoutVersion: LAYOUT_VERSION, record: path.basename(recordPath) }));
  return { rows: n };
}
