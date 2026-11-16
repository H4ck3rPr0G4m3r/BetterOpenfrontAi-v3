import {
  clampWindow, MinimapGeometry, tileToWindowCell, windowCellToTile, type WindowSpec,
} from "../../src/obs/Geometry";
import { FINE, MM, WIN } from "../../src/obs/Layout";

function rng(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
}

test.each([[512, 512], [2000, 1000], [4432, 2528], [2036, 4428], [224, 168]])("extreme maps keep aspect and letterbox %ix%i", (w, h) => {
  const g = new MinimapGeometry(w, h);
  for (const [x, y] of [[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]]) {
    const c = g.cellOfTile(x, y);
    expect(c).toBeGreaterThanOrEqual(0);
    expect(c).toBeLessThan(MM * MM);
    expect(g.tileRectOfCell(c)).not.toBeNull();
  }
  let letterbox = 0;
  for (let c = 0; c < MM * MM; c++) if (g.tileRectOfCell(c) === null) letterbox++;
  expect(letterbox).toBe(MM * MM - g.cellsW * g.cellsH);
  // Aspect kept within one cell.
  expect(Math.abs(g.cellsW / g.cellsH - w / h)).toBeLessThan(0.05 * Math.max(1, w / h));
  if (h > w) expect(g.cellsH).toBeGreaterThan(g.cellsW);
});

test("cellOfTile and tileRectOfCell agree", () => {
  const r = rng(1);
  for (const [w, h] of [[2000, 1000], [2036, 4428], [512, 512]]) {
    const g = new MinimapGeometry(w, h);
    for (let i = 0; i < 1000; i++) {
      const x = Math.floor(r() * w);
      const y = Math.floor(r() * h);
      const rect = g.tileRectOfCell(g.cellOfTile(x, y))!;
      expect(x >= rect.x0 && x < rect.x1 && y >= rect.y0 && y < rect.y1).toBe(true);
    }
  }
});

test("fineToTile stays inside its cell and tileToFine inverts it", () => {
  const g = new MinimapGeometry(4432, 2528);
  for (let cell = 0; cell < MM * MM; cell += 97) {
    const rect = g.tileRectOfCell(cell);
    if (!rect) continue;
    for (let f = 0; f < FINE * FINE; f += 13) {
      const t = g.fineToTile(cell, f)!;
      expect(t.x >= rect.x0 && t.x < rect.x1 && t.y >= rect.y0 && t.y < rect.y1).toBe(true);
      const back = g.tileToFine(t.x, t.y);
      expect(back.cell).toBe(cell);
      const p = g.fineToTile(back.cell, back.fine)!;
      expect(Math.abs(p.x - t.x) <= g.fineSpan() && Math.abs(p.y - t.y) <= g.fineSpan()).toBe(true);
    }
  }
});

test("edge windows are shifted inside", () => {
  const w: WindowSpec = { x0: -10, y0: -5, tpc: 2, kind: "core", priority: 0 };
  expect(clampWindow(w, 1000, 800)).toMatchObject({ x0: 0, y0: 0 });
  const far = clampWindow({ ...w, x0: 990, y0: 790 }, 1000, 800);
  expect(far.x0).toBe(1000 - WIN * 2);
  expect(far.y0).toBe(800 - WIN * 2);
});

test("window round trip", () => {
  for (const tpc of [2, 3, 4] as const) {
    const w: WindowSpec = { x0: 100, y0: 50, tpc, kind: "front", priority: 1 };
    for (let c = 0; c < WIN * WIN; c++) {
      const t = windowCellToTile(w, c, 2000, 1000)!;
      expect(tileToWindowCell(w, t.x, t.y)).toBe(c);
    }
  }
});

test("window larger than a tiny map marks outside cells", () => {
  const w = clampWindow({ x0: 0, y0: 0, tpc: 4, kind: "core", priority: 0 } as WindowSpec, 100, 60);
  expect(w.x0).toBe(0);
  expect(windowCellToTile(w, 0, 100, 60)).not.toBeNull();
  expect(windowCellToTile(w, WIN - 1, 100, 60)).toBeNull();
});
