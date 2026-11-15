/**
 * Nearest legal tile around a point (spec §10.3): candidates within `radius`
 * tiles, nearest first by Euclidean distance, ties broken row-major (y, then
 * x), so decoding is deterministic. Returns null when nothing qualifies; the
 * caller then drops the action (never retargets it somewhere else).
 */
const offsetCache = new Map<number, Int16Array>();

function offsets(radius: number): Int16Array {
  let o = offsetCache.get(radius);
  if (o) return o;
  const list: [number, number, number][] = [];
  for (let dy = -radius; dy <= radius; dy++) {
    for (let dx = -radius; dx <= radius; dx++) {
      const d2 = dx * dx + dy * dy;
      if (d2 <= radius * radius) list.push([d2, dy, dx]);
    }
  }
  list.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
  o = new Int16Array(list.length * 2);
  list.forEach(([, dy, dx], i) => {
    o![2 * i] = dx;
    o![2 * i + 1] = dy;
  });
  offsetCache.set(radius, o);
  return o;
}

export function snapTile(
  start: { x: number; y: number },
  ok: (tile: number) => boolean,
  radius: number,
  mapW: number,
  mapH: number,
  /** Stop after this many candidates passed the cheap bounds check (bounds expensive predicates). */
  maxTries = Infinity,
): number | null {
  const o = offsets(Math.max(0, Math.round(radius)));
  let tries = 0;
  for (let i = 0; i < o.length; i += 2) {
    const x = start.x + o[i];
    const y = start.y + o[i + 1];
    if (x < 0 || y < 0 || x >= mapW || y >= mapH) continue;
    if (tries++ >= maxTries) return null;
    const t = y * mapW + x;
    if (ok(t)) return t;
  }
  return null;
}
