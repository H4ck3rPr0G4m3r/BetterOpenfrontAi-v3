/** float32 -> IEEE half-precision bits (round to nearest even), for the f16 wire fields. */
const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

export function toF16(value: number): number {
  f32[0] = value;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  let exp = (x >>> 23) & 0xff;
  let mant = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0); // inf / nan
  exp = exp - 127 + 15;
  if (exp >= 0x1f) return sign | 0x7c00; // overflow -> inf
  if (exp <= 0) {
    if (exp < -10) return sign; // underflow -> 0
    mant = (mant | 0x800000) >>> (1 - exp);
    if (mant & 0x1000 && mant & 0x2fff) mant += 0x2000;
    return sign | (mant >>> 13);
  }
  if (mant & 0x1000 && mant & 0x2fff) {
    mant += 0x2000;
    if (mant & 0x800000) {
      mant = 0;
      exp += 1;
      if (exp >= 0x1f) return sign | 0x7c00;
    }
  }
  return sign | (exp << 10) | (mant >>> 13);
}

export function fromF16(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >>> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) return sign * 2 ** -14 * (mant / 1024);
  if (exp === 0x1f) return mant ? NaN : sign * Infinity;
  return sign * 2 ** (exp - 15) * (1 + mant / 1024);
}
