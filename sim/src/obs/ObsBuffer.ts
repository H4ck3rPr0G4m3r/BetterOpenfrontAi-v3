/**
 * Preallocated storage for `rows` observations in wire layout: every field of
 * obsFieldSpecs() as one typed array [rows, ...shape]. `bytes()` serializes
 * the first n rows field by field, in layout order (roadmap §3.4 ObsBlock).
 */
import { type FieldSpec, obsFieldSpecs } from "./Layout";

export type FieldArray = Uint8Array | Uint16Array | Int16Array | Int32Array | Float32Array;

function alloc(f: FieldSpec, n: number): FieldArray {
  switch (f.dtype) {
    case "u8":
      return new Uint8Array(n);
    case "u16":
    case "f16":
      return new Uint16Array(n);
    case "i16":
      return new Int16Array(n);
    case "i32":
      return new Int32Array(n);
    case "f32":
      return new Float32Array(n);
  }
}

export class ObsBuffer {
  readonly specs = obsFieldSpecs();
  readonly fields = new Map<string, FieldArray>();
  readonly sizes = new Map<string, number>();

  constructor(readonly rows: number) {
    for (const f of this.specs) {
      const size = f.shape.reduce((a, b) => a * b, 1);
      this.sizes.set(f.name, size);
      this.fields.set(f.name, alloc(f, size * rows));
    }
  }

  /** The slice of one field belonging to `row`. */
  view<T extends FieldArray = FieldArray>(name: string, row: number): T {
    const arr = this.fields.get(name);
    if (!arr) throw new Error(`no field ${name}`);
    const size = this.sizes.get(name)!;
    return arr.subarray(row * size, (row + 1) * size) as T;
  }

  clearRow(row: number): void {
    for (const f of this.specs) {
      const v = this.view(f.name, row);
      v.fill(f.name === "mm_owner" || f.name === "win_owner" ? 0xffff : f.name === "units" ? -1 : 0);
    }
  }

  copyRow(from: number, to: number, other?: ObsBuffer): void {
    const src = other ?? this;
    for (const f of this.specs) this.view(f.name, to).set(src.view(f.name, from));
  }

  /** Field-by-field bytes of rows [0, n). */
  bytes(n = this.rows): Uint8Array {
    let total = 0;
    for (const f of this.specs) total += this.sizes.get(f.name)! * n * this.fields.get(f.name)!.BYTES_PER_ELEMENT;
    const out = new Uint8Array(total);
    let off = 0;
    for (const f of this.specs) {
      const arr = this.fields.get(f.name)!;
      const len = this.sizes.get(f.name)! * n * arr.BYTES_PER_ELEMENT;
      out.set(new Uint8Array(arr.buffer, arr.byteOffset, len), off);
      off += len;
    }
    return out;
  }
}
