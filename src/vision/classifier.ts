/**
 * Tiny CNN square classifier; architecture mirrors training/train.py.
 * Weights are float16, stored in the order conv1 w,b, conv2 w,b, conv3 w,b,
 * dense1 w,b, dense2 w,b with conv kernels in HWIO layout.
 */
export const CLASSES = ['.', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k'] as const;
export type PieceClass = (typeof CLASSES)[number];

const CONVS: [number, number][] = [
  [1, 16],
  [16, 32],
  [32, 64],
];
const DENSE: [number, number][] = [
  [4 * 4 * 64, 128],
  [128, 13],
];

interface Layer {
  w: Float32Array;
  b: Float32Array;
  cin: number;
  cout: number;
}

export interface Model {
  convs: Layer[];
  dense: Layer[];
}

function f16(h: number): number {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + f / 1024);
}

export function loadModel(buf: ArrayBuffer): Model {
  const u16 = new Uint16Array(buf);
  const all = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) all[i] = f16(u16[i]);
  let off = 0;
  const take = (n: number) => {
    const a = all.subarray(off, off + n);
    off += n;
    return a;
  };
  const convs = CONVS.map(([cin, cout]) => ({ w: take(9 * cin * cout), b: take(cout), cin, cout }));
  const dense = DENSE.map(([cin, cout]) => ({ w: take(cin * cout), b: take(cout), cin, cout }));
  if (off !== all.length) throw new Error(`model size mismatch: ${off} vs ${all.length}`);
  return { convs, dense };
}

/** 3x3 SAME conv + ReLU + 2x2 max-pool, HWC layout. */
function convReluPool(input: Float32Array, size: number, l: Layer): Float32Array {
  const { cin, cout, w, b } = l;
  const conv = new Float32Array(size * size * cout);
  const acc = new Float32Array(cout);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      acc.set(b);
      for (let ky = 0; ky < 3; ky++) {
        const iy = y + ky - 1;
        if (iy < 0 || iy >= size) continue;
        for (let kx = 0; kx < 3; kx++) {
          const ix = x + kx - 1;
          if (ix < 0 || ix >= size) continue;
          const ibase = (iy * size + ix) * cin;
          const wbase = (ky * 3 + kx) * cin * cout;
          for (let ci = 0; ci < cin; ci++) {
            const v = input[ibase + ci];
            if (v === 0) continue;
            const wo = wbase + ci * cout;
            for (let co = 0; co < cout; co++) acc[co] += v * w[wo + co];
          }
        }
      }
      const obase = (y * size + x) * cout;
      for (let co = 0; co < cout; co++) conv[obase + co] = acc[co] > 0 ? acc[co] : 0;
    }
  }
  const half = size / 2;
  const out = new Float32Array(half * half * cout);
  for (let y = 0; y < half; y++) {
    for (let x = 0; x < half; x++) {
      for (let co = 0; co < cout; co++) {
        const a = conv[((2 * y) * size + 2 * x) * cout + co];
        const bb = conv[((2 * y) * size + 2 * x + 1) * cout + co];
        const c = conv[((2 * y + 1) * size + 2 * x) * cout + co];
        const d = conv[((2 * y + 1) * size + 2 * x + 1) * cout + co];
        out[(y * half + x) * cout + co] = Math.max(a, bb, c, d);
      }
    }
  }
  return out;
}

function dense(input: Float32Array, l: Layer, relu: boolean): Float32Array {
  const out = new Float32Array(l.cout);
  out.set(l.b);
  for (let i = 0; i < l.cin; i++) {
    const v = input[i];
    if (v === 0) continue;
    const base = i * l.cout;
    for (let o = 0; o < l.cout; o++) out[o] += v * l.w[base + o];
  }
  if (relu) for (let o = 0; o < l.cout; o++) if (out[o] < 0) out[o] = 0;
  return out;
}

/** Classify one 32x32 square (values 0..1). Returns 13 probabilities. */
export function classifySquare(model: Model, sq: Float32Array): Float32Array {
  let h: Float32Array = new Float32Array(sq.length);
  for (let i = 0; i < sq.length; i++) h[i] = sq[i] - 0.5;
  let size = 32;
  for (const l of model.convs) {
    h = convReluPool(h, size, l);
    size /= 2;
  }
  h = dense(h, model.dense[0], true);
  const logits = dense(h, model.dense[1], false);
  let mx = -Infinity;
  for (const v of logits) mx = Math.max(mx, v);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    logits[i] = Math.exp(logits[i] - mx);
    sum += logits[i];
  }
  for (let i = 0; i < logits.length; i++) logits[i] /= sum;
  return logits;
}
