/** Minimal greyscale image utilities (values 0..1, row-major). */
export interface Gray {
  w: number;
  h: number;
  data: Float32Array;
}

export function makeGray(w: number, h: number): Gray {
  return { w, h, data: new Float32Array(w * h) };
}

export function fromRGBA(rgba: Uint8ClampedArray | Uint8Array, w: number, h: number): Gray {
  const g = makeGray(w, h);
  const d = g.data;
  for (let i = 0, j = 0; i < d.length; i++, j += 4) {
    d[i] = (0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2]) / 255;
  }
  return g;
}

export function fromU8(u8: Uint8Array, w: number, h: number): Gray {
  const g = makeGray(w, h);
  for (let i = 0; i < u8.length; i++) g.data[i] = u8[i] / 255;
  return g;
}

/** Separable box blur with radius r (clamped edges). */
export function boxBlur(src: Gray, r: number): Gray {
  const { w, h } = src;
  const tmp = new Float32Array(w * h);
  const out = makeGray(w, h);
  const n = 2 * r + 1;
  for (let y = 0; y < h; y++) {
    const row = y * w;
    let acc = 0;
    for (let k = -r; k <= r; k++) acc += src.data[row + Math.min(w - 1, Math.max(0, k))];
    for (let x = 0; x < w; x++) {
      tmp[row + x] = acc / n;
      const add = Math.min(w - 1, x + r + 1);
      const sub = Math.max(0, x - r);
      acc += src.data[row + add] - src.data[row + sub];
    }
  }
  for (let x = 0; x < w; x++) {
    let acc = 0;
    for (let k = -r; k <= r; k++) acc += tmp[Math.min(h - 1, Math.max(0, k)) * w + x];
    for (let y = 0; y < h; y++) {
      out.data[y * w + x] = acc / n;
      const add = Math.min(h - 1, y + r + 1);
      const sub = Math.max(0, y - r);
      acc += tmp[add * w + x] - tmp[sub * w + x];
    }
  }
  return out;
}

/** Percentile-based contrast stretch to 0..1 (in place copy). */
export function normalise(src: Gray, loP = 0.02, hiP = 0.98): Gray {
  const bins = 512;
  const hist = new Uint32Array(bins);
  const d = src.data;
  for (let i = 0; i < d.length; i++) {
    const b = Math.min(bins - 1, Math.max(0, Math.floor(d[i] * bins)));
    hist[b]++;
  }
  const lo = percentileFromHist(hist, d.length, loP) / bins;
  const hi = (percentileFromHist(hist, d.length, hiP) + 1) / bins;
  const span = Math.max(hi - lo, 1e-3);
  const out = makeGray(src.w, src.h);
  for (let i = 0; i < d.length; i++) {
    const v = (d[i] - lo) / span;
    out.data[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  return out;
}

function percentileFromHist(hist: Uint32Array, total: number, p: number): number {
  const target = p * total;
  let acc = 0;
  for (let i = 0; i < hist.length; i++) {
    acc += hist[i];
    if (acc >= target) return i;
  }
  return hist.length - 1;
}

/** Bilinear sample with edge clamping. */
export function sample(g: Gray, x: number, y: number): number {
  const { w, h, data } = g;
  if (x < 0) x = 0;
  else if (x > w - 1.001) x = w - 1.001;
  if (y < 0) y = 0;
  else if (y > h - 1.001) y = h - 1.001;
  const x0 = x | 0;
  const y0 = y | 0;
  const fx = x - x0;
  const fy = y - y0;
  const i = y0 * w + x0;
  const a = data[i] + (data[i + 1] - data[i]) * fx;
  const b = data[i + w] + (data[i + w + 1] - data[i + w]) * fx;
  return a + (b - a) * fy;
}

/** Area-average downscale by an arbitrary factor (dst smaller than src). */
export function resizeArea(src: Gray, dw: number, dh: number): Gray {
  const out = makeGray(dw, dh);
  const sx = src.w / dw;
  const sy = src.h / dh;
  for (let y = 0; y < dh; y++) {
    const y0 = y * sy;
    const y1 = y0 + sy;
    for (let x = 0; x < dw; x++) {
      const x0 = x * sx;
      const x1 = x0 + sx;
      let acc = 0;
      let wsum = 0;
      for (let yy = Math.floor(y0); yy < Math.ceil(y1) && yy < src.h; yy++) {
        const wy = Math.min(yy + 1, y1) - Math.max(yy, y0);
        for (let xx = Math.floor(x0); xx < Math.ceil(x1) && xx < src.w; xx++) {
          const wx = Math.min(xx + 1, x1) - Math.max(xx, x0);
          const wgt = wx * wy;
          acc += src.data[yy * src.w + xx] * wgt;
          wsum += wgt;
        }
      }
      out.data[y * dw + x] = wsum > 0 ? acc / wsum : 0;
    }
  }
  return out;
}

export interface Gradients {
  gx: Gray;
  gy: Gray;
}

export function gradients(g: Gray): Gradients {
  const { w, h, data } = g;
  const gx = makeGray(w, h);
  const gy = makeGray(w, h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      gx.data[i] = (data[i + 1] - data[i - 1]) * 0.5;
      gy.data[i] = (data[i + w] - data[i - w]) * 0.5;
    }
  }
  return { gx, gy };
}
