/**
 * Finds the 8x8 playing area of a printed chess diagram inside an image
 * region (the guide box plus a margin).
 *
 * 1. A 1-D "lattice" search on edge projections gives a rough axis-aligned
 *    estimate (9 equally spaced lines in x and in y).
 * 2. A hill climb over the four corners (full perspective) maximises a score
 *    that rewards strong edges on the 18 grid lines, weak edges half a square
 *    away, and a light/dark checker pattern between squares.
 * 3. The final score components decide whether the result is a real board.
 */
import { boxBlur, gradients, normalise, resizeArea, sample, type Gray } from './gray';
import { mapPoint, quadSide, scaleQuad, squareToQuad, type Quad } from './homography';

export interface GridMetrics {
  /** mean edge strength on grid lines divided by mean strength on mid lines */
  ratio: number;
  /** how many of the 18 grid lines clearly stand out */
  support: number;
  /** light/dark difference between neighbouring squares (0..1) */
  checker: number;
  /** board side relative to the image side */
  sizeFrac: number;
  /** weakest row/column alternation */
  rowCol: number;
  /** min/max side length ratio */
  regularity: number;
  /** mean on-line gradient (sharpness proxy) */
  edge: number;
}

export interface GridResult {
  quad: Quad;
  ok: boolean;
  score: number;
  metrics: GridMetrics;
}

export interface DetectOptions {
  /** smallest board side as a fraction of min(w, h) */
  minFrac?: number;
  /** largest board side as a fraction of min(w, h) */
  maxFrac?: number;
  /** previous result to warm-start from */
  init?: Quad;
  /** skip the coarse lattice search (only refine `init`) */
  refineOnly?: boolean;
  /** smallest hill-climb step in pixels */
  minStep?: number;
}

export const THRESHOLDS = {
  ratio: 1.25,
  support: 6,
  checker: 0.035,
  rowCol: 0.012,
  sizeFrac: 0.3,
  regularity: 0.72,
  edge: 0.08,
};

export interface Ctx {
  blur: Gray;
  gx: Gray;
  gy: Gray;
}

export function prepare(src: Gray): Ctx {
  const norm = normalise(src);
  const r = Math.max(1, Math.round(Math.min(src.w, src.h) / 256));
  const blur = boxBlur(boxBlur(norm, r), r);
  const { gx, gy } = gradients(blur);
  return { blur, gx, gy };
}

export function detectGrid(src: Gray, opts: DetectOptions = {}): GridResult {
  const fine = prepare(src);
  // Half-resolution level: smoother score landscape -> wider basins of
  // attraction, so the coarse search converges from rougher starts.
  const half = resizeArea(src, Math.max(16, Math.round(src.w / 2)), Math.max(16, Math.round(src.h / 2)));
  const coarse = prepare(half);
  return detectGridPyramid(fine, coarse, src.w, src.h, opts);
}

type Scored = { quad: Quad; score: number };

export function detectGridPyramid(fine: Ctx, coarse: Ctx, w: number, h: number, opts: DetectOptions = {}): GridResult {
  const n = Math.min(w, h);
  const minStep = opts.minStep ?? 0.2;
  const sc = coarse.blur.w / w;
  let best: Scored | null = null;

  if (opts.init) {
    // tracking: refine the previous result directly at full resolution
    best = refine(fine, opts.init, Math.max(quadSide(opts.init) / 8 / 8, 1), minStep);
  }
  if (!opts.refineOnly && (!best || !passes(measure(fine, best.quad, n)))) {
    let cands = latticeCandidates(coarse, coarse.blur.w, coarse.blur.h, opts.minFrac ?? 0.3, opts.maxFrac ?? 0.99)
      .map((q) => ({ quad: q, score: scoreOf(evaluate(coarse, q)) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 3)
      .map((c) => refine(coarse, c.quad, Math.max(quadSide(c.quad) / 8 / 4, 0.5), 0.5));
    cands.sort((a, b) => b.score - a.score);
    let cbest = cands[0];
    if (cbest) {
      // The hill climb can't jump a whole square: try the lattice shifted by
      // one square in each direction.
      const base = cbest.quad;
      for (const [du, dv] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const q = shiftQuad(base, du, dv);
        const r = refine(coarse, q, Math.max(quadSide(q) / 8 / 8, 0.5), 0.5);
        if (r.score > cbest.score) cbest = r;
      }
      const r = refine(fine, scaleQuad(cbest.quad, 1 / sc), 1, minStep);
      if (!best || r.score > best.score) best = r;
    }
  }
  if (!best) {
    const q: Quad = [[0, 0], [w, 0], [w, h], [0, h]];
    return { quad: q, ok: false, score: -1, metrics: zeroMetrics() };
  }
  const metrics = measure(fine, best.quad, n);
  return { quad: best.quad, score: best.score, metrics, ok: passes(metrics) };
}

/** Move the lattice by whole squares (du, dv) along its own axes. */
export function shiftQuad(q: Quad, du: number, dv: number): Quad {
  const m = squareToQuad(q);
  const a = du / 8;
  const b = dv / 8;
  return [mapPoint(m, a, b), mapPoint(m, 1 + a, b), mapPoint(m, 1 + a, 1 + b), mapPoint(m, a, 1 + b)];
}

function zeroMetrics(): GridMetrics {
  return { ratio: 0, support: 0, checker: 0, rowCol: 0, sizeFrac: 0, regularity: 0, edge: 0 };
}

export function passes(m: GridMetrics): boolean {
  return (
    m.ratio >= THRESHOLDS.ratio &&
    m.support >= THRESHOLDS.support &&
    m.checker >= THRESHOLDS.checker &&
    m.rowCol >= THRESHOLDS.rowCol &&
    m.sizeFrac >= THRESHOLDS.sizeFrac &&
    m.regularity >= THRESHOLDS.regularity &&
    m.edge >= THRESHOLDS.edge
  );
}

// ------------------------------------------------------------ coarse search

/** Dominant edge orientation in degrees (-45..45], i.e. the board's rotation. */
export function dominantAngle(ctx: Ctx): number {
  const { gx, gy } = ctx;
  const bins = 90;
  const hist = new Float32Array(bins);
  const n = gx.data.length;
  for (let i = 0; i < n; i++) {
    const x = gx.data[i];
    const y = gy.data[i];
    const m2 = x * x + y * y;
    if (m2 < 1e-4) continue;
    let a = (Math.atan2(y, x) * 180) / Math.PI; // -180..180
    a = ((a % 90) + 90) % 90; // 0..90
    if (a >= 45) a -= 90; // -45..45
    const b = a + 45;
    const b0 = Math.floor(b) % bins;
    const f = b - Math.floor(b);
    hist[b0] += m2 * (1 - f);
    hist[(b0 + 1) % bins] += m2 * f;
  }
  // circular smoothing
  const sm = new Float32Array(bins);
  for (let i = 0; i < bins; i++) {
    for (let k = -2; k <= 2; k++) sm[i] += hist[(i + k + bins) % bins] * (3 - Math.abs(k));
  }
  // Only consider |angle| <= 20 deg: the user lines the board up with the box,
  // and 45-degree hatching on dark squares must not be mistaken for rotation.
  let best = 45;
  for (let i = 25; i <= 65; i++) if (sm[i] > sm[best]) best = i;
  const l = sm[(best - 1 + bins) % bins];
  const r = sm[(best + 1) % bins];
  const den = l - 2 * sm[best] + r;
  const off = den !== 0 ? (0.5 * (l - r)) / den : 0;
  return best + off - 45;
}

interface Frame {
  cos: number;
  sin: number;
  umin: number;
  vmin: number;
  nu: number;
  nv: number;
}

function makeFrame(w: number, h: number, angleDeg: number): Frame {
  const a = (angleDeg * Math.PI) / 180;
  const cos = Math.cos(a);
  const sin = Math.sin(a);
  const us = [0, w * cos, h * sin, w * cos + h * sin];
  const vs = [0, -w * sin, h * cos, -w * sin + h * cos];
  const umin = Math.min(...us);
  const vmin = Math.min(...vs);
  return { cos, sin, umin, vmin, nu: Math.ceil(Math.max(...us) - umin) + 1, nv: Math.ceil(Math.max(...vs) - vmin) + 1 };
}

/**
 * Edge profile along the rotated u axis (lines of constant u) or v axis,
 * restricted to a band of the other coordinate.
 */
function profile(ctx: Ctx, fr: Frame, axis: 'u' | 'v', lo: number, hi: number): Float32Array {
  const { gx, gy } = ctx;
  const { w, h } = gx;
  const out = new Float32Array(axis === 'u' ? fr.nu : fr.nv);
  const { cos, sin } = fr;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const u = x * cos + y * sin - fr.umin;
      const v = -x * sin + y * cos - fr.vmin;
      const i = y * w + x;
      let pos: number;
      let resp: number;
      if (axis === 'u') {
        if (v < lo || v > hi) continue;
        pos = u;
        resp = Math.abs(gx.data[i] * cos + gy.data[i] * sin);
      } else {
        if (u < lo || u > hi) continue;
        pos = v;
        resp = Math.abs(-gx.data[i] * sin + gy.data[i] * cos);
      }
      const p0 = pos | 0;
      const f = pos - p0;
      out[p0] += resp * (1 - f);
      if (p0 + 1 < out.length) out[p0 + 1] += resp * f;
    }
  }
  // compress so a single very strong edge (page border, shadow) can't
  // dominate the nine lattice lines
  for (let i = 0; i < out.length; i++) out[i] = Math.sqrt(out[i]);
  return out;
}

function interp(p: Float32Array, x: number): number {
  if (x <= 0) return p[0];
  if (x >= p.length - 1) return p[p.length - 1];
  const i = x | 0;
  const f = x - i;
  return p[i] * (1 - f) + p[i + 1] * f;
}

interface Fit {
  o: number;
  s: number;
  score: number;
}

/** Best 9-line lattices (offset o, spacing s) in a 1-D edge profile. */
function fit1D(p: Float32Array, minS: number, maxS: number, top = 3): Fit[] {
  const n = p.length;
  const fits: Fit[] = [];
  for (let s = minS; s <= maxS; s += 0.25) {
    let bestO = 0;
    let bestScore = -Infinity;
    for (let o = 0; o + 8 * s <= n - 1; o += 0.5) {
      let sc = 0;
      for (let k = 0; k <= 8; k++) sc += interp(p, o + k * s);
      for (let k = 0; k < 8; k++) sc -= (9 / 8) * interp(p, o + (k + 0.5) * s);
      if (sc > bestScore) {
        bestScore = sc;
        bestO = o;
      }
    }
    if (bestScore > -Infinity) fits.push({ o: bestO, s, score: bestScore });
  }
  fits.sort((a, b) => b.score - a.score);
  const out: Fit[] = [];
  for (const f of fits) {
    if (out.every((g) => Math.abs(g.s - f.s) > 1.5 || Math.abs(g.o - f.o) > 1.5)) out.push(f);
    if (out.length >= top) break;
  }
  return out;
}

function latticeCandidates(ctx: Ctx, w: number, h: number, minFrac: number, maxFrac: number): Quad[] {
  const n = Math.min(w, h);
  const minS = (n * minFrac) / 8;
  const maxS = (n * maxFrac) / 8;
  const angle = dominantAngle(ctx);
  const fr = makeFrame(w, h, angle);
  // Alternate between axes, restricting each profile to the band the other
  // axis found, which removes most of the surrounding text.
  let fu = fit1D(profile(ctx, fr, 'u', -1e9, 1e9), minS, maxS, 1)[0];
  let fv = fit1D(profile(ctx, fr, 'v', -1e9, 1e9), minS, maxS, 1)[0];
  if (!fu || !fv) return [];
  for (let it = 0; it < 2; it++) {
    fv = fit1D(profile(ctx, fr, 'v', fu.o, fu.o + 8 * fu.s), minS, maxS, 1)[0] ?? fv;
    fu = fit1D(profile(ctx, fr, 'u', fv.o, fv.o + 8 * fv.s), minS, maxS, 1)[0] ?? fu;
  }
  const us = fit1D(profile(ctx, fr, 'u', fv.o, fv.o + 8 * fv.s), minS, maxS, 3);
  const vs = fit1D(profile(ctx, fr, 'v', fu.o, fu.o + 8 * fu.s), minS, maxS, 3);
  const toXY = (u: number, v: number): [number, number] => {
    u += fr.umin;
    v += fr.vmin;
    return [u * fr.cos - v * fr.sin, u * fr.sin + v * fr.cos];
  };
  const quads: Quad[] = [];
  for (const a of us) {
    for (const b of vs) {
      if (a.s / b.s > 1.3 || b.s / a.s > 1.3) continue;
      quads.push([
        toXY(a.o, b.o),
        toXY(a.o + 8 * a.s, b.o),
        toXY(a.o + 8 * a.s, b.o + 8 * b.s),
        toXY(a.o, b.o + 8 * b.s),
      ]);
    }
  }
  if (!quads.length) {
    const s = (fu.s + fv.s) / 2;
    quads.push([toXY(fu.o, fv.o), toXY(fu.o + 8 * s, fv.o), toXY(fu.o + 8 * s, fv.o + 8 * s), toXY(fu.o, fv.o + 8 * s)]);
  }
  return quads;
}

// ------------------------------------------------------------ scoring

const LINE_SAMPLES = 20;
const INSET = [0.15, 0.85];
const CHECKER_WEIGHT = 0.5;

interface Eval {
  on: number;
  mid: number;
  checker: number;
  /** weakest per-row / per-column light-dark alternation (signed) */
  minRowCol: number;
  lines?: Float32Array;
}

export function evaluate(ctx: Ctx, q: Quad, detail = false): Eval {
  const m = squareToQuad(q);
  const { gx, gy, blur } = ctx;
  let on = 0;
  let mid = 0;
  const lines = detail ? new Float32Array(18) : undefined;
  for (let fam = 0; fam < 2; fam++) {
    for (let k = 0; k <= 16; k++) {
      const t = k / 16;
      // a straight line under a homography stays straight: one normal per line
      const a = fam === 0 ? mapPoint(m, 0, t) : mapPoint(m, t, 0);
      const b = fam === 0 ? mapPoint(m, 1, t) : mapPoint(m, t, 1);
      let tx = b[0] - a[0];
      let ty = b[1] - a[1];
      const len = Math.hypot(tx, ty) || 1;
      tx /= len;
      ty /= len;
      const nx = -ty;
      const ny = tx;
      let acc = 0;
      for (let j = 0; j < LINE_SAMPLES; j++) {
        const s = (j + 0.5) / LINE_SAMPLES;
        const p = fam === 0 ? mapPoint(m, s, t) : mapPoint(m, t, s);
        acc += Math.abs(sample(gx, p[0], p[1]) * nx + sample(gy, p[0], p[1]) * ny);
      }
      acc = Math.sqrt(acc / LINE_SAMPLES);
      if (k % 2 === 0) {
        on += acc;
        if (lines) lines[fam * 9 + k / 2] = acc;
      } else mid += acc;
    }
  }
  on /= 18;
  mid /= 16;
  let checker = 0;
  const rows = new Float32Array(8);
  const cols = new Float32Array(8);
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      let acc = 0;
      for (const du of INSET) {
        for (const dv of INSET) {
          const p = mapPoint(m, (c + du) / 8, (r + dv) / 8);
          acc += sample(blur, p[0], p[1]);
        }
      }
      const v = (r + c) % 2 === 0 ? acc : -acc;
      checker += v;
      rows[r] += v;
      cols[c] += v;
    }
  }
  checker /= 256;
  // Every row and column of a real board alternates light/dark with the same
  // phase; a lattice shifted by one square has a row or column of paper.
  const sgn = checker >= 0 ? 1 : -1;
  let minRowCol = Infinity;
  for (let i = 0; i < 8; i++) minRowCol = Math.min(minRowCol, (sgn * rows[i]) / 32, (sgn * cols[i]) / 32);
  return { on, mid, checker, minRowCol, lines };
}

export function scoreOf(e: Eval): number {
  return e.on - e.mid + CHECKER_WEIGHT * (Math.abs(e.checker) + e.minRowCol);
}

function refine(ctx: Ctx, start: Quad, startStep: number, minStep: number): Scored {
  let q = start.map((p) => [p[0], p[1]]) as Quad;
  let best = scoreOf(evaluate(ctx, q));
  for (let step = startStep; step >= minStep; step /= 2) {
    for (let iter = 0; iter < 12; iter++) {
      let improved = false;
      const ms = moves(q, step);
      for (let k = 0; k < ms.length; k++) {
        let cand = ms[k];
        let s = scoreOf(evaluate(ctx, cand));
        // keep going in a direction that helps
        while (s > best + 1e-7) {
          best = s;
          const prev = q;
          q = cand;
          improved = true;
          cand = q.map((p, i) => [2 * p[0] - prev[i][0], 2 * p[1] - prev[i][1]]) as Quad;
          s = scoreOf(evaluate(ctx, cand));
        }
      }
      if (!improved) break;
    }
  }
  return { quad: q, score: best };
}

function moves(q: Quad, step: number): Quad[] {
  const out: Quad[] = [];
  const clone = () => q.map((p) => [p[0], p[1]]) as Quad;
  // individual corner coordinates
  for (let i = 0; i < 4; i++) {
    for (let d = 0; d < 2; d++) {
      for (const sgn of [1, -1]) {
        const c = clone();
        c[i][d] += sgn * step;
        out.push(c);
      }
    }
  }
  // whole-quad translation
  for (const [dx, dy] of [[step, 0], [-step, 0], [0, step], [0, -step]]) {
    out.push(q.map((p) => [p[0] + dx, p[1] + dy]) as Quad);
  }
  // scale and rotation about the centre
  const cx = (q[0][0] + q[1][0] + q[2][0] + q[3][0]) / 4;
  const cy = (q[0][1] + q[1][1] + q[2][1] + q[3][1]) / 4;
  const side = quadSide(q);
  for (const sgn of [1, -1]) {
    const s = 1 + (sgn * 2 * step) / side;
    out.push(q.map((p) => [cx + (p[0] - cx) * s, cy + (p[1] - cy) * s]) as Quad);
    const a = (sgn * 2 * step) / side;
    const ca = Math.cos(a);
    const sa = Math.sin(a);
    out.push(
      q.map((p) => {
        const x = p[0] - cx;
        const y = p[1] - cy;
        return [cx + x * ca - y * sa, cy + x * sa + y * ca];
      }) as Quad,
    );
  }
  return out;
}

function measure(ctx: Ctx, q: Quad, n: number): GridMetrics {
  const e = evaluate(ctx, q, true);
  const lines = e.lines!;
  let support = 0;
  for (let i = 0; i < 18; i++) if (lines[i] > 1.3 * e.mid) support++;
  const d = (i: number, j: number) => Math.hypot(q[i][0] - q[j][0], q[i][1] - q[j][1]);
  const sides = [d(0, 1), d(1, 2), d(2, 3), d(3, 0)];
  const regularity = Math.min(...sides) / Math.max(...sides);
  return {
    ratio: e.on / Math.max(e.mid, 1e-4),
    support,
    checker: Math.abs(e.checker),
    rowCol: e.minRowCol,
    sizeFrac: quadSide(q) / n,
    regularity: isConvex(q) ? regularity : 0,
    edge: e.on,
  };
}

function isConvex(q: Quad): boolean {
  let sign = 0;
  for (let i = 0; i < 4; i++) {
    const a = q[i];
    const b = q[(i + 1) % 4];
    const c = q[(i + 2) % 4];
    const cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
    const s = Math.sign(cross);
    if (s === 0) return false;
    if (sign === 0) sign = s;
    else if (s !== sign) return false;
  }
  return true;
}
