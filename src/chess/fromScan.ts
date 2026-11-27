/**
 * Turns per-square class probabilities into a legal-looking position.
 * Squares are indexed row-major from the top-left of the diagram as printed.
 */
import { CLASSES, type PieceClass } from '../vision/classifier';

export interface ScanPosition {
  /** 64 symbols, index 0 = a8 (after any orientation fix) */
  placement: PieceClass[];
  /** indices (same frame as placement) the model was unsure about */
  uncertain: number[];
  /** true when the diagram was printed from Black's side and we rotated it */
  rotated: boolean;
  /** lowest chosen-class probability */
  minConfidence: number;
}

const UNCERTAIN_BELOW = 0.8;
const idx = (c: PieceClass) => CLASSES.indexOf(c);

/** Average several frames' probability tables (geometric mean). */
export function combineFrames(frames: Float32Array[][]): Float32Array[] {
  const out: Float32Array[] = [];
  for (let s = 0; s < 64; s++) {
    const acc = new Float32Array(13);
    for (const f of frames) for (let k = 0; k < 13; k++) acc[k] += Math.log(Math.max(f[s][k], 1e-6));
    let mx = -Infinity;
    for (let k = 0; k < 13; k++) mx = Math.max(mx, acc[k]);
    let sum = 0;
    for (let k = 0; k < 13; k++) {
      acc[k] = Math.exp((acc[k] - mx) / frames.length);
      sum += acc[k];
    }
    for (let k = 0; k < 13; k++) acc[k] /= sum;
    out.push(acc);
  }
  return out;
}

function argmax(p: Float32Array): number {
  let b = 0;
  for (let k = 1; k < p.length; k++) if (p[k] > p[b]) b = k;
  return b;
}

/** Apply chess constraints and pick a class per square. */
export function decode(probs: Float32Array[]): { placement: PieceClass[]; conf: number[] } {
  const p = probs.map((a) => Float32Array.from(a));
  const renorm = (a: Float32Array) => {
    let s = 0;
    for (const v of a) s += v;
    if (s > 0) for (let k = 0; k < a.length; k++) a[k] /= s;
  };
  // no pawns on the first or last rank (either orientation)
  for (let s = 0; s < 64; s++) {
    const row = s >> 3;
    if (row === 0 || row === 7) {
      p[s][idx('P')] = 0;
      p[s][idx('p')] = 0;
      renorm(p[s]);
    }
  }
  // exactly one king of each colour
  const kingSquare: Record<'K' | 'k', number> = { K: -1, k: -1 };
  const order: ('K' | 'k')[] =
    Math.max(...p.map((a) => a[idx('K')])) >= Math.max(...p.map((a) => a[idx('k')])) ? ['K', 'k'] : ['k', 'K'];
  for (const king of order) {
    let best = -1;
    for (let s = 0; s < 64; s++) {
      if (s === kingSquare.K || s === kingSquare.k) continue;
      if (best < 0 || p[s][idx(king)] > p[best][idx(king)]) best = s;
    }
    kingSquare[king] = best;
  }
  const placement: PieceClass[] = [];
  const conf: number[] = [];
  for (let s = 0; s < 64; s++) {
    if (s === kingSquare.K || s === kingSquare.k) {
      const king = s === kingSquare.K ? 'K' : 'k';
      placement.push(king);
      conf.push(p[s][idx(king)]);
      continue;
    }
    p[s][idx('K')] = 0;
    p[s][idx('k')] = 0;
    renorm(p[s]);
    const k = argmax(p[s]);
    placement.push(CLASSES[k]);
    conf.push(p[s][k]);
  }
  applyCountPriors(p, placement, conf);
  return { placement, conf };
}

/**
 * Real positions rarely have more than two bishops / knights / rooks, one
 * queen or eight pawns per side, and never two same-side bishops on the same
 * square colour without a promotion.  When the reading breaks such a rule,
 * the least confident offenders (if not very confident) are switched to their
 * next best class, and stay flagged through their low confidence.
 */
const SURE = 0.9;
function applyCountPriors(p: Float32Array[], placement: PieceClass[], conf: number[]) {
  const groups: { sym: PieceClass; cap: number; filter?: (s: number) => boolean }[] = [];
  for (const side of ['w', 'b'] as const) {
    const c = (x: string) => (side === 'w' ? x.toUpperCase() : x) as PieceClass;
    groups.push({ sym: c('b'), cap: 1, filter: (s) => ((s >> 3) + s) % 2 === 0 });
    groups.push({ sym: c('b'), cap: 1, filter: (s) => ((s >> 3) + s) % 2 === 1 });
    groups.push({ sym: c('n'), cap: 2 }, { sym: c('r'), cap: 2 }, { sym: c('q'), cap: 1 }, { sym: c('p'), cap: 8 });
  }
  const count = (sym: PieceClass, filter?: (s: number) => boolean) =>
    placement.filter((x, s) => x === sym && (!filter || filter(s))).length;
  for (const g of groups) {
    const squares = placement
      .map((x, s) => s)
      .filter((s) => placement[s] === g.sym && (!g.filter || g.filter(s)))
      .sort((a, b) => conf[b] - conf[a]);
    for (const s of squares.slice(g.cap)) {
      if (conf[s] >= SURE) continue;
      // best alternative that doesn't break its own cap
      const order = Array.from(p[s].keys()).sort((a, b) => p[s][b] - p[s][a]);
      for (const k of order) {
        const sym = CLASSES[k];
        if (sym === g.sym || sym === 'K' || sym === 'k') continue;
        const cap = groups.find((x) => x.sym === sym && (!x.filter || x.filter(s)));
        if (cap && count(sym, cap.filter) >= cap.cap) continue;
        if ((sym === 'P' || sym === 'p') && ((s >> 3) === 0 || (s >> 3) === 7)) continue;
        placement[s] = sym;
        conf[s] = Math.min(conf[s], 0.5);
        break;
      }
    }
  }
}

/**
 * Most books print White at the bottom.  Rotate only on clear evidence that
 * this diagram is printed from Black's side.
 */
export function looksRotated(placement: PieceClass[]): boolean {
  const rows = (sym: PieceClass) => placement.flatMap((c, i) => (c === sym ? [i >> 3] : []));
  const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;
  const wp = rows('P');
  const bp = rows('p');
  let evidence = 0;
  // White pawns normally sit lower on the diagram (bigger row) than Black's.
  if (wp.length >= 2 && bp.length >= 2) evidence += mean(bp) - mean(wp);
  const wk = rows('K')[0];
  const bk = rows('k')[0];
  if (wk !== undefined && bk !== undefined) {
    if (wk <= 1 && bk >= 6) evidence += 1.5;
    if (wk >= 6 && bk <= 1) evidence -= 1.5;
  }
  return evidence >= 2;
}

export function rotate180<T>(a: T[]): T[] {
  return a.slice().reverse();
}

export function scanToPosition(frames: Float32Array[][]): ScanPosition {
  const probs = combineFrames(frames);
  let { placement, conf } = decode(probs);
  const rotated = looksRotated(placement);
  if (rotated) {
    placement = rotate180(placement);
    conf = rotate180(conf);
  }
  const uncertain = conf.flatMap((c, i) => (c < UNCERTAIN_BELOW ? [i] : []));
  return { placement, uncertain, rotated, minConfidence: Math.min(...conf) };
}
