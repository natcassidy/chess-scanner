export type Pt = [number, number];
/** Board corners in image pixels: top-left, top-right, bottom-right, bottom-left. */
export type Quad = [Pt, Pt, Pt, Pt];

/** Projective map from the unit square to a quad (Heckbert's closed form). */
export interface SquareToQuad {
  a: number; b: number; c: number;
  d: number; e: number; f: number;
  g: number; h: number;
}

export function squareToQuad(q: Quad): SquareToQuad {
  const [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = q;
  const dx3 = x0 - x1 + x2 - x3;
  const dy3 = y0 - y1 + y2 - y3;
  if (Math.abs(dx3) < 1e-9 && Math.abs(dy3) < 1e-9) {
    return { a: x1 - x0, b: x3 - x0, c: x0, d: y1 - y0, e: y3 - y0, f: y0, g: 0, h: 0 };
  }
  const dx1 = x1 - x2;
  const dx2 = x3 - x2;
  const dy1 = y1 - y2;
  const dy2 = y3 - y2;
  const den = dx1 * dy2 - dx2 * dy1;
  const g = (dx3 * dy2 - dx2 * dy3) / den;
  const h = (dx1 * dy3 - dx3 * dy1) / den;
  return {
    a: x1 - x0 + g * x1,
    b: x3 - x0 + h * x3,
    c: x0,
    d: y1 - y0 + g * y1,
    e: y3 - y0 + h * y3,
    f: y0,
    g,
    h,
  };
}

export function mapPoint(m: SquareToQuad, u: number, v: number): Pt {
  const w = m.g * u + m.h * v + 1;
  return [(m.a * u + m.b * v + m.c) / w, (m.d * u + m.e * v + m.f) / w];
}

export function scaleQuad(q: Quad, s: number, dx = 0, dy = 0): Quad {
  return q.map(([x, y]) => [x * s + dx, y * s + dy]) as Quad;
}

export function quadSide(q: Quad): number {
  const d = (a: Pt, b: Pt) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  return (d(q[0], q[1]) + d(q[1], q[2]) + d(q[2], q[3]) + d(q[3], q[0])) / 4;
}
