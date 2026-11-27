import { resizeArea, type Gray } from './gray';
import { detectGrid, type GridResult } from './grid';
import { quadSide, scaleQuad, type Quad } from './homography';
import { classifySquare, type Model } from './classifier';
import { squares, warpBoard } from './warp';

/** Resolution used for live (per-frame) detection. */
export const LIVE_PX = 256;
/** The board is resampled to about this width before reading pieces. */
const READ_BOARD_PX = 600;

/** Coarse detection on a small image, then refinement at higher resolution. */
export function locateBoard(capture: Gray, hint?: Quad): GridResult {
  const n = Math.min(capture.w, capture.h);
  const f = Math.min(1, LIVE_PX / n);
  const small = f < 1 ? resizeArea(capture, Math.round(capture.w * f), Math.round(capture.h * f)) : capture;
  const coarse = detectGrid(small, { init: hint ? scaleQuad(hint, f) : undefined });
  if (!coarse.ok) return { ...coarse, quad: scaleQuad(coarse.quad, 1 / f) };
  return refineAt(capture, scaleQuad(coarse.quad, 1 / f));
}

/** Refine a quad on an image resampled so the board is ~READ_BOARD_PX wide. */
export function refineAt(capture: Gray, quad: Quad): GridResult {
  const { img, f } = shrinkFor(capture, quad);
  const r = detectGrid(img, { init: scaleQuad(quad, f), refineOnly: true, minStep: 0.25 });
  return { ...r, quad: scaleQuad(r.quad, 1 / f) };
}

function shrinkFor(capture: Gray, quad: Quad): { img: Gray; f: number } {
  const side = quadSide(quad);
  const f = Math.min(1, READ_BOARD_PX / side);
  if (f > 0.95) return { img: capture, f: 1 };
  const w = Math.round(capture.w * f);
  const h = Math.round(capture.h * f);
  return { img: resizeArea(capture, w, h), f: w / capture.w };
}

/** Classify all 64 squares. Returns probabilities per square (row-major from top-left). */
export function readBoard(model: Model, capture: Gray, quad: Quad): Float32Array[] {
  const { img, f } = shrinkFor(capture, quad);
  const board = warpBoard(img, scaleQuad(quad, f));
  return squares(board).map((s) => classifySquare(model, s));
}
