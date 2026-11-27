import { makeGray, normalise, sample, type Gray } from './gray';
import { mapPoint, squareToQuad, type Quad } from './homography';

export const BOARD_PX = 256;
export const SQ_PX = BOARD_PX / 8;

/**
 * Warp the 8x8 area to BOARD_PX x BOARD_PX: bilinear samples on a 2x grid,
 * averaged 2x2, then percentile-normalised.  Mirrors training/synth.py.
 */
export function warpBoard(src: Gray, quad: Quad): Gray {
  const big = BOARD_PX * 2;
  const m = squareToQuad(quad);
  const out = makeGray(BOARD_PX, BOARD_PX);
  for (let y = 0; y < BOARD_PX; y++) {
    for (let x = 0; x < BOARD_PX; x++) {
      let acc = 0;
      for (let dy = 0; dy < 2; dy++) {
        for (let dx = 0; dx < 2; dx++) {
          // cv2.warpPerspective maps pixel centres: dst (X,Y) <- H^-1 (X,Y)
          // with corners at dst 0 and `big`, so u = X / big.
          const u = (2 * x + dx) / big;
          const v = (2 * y + dy) / big;
          const p = mapPoint(m, u, v);
          acc += sample(src, p[0], p[1]);
        }
      }
      out.data[y * BOARD_PX + x] = acc / 4;
    }
  }
  return normalise(out);
}

/** Split a warped board into 64 squares (row-major from the top-left). */
export function squares(board: Gray): Float32Array[] {
  const out: Float32Array[] = [];
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const sq = new Float32Array(SQ_PX * SQ_PX);
      for (let y = 0; y < SQ_PX; y++) {
        for (let x = 0; x < SQ_PX; x++) {
          sq[y * SQ_PX + x] = board.data[(r * SQ_PX + y) * BOARD_PX + c * SQ_PX + x];
        }
      }
      out.push(sq);
    }
  }
  return out;
}
