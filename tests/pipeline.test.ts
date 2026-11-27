/**
 * End-to-end vision test on synthetic guide-box captures
 * (training/gen_testframes.py).  Uses tests/fixtures by default; set
 * CS_TESTFRAMES to a directory of freshly generated frames for a bigger run.
 */
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { describe, expect, it } from 'vitest';
import { fromU8 } from '../src/vision/gray';
import { loadModel } from '../src/vision/classifier';
import { locateBoard, readBoard } from '../src/vision/recognize';
import { scanToPosition } from '../src/chess/fromScan';
import type { Quad } from '../src/vision/homography';

const dir = process.env.CS_TESTFRAMES ?? path.join(__dirname, 'fixtures');
const meta: { file: string; w: number; h: number; corners: Quad | null; placement: string | null }[] = JSON.parse(
  fs.readFileSync(path.join(dir, 'frames.json'), 'utf8'),
);
const modelPath = path.join(__dirname, '../src/vision/model/pieces.bin');
const model = fs.existsSync(modelPath) ? loadModel(new Uint8Array(fs.readFileSync(modelPath)).buffer) : null;

function load(file: string, w: number, h: number) {
  const p = path.join(dir, file);
  let buf = fs.existsSync(p) ? fs.readFileSync(p) : zlib.gunzipSync(fs.readFileSync(p + '.gz'));
  return fromU8(new Uint8Array(buf), w, h);
}

function cornerError(a: Quad, b: Quad): number {
  let m = 0;
  for (let i = 0; i < 4; i++) m = Math.max(m, Math.hypot(a[i][0] - b[i][0], a[i][1] - b[i][1]));
  return m;
}

describe('vision pipeline on synthetic captures', () => {
  it('locates boards, rejects non-boards, reads positions', () => {
    let found = 0;
    let boards = 0;
    let falsePos = 0;
    let negatives = 0;
    let sqWrong = 0;
    let perfect = 0;
    const errs: number[] = [];
    const t0 = Date.now();
    for (const m of meta) {
      const img = load(m.file, m.w, m.h);
      const loc = locateBoard(img);
      if (!m.corners) {
        if (process.env.CS_VERBOSE) console.log(m.file, loc.ok, JSON.stringify(loc.metrics));
        negatives++;
        if (loc.ok) falsePos++;
        continue;
      }
      boards++;
      const sqSize = Math.hypot(m.corners[1][0] - m.corners[0][0], m.corners[1][1] - m.corners[0][1]) / 8;
      const err = cornerError(loc.quad, m.corners) / sqSize;
      errs.push(err);
      if (loc.ok && err < 0.25) found++;
      if (process.env.CS_VERBOSE) console.log(m.file, loc.ok, err.toFixed(3), JSON.stringify(loc.metrics));
      if (process.env.CS_DUMP) fs.appendFileSync(process.env.CS_DUMP, JSON.stringify({ file: m.file, quad: loc.quad, ok: loc.ok }) + '\n');
      if (model && loc.ok && m.placement) {
        const probs = readBoard(model, img, loc.quad);
        const pos = scanToPosition([probs]);
        // compare in the printed frame
        const printed = pos.rotated ? pos.placement.slice().reverse() : pos.placement;
        let wrong = 0;
        for (let i = 0; i < 64; i++) if (printed[i] !== m.placement[i]) wrong++;
        sqWrong += wrong;
        if (wrong === 0) perfect++;
        if (process.env.CS_VERBOSE && wrong) console.log('  wrong squares', wrong);
      }
    }
    const ms = (Date.now() - t0) / meta.length;
    errs.sort((a, b) => a - b);
    console.log(
      `boards ${boards}: located ${found}, median err ${errs[errs.length >> 1]?.toFixed(3)} sq; ` +
        `false positives ${falsePos}/${negatives}; perfect boards ${perfect}; wrong squares ${sqWrong}; ${ms.toFixed(0)} ms/frame`,
    );
    // The synthetic captures are deliberately harsher than real use (strong
    // tilt, glare, blur, partial neighbours). These are regression guards.
    expect(found / boards).toBeGreaterThanOrEqual(0.75);
    expect(falsePos).toBe(0);
    if (model) expect(perfect / found).toBeGreaterThanOrEqual(0.75);
  }, 600_000);
});
