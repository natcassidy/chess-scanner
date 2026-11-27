import fs from 'node:fs';
import path from 'node:path';
import { it } from 'vitest';
import { fromU8, resizeArea } from '../src/vision/gray';
import { detectGrid } from '../src/vision/grid';

it.skipIf(!process.env.CS_TESTFRAMES)('timing of live detection at 256px', () => {
  const dir = process.env.CS_TESTFRAMES!;
  const meta = JSON.parse(fs.readFileSync(path.join(dir, 'frames.json'), 'utf8'));
  let full = 0, track = 0;
  const N = 20;
  for (let i = 0; i < N; i++) {
    const m = meta[i];
    const img = resizeArea(fromU8(new Uint8Array(fs.readFileSync(path.join(dir, m.file))), m.w, m.h), 256, 256);
    let t = performance.now();
    const r = detectGrid(img);
    full += performance.now() - t;
    t = performance.now();
    detectGrid(img, { init: r.quad });
    track += performance.now() - t;
  }
  console.log(`full search ${(full / N).toFixed(0)} ms, tracking ${(track / N).toFixed(0)} ms`);
});
