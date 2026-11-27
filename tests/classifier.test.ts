import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { classifySquare, loadModel } from '../src/vision/classifier';

describe('square classifier', () => {
  it('matches the training framework output', () => {
    const model = loadModel(new Uint8Array(fs.readFileSync(path.join(__dirname, '../src/vision/model/pieces.bin'))).buffer);
    const { squares, probs } = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures/parity.json'), 'utf8'));
    squares.forEach((sq: number[], i: number) => {
      const out = classifySquare(model, Float32Array.from(sq, (v) => v / 255));
      for (let k = 0; k < 13; k++) expect(out[k]).toBeCloseTo(probs[i][k], 3);
    });
  });
});
