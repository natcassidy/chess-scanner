"""Write synthetic guide-box captures for the TypeScript pipeline tests.

usage: python gen_testframes.py OUT_DIR N SEED [--only sets] [--size 900]
Each frame: frame_XXX.gray (uint8, size*size) + frames.json with the true
board corners and the 64-square placement (row-major from the top-left).
Also writes a few board-free frames (text only) as negatives.
"""
import argparse
import json
import os
import random

import numpy as np

import synth

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('n', type=int); ap.add_argument('seed', type=int)
ap.add_argument('--only'); ap.add_argument('--size', type=int, default=900)
ap.add_argument('--negatives', type=int, default=10)
ap.add_argument('--pieces', default='/home/user/cs-data/pieces-png')
ap.add_argument('--png', action='store_true', help='also write PNGs for viewing')
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
sets = synth.load_piece_sets(a.pieces, only=a.only.split(',') if a.only else None)
rng = random.Random(a.seed)
meta = []
for i in range(a.n):
    diag = synth.render_diagram(rng, sets, legal=True)
    page, corners = synth.render_page(rng, sets, diag)
    # the guide box has a 10% margin around the region the user aims at;
    # the board fills 45-85% of the crop
    for _ in range(20):
        img, c = synth.camera(rng, page, corners, a.size, board_frac=(0.45, 0.85))
        if c.min() > 0.02 * a.size and c.max() < 0.98 * a.size:
            break
    name = f'frame_{i:03d}.gray'
    img.tofile(os.path.join(a.out, name))
    if a.png:
        import cv2
        cv2.imwrite(os.path.join(a.out, name.replace('.gray', '.png')), img)
    # position stored in the diagram's printed frame
    meta.append({'file': name, 'w': a.size, 'h': a.size, 'corners': c.tolist(),
                 'placement': ''.join(diag.position)})
for j in range(a.negatives):
    size = a.size
    page = np.full((size, size), rng.uniform(0.85, 1.0), np.float32)
    synth._random_text(rng, page, np.zeros_like(page), line_h=rng.randint(12, 30))
    img, _ = synth.camera(rng, page, np.array([[300, 300], [600, 300], [600, 600], [300, 600]], np.float32), size,
                          board_frac=(0.5, 0.8))
    name = f'neg_{j:03d}.gray'
    img.tofile(os.path.join(a.out, name))
    meta.append({'file': name, 'w': size, 'h': size, 'corners': None, 'placement': None})
with open(os.path.join(a.out, 'frames.json'), 'w') as f:
    json.dump(meta, f)
print('wrote', len(meta))
