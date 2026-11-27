"""Build a fake-camera video (y4m) of a synthetic book page for the e2e test.

usage: python make_fake_camera.py OUT_DIR [--set font-merida] [--seed 5]
Writes OUT_DIR/camera.y4m (720x1280 portrait, 40 frames with hand shake)
and OUT_DIR/truth.json with the board FEN the app should find.
"""
import argparse
import json
import os
import random
import subprocess

import cv2
import numpy as np

import synth

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--set', default='font-merida')
ap.add_argument('--seed', type=int, default=5)
ap.add_argument('--fen', default='r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N2N2/PP2BPPP/R2QKB1R')
ap.add_argument('--pieces', default='/home/user/cs-data/pieces-png')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = random.Random(a.seed)
sets = synth.load_piece_sets(a.pieces, only=[a.set])

pos = []
for row in a.fen.split('/'):
    for ch in row:
        pos.extend(['.'] * int(ch) if ch.isdigit() else [ch])
diag = synth.render_diagram(rng, sets, position=pos, sq=60)
page, corners = synth.render_page(rng, sets, diag, neighbours=False)

W, H = 720, 1280
# a 720x720 "photo" of the page region, then extend to portrait with more page
sq_img, c = synth.camera(rng, page, corners, 720, board_frac=(0.6, 0.6), max_rot=2, max_persp=0.03, strength=0.4)
frame = np.full((H, W), int(np.median(sq_img[:40])), np.uint8)
y0 = (H - 720) // 2 - 40
frame[y0:y0 + 720] = sq_img
frames_dir = os.path.join(a.out, 'frames')
os.makedirs(frames_dir, exist_ok=True)
for i in range(40):
    dx, dy = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    f = cv2.warpAffine(frame, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(os.path.join(frames_dir, f'f{i:03d}.png'), cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '15', '-i', os.path.join(frames_dir, 'f%03d.png'),
                '-pix_fmt', 'yuv420p', os.path.join(a.out, 'camera.y4m')], check=True)
json.dump({'fen': a.fen, 'set': a.set}, open(os.path.join(a.out, 'truth.json'), 'w'))
cv2.imwrite(os.path.join(a.out, 'frame.png'), frame)
print('ok', a.out)
