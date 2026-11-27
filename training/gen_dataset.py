"""Generate square-crop training data: python gen_dataset.py OUT.npz N_BOARDS SEED [--exclude a,b] [--only a,b]"""
import argparse
import multiprocessing as mp
import random

import numpy as np

import synth

PIECES = '/home/user/cs-data/pieces-png'


def work(args):
    seed, n, only, exclude = args
    sets = synth.load_piece_sets(PIECES, only=only, exclude=exclude)
    rng = random.Random(seed)
    xs, ys = [], []
    for _ in range(n):
        x, y = synth.training_sample(rng, sets)
        xs.append(x); ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('n', type=int); ap.add_argument('seed', type=int)
    ap.add_argument('--pieces', default=PIECES)
    ap.add_argument('--only'); ap.add_argument('--exclude')
    a = ap.parse_args()
    PIECES = a.pieces
    only = a.only.split(',') if a.only else None
    exclude = a.exclude.split(',') if a.exclude else None
    chunks = 32
    jobs = [(a.seed * 1000 + i, a.n // chunks, only, exclude) for i in range(chunks)]
    with mp.Pool(mp.cpu_count()) as pool:
        res = pool.map(work, jobs)
    x = np.concatenate([r[0] for r in res]); y = np.concatenate([r[1] for r in res])
    np.savez_compressed(a.out, x=x, y=y)
    print(a.out, x.shape, np.bincount(y, minlength=13))
