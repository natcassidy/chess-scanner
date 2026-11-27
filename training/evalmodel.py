"""python evalmodel.py weights.bin data.npz -> accuracy + confusion"""
import json, sys
import numpy as np
import jax.numpy as jnp
from train import forward, prep, evaluate
from synth import CLASSES

def load(path):
    meta = json.load(open(path.replace('.bin', '.json')))
    flat = np.fromfile(path, np.float16).astype(np.float32)
    arrs, off = [], 0
    for sh in meta['shapes']:
        n = int(np.prod(sh)); arrs.append(jnp.asarray(flat[off:off+n].reshape(sh))); off += n
    return [(arrs[i], arrs[i+1]) for i in range(0, len(arrs), 2)]

if __name__ == '__main__':
    params = load(sys.argv[1])
    for p in sys.argv[2:]:
        d = np.load(p)
        acc, bok, pred = evaluate(params, d['x'], d['y'].astype(np.int32))
        print(p, f'sq {acc:.4f} board {bok:.3f}')
        cm = np.zeros((13, 13), int)
        for t, q in zip(d['y'], pred): cm[t, q] += 1
        print('     ' + ' '.join(f'{c:>5}' for c in CLASSES))
        for i, c in enumerate(CLASSES): print(f'{c:>4} ' + ' '.join(f'{v:5d}' for v in cm[i]))
