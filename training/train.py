"""Train the 13-class square classifier (JAX, CPU).

Architecture (mirrored exactly by src/vision/classifier.ts):
  input 32x32x1 (value - 0.5)
  conv3x3 16  + ReLU + maxpool2   -> 16x16x16
  conv3x3 32  + ReLU + maxpool2   -> 8x8x32
  conv3x3 64  + ReLU + maxpool2   -> 4x4x64
  flatten (HWC order) -> dense 128 + ReLU -> dense 13

usage: python train.py TRAIN.npz [VAL.npz ...] --out weights.bin --epochs 8
"""
import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

CONVS = [(1, 16), (16, 32), (32, 64)]
DENSE = [(4 * 4 * 64, 128), (128, 13)]


def init(key):
    params = []
    for cin, cout in CONVS:
        key, k = jax.random.split(key)
        w = jax.random.normal(k, (3, 3, cin, cout)) * np.sqrt(2.0 / (9 * cin))
        params.append((w, jnp.zeros(cout)))
    for din, dout in DENSE:
        key, k = jax.random.split(key)
        w = jax.random.normal(k, (din, dout)) * np.sqrt(2.0 / din)
        params.append((w, jnp.zeros(dout)))
    return params


def forward(params, x, key=None, train=False):
    h = x
    for w, b in params[:3]:
        h = jax.lax.conv_general_dilated(h, w, (1, 1), 'SAME', dimension_numbers=('NHWC', 'HWIO', 'NHWC')) + b
        h = jax.nn.relu(h)
        h = jax.lax.reduce_window(h, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')
    h = h.reshape(h.shape[0], -1)
    w, b = params[3]
    h = jax.nn.relu(h @ w + b)
    if train:
        keep = jax.random.bernoulli(key, 0.7, h.shape)
        h = jnp.where(keep, h / 0.7, 0)
    w, b = params[4]
    return h @ w + b


def augment(rng, xb):
    """xb uint8 (B,32,32) -> float (B,32,32,1) centred."""
    b = xb.shape[0]
    x = xb.astype(np.float32) / 255.0
    # horizontal flip keeps the class for every piece
    flip = rng.random(b) < 0.5
    x[flip] = x[flip, :, ::-1]
    # small shifts (grid error)
    out = np.empty_like(x)
    pad = np.pad(x, ((0, 0), (2, 2), (2, 2)), mode='edge')
    dx = rng.integers(-2, 3, b); dy = rng.integers(-2, 3, b)
    for i in range(b):
        out[i] = pad[i, 2 + dy[i]:34 + dy[i], 2 + dx[i]:34 + dx[i]]
    # contrast / brightness / gamma
    c = rng.uniform(0.7, 1.15, (b, 1, 1)); m = rng.uniform(-0.1, 0.1, (b, 1, 1))
    out = np.clip((out - 0.5) * c + 0.5 + m, 0, 1) ** rng.uniform(0.8, 1.25, (b, 1, 1))
    out += rng.normal(0, 0.02, out.shape)
    return (out - 0.5)[..., None].astype(np.float32)


def prep(x):
    return (x.astype(np.float32) / 255.0 - 0.5)[..., None]


def evaluate(params, x, y, bs=2048):
    preds = []
    f = jax.jit(forward)
    for i in range(0, len(x), bs):
        preds.append(np.asarray(jnp.argmax(f(params, prep(x[i:i + bs])), -1)))
    p = np.concatenate(preds)
    acc = (p == y).mean()
    # board-level: fraction of 64-square boards with zero errors
    nb = len(y) // 64
    board_ok = (p[:nb * 64].reshape(nb, 64) == y[:nb * 64].reshape(nb, 64)).all(1).mean()
    return acc, board_ok, p


def fp16_export(params, path):
    arrs = []
    meta = []
    for w, b in params:
        for a in (w, b):
            a = np.asarray(a, np.float32)
            meta.append(list(a.shape))
            arrs.append(a.ravel())
    flat = np.concatenate(arrs).astype(np.float16)
    with open(path, 'wb') as f:
        f.write(flat.tobytes())
    with open(path.replace('.bin', '.json'), 'w') as f:
        json.dump({'shapes': meta, 'dtype': 'float16', 'count': int(flat.size)}, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('train', nargs='+')
    ap.add_argument('--val', nargs='*', default=[])
    ap.add_argument('--out', default=None)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-3)
    a = ap.parse_args()

    xs, ys = [], []
    for p in a.train:
        d = np.load(p); xs.append(d['x']); ys.append(d['y'])
    x, y = np.concatenate(xs), np.concatenate(ys).astype(np.int32)
    vals = [(p, np.load(p)) for p in a.val]
    print('train', x.shape)

    key = jax.random.PRNGKey(0)
    params = init(key)
    steps = a.epochs * (len(x) // a.bs)
    sched = optax.warmup_cosine_decay_schedule(0, a.lr, 300, steps, a.lr * 0.02)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4))
    state = opt.init(params)

    @jax.jit
    def step(params, state, xb, yb, key):
        def loss_fn(p):
            logits = forward(p, xb, key, train=True)
            onehot = optax.smooth_labels(jax.nn.one_hot(yb, 13), 0.03)
            return optax.softmax_cross_entropy(logits, onehot).mean()
        loss, g = jax.value_and_grad(loss_fn)(params)
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state, loss

    rng = np.random.default_rng(0)
    for ep in range(a.epochs):
        t = time.time()
        perm = rng.permutation(len(x))
        tot = 0.0
        for i in range(len(x) // a.bs):
            idx = perm[i * a.bs:(i + 1) * a.bs]
            key, k = jax.random.split(key)
            params, state, loss = step(params, state, augment(rng, x[idx]), y[idx], k)
            tot += float(loss)
        msg = f'epoch {ep} loss {tot / (len(x) // a.bs):.4f} {time.time() - t:.0f}s'
        for name, d in vals:
            acc, bok, _ = evaluate(params, d['x'], d['y'].astype(np.int32))
            msg += f' | {name.split("/")[-1]} sq {acc:.4f} board {bok:.3f}'
        print(msg, flush=True)
    if a.out:
        fp16_export(params, a.out)
        print('saved', a.out)


if __name__ == '__main__':
    main()
