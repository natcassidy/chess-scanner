"""Write tests/fixtures/parity.json: a few squares with the JAX model's
probabilities, so tests can check the TypeScript inference matches."""
import json, sys
import numpy as np
import jax
from evalmodel import load
from train import forward, prep

weights, data, out = sys.argv[1:4]
params = load(weights)
d = np.load(data)
idx = np.random.default_rng(0).choice(len(d['x']), 12, replace=False)
x = d['x'][idx]
probs = np.asarray(jax.nn.softmax(forward(params, prep(x)), -1))
json.dump({'squares': [s.astype(int).ravel().tolist() for s in x], 'probs': probs.round(6).tolist()}, open(out, 'w'))
print('wrote', out)
