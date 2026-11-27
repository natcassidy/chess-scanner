/// <reference lib="webworker" />
import { fromRGBA, resizeArea } from './gray';
import { detectGrid } from './grid';
import { loadModel, type Model } from './classifier';
import { locateBoard, readBoard, refineAt } from './recognize';
import { scaleQuad, type Quad } from './homography';
import type { WorkerRequest, WorkerResponse } from './protocol';

let model: Model | null = null;
let modelPromise: Promise<Model> | null = null;

function getModel(url: string): Promise<Model> {
  if (model) return Promise.resolve(model);
  modelPromise ??= fetch(url)
    .then((r) => {
      if (!r.ok) throw new Error(`model ${r.status}`);
      return r.arrayBuffer();
    })
    .then((b) => (model = loadModel(b)));
  return modelPromise;
}

const post = (m: WorkerResponse, transfer: Transferable[] = []) => (self as DedicatedWorkerGlobalScope).postMessage(m, transfer);

self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data;
  try {
    switch (msg.type) {
      case 'init':
        await getModel(msg.modelUrl);
        post({ type: 'ready', id: msg.id });
        break;
      case 'detect': {
        const g = fromRGBA(msg.rgba, msg.w, msg.h);
        // tracks from the previous quad; falls back to a full search
        const res = detectGrid(g, { init: msg.init ?? undefined, minFrac: 0.3 });
        post({ type: 'detected', id: msg.id, quad: res.ok ? res.quad : null, metrics: res.metrics });
        break;
      }
      case 'read': {
        const m = await getModel(msg.modelUrl);
        const g = fromRGBA(msg.rgba, msg.w, msg.h);
        const r = refineAt(g, msg.quad);
        if (!r.ok) {
          post({ type: 'read', id: msg.id, probs: null, quad: null });
          break;
        }
        const probs = readBoard(m, g, r.quad);
        post({ type: 'read', id: msg.id, probs: flatten(probs), quad: r.quad });
        break;
      }
      case 'scanImage': {
        const m = await getModel(msg.modelUrl);
        let g = fromRGBA(msg.rgba, msg.w, msg.h);
        const n = Math.max(g.w, g.h);
        if (n > 1600) g = resizeArea(g, Math.round((g.w * 1600) / n), Math.round((g.h * 1600) / n));
        const loc = locateBoard(g);
        if (!loc.ok) {
          post({ type: 'read', id: msg.id, probs: null, quad: null });
          break;
        }
        const probs = readBoard(m, g, loc.quad);
        post({ type: 'read', id: msg.id, probs: flatten(probs), quad: scaleQuad(loc.quad, msg.w / g.w) as Quad });
        break;
      }
    }
  } catch (err) {
    post({ type: 'error', id: msg.id, message: String((err as Error)?.message ?? err) });
  }
};

function flatten(p: Float32Array[]): Float32Array {
  const out = new Float32Array(64 * 13);
  p.forEach((a, i) => out.set(a, i * 13));
  return out;
}
