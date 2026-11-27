import type { Quad } from './homography';
import type { WorkerRequest, WorkerResponse } from './protocol';
import type { GridMetrics } from './grid';
import modelUrl from './model/pieces.bin?url';

type DistributiveOmit<T, K extends keyof never> = T extends unknown ? Omit<T, K> : never;
type Pending = { resolve: (r: WorkerResponse) => void; reject: (e: Error) => void };

/** Promise wrapper around the vision worker. */
class VisionClient {
  private worker: Worker | null = null;
  private seq = 1;
  private pending = new Map<number, Pending>();

  private get w(): Worker {
    if (!this.worker) {
      this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
      this.worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
        const p = this.pending.get(e.data.id);
        if (!p) return;
        this.pending.delete(e.data.id);
        if (e.data.type === 'error') p.reject(new Error(e.data.message));
        else p.resolve(e.data);
      };
      this.worker.onerror = (e) => {
        for (const p of this.pending.values()) p.reject(new Error(e.message || 'worker error'));
        this.pending.clear();
      };
    }
    return this.worker;
  }

  private call(msg: DistributiveOmit<WorkerRequest, 'id'>, transfer: Transferable[] = []): Promise<WorkerResponse> {
    const id = this.seq++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.w.postMessage({ ...msg, id }, transfer);
    });
  }

  /** Load the model early so the first scan doesn't wait for it. */
  warmUp(): Promise<void> {
    return this.call({ type: 'init', modelUrl }).then(() => undefined);
  }

  async detect(img: ImageData, init: Quad | null): Promise<{ quad: Quad | null; metrics: GridMetrics }> {
    const r = await this.call({ type: 'detect', rgba: img.data, w: img.width, h: img.height, init }, [img.data.buffer]);
    if (r.type !== 'detected') throw new Error('unexpected reply');
    return { quad: r.quad, metrics: r.metrics };
  }

  async read(img: ImageData, quad: Quad): Promise<{ probs: Float32Array[] | null; quad: Quad | null }> {
    const r = await this.call({ type: 'read', modelUrl, rgba: img.data, w: img.width, h: img.height, quad }, [
      img.data.buffer,
    ]);
    if (r.type !== 'read') throw new Error('unexpected reply');
    return { probs: r.probs ? unflatten(r.probs) : null, quad: r.quad };
  }

  async scanImage(img: ImageData): Promise<{ probs: Float32Array[] | null; quad: Quad | null }> {
    const r = await this.call({ type: 'scanImage', modelUrl, rgba: img.data, w: img.width, h: img.height }, [
      img.data.buffer,
    ]);
    if (r.type !== 'read') throw new Error('unexpected reply');
    return { probs: r.probs ? unflatten(r.probs) : null, quad: r.quad };
  }
}

function unflatten(f: Float32Array): Float32Array[] {
  const out: Float32Array[] = [];
  for (let i = 0; i < 64; i++) out.push(f.slice(i * 13, (i + 1) * 13));
  return out;
}

export const vision = new VisionClient();
