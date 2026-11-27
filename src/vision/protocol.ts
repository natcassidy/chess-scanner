import type { GridMetrics } from './grid';
import type { Quad } from './homography';

export type WorkerRequest =
  | { type: 'init'; id: number; modelUrl: string }
  | { type: 'detect'; id: number; rgba: Uint8ClampedArray; w: number; h: number; init: Quad | null }
  | { type: 'read'; id: number; modelUrl: string; rgba: Uint8ClampedArray; w: number; h: number; quad: Quad }
  | { type: 'scanImage'; id: number; modelUrl: string; rgba: Uint8ClampedArray; w: number; h: number };

export type WorkerResponse =
  | { type: 'ready'; id: number }
  | { type: 'detected'; id: number; quad: Quad | null; metrics: GridMetrics }
  | { type: 'read'; id: number; probs: Float32Array | null; quad: Quad | null }
  | { type: 'error'; id: number; message: string };
