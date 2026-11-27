import { useEffect, useRef, useState } from 'preact/hooks';
import { vision } from '../vision/client';
import { LIVE_PX } from '../vision/recognize';
import { quadSide, scaleQuad, type Quad } from '../vision/homography';
import { scanToPosition, type ScanPosition } from '../chess/fromScan';
import { IconClose, IconHistory, IconImage } from './icons';

interface Props {
  onScanned: (pos: ScanPosition) => void;
  onRecent: () => void;
  /** back to the board that was open before (if any) */
  onClose?: () => void;
}

type Status = 'starting' | 'searching' | 'reading' | 'denied' | 'nocamera';

/** Extra margin analysed around the guide box (fraction of the box side). */
const MARGIN = 0.12;
/** Consecutive steady detections needed before capturing. */
const STEADY_FRAMES = 3;
/** Frames read and combined for one scan. */
const READ_FRAMES = 3;
const ZOOM_KEY = 'chess-scanner.zoom';

interface Box {
  x: number;
  y: number;
  s: number;
}
interface Geometry {
  crop: Box; // in video pixels
  screen: Box; // in CSS pixels
}

const nextFrame = () => new Promise<void>((r) => requestAnimationFrame(() => r()));
const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

function loadZoom(): number {
  try {
    const z = Number(localStorage.getItem(ZOOM_KEY));
    return z >= 1 && z <= 3 ? z : 1;
  } catch {
    return 1;
  }
}

function grab(ctx: CanvasRenderingContext2D, video: HTMLVideoElement, crop: Box, size: number) {
  ctx.fillStyle = '#888';
  ctx.fillRect(0, 0, size, size);
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const sx0 = Math.max(0, crop.x);
  const sy0 = Math.max(0, crop.y);
  const sx1 = Math.min(vw, crop.x + crop.s);
  const sy1 = Math.min(vh, crop.y + crop.s);
  if (sx1 <= sx0 || sy1 <= sy0) return;
  const k = size / crop.s;
  ctx.drawImage(video, sx0, sy0, sx1 - sx0, sy1 - sy0, (sx0 - crop.x) * k, (sy0 - crop.y) * k, (sx1 - sx0) * k, (sy1 - sy0) * k);
}

function maxCornerMove(a: Quad, b: Quad): number {
  let m = 0;
  for (let i = 0; i < 4; i++) m = Math.max(m, Math.hypot(a[i][0] - b[i][0], a[i][1] - b[i][1]));
  return m;
}

function plausible(p: ScanPosition): boolean {
  const white = p.placement.filter((s) => s !== '.' && s === s.toUpperCase()).length;
  const black = p.placement.filter((s) => s !== '.' && s === s.toLowerCase()).length;
  return white <= 16 && black <= 16;
}

export function ScannerScreen({ onScanned, onRecent, onClose }: Props) {
  const rootRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const [status, setStatus] = useState<Status>('starting');
  const [hint, setHint] = useState<string | null>(null);
  const [outline, setOutline] = useState<[number, number][] | null>(null);
  const [steady, setSteady] = useState(0);
  const [box, setBox] = useState<Box>({ x: 0, y: 0, s: 0 });
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [zoom, setZoom] = useState(loadZoom);
  const [zoomBadge, setZoomBadge] = useState(false);
  const boxRef = useRef(box);
  const zoomRef = useRef(zoom);
  const pausedRef = useRef(false);
  const onScannedRef = useRef(onScanned);
  onScannedRef.current = onScanned;
  boxRef.current = box;
  zoomRef.current = zoom;

  // layout: square guide box, a bit above centre
  useEffect(() => {
    const el = rootRef.current!;
    const update = () => {
      const W = el.clientWidth;
      const H = el.clientHeight;
      const s = Math.round(Math.min(W * 0.88, H * 0.56));
      setSize({ w: W, h: H });
      setBox({ x: Math.round((W - s) / 2), y: Math.round(Math.max(H * 0.12, (H - s) / 2 - H * 0.05)), s });
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    vision.warmUp().catch(() => {});
  }, []);

  // camera + detection loop
  useEffect(() => {
    const video = videoRef.current!;
    let stream: MediaStream | null = null;
    let gen = 0;
    const small = document.createElement('canvas');
    small.width = small.height = LIVE_PX;
    const sctx = small.getContext('2d', { willReadFrequently: true })!;
    sctx.imageSmoothingQuality = 'high';

    const geometry = (): Geometry | null => {
      const el = rootRef.current;
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      const b = boxRef.current;
      if (!el || !vw || !vh || !b.s) return null;
      const W = el.clientWidth;
      const H = el.clientHeight;
      const c = Math.max(W / vw, H / vh) * zoomRef.current;
      const ox = (W - vw * c) / 2;
      const oy = (H - vh * c) / 2;
      const m = b.s * MARGIN;
      const screen = { x: b.x - m, y: b.y - m, s: b.s + 2 * m };
      return { screen, crop: { x: (screen.x - ox) / c, y: (screen.y - oy) / c, s: screen.s / c } };
    };

    const toScreen = (q: Quad, g: Geometry) =>
      q.map(([x, y]) => [g.screen.x + (x * g.screen.s) / LIVE_PX, g.screen.y + (y * g.screen.s) / LIVE_PX]) as [number, number][];

    const stopStream = () => {
      stream?.getTracks().forEach((t) => t.stop());
      stream = null;
      video.srcObject = null;
    };

    const run = async (myGen: number) => {
      const alive = () => myGen === gen;
      const st = { prev: null as Quad | null, steady: 0, misses: 0, lastFound: performance.now() };
      const reset = () => {
        st.prev = null;
        st.steady = 0;
        setOutline(null);
        setSteady(0);
      };
      while (alive()) {
        if (pausedRef.current || video.readyState < 2) {
          await nextFrame();
          continue;
        }
        const g = geometry();
        if (!g) {
          await nextFrame();
          continue;
        }
        grab(sctx, video, g.crop, LIVE_PX);
        let res;
        try {
          res = await vision.detect(sctx.getImageData(0, 0, LIVE_PX, LIVE_PX), st.prev);
        } catch {
          await sleep(300);
          continue;
        }
        if (!alive() || pausedRef.current) continue;
        const now = performance.now();
        if (res.quad) {
          st.lastFound = now;
          st.misses = 0;
          const moved = st.prev ? maxCornerMove(st.prev, res.quad) : Infinity;
          st.steady = moved < LIVE_PX * 0.02 ? st.steady + 1 : 0;
          st.prev = res.quad;
          setOutline(toScreen(res.quad, g));
          setSteady(Math.min(1, st.steady / STEADY_FRAMES));
          setHint(res.metrics.sizeFrac < 0.42 ? 'Move a little closer' : null);
          if (st.steady >= STEADY_FRAMES) {
            const ok = await capture(st.prev, alive);
            if (!alive()) return;
            if (ok) return;
            reset();
          }
        } else {
          st.misses++;
          if (st.misses >= 2) reset();
          if (now - st.lastFound > 3500) setHint('Fit the diagram inside the box');
        }
        await nextFrame();
      }
    };

    /** Read a few frames at high resolution and hand the result over. */
    const capture = async (quadLive: Quad, alive: () => boolean): Promise<boolean> => {
      setStatus('reading');
      setHint(null);
      const frames: Float32Array[][] = [];
      let quad = quadLive;
      const big = document.createElement('canvas');
      const bctx = big.getContext('2d', { willReadFrequently: true })!;
      for (let k = 0; k < READ_FRAMES && alive(); k++) {
        const g = geometry();
        if (!g) break;
        // enough resolution for the board to be ~720px wide, but no more
        // than the camera actually delivers for this region
        const S = Math.round(Math.min(Math.max(g.crop.s, LIVE_PX), Math.max(LIVE_PX, (LIVE_PX * 720) / quadSide(quad))));
        big.width = big.height = S;
        bctx.imageSmoothingQuality = 'high';
        grab(bctx, video, g.crop, S);
        try {
          const r = await vision.read(bctx.getImageData(0, 0, S, S), scaleQuad(quad, S / LIVE_PX));
          if (r.probs) frames.push(r.probs);
          if (r.quad) quad = scaleQuad(r.quad, LIVE_PX / S);
        } catch {
          /* try the next frame */
        }
        await nextFrame();
      }
      if (!alive()) return false;
      if (frames.length) {
        const pos = scanToPosition(frames);
        if (plausible(pos)) {
          gen++;
          stopStream();
          onScannedRef.current(pos);
          return true;
        }
      }
      setStatus('searching');
      setHint('Couldn’t read that – hold steady');
      return false;
    };

    const start = async () => {
      const myGen = ++gen;
      setStatus('starting');
      if (!navigator.mediaDevices?.getUserMedia) {
        setStatus('nocamera');
        return;
      }
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { facingMode: { ideal: 'environment' }, width: { ideal: 3840 }, height: { ideal: 2160 } },
        });
      } catch (e) {
        if (myGen !== gen) return;
        setStatus((e as Error)?.name === 'NotAllowedError' ? 'denied' : 'nocamera');
        return;
      }
      if (myGen !== gen) {
        stopStream();
        return;
      }
      video.srcObject = stream;
      try {
        await video.play();
      } catch {
        /* autoplay of a muted inline video is allowed; ignore */
      }
      setStatus('searching');
      run(myGen);
    };

    const onVis = () => {
      if (document.hidden) {
        gen++;
        stopStream();
      } else start();
    };
    document.addEventListener('visibilitychange', onVis);
    start();
    return () => {
      gen++;
      document.removeEventListener('visibilitychange', onVis);
      stopStream();
    };
  }, []);

  // pinch to zoom (digital)
  useEffect(() => {
    const el = rootRef.current!;
    let d0 = 0;
    let z0 = 1;
    let badgeTimer = 0;
    const dist = (t: TouchList) => Math.hypot(t[0].clientX - t[1].clientX, t[0].clientY - t[1].clientY);
    const onStart = (e: TouchEvent) => {
      if (e.touches.length === 2) {
        d0 = dist(e.touches);
        z0 = zoomRef.current;
      }
    };
    const onMove = (e: TouchEvent) => {
      if (e.touches.length === 2 && d0) {
        e.preventDefault();
        const z = Math.min(3, Math.max(1, (z0 * dist(e.touches)) / d0));
        setZoom(Math.round(z * 20) / 20);
        setZoomBadge(true);
        clearTimeout(badgeTimer);
        badgeTimer = window.setTimeout(() => setZoomBadge(false), 900);
      }
    };
    const onEnd = () => {
      if (d0) {
        d0 = 0;
        try {
          localStorage.setItem(ZOOM_KEY, String(zoomRef.current));
        } catch {
          /* ignore */
        }
      }
    };
    el.addEventListener('touchstart', onStart, { passive: true });
    el.addEventListener('touchmove', onMove, { passive: false });
    el.addEventListener('touchend', onEnd);
    return () => {
      el.removeEventListener('touchstart', onStart);
      el.removeEventListener('touchmove', onMove);
      el.removeEventListener('touchend', onEnd);
    };
  }, []);

  const importPhoto = async (file: File) => {
    pausedRef.current = true;
    setStatus('reading');
    setOutline(null);
    try {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.src = url;
      await img.decode();
      const n = Math.max(img.naturalWidth, img.naturalHeight);
      const k = Math.min(1, 1600 / n);
      const c = document.createElement('canvas');
      c.width = Math.round(img.naturalWidth * k);
      c.height = Math.round(img.naturalHeight * k);
      const ctx = c.getContext('2d', { willReadFrequently: true })!;
      ctx.drawImage(img, 0, 0, c.width, c.height);
      URL.revokeObjectURL(url);
      const r = await vision.scanImage(ctx.getImageData(0, 0, c.width, c.height));
      if (r.probs) {
        const pos = scanToPosition([r.probs]);
        if (plausible(pos)) {
          onScanned(pos);
          return;
        }
      }
      setHint('Couldn’t find a diagram in that photo');
    } catch {
      setHint('Couldn’t open that photo');
    }
    setStatus('searching');
    pausedRef.current = false;
  };

  const { w: W, h: H } = size;
  const locked = steady >= 1 || status === 'reading';
  let message: string;
  if (status === 'starting') message = 'Starting camera…';
  else if (status === 'reading') message = 'Reading position…';
  else if (hint) message = hint;
  else if (outline) message = 'Hold still…';
  else message = 'Point at a diagram';

  return (
    <div class="scanner" ref={rootRef}>
      <video ref={videoRef} class="scanner-video" playsInline muted autoPlay style={{ transform: `scale(${zoom})` }} />
      {W > 0 && (
        <svg class="scanner-overlay" width={W} height={H} viewBox={`0 0 ${W} ${H}`}>
          <path
            fill-rule="evenodd"
            fill="rgba(0,0,0,0.55)"
            d={`M0 0H${W}V${H}H0Z M${box.x} ${box.y}h${box.s}v${box.s}h${-box.s}Z`}
          />
          <GuideCorners box={box} active={locked} />
          {outline && (
            <polygon
              class={'scanner-outline' + (locked ? ' locked' : '')}
              points={outline.map((p) => p.join(',')).join(' ')}
              style={{ opacity: 0.55 + 0.45 * steady }}
            />
          )}
        </svg>
      )}
      {status === 'reading' && <div class="scanner-flash" />}

      {onClose && (
        <button class="icon-btn scanner-close" onClick={onClose} aria-label="Back to board">
          <IconClose />
        </button>
      )}
      {zoomBadge && <div class="zoom-badge">{zoom.toFixed(1)}×</div>}

      {(status === 'denied' || status === 'nocamera') ? (
        <div class="scanner-error">
          <h2>{status === 'denied' ? 'Camera access is off' : 'No camera available'}</h2>
          <p>
            {status === 'denied'
              ? 'Allow camera access for this app (iPhone Settings → Safari → Camera), then reopen it. You can also pick a photo instead.'
              : 'You can still scan a diagram from a photo.'}
          </p>
          <button class="btn primary" onClick={() => fileRef.current?.click()}>
            <IconImage /> Choose photo
          </button>
        </div>
      ) : (
        <div class="scanner-status" style={{ top: `${box.y + box.s + 18}px` }}>
          {status === 'reading' && <span class="spinner" />}
          {message}
        </div>
      )}

      <div class="scanner-bar">
        <button class="pill-btn" onClick={onRecent}>
          <IconHistory /> Recent
        </button>
        <button class="pill-btn" onClick={() => fileRef.current?.click()}>
          <IconImage /> Photo
        </button>
      </div>
      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => {
          const f = (e.currentTarget as HTMLInputElement).files?.[0];
          (e.currentTarget as HTMLInputElement).value = '';
          if (f) importPhoto(f);
        }}
      />
    </div>
  );
}

function GuideCorners({ box, active }: { box: Box; active: boolean }) {
  const l = Math.round(box.s * 0.12);
  const { x, y, s } = box;
  const d = [
    `M${x} ${y + l}V${y}H${x + l}`,
    `M${x + s - l} ${y}H${x + s}V${y + l}`,
    `M${x + s} ${y + s - l}V${y + s}H${x + s - l}`,
    `M${x + l} ${y + s}H${x}V${y + s - l}`,
  ].join(' ');
  return <path class={'guide-corners' + (active ? ' active' : '')} d={d} />;
}
