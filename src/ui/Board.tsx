import { useEffect, useRef } from 'preact/hooks';
import { Chessground } from 'chessground';
import type { Api } from 'chessground/api';
import type { Config } from 'chessground/config';
import type { Key } from 'chessground/types';

export interface BoardProps {
  fen: string;
  orientation: 'white' | 'black';
  /** null = no moves allowed */
  movable: 'white' | 'black' | 'both' | null;
  dests?: Map<Key, Key[]>;
  lastMove?: [Key, Key];
  check?: boolean;
  turnColor?: 'white' | 'black';
  highlights?: Map<Key, string>;
  /** edit mode: pieces move freely and can be dragged off */
  free?: boolean;
  /** edit mode with a palette tool selected: taps place pieces, no dragging */
  tapMode?: boolean;
  /** bump to force the board back in sync with `fen` */
  sync?: number;
  onMove?: (from: Key, to: Key) => void;
  onSelect?: (key: Key) => void;
  onEditChange?: (fen: string) => void;
}

export function Board(p: BoardProps) {
  const el = useRef<HTMLDivElement>(null);
  const api = useRef<Api | null>(null);
  const cb = useRef(p);
  cb.current = p;

  const config = (): Config => ({
    fen: p.fen,
    orientation: p.orientation,
    turnColor: p.turnColor ?? 'white',
    check: p.check ?? false,
    lastMove: p.lastMove,
    coordinates: true,
    autoCastle: true,
    highlight: { lastMove: true, check: true, custom: p.highlights ?? new Map() },
    animation: { enabled: true, duration: 160 },
    premovable: { enabled: false },
    draggable: { enabled: !p.tapMode, showGhost: true, deleteOnDropOff: !!p.free },
    selectable: { enabled: true },
    movable: p.free
      ? { free: true, color: p.tapMode ? undefined : 'both', showDests: false }
      : {
          free: false,
          color: p.movable ?? undefined,
          dests: p.movable ? p.dests ?? new Map() : new Map(),
          showDests: true,
        },
    drawable: { enabled: false },
    events: {
      move: () => {
        if (cb.current.free) cb.current.onEditChange?.(api.current!.getFen());
      },
      change: () => {
        if (cb.current.free) cb.current.onEditChange?.(api.current!.getFen());
      },
      select: (key: Key) => cb.current.onSelect?.(key),
    },
  });

  useEffect(() => {
    const c = config();
    c.movable!.events = { after: (o, d) => cb.current.onMove?.(o, d) };
    api.current = Chessground(el.current!, c);
    return () => api.current?.destroy();
  }, []);

  useEffect(() => {
    const c = config();
    // chessground merges configs; keep the move callback
    c.movable!.events = { after: (o, d) => cb.current.onMove?.(o, d) };
    // `lastMove` must be cleared explicitly
    if (!p.lastMove) api.current?.set({ lastMove: undefined });
    api.current?.set(c);
  }, [p.fen, p.orientation, p.movable, p.dests, p.lastMove?.join(), p.check, p.turnColor, p.highlights, p.free, p.tapMode, p.sync]);

  return <div class="cg-wrap-outer"><div ref={el} class="cg-wrap" /></div>;
}

