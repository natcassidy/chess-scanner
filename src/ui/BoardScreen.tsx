import { useEffect, useMemo, useState } from 'preact/hooks';
import type { Key } from 'chessground/types';
import type { Square } from 'chess.js';
import { Board } from './Board';
import { MoveList } from './MoveList';
import { EditPanel } from './EditPanel';
import { GameTree, isRoot, nodeId, type TreeNode } from '../chess/tree';
import { load, type Color } from '../chess/position';
import type { Session } from '../storage';
import {
  IconCamera, IconCopy, IconEdit, IconExternal, IconFirst, IconFlip, IconHistory, IconLast, IconMore, IconNext,
  IconPrev, IconReset, IconStar, IconTrash,
} from './icons';

interface Props {
  session: Session;
  onChange: (s: Session) => void;
  onScan: () => void;
  onRecent: () => void;
}

const FILES = 'abcdefgh';
export const squareKey = (i: number) => `${FILES[i % 8]}${8 - (i >> 3)}` as Key;
const colorName = (c: Color) => (c === 'w' ? 'White' : 'Black');

export function BoardScreen({ session, onChange, onScan, onRecent }: Props) {
  const tree = useMemo(() => new GameTree(session.tree), [session.id, session.tree]);
  const [currentId, setCurrentId] = useState(session.current);
  const [orientation, setOrientation] = useState(session.orientation);
  const [uncertain, setUncertain] = useState(session.uncertain);
  const [version, setVersion] = useState(0);
  const [menu, setMenu] = useState(false);
  const [editing, setEditing] = useState(false);
  const [promo, setPromo] = useState<{ from: Square; to: Square; color: Color } | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  const node: TreeNode = tree.find(currentId) ?? tree.root;

  // persist on every change
  useEffect(() => {
    onChange({
      ...session,
      tree: tree.root,
      current: nodeId(node),
      orientation,
      uncertain,
      updated: Date.now(),
    });
  }, [version, currentId, orientation, uncertain]);

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 1800);
    return () => clearTimeout(t);
  }, [toast]);

  const view = useMemo(() => {
    const colors = tree.colorsToMove(node);
    const fen = tree.fenAt(node, colors[0]);
    const g = load(fen);
    const over = !isRoot(node) && g ? (g.isCheckmate() ? 'Checkmate' : g.isStalemate() ? 'Stalemate' : g.isDraw() ? 'Draw' : null) : null;
    let status: string;
    if (over) status = over;
    else if (colors.length === 2) status = 'Either side can move';
    else status = `${colorName(colors[0])} to move`;
    return {
      boardFen: tree.boardFenAt(node),
      colors,
      dests: tree.dests(node) as unknown as Map<Key, Key[]>,
      check: !!g?.inCheck(),
      status,
      lastMove: isRoot(node) ? undefined : ([node.from, node.to] as [Key, Key]),
    };
  }, [nodeId(node), version]);

  const highlights = useMemo(() => {
    const m = new Map<Key, string>();
    if (isRoot(node)) for (const i of uncertain) m.set(squareKey(i), 'uncertain');
    return m;
  }, [uncertain, nodeId(node)]);

  const go = (n: TreeNode | undefined) => n && setCurrentId(nodeId(n));
  const prev = () => go(tree.parent(node));
  const next = () => go(node.children[0]);
  const first = () => go(tree.root);
  const last = () => go(tree.mainlineEnd(node));

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (editing || menu) return;
      if (e.key === 'ArrowLeft') prev();
      else if (e.key === 'ArrowRight') next();
      else if (e.key === 'ArrowUp') first();
      else if (e.key === 'ArrowDown') last();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  const commit = (from: Square, to: Square, promotion?: string) => {
    const child = tree.play(node, from, to, promotion);
    if (child) {
      setCurrentId(child.id);
      if (uncertain.length) setUncertain([]);
    }
    setVersion((v) => v + 1);
  };

  const onMove = (from: Key, to: Key) => {
    const piece = load(tree.fenAt(node, 'w'))?.get(from as Square);
    if (piece?.type === 'p' && (to[1] === '8' || to[1] === '1')) {
      setPromo({ from: from as Square, to: to as Square, color: piece.color });
      return;
    }
    commit(from as Square, to as Square);
  };

  const inVariation = !isRoot(node) && tree.inVariation(node);

  const copyFen = async () => {
    const fen = tree.fenAt(node, view.colors[0]);
    try {
      await navigator.clipboard.writeText(fen);
      setToast('FEN copied');
    } catch {
      setToast(fen);
    }
  };

  const lichessUrl = () => `https://lichess.org/analysis/standard/${tree.fenAt(node, view.colors[0]).replace(/ /g, '_')}`;

  if (editing) {
    return (
      <EditPanel
        boardFen={tree.root.boardFen}
        orientation={orientation}
        uncertain={uncertain}
        hasMoves={tree.root.children.length > 0}
        onCancel={() => setEditing(false)}
        onDone={(boardFen) => {
          const t = GameTree.fromBoardFen(boardFen);
          onChange({ ...session, tree: t.root, current: 0, orientation, uncertain: [], updated: Date.now() });
          setUncertain([]);
          setCurrentId(0);
          setEditing(false);
        }}
      />
    );
  }

  return (
    <div class="screen board-screen">
      <header class="topbar">
        <button class="icon-btn" onClick={onScan} aria-label="Scan a new position">
          <IconCamera />
        </button>
        <div class={'status' + (view.colors.length === 2 && isRoot(node) ? ' hint' : '')}>
          <span class={'turn-dot ' + (view.colors.length === 2 ? 'both' : view.colors[0])} />
          {view.status}
        </div>
        <button class="icon-btn" onClick={() => setMenu(true)} aria-label="More">
          <IconMore />
        </button>
      </header>

      <div class="board-area">
        <Board
          fen={view.boardFen}
          orientation={orientation}
          movable={view.colors.length === 2 ? 'both' : view.colors[0] === 'w' ? 'white' : 'black'}
          turnColor={view.colors[0] === 'w' ? 'white' : 'black'}
          dests={view.dests}
          lastMove={view.lastMove}
          check={view.check}
          highlights={highlights}
          sync={version}
          onMove={onMove}
        />
        {promo && (
          <div class="promo" onClick={() => { setPromo(null); setVersion((v) => v + 1); }}>
            <div class="promo-box cg-wrap" onClick={(e) => e.stopPropagation()}>
              {(['queen', 'rook', 'bishop', 'knight'] as const).map((role) => (
                <button
                  class="promo-btn"
                  onClick={() => {
                    const p = promo;
                    setPromo(null);
                    commit(p.from, p.to, role === 'knight' ? 'n' : role[0]);
                  }}
                >
                  <piece class={`${role} ${promo.color === 'w' ? 'white' : 'black'}`} />
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {uncertain.length > 0 && isRoot(node) && (
        <div class="notice">
          <span>
            {uncertain.length === 1 ? '1 highlighted square' : `${uncertain.length} highlighted squares`} may be misread.
          </span>
          <button class="btn small" onClick={() => setEditing(true)}>Fix</button>
          <button class="btn small ghost" onClick={() => setUncertain([])}>Looks right</button>
        </div>
      )}

      <MoveList tree={tree} current={node} version={version} onJump={(id) => setCurrentId(id)} />

      <nav class="navbar">
        <button class="nav-btn" onClick={first} disabled={isRoot(node)} aria-label="Start">
          <IconFirst />
        </button>
        <button class="nav-btn big" onClick={prev} disabled={isRoot(node)} aria-label="Back">
          <IconPrev />
        </button>
        <button class="nav-btn big" onClick={next} disabled={!node.children.length} aria-label="Forward">
          <IconNext />
        </button>
        <button class="nav-btn" onClick={last} disabled={!node.children.length} aria-label="End of line">
          <IconLast />
        </button>
      </nav>

      {menu && (
        <div class="sheet-backdrop" onClick={() => setMenu(false)}>
          <div class="sheet" onClick={(e) => e.stopPropagation()}>
            <MenuItem icon={<IconFlip />} label="Flip board" onClick={() => { setOrientation(orientation === 'white' ? 'black' : 'white'); setMenu(false); }} />
            <MenuItem icon={<IconEdit />} label="Edit position" onClick={() => { setEditing(true); setMenu(false); }} />
            {inVariation && (
              <MenuItem icon={<IconStar />} label="Make this the main line" onClick={() => { if (!isRoot(node)) tree.promote(node); setVersion((v) => v + 1); setMenu(false); }} />
            )}
            {!isRoot(node) && (
              <MenuItem icon={<IconTrash />} label="Delete from this move" onClick={() => { if (!isRoot(node)) setCurrentId(nodeId(tree.remove(node))); setVersion((v) => v + 1); setMenu(false); }} />
            )}
            {tree.root.children.length > 0 && (
              <MenuItem icon={<IconReset />} label="Clear all moves" onClick={() => { tree.root.children = []; setCurrentId(0); setVersion((v) => v + 1); setMenu(false); }} />
            )}
            <MenuItem icon={<IconCopy />} label="Copy FEN" onClick={() => { copyFen(); setMenu(false); }} />
            <MenuItem icon={<IconExternal />} label="Open in Lichess" onClick={() => { window.open(lichessUrl(), '_blank'); setMenu(false); }} />
            <MenuItem icon={<IconHistory />} label="Recent scans" onClick={() => { setMenu(false); onRecent(); }} />
            <button class="sheet-cancel" onClick={() => setMenu(false)}>Close</button>
          </div>
        </div>
      )}
      {toast && <div class="toast">{toast}</div>}
    </div>
  );
}

function MenuItem({ icon, label, onClick }: { icon: preact.JSX.Element; label: string; onClick: () => void }) {
  return (
    <button class="sheet-item" onClick={onClick}>
      {icon}
      <span>{label}</span>
    </button>
  );
}
