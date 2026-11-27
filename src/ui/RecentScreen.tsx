import { useState } from 'preact/hooks';
import { deleteSession, listSessions, type Session } from '../storage';
import { boardFenToPlacement } from '../chess/position';
import { IconCamera, IconTrash } from './icons';

interface Props {
  onOpen: (s: Session) => void;
  onScan: () => void;
}

const ROLE: Record<string, string> = { k: 'king', q: 'queen', r: 'rook', b: 'bishop', n: 'knight', p: 'pawn' };

function ago(t: number): string {
  const s = (Date.now() - t) / 1000;
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  const d = Math.floor(s / 86400);
  if (d < 7) return d === 1 ? 'yesterday' : `${d} days ago`;
  return new Date(t).toLocaleDateString();
}

function countMoves(s: Session): number {
  let n = 0;
  const walk = (x: { children: { children: unknown[] }[] }) => {
    for (const c of x.children) {
      n++;
      walk(c as never);
    }
  };
  walk(s.tree as never);
  return n;
}

export function MiniBoard({ boardFen, flipped }: { boardFen: string; flipped?: boolean }) {
  let p = boardFenToPlacement(boardFen);
  if (flipped) p = p.slice().reverse();
  return (
    <div class="mini cg-wrap">
      {p.map((s, i) => (
        <div class={'mini-sq ' + (((i >> 3) + (i % 8)) % 2 ? 'd' : 'l')}>
          {s !== '.' && <piece class={`${ROLE[s.toLowerCase()]} ${s === s.toUpperCase() ? 'white' : 'black'}`} />}
        </div>
      ))}
    </div>
  );
}

export function RecentScreen({ onOpen, onScan }: Props) {
  const [items, setItems] = useState(listSessions);
  return (
    <div class="screen recent-screen">
      <header class="topbar">
        <button class="icon-btn" onClick={onScan} aria-label="Scan">
          <IconCamera />
        </button>
        <div class="status">Recent scans</div>
        <span class="icon-btn-spacer" />
      </header>
      <div class="recent-list">
        {items.length === 0 && (
          <div class="empty">
            <p>No scans yet.</p>
            <button class="btn primary" onClick={onScan}>
              <IconCamera /> Scan a diagram
            </button>
          </div>
        )}
        {items.map((s) => {
          const moves = countMoves(s);
          return (
            <div class="recent-item" onClick={() => onOpen(s)}>
              <MiniBoard boardFen={s.tree.boardFen} flipped={s.orientation === 'black'} />
              <div class="recent-meta">
                <div class="when">{ago(s.created)}</div>
                <div class="what">{moves ? `${moves} move${moves === 1 ? '' : 's'} analysed` : 'No moves yet'}</div>
              </div>
              <button
                class="icon-btn"
                aria-label="Delete"
                onClick={(e) => {
                  e.stopPropagation();
                  deleteSession(s.id);
                  setItems(listSessions());
                }}
              >
                <IconTrash />
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
