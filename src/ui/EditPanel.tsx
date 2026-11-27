import { useMemo, useState } from 'preact/hooks';
import type { Key } from 'chessground/types';
import { Board } from './Board';
import { boardFenToPlacement, checkPlacement, placementToBoardFen } from '../chess/position';
import type { PieceClass } from '../vision/classifier';
import { IconEraser, IconRotate } from './icons';

interface Props {
  boardFen: string;
  orientation: 'white' | 'black';
  uncertain: number[];
  hasMoves: boolean;
  onCancel: () => void;
  onDone: (boardFen: string) => void;
}

const ROLES: Record<string, string> = { k: 'king', q: 'queen', r: 'rook', b: 'bishop', n: 'knight', p: 'pawn' };
type Tool = PieceClass | 'erase' | null;

const keyIndex = (k: Key) => (8 - Number(k[1])) * 8 + (k.charCodeAt(0) - 97);
const FILES = 'abcdefgh';
const squareKey = (i: number) => `${FILES[i % 8]}${8 - (i >> 3)}` as Key;

export function EditPanel({ boardFen, orientation, uncertain, hasMoves, onCancel, onDone }: Props) {
  const [fen, setFen] = useState(boardFen);
  const [tool, setTool] = useState<Tool>(null);
  const [error, setError] = useState<string | null>(null);
  const [sync, setSync] = useState(0);
  const [todo, setTodo] = useState(uncertain);

  const highlights = useMemo(() => new Map(todo.map((i) => [squareKey(i), 'uncertain'])), [todo]);

  const update = (next: string) => {
    setFen(next);
    setError(null);
  };

  const onSelect = (key: Key) => {
    if (!tool) {
      setTodo((t) => t.filter((i) => i !== keyIndex(key)));
      return;
    }
    const p = boardFenToPlacement(fen);
    const i = keyIndex(key);
    const piece: PieceClass = tool === 'erase' ? '.' : tool;
    // tapping the same piece again removes it
    p[i] = p[i] === piece ? '.' : piece;
    setTodo((t) => t.filter((x) => x !== i));
    update(placementToBoardFen(p));
    setSync((s) => s + 1);
  };

  const done = () => {
    const problem = checkPlacement(boardFenToPlacement(fen));
    if (problem) {
      setError(problem.message);
      return;
    }
    onDone(fen);
  };

  const palette = (color: 'w' | 'b') =>
    (['k', 'q', 'r', 'b', 'n', 'p'] as const).map((r) => {
      const sym = (color === 'w' ? r.toUpperCase() : r) as PieceClass;
      return (
        <button
          class={'pal-btn' + (tool === sym ? ' active' : '')}
          onClick={() => setTool(tool === sym ? null : sym)}
          aria-label={`${color === 'w' ? 'White' : 'Black'} ${ROLES[r]}`}
        >
          <piece class={`${ROLES[r]} ${color === 'w' ? 'white' : 'black'}`} />
        </button>
      );
    });

  return (
    <div class="screen board-screen editing">
      <header class="topbar">
        <button class="text-btn" onClick={onCancel}>Cancel</button>
        <div class="status">Edit position</div>
        <button class="text-btn strong" onClick={done}>Done</button>
      </header>
      <div class="board-area">
        <Board
          fen={fen}
          orientation={orientation}
          movable={null}
          free
          tapMode={tool !== null}
          highlights={highlights}
          sync={sync}
          onSelect={onSelect}
          onEditChange={(f) => {
            if (f !== fen) update(f);
          }}
        />
      </div>
      <div class="edit-help">
        {error ? (
          <span class="error">{error}</span>
        ) : tool ? (
          'Tap squares to place. Tap the piece again to deselect.'
        ) : (
          'Drag pieces to move them, or off the board to remove. Pick a piece below to place it.'
        )}
        {hasMoves && !error && <div class="sub">Changing the position clears the moves you’ve played.</div>}
      </div>
      <div class="palette cg-wrap">
        <div class="pal-row">{palette('w')}</div>
        <div class="pal-row">{palette('b')}</div>
        <div class="pal-row tools">
          <button class={'pal-btn tool' + (tool === 'erase' ? ' active' : '')} onClick={() => setTool(tool === 'erase' ? null : 'erase')}>
            <IconEraser /> <span>Erase</span>
          </button>
          <button
            class="pal-btn tool"
            onClick={() => {
              update(placementToBoardFen(boardFenToPlacement(fen).reverse()));
              setTodo((t) => t.map((i) => 63 - i));
              setSync((s) => s + 1);
            }}
          >
            <IconRotate /> <span>Rotate 180°</span>
          </button>
        </div>
      </div>
    </div>
  );
}
