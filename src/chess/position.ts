/**
 * Position helpers around chess.js for scanned diagrams, where the side to
 * move is unknown until the user plays a move.
 */
import { Chess, type Square } from 'chess.js';
import type { PieceClass } from '../vision/classifier';

export type Color = 'w' | 'b';

/** 64 symbols (index 0 = a8) -> FEN piece-placement field. */
export function placementToBoardFen(p: readonly PieceClass[]): string {
  const rows: string[] = [];
  for (let r = 0; r < 8; r++) {
    let row = '';
    let empty = 0;
    for (let c = 0; c < 8; c++) {
      const s = p[r * 8 + c];
      if (s === '.') empty++;
      else {
        if (empty) row += empty;
        empty = 0;
        row += s;
      }
    }
    if (empty) row += empty;
    rows.push(row);
  }
  return rows.join('/');
}

export function boardFenToPlacement(boardFen: string): PieceClass[] {
  const out: PieceClass[] = [];
  for (const row of boardFen.split(' ')[0].split('/')) {
    for (const ch of row) {
      if (/\d/.test(ch)) for (let i = 0; i < Number(ch); i++) out.push('.');
      else out.push(ch as PieceClass);
    }
  }
  return out;
}

/** Castling rights implied by kings and rooks on their home squares. */
export function inferCastling(p: readonly PieceClass[]): string {
  const at = (sq: string) => p[(8 - Number(sq[1])) * 8 + (sq.charCodeAt(0) - 97)];
  let s = '';
  if (at('e1') === 'K') {
    if (at('h1') === 'R') s += 'K';
    if (at('a1') === 'R') s += 'Q';
  }
  if (at('e8') === 'k') {
    if (at('h8') === 'r') s += 'k';
    if (at('a8') === 'r') s += 'q';
  }
  return s || '-';
}

export function fullFen(boardFen: string, turn: Color, castling?: string): string {
  const castle = castling ?? inferCastling(boardFenToPlacement(boardFen));
  return `${boardFen} ${turn} ${castle} - 0 1`;
}

/** Load a FEN leniently (diagrams can be "illegal": extra pawns, etc.). */
export function load(fen: string): Chess | null {
  try {
    return new Chess(fen);
  } catch {
    try {
      return new Chess(fen, { skipValidation: true });
    } catch {
      return null;
    }
  }
}

export interface PositionProblem {
  message: string;
}

/** Problems that make a position unplayable. */
export function checkPlacement(p: readonly PieceClass[]): PositionProblem | null {
  const count = (s: PieceClass) => p.filter((x) => x === s).length;
  if (count('K') !== 1) return { message: count('K') ? 'There can only be one white king.' : 'Add a white king.' };
  if (count('k') !== 1) return { message: count('k') ? 'There can only be one black king.' : 'Add a black king.' };
  for (let c = 0; c < 8; c++) {
    if (p[c] === 'P' || p[c] === 'p' || p[56 + c] === 'P' || p[56 + c] === 'p')
      return { message: 'Pawns can’t stand on the first or last rank.' };
  }
  return null;
}

/** Which colours may make the first move from a scanned position. */
export function startingColors(boardFen: string): Color[] {
  const out: Color[] = [];
  for (const turn of ['w', 'b'] as Color[]) {
    const other: Color = turn === 'w' ? 'b' : 'w';
    // The side NOT to move must not be in check.
    const probe = load(fullFen(boardFen, other));
    if (probe && probe.inCheck()) continue;
    const g = load(fullFen(boardFen, turn));
    if (g && g.moves().length) out.push(turn);
  }
  return out.length ? out : ['w'];
}

export type Dests = Map<Square, Square[]>;

export function destsFor(fen: string): Dests {
  const g = load(fen);
  const d: Dests = new Map();
  if (!g) return d;
  for (const m of g.moves({ verbose: true })) {
    const list = d.get(m.from) ?? [];
    list.push(m.to);
    d.set(m.from, list);
  }
  return d;
}

export function mergeDests(a: Dests, b: Dests): Dests {
  const out: Dests = new Map(a);
  for (const [k, v] of b) out.set(k, [...(out.get(k) ?? []), ...v]);
  return out;
}

export function pieceColorAt(boardFen: string, sq: Square): Color | null {
  const p = boardFenToPlacement(boardFen)[(8 - Number(sq[1])) * 8 + (sq.charCodeAt(0) - 97)];
  if (!p || p === '.') return null;
  return p === p.toUpperCase() ? 'w' : 'b';
}
