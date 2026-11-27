/**
 * Move tree with variations.  The root is the scanned position, which has no
 * fixed side to move: its children may be White or Black moves.
 */
import type { Square } from 'chess.js';
import { destsFor, fullFen, load, mergeDests, pieceColorAt, startingColors, type Color, type Dests } from './position';

export interface MoveNode {
  id: number;
  san: string;
  from: Square;
  to: Square;
  promotion?: string;
  /** side that played this move */
  color: Color;
  /** full-move number of this move */
  num: number;
  /** FEN after the move */
  fen: string;
  children: MoveNode[];
}

export interface Root {
  /** piece placement only (first FEN field) */
  boardFen: string;
  castling: string;
  children: MoveNode[];
}

export type TreeNode = Root | MoveNode;

export const isRoot = (n: TreeNode): n is Root => (n as MoveNode).san === undefined;

export class GameTree {
  root: Root;
  private nextId = 1;
  private parents = new Map<TreeNode, TreeNode>();

  constructor(root: Root) {
    this.root = root;
    const walk = (n: TreeNode) => {
      for (const c of n.children) {
        this.nextId = Math.max(this.nextId, c.id + 1);
        this.parents.set(c, n);
        walk(c);
      }
    };
    walk(root);
  }

  static fromBoardFen(boardFen: string, castling?: string): GameTree {
    const c = castling ?? fullFen(boardFen, 'w').split(' ')[2];
    return new GameTree({ boardFen, castling: c, children: [] });
  }

  parent(n: TreeNode): TreeNode | undefined {
    return this.parents.get(n);
  }

  /** Path from the root (exclusive) to n (inclusive). */
  path(n: TreeNode): MoveNode[] {
    const out: MoveNode[] = [];
    let cur: TreeNode | undefined = n;
    while (cur && !isRoot(cur)) {
      out.push(cur);
      cur = this.parents.get(cur);
    }
    return out.reverse();
  }

  /** FEN of node n (for the root, using `turn`). */
  fenAt(n: TreeNode, turn: Color = 'w'): string {
    return isRoot(n) ? fullFen(n.boardFen, turn, n.castling) : n.fen;
  }

  boardFenAt(n: TreeNode): string {
    return isRoot(n) ? n.boardFen : n.fen.split(' ')[0];
  }

  /** Colours allowed to move at n. */
  colorsToMove(n: TreeNode): Color[] {
    if (isRoot(n)) return startingColors(n.boardFen);
    return [n.fen.split(' ')[1] as Color];
  }

  dests(n: TreeNode): Dests {
    let d: Dests = new Map();
    for (const c of this.colorsToMove(n)) d = mergeDests(d, destsFor(this.fenAt(n, c)));
    return d;
  }

  /** Play from->to at node n; returns the (new or existing) child. */
  play(n: TreeNode, from: Square, to: Square, promotion?: string): MoveNode | null {
    const color = isRoot(n) ? pieceColorAt(n.boardFen, from) : (n.fen.split(' ')[1] as Color);
    if (!color) return null;
    const g = load(this.fenAt(n, color));
    if (!g) return null;
    let mv;
    try {
      mv = g.move({ from, to, promotion: promotion ?? 'q' });
    } catch {
      return null;
    }
    const existing = n.children.find((c) => c.from === from && c.to === to && (c.promotion ?? '') === (mv.promotion ?? ''));
    if (existing) return existing;
    const before = this.fenAt(n, color).split(' ');
    const child: MoveNode = {
      id: this.nextId++,
      san: mv.san,
      from,
      to,
      promotion: mv.promotion,
      color,
      num: Number(before[5]) || 1,
      fen: g.fen(),
      children: [],
    };
    n.children.push(child);
    this.parents.set(child, n);
    return child;
  }

  /** Remove n and everything after it. Returns its parent. */
  remove(n: MoveNode): TreeNode {
    const p = this.parents.get(n)!;
    p.children = p.children.filter((c) => c !== n);
    this.parents.delete(n);
    return p;
  }

  /** Make the variation containing n the main line at its branch point. */
  promote(n: MoveNode): void {
    let cur: MoveNode = n;
    let p = this.parents.get(cur);
    while (p) {
      const i = p.children.indexOf(cur);
      if (i > 0) {
        p.children.splice(i, 1);
        p.children.unshift(cur);
        return;
      }
      if (isRoot(p)) return;
      cur = p;
      p = this.parents.get(cur);
    }
  }

  /** True if n is not on the main line. */
  inVariation(n: TreeNode): boolean {
    let cur: TreeNode = n;
    let p = this.parents.get(cur);
    while (p) {
      if (p.children[0] !== cur) return true;
      cur = p;
      p = this.parents.get(cur);
    }
    return false;
  }

  mainlineEnd(n: TreeNode): TreeNode {
    let cur = n;
    while (cur.children.length) cur = cur.children[0];
    return cur;
  }

  find(id: number): TreeNode | undefined {
    if (id === 0) return this.root;
    let found: TreeNode | undefined;
    const walk = (n: TreeNode) => {
      for (const c of n.children) {
        if (c.id === id) found = c;
        if (!found) walk(c);
      }
    };
    walk(this.root);
    return found;
  }

  toJSON(): Root {
    return this.root;
  }
}

export const nodeId = (n: TreeNode) => (isRoot(n) ? 0 : n.id);
