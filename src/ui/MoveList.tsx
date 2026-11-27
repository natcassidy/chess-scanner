import type { JSX } from 'preact';
import { useEffect, useRef } from 'preact/hooks';
import { isRoot, nodeId, type GameTree, type MoveNode, type TreeNode } from '../chess/tree';

interface Props {
  tree: GameTree;
  current: TreeNode;
  version: number;
  onJump: (id: number) => void;
}

/**
 * Moves inline, book style: 1. e4 e5 2. Nf3 (2. Bc4 Nf6) 2... Nc6
 * Variations are shown in parentheses right after the move they replace.
 */
export function MoveList({ tree, current, onJump }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const curId = nodeId(current);

  useEffect(() => {
    const el = ref.current?.querySelector('.mv.current') as HTMLElement | null;
    el?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }, [curId]);

  const out: JSX.Element[] = [];
  const token = (n: MoveNode, withNumber: boolean, depth: number) => {
    const label = n.color === 'w' ? `${n.num}.` : withNumber ? `${n.num}…` : '';
    out.push(
      <span
        key={n.id}
        class={'mv' + (n.id === curId ? ' current' : '') + (depth > 0 ? ' var' : '')}
        onClick={() => onJump(n.id)}
      >
        {label && <span class="num">{label}</span>}
        {n.san}
      </span>,
    );
  };

  /** Render the line continuing after `parent`, starting with `first`. */
  const line = (parent: TreeNode, first: MoveNode, depth: number, forceNumber: boolean) => {
    let node: TreeNode = parent;
    let next: MoveNode | undefined = first;
    let needNumber = forceNumber;
    while (next) {
      token(next, needNumber, depth);
      needNumber = false;
      // alternatives to `next` (siblings), only for the main child
      if (node.children[0] === next) {
        for (const alt of node.children.slice(1)) {
          out.push(<span key={`o${alt.id}`} class="paren">(</span>);
          line(node, alt, depth + 1, true);
          out.push(<span key={`c${alt.id}`} class="paren">)</span>);
          needNumber = true;
        }
      }
      node = next;
      next = node.children[0];
    }
  };

  const root = tree.root;
  if (root.children.length) line(root, root.children[0], 0, true);

  return (
    <div class="movelist" ref={ref}>
      {out.length ? out : <span class="movelist-empty">{isRoot(current) ? 'Make a move on the board to start analysing.' : ''}</span>}
    </div>
  );
}
