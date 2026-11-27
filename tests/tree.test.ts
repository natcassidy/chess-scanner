import { describe, expect, it } from 'vitest';
import { GameTree, isRoot } from '../src/chess/tree';
import { checkPlacement, boardFenToPlacement, inferCastling, placementToBoardFen, startingColors } from '../src/chess/position';
import { decode, looksRotated } from '../src/chess/fromScan';
import { CLASSES } from '../src/vision/classifier';

const START = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR';

describe('position helpers', () => {
  it('round-trips placement and infers castling', () => {
    const p = boardFenToPlacement(START);
    expect(p).toHaveLength(64);
    expect(placementToBoardFen(p)).toBe(START);
    expect(inferCastling(p)).toBe('KQkq');
    expect(checkPlacement(p)).toBeNull();
  });

  it('lets either side start unless one is in check', () => {
    expect(startingColors(START).sort()).toEqual(['b', 'w']);
    // black king on e8 checked by a white rook on e1: only Black can move
    expect(startingColors('4k3/8/8/8/8/8/8/4RK2')).toEqual(['b']);
  });

  it('flags missing kings and back-rank pawns', () => {
    expect(checkPlacement(boardFenToPlacement('8/8/8/8/8/8/8/4K3'))?.message).toMatch(/black king/);
    expect(checkPlacement(boardFenToPlacement('P3k3/8/8/8/8/8/8/4K3'))?.message).toMatch(/Pawns/);
  });
});

describe('game tree', () => {
  it('decides the side to move from the first move and builds variations', () => {
    const t = GameTree.fromBoardFen(START);
    const e5 = t.play(t.root, 'e7', 'e5')!;
    expect(e5.color).toBe('b');
    expect(e5.san).toBe('e5');
    const nf3 = t.play(e5, 'g1', 'f3')!;
    expect(nf3.num).toBe(2);
    const e4 = t.play(t.root, 'e2', 'e4')!;
    expect(e4.color).toBe('w');
    expect(t.root.children.map((c) => c.san)).toEqual(['e5', 'e4']);
    expect(t.inVariation(e4)).toBe(true);
    t.promote(e4);
    expect(t.root.children[0]).toBe(e4);
    // replaying an existing move reuses the node
    expect(t.play(t.root, 'e7', 'e5')).toBe(e5);
    expect(t.path(nf3).map((n) => n.san)).toEqual(['e5', 'Nf3']);
    const back = t.remove(e5);
    expect(isRoot(back)).toBe(true);
    expect(t.root.children).toHaveLength(1);
  });

  it('serialises and restores', () => {
    const t = GameTree.fromBoardFen(START);
    const a = t.play(t.root, 'e2', 'e4')!;
    t.play(a, 'c7', 'c5');
    const t2 = new GameTree(JSON.parse(JSON.stringify(t.toJSON())));
    const n = t2.find(2)!;
    expect(isRoot(n)).toBe(false);
    expect(t2.path(n).map((x) => x.san)).toEqual(['e4', 'c5']);
    expect(t2.play(n, 'g1', 'f3')!.id).toBe(3);
  });

  it('promotes pawns (queen by default)', () => {
    const t = GameTree.fromBoardFen('4k3/P7/8/8/8/8/8/4K3');
    const m = t.play(t.root, 'a7', 'a8')!;
    expect(m.san).toBe('a8=Q+');
    const n = t.play(t.root, 'a7', 'a8', 'n')!;
    expect(n.san).toBe('a8=N');
  });
});

describe('scan decoding', () => {
  const onehot = (sym: string, p = 0.99) => {
    const a = new Float32Array(13).fill((1 - p) / 12);
    a[CLASSES.indexOf(sym as never)] = p;
    return a;
  };
  it('forces exactly one king per side and no back-rank pawns', () => {
    const probs = boardFenToPlacement(START).map((s) => onehot(s));
    // second "white king" that is slightly less likely
    probs[35] = onehot('K', 0.6);
    // pawn on the back rank with a rook as runner-up
    probs[0] = new Float32Array(13);
    probs[0][CLASSES.indexOf('p')] = 0.6;
    probs[0][CLASSES.indexOf('r')] = 0.4;
    const { placement } = decode(probs);
    expect(placement.filter((s) => s === 'K')).toHaveLength(1);
    expect(placement[60]).toBe('K');
    expect(placement[0]).toBe('r');
  });
  it('detects diagrams printed from Black’s side', () => {
    const flipped = boardFenToPlacement(START).slice().reverse();
    expect(looksRotated(flipped)).toBe(true);
    expect(looksRotated(boardFenToPlacement(START))).toBe(false);
  });
});

describe('count priors', () => {
  const mk = (best: string, pBest: number, second: string) => {
    const a = new Float32Array(13).fill(0.001);
    a[CLASSES.indexOf(best as never)] = pBest;
    a[CLASSES.indexOf(second as never)] = 1 - pBest - 0.011;
    return a;
  };
  it('turns unsure extra bishops back into pawns', () => {
    const probs = Array.from({ length: 64 }, () => mk('.', 0.98, 'p'));
    probs[4] = mk('k', 0.99, 'q');
    probs[60] = mk('K', 0.99, 'Q');
    // a row of black "bishops" on rank 7 that are really pawns
    for (let c = 0; c < 8; c++) probs[8 + c] = mk('b', 0.6, 'p');
    const { placement, conf } = decode(probs);
    expect(placement.slice(8, 16).filter((s) => s === 'b')).toHaveLength(2);
    expect(placement.slice(8, 16).filter((s) => s === 'p')).toHaveLength(6);
    expect(conf.filter((c) => c <= 0.5)).toHaveLength(6);
  });
  it('keeps confident extra pieces (promotions)', () => {
    const probs = Array.from({ length: 64 }, () => mk('.', 0.98, 'p'));
    probs[4] = mk('k', 0.99, 'q');
    probs[60] = mk('K', 0.99, 'Q');
    probs[20] = mk('Q', 0.97, 'K');
    probs[30] = mk('Q', 0.97, 'K');
    const { placement } = decode(probs);
    expect(placement.filter((s) => s === 'Q')).toHaveLength(2);
  });
});
