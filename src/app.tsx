import { useEffect, useState } from 'preact/hooks';
import { ScannerScreen } from './ui/ScannerScreen';
import { BoardScreen } from './ui/BoardScreen';
import { RecentScreen } from './ui/RecentScreen';
import { GameTree } from './chess/tree';
import { placementToBoardFen } from './chess/position';
import type { ScanPosition } from './chess/fromScan';
import { getSession, loadLast, newId, saveLast, saveSession, type Session } from './storage';

type Screen = 'scan' | 'board' | 'recent';

/** Reopen the board you were on if you come back within this time. */
const RESUME_MS = 30 * 60 * 1000;

function initial(): { screen: Screen; session: Session | null } {
  const params = new URLSearchParams(location.search);
  const fen = params.get('fen');
  if (fen) {
    const s = makeSession(fen.split(' ')[0], []);
    saveSession(s);
    return { screen: 'board', session: s };
  }
  const last = loadLast();
  if (last?.screen === 'board' && last.sessionId && Date.now() - last.at < RESUME_MS) {
    const s = getSession(last.sessionId);
    if (s) return { screen: 'board', session: s };
  }
  return { screen: 'scan', session: last?.sessionId ? getSession(last.sessionId) ?? null : null };
}

function makeSession(boardFen: string, uncertain: number[]): Session {
  const now = Date.now();
  return {
    id: newId(),
    created: now,
    updated: now,
    tree: GameTree.fromBoardFen(boardFen).root,
    current: 0,
    orientation: 'white',
    uncertain,
  };
}

export function App() {
  const [{ screen, session }, setState] = useState(initial);

  useEffect(() => {
    saveLast({ screen, sessionId: session?.id ?? null, at: Date.now() });
  }, [screen, session?.id, session?.updated]);

  const onScanned = (pos: ScanPosition) => {
    const s = makeSession(placementToBoardFen(pos.placement), pos.uncertain);
    saveSession(s);
    setState({ screen: 'board', session: s });
  };

  if (screen === 'scan') {
    return (
      <ScannerScreen
        onScanned={onScanned}
        onRecent={() => setState({ screen: 'recent', session })}
        onClose={session ? () => setState({ screen: 'board', session }) : undefined}
      />
    );
  }
  if (screen === 'recent' || !session) {
    return (
      <RecentScreen
        onOpen={(s) => setState({ screen: 'board', session: s })}
        onScan={() => setState({ screen: 'scan', session })}
      />
    );
  }
  return (
    <BoardScreen
      key={session.id}
      session={session}
      onChange={(s) => {
        saveSession(s);
        setState((st) => (st.session?.id === s.id ? { ...st, session: s } : st));
      }}
      onScan={() => setState({ screen: 'scan', session })}
      onRecent={() => setState({ screen: 'recent', session })}
    />
  );
}
