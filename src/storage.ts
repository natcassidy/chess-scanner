import type { Root } from './chess/tree';

export interface Session {
  id: string;
  created: number;
  updated: number;
  tree: Root;
  /** id of the node the user was looking at */
  current: number;
  orientation: 'white' | 'black';
  /** squares (0 = a8) the scanner was unsure about; cleared once edited */
  uncertain: number[];
}

const KEY = 'chess-scanner.sessions.v1';
const LAST = 'chess-scanner.last.v1';
const MAX = 100;

function read(): Session[] {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as Session[]) : [];
  } catch {
    return [];
  }
}

function write(list: Session[]) {
  try {
    localStorage.setItem(KEY, JSON.stringify(list.slice(0, MAX)));
  } catch {
    /* storage full or unavailable: the app still works, just without history */
  }
}

export function listSessions(): Session[] {
  return read().sort((a, b) => b.updated - a.updated);
}

export function saveSession(s: Session) {
  const list = read().filter((x) => x.id !== s.id);
  list.unshift(s);
  list.sort((a, b) => b.updated - a.updated);
  write(list);
}

export function deleteSession(id: string) {
  write(read().filter((x) => x.id !== id));
}

export function getSession(id: string): Session | undefined {
  return read().find((x) => x.id === id);
}

export interface LastState {
  screen: 'scan' | 'board' | 'recent';
  sessionId: string | null;
  at: number;
}

export function saveLast(l: LastState) {
  try {
    localStorage.setItem(LAST, JSON.stringify(l));
  } catch {
    /* ignore */
  }
}

export function loadLast(): LastState | null {
  try {
    const raw = localStorage.getItem(LAST);
    return raw ? (JSON.parse(raw) as LastState) : null;
  } catch {
    return null;
  }
}

export function newId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
}
