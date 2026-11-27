"""Synthetic book-diagram generator.

Renders chess diagrams the way books print them (hatched / grey / dotted dark
squares, thin or thick frames, many piece styles), places them on a fake page
with text, then simulates an iPhone camera (perspective, lighting, blur,
noise, JPEG).  Used both to train the square classifier and to produce test
frames for the grid detector.
"""

from __future__ import annotations

import os
import random
import string
from dataclasses import dataclass

import cv2
import numpy as np

CLASSES = ['.', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
PIECE_FILE = {
    'P': 'wP', 'N': 'wN', 'B': 'wB', 'R': 'wR', 'Q': 'wQ', 'K': 'wK',
    'p': 'bP', 'n': 'bN', 'b': 'bB', 'r': 'bR', 'q': 'bQ', 'k': 'bK',
}

# Sets that look like printed book diagrams get more weight.
SET_WEIGHTS = {
    'merida': 6, 'alpha': 6, 'leipzig': 6, 'cburnett': 5, 'companion': 3,
    'chess7': 3, 'maestro': 3, 'cardinal': 2, 'governor': 2, 'kosal': 2,
    'reillycraig': 2, 'riohacha': 2, 'tatiana': 2, 'mpchess': 2, 'staunty': 2,
    'gioco': 1, 'fresca': 1, 'cooke': 1, 'icpieces': 1, 'chessnut': 1,
    'california': 1, 'caliente': 1, 'celtic': 1, 'dubrovny': 1, 'spatial': 1,
    # classic book diagram fonts (training/render_fonts.py)
    'font-adventurer': 3, 'font-alpha': 6, 'font-berlin': 4, 'font-cases': 6, 'font-condal': 4,
    'font-habsburg': 4, 'font-kingdom': 3, 'font-leipzig': 6, 'font-lucena': 3, 'font-magnetic': 3,
    'font-marroquin': 3, 'font-maya': 3, 'font-merida': 6, 'font-motif': 3, 'font-smart': 4,
    'font-traveller': 4, 'font-zurich': 4, 'ttf-casefont': 6,
}


def load_piece_sets(root: str, only: list[str] | None = None, exclude: list[str] | None = None):
    sets = {}
    for name in sorted(os.listdir(root)):
        if name not in SET_WEIGHTS:
            continue
        if only is not None and name not in only:
            continue
        if exclude is not None and name in exclude:
            continue
        pieces = {}
        for sym, f in PIECE_FILE.items():
            rgba = cv2.imread(os.path.join(root, name, f + '.png'), cv2.IMREAD_UNCHANGED)
            rgba = rgba.astype(np.float32) / 255.0
            # BGRA -> luminance + alpha
            lum = 0.114 * rgba[..., 0] + 0.587 * rgba[..., 1] + 0.299 * rgba[..., 2]
            pieces[sym] = (lum, rgba[..., 3])
        sets[name] = pieces
    return sets


# ---------------------------------------------------------------- positions

def random_position(rng: random.Random, legal: bool = False) -> list[str]:
    """64 symbols, index = row*8+col with row 0 at the top (rank 8).
    legal=True: exactly one king per side and no pawns on the edge ranks."""
    p_empty = rng.uniform(0.35, 0.85)
    weights = [8, 2, 2, 2, 1, 1, 8, 2, 2, 2, 1, 1]
    if legal:
        weights = [8, 2, 2, 2, 1, 0, 8, 2, 2, 2, 1, 0]
    syms = CLASSES[1:]
    out = []
    for i in range(64):
        if rng.random() < p_empty:
            out.append('.')
        else:
            s = rng.choices(syms, weights)[0]
            if legal and s in 'Pp' and i // 8 in (0, 7):
                s = '.'
            out.append(s)
    if legal:
        a, b = rng.sample(range(64), 2)
        out[a], out[b] = 'K', 'k'
    return out


# ---------------------------------------------------------------- diagram

@dataclass
class Diagram:
    img: np.ndarray        # float32 grey, 0..1
    corners: np.ndarray    # 4x2 corners of the 8x8 playing area (TL, TR, BR, BL)
    position: list[str]


def _dark_square_texture(rng: random.Random, sq: int, style: str, params: dict) -> np.ndarray:
    t = np.full((sq, sq), params['light'], np.float32)
    yy, xx = np.mgrid[0:sq, 0:sq].astype(np.float32)
    if style == 'solid':
        t[:] = params['dark']
    elif style == 'hatch':
        sp, th, ang, ink = params['spacing'], params['thick'], params['angle'], params['ink']
        if ang == 45:
            d = (xx + yy)
        elif ang == 135:
            d = (xx - yy)
        elif ang == 0:
            d = yy
        else:
            d = xx
        d = d / np.sqrt(2) if ang in (45, 135) else d
        ph = np.mod(d + params['phase'], sp)
        # anti-aliased line profile
        cov = np.clip(th / 2 + 0.5 - np.abs(ph - sp / 2), 0, 1)
        t = t * (1 - cov) + ink * cov
    elif style == 'dots':
        sp, r, ink = params['spacing'], params['radius'], params['ink']
        dx = np.mod(xx + params['phase'], sp) - sp / 2
        dy = np.mod(yy + params['phase'], sp) - sp / 2
        cov = np.clip(r + 0.5 - np.sqrt(dx * dx + dy * dy), 0, 1)
        t = t * (1 - cov) + ink * cov
    return t


def render_diagram(rng: random.Random, sets: dict, position: list[str] | None = None,
                   sq: int | None = None, legal: bool = False) -> Diagram:
    sq = sq or rng.randint(36, 80)
    position = position or random_position(rng, legal)
    set_names = list(sets)
    set_name = rng.choices(set_names, [SET_WEIGHTS[s] for s in set_names])[0]
    pieces = sets[set_name]

    light = rng.uniform(0.9, 1.0)
    style = rng.choices(['hatch', 'solid', 'dots'], [5, 4, 1])[0]
    params = {'light': light, 'dark': rng.uniform(0.5, 0.85)}
    if style == 'hatch':
        params.update(spacing=sq / rng.uniform(5, 14), angle=rng.choice([45, 45, 45, 135, 0, 90]),
                      ink=rng.uniform(0.0, 0.35), phase=rng.uniform(0, 10))
        params['thick'] = params['spacing'] * rng.uniform(0.2, 0.55)
    elif style == 'dots':
        params.update(spacing=sq / rng.uniform(6, 12), ink=rng.uniform(0.0, 0.4), phase=rng.uniform(0, 5))
        params['radius'] = params['spacing'] * rng.uniform(0.2, 0.4)
    dark_tex = _dark_square_texture(rng, sq, style, params)
    # Some books use a continuous pattern across squares; approximate by a
    # single texture tile.  Swap colours occasionally (light grey / white).
    board = np.empty((8 * sq, 8 * sq), np.float32)
    for r in range(8):
        for c in range(8):
            tile = dark_tex if (r + c) % 2 == 1 else light
            board[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq] = tile

    binarize = rng.random() < 0.4
    weight = rng.choice([0, 0, 0, 1, -1]) if sq >= 40 else 0
    scale = rng.uniform(0.82, 1.0)
    psz = max(8, int(round(sq * scale)))
    cache = {}
    for i, s in enumerate(position):
        if s == '.':
            continue
        if s not in cache:
            lum, alpha = pieces[s]
            lum = cv2.resize(lum, (psz, psz), interpolation=cv2.INTER_AREA)
            alpha = cv2.resize(alpha, (psz, psz), interpolation=cv2.INTER_AREA)
            if binarize:
                lum = np.where(lum > 0.55, 1.0, 0.0).astype(np.float32)
            if weight != 0:
                k = np.ones((2, 2), np.uint8)
                # erode on luminance = thicker ink (bold print); dilate = thinner
                lum = cv2.erode(lum, k) if weight > 0 else cv2.dilate(lum, k)
            cache[s] = (lum, alpha)
        lum, alpha = cache[s]
        r, c = divmod(i, 8)
        off = (sq - psz) / 2
        y0 = int(round(r * sq + off + rng.uniform(-0.03, 0.03) * sq))
        x0 = int(round(c * sq + off + rng.uniform(-0.03, 0.03) * sq))
        y0 = min(max(y0, r * sq), (r + 1) * sq - psz)
        x0 = min(max(x0, c * sq), (c + 1) * sq - psz)
        reg = board[y0:y0 + psz, x0:x0 + psz]
        board[y0:y0 + psz, x0:x0 + psz] = reg * (1 - alpha) + lum * alpha

    # Frame
    paper = rng.uniform(0.92, 1.0)
    frame = rng.choice([0, 1, 2, 2, 3, 4, 6]) * sq / 40.0
    gap = rng.choice([0, 0, 0, sq * 0.08])
    outer = 0 if gap == 0 else max(1.0, sq / 40)
    pad = int(np.ceil(frame + gap + outer)) + 2
    size = 8 * sq + 2 * pad
    img = np.full((size, size), paper, np.float32)
    img[pad:pad + 8 * sq, pad:pad + 8 * sq] = board
    ink = rng.uniform(0.0, 0.2)
    if frame > 0:
        f = int(round(frame))
        if f > 0:
            p0, p1 = pad - f, pad + 8 * sq + f
            img[p0:pad, p0:p1] = ink
            img[pad + 8 * sq:p1, p0:p1] = ink
            img[p0:p1, p0:pad] = ink
            img[p0:p1, pad + 8 * sq:p1] = ink
    if outer > 0:
        o = int(round(outer))
        q0 = int(pad - frame - gap - o)
        q1 = int(pad + 8 * sq + frame + gap + o)
        q0 = max(q0, 0); q1 = min(q1, size)
        img[q0:q0 + o, q0:q1] = ink
        img[q1 - o:q1, q0:q1] = ink
        img[q0:q1, q0:q0 + o] = ink
        img[q0:q1, q1 - o:q1] = ink
    corners = np.array([[pad, pad], [pad + 8 * sq, pad], [pad + 8 * sq, pad + 8 * sq], [pad, pad + 8 * sq]],
                       np.float32)
    return Diagram(img, corners, position)


# ---------------------------------------------------------------- page

def _random_text(rng: random.Random, img: np.ndarray, region_mask: np.ndarray, line_h: int):
    h, w = img.shape
    y = rng.randint(2, line_h + 4)
    font = rng.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX])
    fs = line_h / 30.0
    th = max(1, int(round(line_h / 14)))
    while y < h:
        words = ' '.join(''.join(rng.choices(string.ascii_letters + '0123456789.,', k=rng.randint(1, 9)))
                         for _ in range(rng.randint(4, 30)))
        canvas = np.zeros((line_h * 2, w), np.uint8)
        cv2.putText(canvas, words, (rng.randint(0, 10), int(line_h * 1.3)), font, fs, 255, th, cv2.LINE_AA)
        y0 = y - int(line_h * 1.3)
        y1 = y0 + canvas.shape[0]
        a, b = max(0, y0), min(h, y1)
        if b > a:
            cov = canvas[a - y0:b - y0].astype(np.float32) / 255 * (1 - region_mask[a:b])
            img[a:b] = img[a:b] * (1 - cov) + 0.08 * cov
        y += int(line_h * rng.uniform(1.4, 1.9))


def render_page(rng: random.Random, sets: dict, diag: Diagram | None = None, neighbours: bool = True):
    """A page region with the target diagram roughly centred.  Returns
    (page, corners) where corners are the 8x8 area in page pixels."""
    diag = diag or render_diagram(rng, sets)
    d = diag.img
    ds = d.shape[0]
    margin = int(ds * rng.uniform(0.35, 0.8))
    size = ds + 2 * margin
    paper = rng.uniform(0.82, 1.0)
    page = np.full((size, size), paper, np.float32)
    mask = np.zeros_like(page)
    ox = margin + int(rng.uniform(-0.1, 0.1) * margin)
    oy = margin + int(rng.uniform(-0.1, 0.1) * margin)
    page[oy:oy + ds, ox:ox + ds] = d * paper
    mask[oy - 6:oy + ds + 6, ox - 6:ox + ds + 6] = 1
    corners = diag.corners + np.array([ox, oy], np.float32)
    # Neighbouring diagrams (puzzle books)
    if neighbours and rng.random() < 0.35:
        other = render_diagram(rng, sets, sq=int(diag.corners[1, 0] - diag.corners[0, 0]) // 8)
        o = other.img
        gap = int(ds * rng.uniform(0.08, 0.25))
        side = rng.choice(['right', 'left', 'below', 'above'])
        if side == 'right':
            x, y = ox + ds + gap, oy
        elif side == 'left':
            x, y = ox - gap - o.shape[1], oy
        elif side == 'below':
            x, y = ox, oy + ds + gap
        else:
            x, y = ox, oy - gap - o.shape[0]
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + o.shape[1], size), min(y + o.shape[0], size)
        if x1 > x0 and y1 > y0:
            page[y0:y1, x0:x1] = o[y0 - y:y1 - y, x0 - x:x1 - x] * paper
            mask[y0:y1, x0:x1] = 1
    if rng.random() < 0.85:
        _random_text(rng, page, mask, line_h=max(8, int(ds / rng.uniform(14, 30))))
    return page, corners


# ---------------------------------------------------------------- camera

def camera(rng: random.Random, page: np.ndarray, corners: np.ndarray, out_size: int,
           board_frac: tuple[float, float] = (0.55, 0.95), max_rot: float = 6, max_persp: float = 0.08,
           strength: float = 1.0):
    """Photograph the page. The board ends up roughly centred in an
    out_size x out_size frame, covering board_frac of its width."""
    bw = np.linalg.norm(corners[1] - corners[0])
    frac = rng.uniform(*board_frac)
    s = out_size * frac / bw
    ang = np.deg2rad(rng.uniform(-max_rot, max_rot))
    ca, sa = np.cos(ang), np.sin(ang)
    centre = corners.mean(0)
    # similarity: page -> frame
    A = np.array([[s * ca, -s * sa], [s * sa, s * ca]], np.float32)
    shift = np.array([out_size / 2, out_size / 2]) + np.array(
        [rng.uniform(-0.04, 0.04), rng.uniform(-0.04, 0.04)]) * out_size
    def sim(p):
        return (p - centre) @ A.T + shift
    src = np.array([[0, 0], [page.shape[1], 0], [page.shape[1], page.shape[0]], [0, page.shape[0]]], np.float32)
    dst = sim(src).astype(np.float32)
    # perspective: jitter frame corners
    jit = (np.array([[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(4)]) * max_persp * out_size * frac)
    bc = sim(corners)
    bc2 = (bc + jit).astype(np.float32)
    H_sim = cv2.getPerspectiveTransform(src, dst)
    H_j = cv2.getPerspectiveTransform(bc.astype(np.float32), bc2)
    H = H_j @ H_sim
    paper_bg = rng.uniform(0.2, 0.6)
    img = cv2.warpPerspective(page, H, (out_size, out_size), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=paper_bg)
    if s < 0.7:
        # INTER_AREA isn't supported for warpPerspective; emulate by pre-blur
        pass
    new_corners = cv2.perspectiveTransform(corners.reshape(1, 4, 2).astype(np.float32), H)[0]

    # page curl near the spine: smooth displacement
    if rng.random() < 0.3 * strength:
        amp = rng.uniform(0, 0.015) * out_size
        yy, xx = np.mgrid[0:out_size, 0:out_size].astype(np.float32)
        ph = rng.uniform(0, np.pi)
        mx = xx + amp * np.sin(yy / out_size * np.pi + ph)
        my = yy + amp * 0.5 * np.sin(xx / out_size * np.pi + ph)
        img = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # corners shift approximately inversely
        for k in range(4):
            x, y = new_corners[k]
            new_corners[k] = [x - amp * np.sin(y / out_size * np.pi + ph),
                              y - amp * 0.5 * np.sin(x / out_size * np.pi + ph)]

    # lighting: low-frequency gain field + optional hard shadow + glare
    g = cv2.resize(np.random.default_rng(rng.randint(0, 2**31)).uniform(0, 1, (3, 3)).astype(np.float32),
                   (out_size, out_size), interpolation=cv2.INTER_CUBIC)
    lo = rng.uniform(0.45, 0.95) if strength > 0.5 else rng.uniform(0.8, 0.95)
    gain = lo + (1.0 - lo) * g
    img = img * gain * rng.uniform(0.75, 1.1)
    if rng.random() < 0.25 * strength:
        yy, xx = np.mgrid[0:out_size, 0:out_size].astype(np.float32)
        nx, ny = rng.uniform(-1, 1), rng.uniform(-1, 1)
        edge = (xx - out_size / 2) * nx + (yy - out_size / 2) * ny - rng.uniform(-0.4, 0.4) * out_size
        sh = 1 - rng.uniform(0.2, 0.5) * (1 / (1 + np.exp(-edge / (out_size * 0.02))))
        img = img * sh
    if rng.random() < 0.2 * strength:
        cx, cy = rng.uniform(0, out_size), rng.uniform(0, out_size)
        r = rng.uniform(0.05, 0.25) * out_size
        yy, xx = np.mgrid[0:out_size, 0:out_size].astype(np.float32)
        img = img + rng.uniform(0.2, 0.7) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
    # blur (defocus / motion)
    sig = rng.uniform(0, 1.6) * strength
    if sig > 0.3:
        img = cv2.GaussianBlur(img, (0, 0), sig)
    if rng.random() < 0.15 * strength:
        k = rng.randint(3, 7)
        kern = np.zeros((k, k), np.float32)
        if rng.random() < 0.5:
            kern[k // 2, :] = 1
        else:
            kern[:, k // 2] = 1
        img = cv2.filter2D(img, -1, kern / k)
    # gamma, noise, jpeg
    img = np.clip(img, 0, 1) ** rng.uniform(0.8, 1.25)
    img = img + np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(0, 0.03) * strength,
                                                                   img.shape).astype(np.float32)
    img8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    if rng.random() < 0.8:
        q = rng.randint(45, 95)
        ok, enc = cv2.imencode('.jpg', img8, [cv2.IMWRITE_JPEG_QUALITY, q])
        img8 = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return img8, new_corners.astype(np.float32)


# ---------------------------------------------------------------- crops

BOARD_PX = 256
SQ_PX = BOARD_PX // 8


def warp_board(gray8: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Mirror of the app pipeline (src/vision/warp.ts): bilinear warp of the 8x8
    area to 512 px followed by a 2x2 box filter to 256 px, then percentile
    normalisation to 0..1."""
    big = 2 * BOARD_PX
    dst = np.array([[0, 0], [big, 0], [big, big], [0, big]], np.float32)
    H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    w = cv2.warpPerspective(gray8.astype(np.float32), H, (big, big), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    w = w.reshape(BOARD_PX, 2, BOARD_PX, 2).mean(axis=(1, 3))
    return normalise(w)


def normalise(board: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(board, 2), np.percentile(board, 98)
    return np.clip((board - lo) / max(hi - lo, 1e-3), 0, 1).astype(np.float32)


def board_to_squares(board01: np.ndarray) -> np.ndarray:
    return board01.reshape(8, SQ_PX, 8, SQ_PX).transpose(0, 2, 1, 3).reshape(64, SQ_PX, SQ_PX)


def pre_shrink(gray8: np.ndarray, corners: np.ndarray, target_board_px: int = 600):
    """App downsizes the capture so the board is ~600 px wide before warping."""
    bw = max(np.linalg.norm(corners[1] - corners[0]), np.linalg.norm(corners[2] - corners[3]))
    if bw <= target_board_px * 1.05:
        return gray8, corners
    f = target_board_px / bw
    small = cv2.resize(gray8, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return small, corners * f


def training_sample(rng: random.Random, sets: dict):
    diag = render_diagram(rng, sets)
    page, corners = render_page(rng, sets, diag, neighbours=False)
    out = rng.choice([480, 640, 800, 1000])
    img, c = camera(rng, page, corners, out)
    img, c = pre_shrink(img, c)
    # simulate grid-detector error: up to ~8% of a square per corner
    sqpx = np.linalg.norm(c[1] - c[0]) / 8
    err = np.array([[rng.gauss(0, 0.035), rng.gauss(0, 0.035)] for _ in range(4)]) * sqpx
    board = warp_board(img, c + err.astype(np.float32))
    sq = board_to_squares(board)
    labels = np.array([CLASSES.index(s) for s in diag.position], np.uint8)
    return (sq * 255).astype(np.uint8), labels
