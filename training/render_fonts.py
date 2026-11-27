"""Render chess diagram fonts (FontForge .sfd sources, e.g. from Scidb) and
Marroquin-style .ttf fonts to RGBA piece sprites like the lichess sets.

usage: python render_fonts.py SFD_DIR OUT_DIR [--size 160] [--ttf casefont.ttf ...]

White piece  = silhouette filled white + glyph ink.
Black piece  = the font's own black glyph when it has one, otherwise the
               silhouette filled black with interior detail lines in white
               (how Merida-style fonts print black pieces).
"""
import argparse
import os
import re

import cv2
import numpy as np

NAMES = {'King': 'K', 'Queen': 'Q', 'Rook': 'R', 'Bishop': 'B', 'Knight': 'N', 'Pawn': 'P'}


def parse_sfd(path):
    txt = open(path, encoding='latin-1').read()
    ascent = int(re.search(r'^Ascent: (\d+)', txt, re.M).group(1))
    descent = int(re.search(r'^Descent: (\d+)', txt, re.M).group(1))
    glyphs = {}
    by_index = {}
    for block in re.findall(r'StartChar: (.*?)\nEndChar', txt, re.S):
        name = block.split('\n', 1)[0].strip()
        enc = re.search(r'^Encoding: (-?\d+) (-?\d+) (\d+)', block, re.M)
        idx = int(enc.group(3)) if enc else None
        contours = []
        fore = block.split('Fore', 1)[1] if '\nFore' in block else ''
        m = re.search(r'^(?:\s*SplineSet\n)?(.*?)EndSplineSet', fore, re.S)
        if m:
            cur = None
            for line in m.group(1).strip().split('\n'):
                parts = line.split()
                if not parts:
                    continue
                op = parts[-2] if len(parts) >= 2 and parts[-2] in 'mlc' else None
                if op is None:
                    continue
                nums = [float(x) for x in parts[:-2]]
                if op == 'm':
                    cur = [('m', nums)]
                    contours.append(cur)
                elif cur is not None:
                    cur.append((op, nums))
        refs = []
        for r in re.findall(r'^Refer: (\d+) (-?\d+) \w ([-\d.e]+) ([-\d.e]+) ([-\d.e]+) ([-\d.e]+) ([-\d.e]+) ([-\d.e]+)',
                            block, re.M):
            refs.append((int(r[0]), [float(x) for x in r[2:]]))
        g = {'name': name, 'contours': contours, 'refs': refs}
        glyphs[name] = g
        if idx is not None:
            by_index[idx] = g
    return ascent, descent, glyphs, by_index


def flatten(contours, steps=12):
    polys = []
    for c in contours:
        pts = []
        x0 = y0 = None
        for op, n in c:
            if op in 'ml':
                x0, y0 = n[0], n[1]
                pts.append((x0, y0))
            else:
                x1, y1, x2, y2, x3, y3 = n
                for i in range(1, steps + 1):
                    t = i / steps
                    a = (1 - t) ** 3
                    b = 3 * (1 - t) ** 2 * t
                    cc = 3 * (1 - t) * t ** 2
                    d = t ** 3
                    pts.append((a * x0 + b * x1 + cc * x2 + d * x3, a * y0 + b * y1 + cc * y2 + d * y3))
                x0, y0 = x3, y3
        if len(pts) >= 3:
            polys.append(np.array(pts))
    return polys


def glyph_polys(g, by_index, depth=0):
    polys = flatten(g['contours'])
    if depth < 4:
        for idx, (a, b, c, d, e, f) in g['refs']:
            if idx in by_index:
                for p in glyph_polys(by_index[idx], by_index, depth + 1):
                    x, y = p[:, 0], p[:, 1]
                    polys.append(np.stack([a * x + c * y + e, b * x + d * y + f], 1))
    return polys


def raster(polys, ascent, descent, size, ss=4):
    S = size * ss
    em = ascent + descent
    img = np.zeros((S, S), np.uint8)
    pts = [np.round(np.stack([p[:, 0] * S / em, (ascent - p[:, 1]) * S / em], 1) * 16).astype(np.int32)
           for p in polys]
    # even-odd fill by XOR-ing each contour
    for p in pts:
        layer = np.zeros_like(img)
        cv2.fillPoly(layer, [p], 255, lineType=cv2.LINE_8, shift=4)
        img ^= layer
    return img


def centre(ink, sil):
    ys, xs = np.nonzero(sil)
    if not len(xs):
        return ink, sil
    S = ink.shape[0]
    dx = int(round(S / 2 - (xs.min() + xs.max()) / 2))
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(ink, M, (S, S)), cv2.warpAffine(sil, M, (S, S))


def silhouette(ink):
    """Everything not reachable from the border = inside the piece."""
    S = ink.shape[0]
    pad = np.pad((ink > 127).astype(np.uint8) * 255, 1)
    mask = np.zeros((S + 4, S + 4), np.uint8)
    flood = pad.copy()
    cv2.floodFill(flood, mask, (0, 0), 128)
    outside = flood[1:-1, 1:-1] == 128
    return (~outside).astype(np.uint8) * 255


def make_black(ink, sil, ss):
    """Merida-style black piece: black silhouette, interior lines white."""
    k = max(3, int(0.06 * ink.shape[0]))
    inner = cv2.erode(sil, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    lum = np.zeros_like(ink, np.float32)
    interior_lines = (ink > 127) & (inner > 127)
    lum[interior_lines] = 1.0
    return lum


def save(out_dir, key, lum, alpha, size, ss):
    lum = cv2.resize(lum.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize((alpha > 127).astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    rgba = np.zeros((size, size, 4), np.uint8)
    rgba[..., :3] = (np.clip(lum, 0, 1) * 255).astype(np.uint8)[..., None]
    rgba[..., 3] = (alpha * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, key + '.png'), rgba)


def render_sfd(path, out_root, size):
    ascent, descent, glyphs, by_index = parse_sfd(path)
    name = os.path.basename(path).replace('ScidbChess', '').replace('.sfd', '').lower()
    out = os.path.join(out_root, 'font-' + name)
    os.makedirs(out, exist_ok=True)
    ss = 4
    for long, sym in NAMES.items():
        alias = {'Bishop': 'uni2657'}
        wg = glyphs.get('White' + long) or glyphs.get(alias.get(long, ''))
        if wg is None:
            print('missing', name, long)
            continue
        ink = raster(glyph_polys(wg, by_index), ascent, descent, size, ss)
        sil = silhouette(ink)
        mg = glyphs.get('Mask' + long)
        if mg is not None:
            sil = np.maximum(sil, raster(glyph_polys(mg, by_index), ascent, descent, size, ss))
        bg = glyphs.get('Black' + long)
        own_black = bg is not None and bool(bg['contours'])
        if own_black:
            bink = raster(glyph_polys(bg, by_index), ascent, descent, size, ss)
            bsil = np.maximum(silhouette(bink), sil)
            bink_c, bsil_c = centre(bink, bsil)
            save(out, 'b' + sym, 1.0 - (bink_c > 127).astype(np.float32), bsil_c, size, ss)
        ink_c, sil_c = centre(ink, sil)
        save(out, 'w' + sym, 1.0 - (ink_c > 127).astype(np.float32), sil_c, size, ss)
        if not own_black:
            save(out, 'b' + sym, make_black(ink_c, sil_c, ss), sil_c, size, ss)
    return out


def render_ttf(path, out_root, size):
    """Marroquin layout: pnbrqk = white pieces, omvtwl = black pieces."""
    from PIL import Image, ImageDraw, ImageFont
    name = os.path.basename(path).split('.')[0].lower()
    out = os.path.join(out_root, 'ttf-' + name)
    os.makedirs(out, exist_ok=True)
    ss = 4
    S = size * ss
    font = ImageFont.truetype(path, int(S * 0.95))
    for wch, bch, sym in zip('pnbrqk', 'omvtwl', 'PNBRQK'):
        for ch, col in ((wch, 'w'), (bch, 'b')):
            im = Image.new('L', (S, S), 0)
            d = ImageDraw.Draw(im)
            bbox = d.textbbox((0, 0), ch, font=font)
            d.text(((S - (bbox[2] - bbox[0])) / 2 - bbox[0], (S - (bbox[3] - bbox[1])) / 2 - bbox[1]), ch, font=font, fill=255)
            ink = np.array(im)
            sil = silhouette(ink)
            if col == 'w':
                save(out, col + sym, 1.0 - (ink > 127).astype(np.float32), sil, size, ss)
            else:
                save(out, col + sym, 1.0 - (ink > 127).astype(np.float32), sil, size, ss)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('sfd_dir'); ap.add_argument('out')
    ap.add_argument('--size', type=int, default=160)
    ap.add_argument('--ttf', nargs='*', default=[])
    a = ap.parse_args()
    for f in sorted(os.listdir(a.sfd_dir)):
        if f.endswith('.sfd'):
            print(render_sfd(os.path.join(a.sfd_dir, f), a.out, a.size))
    for f in a.ttf:
        print(render_ttf(f, a.out, a.size))
