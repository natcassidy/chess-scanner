# Chess Scanner

Point your iPhone at a chess diagram in a book and get an analysis board for that position.

- **No buttons to scan.** Open the app and the camera is already running. Hold the diagram inside the box, the outline turns green, and the position opens on the board.
- **Analysis board.** Drag or tap to move, use the big ◀ ▶ buttons to step through, and go back and try another move to create a variation. No engine, no spoilers.
- **Side to move is decided by your first move.** Move a black piece first and it's Black to move.
- **Unsure squares are highlighted.** If the scanner isn't confident about a square, it highlights it. Tap **Fix** to correct it, or **Looks right** to dismiss.
- **Works offline** once installed. Every scan and the lines you played are saved under **Recent**.

## Install on iPhone

1. Open the app's URL in Safari (see *Deploy* below).
2. Tap **Share → Add to Home Screen**.
3. Open it from the home screen and allow camera access.

## How it works

Everything runs on the phone; nothing is uploaded.

1. **Board finder** (`src/vision/grid.ts`): about 10 times a second, the region inside the guide box is searched for the 8×8 grid. A rotation-aware edge-projection search finds candidate grids. A hill climb then fits the four corners (with perspective), scoring strong edges on the 18 grid lines, weak edges between them, and the light/dark alternation of every row and column. That last check rejects grids that are off by one square and pages of plain text.
2. **Capture.** After 3 steady frames the app grabs 3 higher-resolution frames, refines the corners, and straightens the board to 256×256.
3. **Piece reader** (`src/vision/classifier.ts`): a small CNN (about 156k weights, 312 KB) classifies each of the 64 squares into empty or one of 12 pieces. Results from the 3 frames are combined.
4. **Chess sanity** (`src/chess/fromScan.ts`): exactly one king per side, no pawns on the first or last rank, and automatic detection of diagrams printed from Black's side.

The model is trained only on synthetic images (`training/`): diagrams rendered with 18 classic book diagram fonts (Merida, Alpha, Leipzig, Cases and others) and 25 piece sets. They are drawn with hatched, grey or dotted dark squares and various frames, placed on fake book pages with text and neighbouring diagrams, then "photographed" with perspective, lighting gradients, shadows, glare, blur, noise and JPEG compression.

## Development

```bash
npm install
npm run dev      # http://localhost:5173 (camera needs https or localhost)
npm test         # chess logic + vision pipeline on synthetic captures
npm run build
```

Open `/?fen=<placement>` to jump straight to a board.

### Retraining the piece model

```bash
python3 -m venv .venv && .venv/bin/pip install numpy opencv-python-headless pillow jax optax
node training/rasterize_pieces.cjs <lichess piece svg dir> <png dir>
.venv/bin/python training/render_fonts.py <scidb .sfd font dir> <png dir> --ttf casefont.ttf
.venv/bin/python training/gen_dataset.py train.npz 8000 1 --pieces <png dir>
.venv/bin/python training/train.py train.npz --out src/vision/model/pieces.bin
```

`training/gen_testframes.py` and `training/make_fake_camera.py` produce test captures. `tests/e2e-scan.cjs` runs the built app in Chromium with a fake camera filming a synthetic book page.

## Deploy

Pushing to `main` runs `.github/workflows/deploy.yml`, which tests, builds and publishes to GitHub Pages. One-time setup: in the repository go to **Settings → Pages → Build and deployment → Source** and choose **GitHub Actions**. The app is then served at `https://<user>.github.io/chess-scanner/`.

## Licence

GPL-3.0-or-later, because the board UI uses [chessground](https://github.com/lichess-org/chessground) (GPL-3.0). Move rules come from [chess.js](https://github.com/jhlywa/chess.js) (BSD-2).
