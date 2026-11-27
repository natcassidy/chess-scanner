// End-to-end: run the built app in Chromium with a fake camera that films a
// synthetic book page; wait for the board screen and compare the position.
// usage: node tests/e2e-scan.cjs <dir with camera.y4m + truth.json> [url] [screenshotDir]
const { chromium, devices } = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
(async () => {
  const dir = process.argv[2];
  const url = process.argv[3] || 'http://localhost:4173/';
  const shots = process.argv[4];
  const truth = JSON.parse(fs.readFileSync(path.join(dir, 'truth.json'), 'utf8'));
  const browser = await chromium.launch({
    args: [
      '--use-fake-device-for-media-stream',
      '--use-fake-ui-for-media-stream',
      `--use-file-for-fake-video-capture=${path.join(dir, 'camera.y4m')}`,
    ],
  });
  const ctx = await browser.newContext({ ...devices['iPhone 13'], deviceScaleFactor: 1 });
  await ctx.grantPermissions(['camera']);
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('pageerror:', e.message));
  page.on('console', (m) => { if (m.type() === 'error') console.log('console error:', m.text()); });
  const t0 = Date.now();
  await page.goto(url);
  if (shots) {
    await page.waitForTimeout(1500);
    await page.screenshot({ path: path.join(shots, 'scan-live.png') });
  }
  let ok = false;
  try {
    await page.waitForSelector('.board-screen', { timeout: 30000 });
    ok = true;
  } catch {
    console.log('no board after 30s; status:', await page.locator('.scanner-status').textContent().catch(() => '?'));
  }
  const secs = ((Date.now() - t0) / 1000).toFixed(1);
  if (ok) {
    await page.waitForTimeout(500);
    if (shots) await page.screenshot({ path: path.join(shots, 'scan-result.png') });
    const got = await page.evaluate(() => {
      const list = JSON.parse(localStorage.getItem('chess-scanner.sessions.v1') || '[]');
      return list[0] && { fen: list[0].tree.boardFen, uncertain: list[0].uncertain };
    });
    const match = got.fen === truth.fen;
    console.log(JSON.stringify({ match, secs, got: got.fen, want: truth.fen, uncertain: got.uncertain }));
  }
  await browser.close();
  process.exit(ok ? 0 : 1);
})();
