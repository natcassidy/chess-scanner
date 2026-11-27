// Rasterize piece SVGs (one folder per set) to transparent PNGs using the
// preinstalled Chromium. Usage: node rasterize_pieces.cjs <svgRoot> <outRoot> [size]
const { chromium } = require('playwright');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
const [svgRoot, outRoot, sizeArg] = process.argv.slice(2);
const size = Number(sizeArg || 160);
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: size, height: size } });
for (const set of fs.readdirSync(svgRoot).sort()) {
  const dir = path.join(svgRoot, set);
  if (!fs.statSync(dir).isDirectory()) continue;
  fs.mkdirSync(path.join(outRoot, set), { recursive: true });
  for (const f of fs.readdirSync(dir).filter((x) => x.endsWith('.svg'))) {
    const svg = fs.readFileSync(path.join(dir, f), 'utf8');
    const url = 'data:image/svg+xml;base64,' + Buffer.from(svg).toString('base64');
    await page.setContent(
      `<html><body style="margin:0;background:transparent"><img src="${url}" style="width:${size}px;height:${size}px;display:block"></body></html>`,
    );
    await page.waitForFunction(() => document.querySelector('img').complete);
    await page.screenshot({ path: path.join(outRoot, set, f.replace('.svg', '.png')), omitBackground: true });
  }
  console.log('done', set);
}
await browser.close();
})();
