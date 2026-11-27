// Render public/icon.svg to the PNG sizes iOS and the manifest need.
const { chromium } = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
(async () => {
  const pub = path.join(__dirname, '..', 'public');
  const svg = fs.readFileSync(path.join(pub, 'icon.svg'), 'utf8');
  const browser = await chromium.launch();
  for (const [name, size] of [['apple-touch-icon.png', 180], ['icon-192.png', 192], ['icon-512.png', 512]]) {
    const page = await browser.newPage({ viewport: { width: size, height: size } });
    await page.setContent(`<html><body style="margin:0">${svg.replace('<svg ', `<svg width="${size}" height="${size}" `)}</body></html>`);
    await page.screenshot({ path: path.join(pub, name) });
    await page.close();
  }
  await browser.close();
})();
