// node render.js <viewer url with ?scene=...> <output dir>
// Renders every view of the scene into <output dir>/<view.out>.png with headless Chromium.
const { chromium } = require('playwright');
const path = require('path');

(async () => {
  const [url, outDir] = process.argv.slice(2);
  const browser = await chromium.launch({args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']});
  const page = await browser.newPage({viewport: {width: 1600, height: 1600}});
  page.on('console', m => { if (m.type() === 'error') console.error('page:', m.text()); });
  await page.goto(url);
  await page.waitForFunction(() => document.title !== 'loading', null, {timeout: 120000});
  const title = await page.title();
  if (title !== 'ready') { console.error(title); process.exit(1); }
  const n = await page.evaluate(() => window.viewCount);
  for (let i = 0; i < n; i++) {
    const name = await page.evaluate(i => window.renderView(i), i);
    const canvas = await page.$('canvas');
    const file = path.join(outDir, name + '.png');
    await canvas.screenshot({path: file});
    console.log(file);
  }
  await browser.close();
})();
