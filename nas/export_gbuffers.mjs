// Export G-buffers (albedo, view-space normal, depth) for several animation times and camera angles.
// Usage: node nas/export_gbuffers.mjs <outDir> [threeDir]
// threeDir: local copy of three@0.147.0 to serve instead of the CDN (for offline/sandboxed runs).
import { chromium } from 'playwright';
import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import path from 'path';

const [outDir = 'nas/data', threeDir] = process.argv.slice(2);
const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
mkdirSync(outDir, { recursive: true });

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || undefined,
  args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader'],
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.error('pageerror:', e.message));
if (threeDir) {
  await page.route('https://cdn.jsdelivr.net/npm/three@0.147.0/**', (r) => {
    const rel = new URL(r.request().url()).pathname.replace('/npm/three@0.147.0/', '');
    r.fulfill({ path: path.join(threeDir, rel), contentType: 'application/javascript' });
  });
}
await page.route('http://nas.local/**', (r) => {
  const p = new URL(r.request().url()).pathname;
  const file = p === '/' ? 'nas/gbuffer.html' : p.slice(1);
  r.fulfill({ path: path.join(root, file) });
});
await page.goto('http://nas.local/');
const { duration } = await page.evaluate(() => window.ready);

const FRAMES = 6, AZIMUTHS = [-35, 0, 35];
const views = [];
for (let f = 0; f < FRAMES; f++) {
  for (const az of AZIMUTHS) {
    const t = (duration * f) / FRAMES;
    const name = `f${f}_az${az}`;
    const passes = await page.evaluate(([t, az]) => window.renderView(t, az), [t, az]);
    for (const [pass, url] of Object.entries(passes)) {
      writeFileSync(path.join(outDir, `${name}_${pass}.png`), Buffer.from(url.split(',')[1], 'base64'));
    }
    views.push({ name, t, azimuth: az });
  }
}
writeFileSync(path.join(outDir, 'views.json'), JSON.stringify(views, null, 1));
console.log(`wrote ${views.length} views to ${outDir}`);
await browser.close();
