// Run the GLSL compositor (nas/nas_shader.js) on exported G-buffers and dump the raw RGBA8 result.
// Usage: node nas/parity.mjs <view name> [threeDir]   -> nas/out/parity_<view>.bin (premultiplied RGBA, bottom-up rows)
import { chromium } from 'playwright';
import { writeFileSync } from 'fs';
import path from 'path';

const [view = 'f5_az35', threeDir] = process.argv.slice(2);
const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || undefined,
  args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader'],
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.error('pageerror:', e.message));
page.on('console', (m) => { if (m.type() === 'error') console.error('console:', m.text()); });
if (threeDir) {
  await page.route('https://cdn.jsdelivr.net/npm/three@0.147.0/**', (r) => {
    const rel = new URL(r.request().url()).pathname.replace('/npm/three@0.147.0/', '');
    r.fulfill({ path: path.join(threeDir, rel), contentType: 'application/javascript' });
  });
}
await page.route('http://nas.local/**', (r) => {
  const p = new URL(r.request().url()).pathname;
  if (p === '/') return r.fulfill({ contentType: 'text/html', body: '<!doctype html><body><script src="https://cdn.jsdelivr.net/npm/three@0.147.0/build/three.min.js"></script><script src="/nas/nas_shader.js"></script>' });
  r.fulfill({ path: path.join(root, p.slice(1)) });
});
await page.goto('http://nas.local/');
const bytes = await page.evaluate(async (view) => {
  const P = await (await fetch('/nas/out/params.json')).json();
  const load = (pass) => new Promise((res) => new THREE.TextureLoader().load(`/nas/data/${view}_${pass}.png`, (t) => {
    t.minFilter = t.magFilter = THREE.NearestFilter; t.generateMipmaps = false; res(t);
  }));
  const [albedo, normal, depth] = await Promise.all(['albedo', 'normal', 'depth'].map(load));
  const W = albedo.image.width, H = albedo.image.height;
  const renderer = new THREE.WebGLRenderer({ premultipliedAlpha: false });
  const target = new THREE.WebGLRenderTarget(W, H);
  const mat = createNASMaterial(THREE, P);
  Object.assign(mat.uniforms, { tAlbedo: { value: albedo }, tNormal: { value: normal }, tDepth: { value: depth } });
  mat.uniforms.texel.value.set(1 / W, 1 / H);
  const scene = new THREE.Scene();
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat);
  quad.frustumCulled = false;
  scene.add(quad);
  renderer.setRenderTarget(target);
  renderer.setClearColor(0x000000, 0);
  renderer.clear();
  renderer.render(scene, new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1));
  const buf = new Uint8Array(W * H * 4);
  renderer.readRenderTargetPixels(target, 0, 0, W, H, buf);
  return { W, H, data: Array.from(buf) };
}, view);
writeFileSync(path.join(root, `nas/out/parity_${view}.bin`), Buffer.from(bytes.data));
console.log(`wrote nas/out/parity_${view}.bin (${bytes.W}x${bytes.H})`);
await browser.close();
