// The per-frame work three.js does once per render call (re-checking geometry buffers, InstancedMesh
// matrices, skeleton bone matrices) runs once per Lighting.render frame, not once per pass, in
// headless Chromium (SwiftShader). The frame draws four sun cascades, a spot and a point light's
// six faces and the main pass twice (opaque, then blended under the sky), so per pass it would
// run a dozen times.
//
// Nor does three walk the scene graph once per pass (#168): the frame walks it once
// (`Lighting.frameWork.nodes`) and each pass hands three only the nodes that draw in it, so the
// nodes three visits in a frame follow what the passes draw, not the graph times the passes.
import { afterAll, expect, test } from 'bun:test';
import { resolve } from 'node:path';
import { chromium } from 'playwright';

const root = resolve(import.meta.dir, '../..');

const page = String.raw`<!doctype html>
<script type="importmap">
  { "imports": { "three": "/node_modules/three/build/three.module.js", "three/addons/": "/node_modules/three/examples/jsm/" } }
</script>
<script type="module">
  import * as THREE from 'three';
  import {
    DEFAULT_SKY, DEFAULT_SUN_SHADOWS, Lighting, LitMaterial, loadToneMappingLuts, POINT_LIGHT_SHADOWS, RAW_SUNLIGHT,
    SPOT_LIGHT_SHADOWS,
  } from '/packages/three-lighting/index.js';

  const luts = await loadToneMappingLuts();

  window.run = () => {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 128;
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
    camera.position.set(0, 3, 8);
    camera.lookAt(0, 1, 0);
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new LitMaterial());
    ground.rotation.x = -Math.PI / 2;
    scene.add(ground);
    // A two-bone skinned column.
    const geometry = new THREE.CylinderGeometry(0.3, 0.3, 2, 8, 4);
    const position = geometry.attributes.position;
    const indices = [];
    const weights = [];
    for (let i = 0; i < position.count; i++) {
      const upper = position.getY(i) > 0 ? 1 : 0;
      indices.push(upper, 0, 0, 0);
      weights.push(1, 0, 0, 0);
    }
    geometry.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(indices, 4));
    geometry.setAttribute('skinWeight', new THREE.Float32BufferAttribute(weights, 4));
    const hip = new THREE.Bone();
    const chest = new THREE.Bone();
    chest.position.y = 1;
    hip.add(chest);
    const figure = new THREE.SkinnedMesh(geometry, new LitMaterial());
    figure.add(hip);
    figure.bind(new THREE.Skeleton([hip, chest]));
    figure.position.y = 1;
    scene.add(figure);
    const glass = new THREE.Mesh(new THREE.SphereGeometry(0.5), new LitMaterial({ alpha: 0.5, alphaMode: 'blend' }));
    glass.position.set(-2, 1, 0);
    scene.add(glass);
    // What a pass does not draw: a rig of 200 empty nodes, and a block far outside every view.
    const rig = new THREE.Object3D();
    scene.add(rig);
    for (let i = 0, parent = rig; i < 200; i++) {
      const node = new THREE.Object3D();
      (i % 10 === 0 ? rig : parent).add(node);
      parent = node;
    }
    // A hidden herd: nothing below a hidden node is visited (#177).
    const herd = new THREE.Object3D();
    herd.visible = false;
    for (let i = 0; i < 50; i++) herd.add(new THREE.Mesh(new THREE.BoxGeometry(), new LitMaterial()));
    scene.add(herd);
    const away = new THREE.Mesh(new THREE.BoxGeometry(), new LitMaterial());
    away.position.set(500, 1, 0);
    scene.add(away);
    // 128 posts along x, most of them outside any one shadow view.
    const posts = new THREE.InstancedMesh(new THREE.BoxGeometry(0.2, 1, 0.2), new LitMaterial(), 128);
    for (let i = 0; i < posts.count; i++) posts.setMatrixAt(i, new THREE.Matrix4().makeTranslation((i - 64) * 3, 0.5, -3));
    scene.add(posts);
    let nodes = 0;
    scene.traverse(() => nodes++);

    const l = new Lighting(luts, { shadowMapSize: 256, pointShadowMapSize: 64 });
    l.sun = { direction: [0.4, -1, 0.3], color: [1, 1, 1], illuminance: RAW_SUNLIGHT, shadows: DEFAULT_SUN_SHADOWS };
    l.sky = DEFAULT_SKY;
    l.spotLights = [{ position: [0, 6, 0], direction: [0, -1, 0], color: [1, 1, 1], intensity: 4000, range: 20, radius: 0, innerAngle: 0.3, outerAngle: 0.6, shadows: SPOT_LIGHT_SHADOWS }];
    l.pointLights = [{ position: [1, 2, 1], color: [1, 0.8, 0.6], intensity: 2000, range: 15, radius: 0.1, shadows: POINT_LIGHT_SHADOWS }];

    const skeleton = figure.skeleton;
    const update = skeleton.update.bind(skeleton);
    let updates = 0;
    skeleton.update = () => {
      updates++;
      update();
    };
    const render = renderer.render.bind(renderer);
    let renders = 0;
    let inScene = false;
    renderer.render = (s, c) => {
      if (s === scene) renders++;
      inScene = s === scene;
      render(s, c);
      inScene = false;
    };
    // three tests a node's layers each time a render call's walk reaches it, and again when it
    // draws it; counted for the scene's passes (the sky's and the output's each draw one quad).
    const layersTest = THREE.Layers.prototype.test;
    let visited = 0;
    THREE.Layers.prototype.test = function (layers) {
      if (inScene) visited++;
      return layersTest.call(this, layers);
    };
    const tree = () => {
      const out = [];
      scene.traverse((o) => out.push(o.id, o.parent?.id ?? 0, o.children.length));
      return out.join();
    };
    const before = tree();
    const frame = () => {
      [updates, renders, visited] = [0, 0, 0];
      l.render(renderer, scene, camera);
      return { updates, renders, visited, work: l.frameWork };
    };
    const first = frame();
    chest.rotation.z = 0.5;
    const bent = frame();
    const bones = Array.from(skeleton.boneMatrices);
    // Without the glass nothing blended is left: the blended half of the main pass is not rendered.
    glass.visible = false;
    const unblended = frame();
    glass.visible = true;
    // Outside Lighting, three's own count goes on.
    const plain = renderer.info.render.frame;
    render(scene, camera);
    return { nodes, first, bent, unblended, bones, restored: tree() === before, counted: renderer.info.render.frame - plain, error: renderer.getContext().getError() };
  };
  window.ready = true;
</script>`;

const server = Bun.serve({
  port: 0,
  hostname: '127.0.0.1',
  async fetch(request) {
    const path = decodeURIComponent(new URL(request.url).pathname);
    if (path === '/') return new Response(page, { headers: { 'content-type': 'text/html' } });
    const file = resolve(root, `.${path}`);
    if (!file.startsWith(`${root}/`)) return new Response(null, { status: 403 });
    const body = Bun.file(file);
    return (await body.exists()) ? new Response(body) : new Response(null, { status: 404 });
  },
});

// CI installs Chromium for this test; elsewhere it is skipped without one.
let browser = null;
try {
  browser = await chromium.launch({ headless: true, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--use-gl=angle'] });
} catch (error) {
  if (process.env.CI) throw error;
  console.warn(`frame-work.test.js skipped: no Playwright Chromium (${error.message.split('\n')[0]})`);
}

afterAll(async () => {
  await browser?.close();
  server.stop(true);
});

test.skipIf(!browser)('a frame of many passes updates each skeleton once', async () => {
  const tab = await browser.newPage();
  const errors = [];
  tab.on('pageerror', (e) => errors.push(e.message));
  tab.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  await tab.goto(`http://127.0.0.1:${server.port}/`);
  await tab.waitForFunction(() => window.ready || null, null, { timeout: 60_000 });
  const r = await tab.evaluate(() => window.run());
  console.log(JSON.stringify({ nodes: r.nodes, first: r.first, bent: r.bent }));
  expect(errors).toEqual([]);
  expect(r.error).toBe(0);
  // Sun cascades, the spot view, six point faces and the main pass's two draws.
  expect(r.first.renders).toBeGreaterThanOrEqual(10);
  expect(r.first.updates).toBe(1);
  // The next frame takes the moved bone.
  expect(r.bent.updates).toBe(1);
  expect(r.bones.slice(16).some((v, i) => Math.abs(v - [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1][i]) > 1e-3)).toBe(true);
  expect(r.counted).toBe(1);

  // The frame walks the graph once; three visits, per pass, the scene and what the pass draws.
  const passes = r.bent.work.passes;
  expect(r.nodes).toBeGreaterThan(200);
  // All but the hidden herd's 50 meshes.
  expect(r.bent.work.nodes).toBe(r.nodes - 50);
  expect(passes.map((p) => p.pass)).toEqual([
    'cascade 0', 'cascade 1', 'cascade 2', 'cascade 3', 'spot 0',
    ...[0, 1, 2, 3, 4, 5].map((face) => `point 0 face ${face}`),
    'main opaque', 'main blended',
  ]);
  const listed = passes.reduce((sum, p) => sum + p.objects + 1, 0);
  expect(r.bent.visited).toBeGreaterThanOrEqual(listed);
  // three tests a listed node's layers again when it draws it; a shadow view's stand-in for the
  // posts is the one node it reaches below a listed one.
  expect(r.bent.visited).toBeLessThanOrEqual(2 * (listed + passes.length));
  expect(r.bent.visited).toBeLessThan(r.nodes);
  const objects = Object.fromEntries(passes.map((p) => [p.pass, p.objects]));
  // The ground, the figure and the posts; then the glass alone. The far block is in neither.
  expect(objects['main opaque']).toBe(3);
  expect(objects['main blended']).toBe(1);
  for (const p of passes.slice(0, 11)) expect(p.objects).toBeLessThanOrEqual(5);
  // Posts straddle the views: tested one by one in the first frame, and not again while neither
  // they nor the view change.
  const tested = (frame) => frame.work.passes.reduce((sum, p) => sum + p.instances, 0);
  expect(tested(r.first)).toBeGreaterThanOrEqual(128);
  expect(tested(r.bent)).toBe(0);
  // The scene graph is the consumer's again after a frame.
  expect(r.restored).toBe(true);
  // A pass with nothing to draw is listed, not rendered (#179).
  expect(r.unblended.work.passes.at(-1)).toEqual({ pass: 'main blended', objects: 0, instances: 0 });
  expect(r.unblended.renders).toBe(r.bent.renders - 1);
}, 180_000);
