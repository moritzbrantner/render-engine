// Lighting.prepare (#147) in headless Chromium (SwiftShader): after it resolves, the first frame
// of the same scene creates no program and links none. Every pass is covered: the sky, sun
// cascades, spot and point-light views (with masked, double-sided and instanced casters, whose
// culled stand-ins draw without instance colours), the HDR pass with blended draws, tone mapping
// and output. A control shows three's compileAsync leaves that frame programs to link.
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
    DEFAULT_SKY, DEFAULT_SUN_SHADOWS, FogFalloff, Lighting, LitMaterial, loadToneMappingLuts,
    POINT_LIGHT_SHADOWS, RAW_SUNLIGHT, SPOT_LIGHT_SHADOWS,
  } from '/packages/three-lighting/index.js';

  const luts = await loadToneMappingLuts();

  function world() {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
    camera.position.set(0, 5, 15);
    camera.lookAt(0, 0, 0);
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), new LitMaterial({ baseColor: [0.5, 0.5, 0.5] }));
    ground.rotation.x = -Math.PI / 2;
    scene.add(ground);
    const box = new THREE.BoxGeometry(1, 1, 1);
    scene.add(new THREE.Mesh(box, new LitMaterial({ baseColor: [0.8, 0.2, 0.1], metallic: 1 })));
    const texture = new THREE.DataTexture(new Uint8Array([255, 255, 255, 255, 255, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255, 255]), 2, 2);
    texture.needsUpdate = true;
    const leaves = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new LitMaterial({ baseColorTexture: texture, alphaMode: 'mask', doubleSided: true }));
    leaves.position.set(2, 1, 0);
    scene.add(leaves);
    const glass = new THREE.Mesh(new THREE.SphereGeometry(0.5), new LitMaterial({ alpha: 0.5, alphaMode: 'blend' }));
    glass.position.set(-2, 1, 0);
    scene.add(glass);
    // A row of instances straddling every shadow view: they draw through culled stand-ins.
    const row = new THREE.InstancedMesh(box, new LitMaterial(), 200);
    for (let i = 0; i < 200; i++) {
      row.setMatrixAt(i, new THREE.Matrix4().makeTranslation(-300 + 3 * i, 0.5, -3));
      row.setColorAt(i, new THREE.Color(i / 200, 0.5, 0.5));
    }
    scene.add(row);
    return { scene, camera };
  }

  function lighting() {
    const l = new Lighting(luts, { shadowMapSize: 256, pointShadowMapSize: 64 });
    l.sun = { direction: [0.4, -1, 0.3], color: [1, 1, 1], illuminance: RAW_SUNLIGHT, shadows: DEFAULT_SUN_SHADOWS };
    l.sky = DEFAULT_SKY;
    l.fog = { color: [0.5, 0.6, 0.7, 1], falloff: FogFalloff.fromVisibility(200) };
    l.spotLights = [{ position: [0, 6, 0], direction: [0, -1, 0], color: [1, 1, 1], intensity: 4000, range: 20, radius: 0, innerAngle: 0.3, outerAngle: 0.6, shadows: SPOT_LIGHT_SHADOWS }];
    l.pointLights = [{ position: [1, 2, 1], color: [1, 0.8, 0.6], intensity: 2000, range: 15, radius: 0.1, shadows: POINT_LIGHT_SHADOWS }];
    return l;
  }

  function counted() {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 128;
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
    const gl = renderer.getContext();
    const link = gl.linkProgram.bind(gl);
    const counts = { links: 0 };
    gl.linkProgram = (program) => {
      counts.links++;
      link(program);
    };
    // Programs and links one call creates.
    const during = (f) => {
      const [programs, links] = [renderer.info.programs.length, counts.links];
      f();
      return { programs: renderer.info.programs.length - programs, links: counts.links - links };
    };
    return { renderer, during };
  }

  window.run = async () => {
    const { renderer, during } = counted();
    const { scene, camera } = world();
    const l = lighting();
    const prepared = await l.prepare(renderer, scene, camera);
    const first = during(() => l.render(renderer, scene, camera));
    // A light more changes the lit materials' defines: prepare again before that frame.
    l.pointLights = [...l.pointLights, { position: [-3, 1, 2], color: [0.6, 0.8, 1], intensity: 800, range: 8, radius: 0, shadows: null }];
    const reprepared = await l.prepare(renderer, scene, camera);
    const afterChange = during(() => l.render(renderer, scene, camera));

    // The control: three's own compileAsync over the same kind of scene.
    const control = counted();
    const other = world();
    await control.renderer.compileAsync(other.scene, other.camera);
    const controlFirst = control.during(() => lighting().render(control.renderer, other.scene, other.camera));

    return {
      parallel: renderer.extensions.has('KHR_parallel_shader_compile'),
      error: renderer.getContext().getError(),
      prepared, first, reprepared, afterChange, controlFirst,
    };
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
  console.warn(`prepare.test.js skipped: no Playwright Chromium (${error.message.split('\n')[0]})`);
}

afterAll(async () => {
  await browser?.close();
  server.stop(true);
});

test.skipIf(!browser)('after Lighting.prepare the first frame creates and links no program', async () => {
  const tab = await browser.newPage();
  const errors = [];
  tab.on('pageerror', (e) => errors.push(e.message));
  tab.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  await tab.goto(`http://127.0.0.1:${server.port}/`);
  await tab.waitForFunction(() => window.ready || null, null, { timeout: 60_000 });
  const r = await tab.evaluate(() => window.run());
  console.log(JSON.stringify(r));
  expect(errors).toEqual([]);
  expect(r.error).toBe(0);
  expect(r.prepared).toBeGreaterThan(0);
  expect(r.first).toEqual({ programs: 0, links: 0 });
  expect(r.reprepared).toBeGreaterThan(0);
  expect(r.afterChange).toEqual({ programs: 0, links: 0 });
  // three's compileAsync compiles variants the frame does not draw: the frame links its own.
  expect(r.controlFirst.programs).toBeGreaterThan(0);
}, 180_000);
