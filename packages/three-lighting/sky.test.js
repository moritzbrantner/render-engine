// The sky's LUTs are drawn only when their inputs change (mmorpg#1115), the aerial-view LUT
// several slices per pass, and its environment cubes are filtered once for a static sky and again
// when it changes (render-engine#54). Runs without a GPU: a stand-in renderer counts the passes.
import { expect, test } from 'bun:test';
import { Matrix4, Vector3 } from 'three';
import { ENVIRONMENT_ALTITUDE_TOLERANCE, ENVIRONMENT_TURN_TOLERANCE, SkyPasses } from './sky.js';

/** A stand-in WebGL 2 context with `drawBuffers` draw buffers: it records the aerial LUT layers attached. */
function fakeGl(drawBuffers) {
  const gl = {
    MAX_DRAW_BUFFERS: 0x8824,
    MAX_COLOR_ATTACHMENTS: 0x8cdf,
    FRAMEBUFFER: 0x8d40,
    TEXTURE_2D: 0x0de1,
    COLOR_ATTACHMENT0: 0x8ce0,
    COLOR_ATTACHMENT2: 0x8ce2,
    layers: [],
    framebuffers: 0,
  };
  gl.getParameter = () => drawBuffers;
  gl.createFramebuffer = () => (gl.framebuffers++, {});
  gl.isFramebuffer = () => true;
  gl.deleteFramebuffer = () => gl.framebuffers--;
  gl.framebufferTexture2D = () => {};
  gl.framebufferTextureLayer = (_target, _attachment, _texture, _level, layer) => gl.layers.push(layer);
  gl.drawBuffers = () => {};
  return gl;
}

function countingRenderer(drawBuffers = 8) {
  const r = { draws: 0, targets: [], gl: fakeGl(drawBuffers) };
  // A new map stands for a restored context: every GL texture is new.
  r.loseContext = () => (r.props = new WeakMap());
  r.loseContext();
  r.properties = { get: (o) => (r.props.has(o) || r.props.set(o, {}), r.props.get(o)) };
  r.initRenderTarget = (target) => {
    for (const t of target.textures) r.properties.get(t).__webglTexture ??= {};
  };
  r.getContext = () => r.gl;
  r.state = { bindFramebuffer: () => true };
  r.setRenderTargetFramebuffer = () => {};
  r.setRenderTarget = (target) => r.targets.push(target);
  r.render = () => r.draws++;
  return r;
}

const sunAt = (elevation) => ({
  directionToLight: new Vector3(0, Math.sin(elevation), -Math.cos(elevation)),
  color: [130_000, 130_000, 130_000],
  diskSize: 0.0093,
  diskIntensity: 1,
});

const frame = (sun, cameraY = 2, yaw = 0) => ({
  groundAlbedo: [0.3, 0.3, 0.3],
  lights: [sun],
  worldFromView: new Matrix4().makeRotationY(yaw).setPosition(0, cameraY, 0),
  clipFromView: new Matrix4().makePerspective(-0.1, 0.1, 0.1, -0.1, 0.1, 1000),
  exposure: 1,
  environment: true,
});

/** The LUT passes one `render` draws. */
const lutPasses = (sky, renderer, f) => {
  const before = sky.lutPasses;
  sky.render(renderer, f);
  return sky.lutPasses - before;
};

test('a static sky filters its environment cubes once', () => {
  const sky = new SkyPasses();
  const renderer = countingRenderer();
  sky.render(renderer, frame(sunAt(0.5)));
  const first = renderer.draws;
  expect(sky.environmentFilters).toBe(1);
  for (let i = 0; i < 10; i++) sky.render(renderer, frame(sunAt(0.5), 2 + i));
  expect(sky.environmentFilters).toBe(1);
  const perFrame = (renderer.draws - first) / 10;
  console.log(`sky render calls: first frame ${first}, static frames ${perFrame}`);
  expect(perFrame).toBeLessThan(first - 100);
  sky.dispose();
});

test('each LUT is drawn only when its inputs change, the aerial view in six passes', () => {
  const sky = new SkyPasses();
  const renderer = countingRenderer();
  // Transmittance, multiscattering, sky view and the aerial view's 6 passes.
  expect(lutPasses(sky, renderer, frame(sunAt(0.5)))).toBe(9);
  for (let i = 0; i < 5; i++) expect(lutPasses(sky, renderer, frame(sunAt(0.5)))).toBe(0);
  // The camera turns: only the aerial view (camera space).
  expect(lutPasses(sky, renderer, frame(sunAt(0.5), 2, 0.1))).toBe(6);
  // The camera moves or the sun turns: the sky view too.
  expect(lutPasses(sky, renderer, frame(sunAt(0.5), 3, 0.1))).toBe(7);
  expect(lutPasses(sky, renderer, frame(sunAt(0.6), 3, 0.1))).toBe(7);
  // A new ground albedo: the multiscattering LUT too; the transmittance LUT never again.
  expect(lutPasses(sky, renderer, { ...frame(sunAt(0.6), 3, 0.1), groundAlbedo: [0.2, 0.3, 0.3] })).toBe(8);
  // A restored context holds no LUT: all of them again, and the environment cubes.
  const filters = sky.environmentFilters;
  renderer.loseContext();
  expect(lutPasses(sky, renderer, frame(sunAt(0.6), 3, 0.1))).toBe(9);
  expect(sky.environmentFilters).toBe(filters + 1);
  sky.dispose();
  expect(renderer.gl.framebuffers).toBe(0);
});

test('with reuse tolerances the sky and aerial views are kept while the sun and camera drift within them', () => {
  const sky = new SkyPasses();
  const renderer = countingRenderer();
  const degree = Math.PI / 180;
  const reuse = { lightTurn: 0.1 * degree, cameraMove: 1, cameraTurn: 0.5 * degree };
  const at = (elevation, altitude = 2, yaw = 0) => ({ ...frame(sunAt(elevation), altitude, yaw), reuse });
  expect(lutPasses(sky, renderer, at(0.5))).toBe(9);
  // The sun creeps and the camera drifts and turns a little: nothing is drawn.
  expect(lutPasses(sky, renderer, at(0.5 + 0.04 * degree, 2.4, 0.2 * degree))).toBe(0);
  expect(lutPasses(sky, renderer, at(0.5 + 0.08 * degree, 2.8, 0.4 * degree))).toBe(0);
  // Past the camera's turn since the aerial view was drawn: the aerial view alone.
  expect(lutPasses(sky, renderer, at(0.5 + 0.08 * degree, 2.8, 0.6 * degree))).toBe(6);
  // Past the sun's turn since the sky view was drawn; the aerial view, drawn later, stands.
  expect(lutPasses(sky, renderer, at(0.5 + 0.12 * degree, 2.8, 0.6 * degree))).toBe(1);
  // Past the camera's move since either was drawn.
  expect(lutPasses(sky, renderer, at(0.5 + 0.12 * degree, 4.2, 0.6 * degree))).toBe(7);
  // A light's colour is never within a tolerance.
  const dimmer = at(0.5 + 0.12 * degree, 4.2, 0.6 * degree);
  dimmer.lights = dimmer.lights.map((l) => ({ ...l, color: l.color.map((c) => c * 0.99) }));
  expect(lutPasses(sky, renderer, dimmer)).toBe(7);
  // Without tolerances the smallest drift draws both.
  lutPasses(sky, renderer, frame(sunAt(0.6), 4.2));
  expect(lutPasses(sky, renderer, frame(sunAt(0.6 + 0.001 * degree), 4.2))).toBe(7);
  sky.dispose();
});

test('the aerial march stores every slice once, as many a pass as the draw buffers allow', () => {
  for (const [drawBuffers, passes] of [
    [8, 6],
    [4, 16],
  ]) {
    const sky = new SkyPasses();
    const renderer = countingRenderer(drawBuffers);
    expect(lutPasses(sky, renderer, frame(sunAt(0.5)))).toBe(3 + passes);
    expect(renderer.gl.layers).toEqual(Array.from({ length: 32 }, (_, i) => i));
    sky.dispose();
  }
});

test('a moving sun, a new ground albedo or a climbing camera re-filters', () => {
  const sky = new SkyPasses();
  const renderer = countingRenderer();
  sky.render(renderer, frame(sunAt(0.5)));
  // Within the tolerance: kept.
  sky.render(renderer, frame(sunAt(0.5 + ENVIRONMENT_TURN_TOLERANCE / 2)));
  expect(sky.environmentFilters).toBe(1);
  sky.render(renderer, frame(sunAt(0.5 + 2 * ENVIRONMENT_TURN_TOLERANCE)));
  expect(sky.environmentFilters).toBe(2);
  // A slowly moving sun re-filters once it has turned past the tolerance since the last filter.
  let elevation = 0.5 + 2 * ENVIRONMENT_TURN_TOLERANCE;
  for (let i = 0; i < 4; i++) {
    elevation += 0.6 * ENVIRONMENT_TURN_TOLERANCE;
    sky.render(renderer, frame(sunAt(elevation)));
  }
  expect(sky.environmentFilters).toBe(4);
  sky.render(renderer, { ...frame(sunAt(elevation)), groundAlbedo: [0.2, 0.3, 0.3] });
  expect(sky.environmentFilters).toBe(5);
  const high = { ...frame(sunAt(elevation), 3 * ENVIRONMENT_ALTITUDE_TOLERANCE), groundAlbedo: [0.2, 0.3, 0.3] };
  sky.render(renderer, high);
  expect(sky.environmentFilters).toBe(6);
  // Switching the environment light off and on filters again.
  sky.render(renderer, { ...high, environment: false });
  sky.render(renderer, high);
  expect(sky.environmentFilters).toBe(7);
  sky.dispose();
});
