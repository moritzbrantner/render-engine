// The sky's environment cubes are filtered once for a static sky and again when it changes
// (render-engine#54). Runs without a GPU: a stand-in renderer counts the passes.
import { expect, test } from 'bun:test';
import { Matrix4, Vector3 } from 'three';
import { ENVIRONMENT_ALTITUDE_TOLERANCE, ENVIRONMENT_TURN_TOLERANCE, SkyPasses } from './sky.js';

function countingRenderer() {
  const r = { draws: 0, targets: [] };
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

const frame = (sun, cameraY = 2) => ({
  groundAlbedo: [0.3, 0.3, 0.3],
  lights: [sun],
  worldFromView: new Matrix4().makeTranslation(0, cameraY, 0),
  clipFromView: new Matrix4().makePerspective(-0.1, 0.1, 0.1, -0.1, 0.1, 1000),
  exposure: 1,
  environment: true,
});

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
