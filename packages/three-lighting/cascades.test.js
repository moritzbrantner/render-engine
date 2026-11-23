// Staggered sun cascades (#149): which cascades a frame redraws, what a kept one holds, and when the
// view or the sun forces a redraw. Runs without a GPU.
import { expect, test } from 'bun:test';
import { PerspectiveCamera } from 'three';
import { cascadeRanges, fitCascade, sliceInside, STAGGER_MARGIN, StaggeredCascades } from './cascades.js';

const SHADOWS = {
  numCascades: 3,
  minDistance: 0.2,
  firstCascadeFarBound: 12,
  maxDistance: 160,
  overlapProportion: 0.2,
  depthBias: 0.02,
  normalBias: 1.8,
};
const MAP = 2048;
const SUN = [-0.4, -0.8, -0.3];
const FOV = (55 * Math.PI) / 180;
const ASPECT = 1.6;

function view(x, z, yaw = 0) {
  const camera = new PerspectiveCamera(55, ASPECT, 0.1, 1000);
  camera.position.set(x, 3, z);
  camera.rotation.set(-0.2, yaw, 0, 'YXZ');
  camera.updateMatrixWorld();
  return camera.matrixWorld.elements;
}

const frame = (cascades, worldFromView, extra = {}) =>
  cascades.update({ shadows: SHADOWS, mapSize: MAP, sunDirection: SUN, fovY: FOV, aspect: ASPECT, worldFromView, intervals: [1, 2, 2], ...extra });

const redraws = (frameCascades) => frameCascades.map((c) => c.redraw);

test('a margin widens the fit by whole metres on every side, and none is Bevy\'s fit', () => {
  const world = view(0, 0);
  const [near, far] = cascadeRanges(SHADOWS)[2];
  const plain = fitCascade(near, far, MAP, SUN, FOV, ASPECT, world);
  expect(fitCascade(near, far, MAP, SUN, FOV, ASPECT, world, 0)).toEqual(plain);
  const wide = fitCascade(near, far, MAP, SUN, FOV, ASPECT, world, STAGGER_MARGIN);
  const margin = Math.ceil(plain.diameter * STAGGER_MARGIN);
  expect(wide.diameter).toBe(plain.diameter + 2 * margin);
  expect(wide.texelSize).toBe(Math.fround(wide.diameter / MAP));
  expect(wide.depthRange).toBeCloseTo(plain.depthRange + 2 * margin, 3);
  expect(sliceInside(wide, near, far, FOV, ASPECT, world, MAP)).toBe(true);
  expect(sliceInside(wide, near, far, FOV, ASPECT, view(400, 0), MAP)).toBe(false);
});

test('cascades of one interval take turns, and a kept cascade keeps the matrices it was drawn with', () => {
  const cascades = new StaggeredCascades();
  const first = frame(cascades, view(0, 0));
  expect(redraws(first)).toEqual([true, true, true]);
  let last = first;
  // Walking on at about 6 m/s: no forced redraw, exactly two cascades a frame.
  for (let f = 1; f <= 60; f++) {
    const now = frame(cascades, view(0, -0.1 * f));
    expect(redraws(now)).toEqual(f % 2 === 1 ? [true, true, false] : [true, false, true]);
    for (let i = 0; i < 3; i++) {
      if (!now[i].redraw) expect(now[i].clipFromWorld).toEqual(last[i].clipFromWorld);
    }
    last = now;
  }
});

test('intervals of one redraw every cascade with Bevy\'s fit', () => {
  const cascades = new StaggeredCascades();
  const world = view(3, 4, 0.5);
  for (let f = 0; f < 4; f++) {
    const now = frame(cascades, world, { intervals: [] });
    expect(redraws(now)).toEqual([true, true, true]);
    cascadeRanges(SHADOWS).forEach(([near, far], i) => {
      expect(now[i].clipFromWorld).toEqual(fitCascade(near, far, MAP, SUN, FOV, ASPECT, world).clipFromWorld);
    });
  }
});

test('leaving a kept cascade, turning the sun or another scene redraws early', () => {
  const cascades = new StaggeredCascades();
  frame(cascades, view(0, 0));
  frame(cascades, view(0, 0));
  // A teleport: the view leaves both kept cascades.
  expect(redraws(frame(cascades, view(500, 0)))).toEqual([true, true, true]);
  frame(cascades, view(500, 0));
  // A turn of the camera on the spot moves the far slices across the map.
  expect(redraws(frame(cascades, view(500, 0, Math.PI)))).toEqual([true, true, true]);
  frame(cascades, view(500, 0, Math.PI));
  // The sun creeping (a tenth of a degree about the vertical) keeps them, three degrees do not.
  const turn = (deg) => {
    const a = (deg * Math.PI) / 180;
    return [SUN[0] * Math.cos(a) - SUN[2] * Math.sin(a), SUN[1], SUN[0] * Math.sin(a) + SUN[2] * Math.cos(a)];
  };
  const kept = frame(cascades, view(500, 0, Math.PI), { sunDirection: turn(0.1) });
  expect(kept.filter((c) => c.redraw).length).toBe(2);
  expect(redraws(frame(cascades, view(500, 0, Math.PI), { sunDirection: turn(3) }))).toEqual([true, true, true]);
  // What the maps hold belongs to one scene.
  frame(cascades, view(500, 0, Math.PI), { scene: 'a' });
  expect(redraws(frame(cascades, view(500, 0, Math.PI), { scene: 'b' }))).toEqual([true, true, true]);
});

test('a peek fits afresh and changes nothing kept', () => {
  const cascades = new StaggeredCascades();
  const first = frame(cascades, view(0, 0));
  const peek = frame(cascades, view(0, -0.1), { peek: true });
  expect(redraws(peek)).toEqual([true, true, true]);
  // The frame after the first, as if the peek had not been.
  const next = frame(cascades, view(0, -0.1));
  expect(redraws(next)).toEqual([true, true, false]);
  expect(next[2].clipFromWorld).toEqual(first[2].clipFromWorld);
});
