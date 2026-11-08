// The per-object shadow-caster opt-out (Bevy's NotShadowCaster). Runs without a GPU.
import { expect, test } from 'bun:test';
import { Group, Mesh } from 'three';
import { notShadowCaster, shadowCascades } from './index.js';

test('only an object marked notShadowCaster opts out, and its children still cast', () => {
  const parent = new Group();
  const child = new Mesh();
  parent.add(child);
  expect(notShadowCaster(parent)).toBe(false);
  parent.userData.notShadowCaster = true;
  expect(notShadowCaster(parent)).toBe(true);
  expect(notShadowCaster(child)).toBe(false);
  // three's castShadow defaults to false and is not the switch.
  expect(child.castShadow).toBe(false);
});

test('shadowCascades limits an object to the nearest sun cascades, unset or invalid means all', () => {
  const mesh = new Mesh();
  expect(shadowCascades(mesh)).toBe(Infinity);
  mesh.userData.shadowCascades = 2;
  expect(shadowCascades(mesh)).toBe(2);
  mesh.userData.shadowCascades = 0;
  expect(shadowCascades(mesh)).toBe(0);
  for (const invalid of [-1, 1.5, '2', null]) {
    mesh.userData.shadowCascades = invalid;
    expect(shadowCascades(mesh)).toBe(Infinity);
  }
});
