// The per-object shadow-caster opt-out (Bevy's NotShadowCaster). Runs without a GPU.
import { expect, test } from 'bun:test';
import { Group, Mesh } from 'three';
import { notShadowCaster } from './index.js';

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
