// The per-object shadow-caster opt-out (Bevy's NotShadowCaster). Runs without a GPU.
import { expect, test } from 'bun:test';
import { BoxGeometry, Group, InstancedMesh, Mesh } from 'three';
import { casterRadius, notShadowCaster, shadowCascades, shadowGeometry } from './index.js';

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

test('shadowGeometry picks the proxy per cascade, the last entry for later ones, else the own geometry', () => {
  const mesh = new Mesh(new BoxGeometry(1, 1, 1));
  expect(shadowGeometry(mesh, 2)).toBe(mesh.geometry);
  const near = new BoxGeometry(1, 1, 1);
  const far = new BoxGeometry(1, 1, 1);
  mesh.userData.shadowGeometry = near;
  expect(shadowGeometry(mesh, 0)).toBe(near);
  expect(shadowGeometry(mesh, 3)).toBe(near);
  mesh.userData.shadowGeometry = [near, far];
  expect(shadowGeometry(mesh, 0)).toBe(near);
  expect(shadowGeometry(mesh, 1)).toBe(far);
  expect(shadowGeometry(mesh, 2)).toBe(far);
  mesh.userData.shadowGeometry = [];
  expect(shadowGeometry(mesh, 1)).toBe(mesh.geometry);
});

test('casterRadius scales the bounding sphere by the largest world axis; instanced and batched meshes are never culled whole', () => {
  const mesh = new Mesh(new BoxGeometry(2, 2, 2));
  mesh.scale.set(1, 3, 1);
  mesh.updateMatrixWorld();
  expect(casterRadius(mesh)).toBeCloseTo(Math.sqrt(3) * 3, 6);
  expect(casterRadius(new InstancedMesh(new BoxGeometry(), undefined, 1))).toBe(Infinity);
  expect(casterRadius(new Group())).toBe(Infinity);
});
