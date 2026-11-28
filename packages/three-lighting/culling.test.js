// Projected-size culling of InstancedMesh casters per shadow view (#149). Runs without a GPU.
import { expect, test } from 'bun:test';
import { BoxGeometry, InstancedMesh, Matrix4, MeshBasicMaterial, OrthographicCamera, Scene, ShaderMaterial } from 'three';
import { ShadowCulling } from './culling.js';

// A field of 100 unit boxes (radius about 0.87), the odd ones scaled to a tenth.
function field() {
  const mesh = new InstancedMesh(new BoxGeometry(1, 1, 1), new MeshBasicMaterial(), 100);
  for (let i = 0; i < 100; i++) {
    const s = i % 2 ? 0.1 : 1;
    mesh.setMatrixAt(i, new Matrix4().makeScale(s, s, s).setPosition((i % 10) * 3, 0, Math.floor(i / 10) * 3));
  }
  mesh.updateMatrixWorld();
  return mesh;
}

function camera() {
  const cam = new OrthographicCamera(-100, 100, 100, -100, -100, 100);
  cam.updateMatrixWorld();
  cam.updateProjectionMatrix();
  return cam;
}

function standInCount(scene) {
  const group = scene.children.find((c) => c.name === 'three-lighting shadow stand-ins');
  return group.children.map((s) => s.count);
}

test('instances below the radius are left out, all of them drops the mesh, none keeps it whole', () => {
  const scene = new Scene();
  const mesh = field();
  scene.add(mesh);
  const culling = new ShadowCulling(new ShaderMaterial({ visible: false }));
  culling.begin([mesh], scene);

  culling.view(0, camera(), 0);
  expect(mesh.visible).toBe(true);
  expect(standInCount(scene)).toEqual([]);

  // Only the 50 large boxes are drawn, through a stand-in.
  culling.view(0, camera(), 0.5);
  expect(mesh.visible).toBe(false);
  expect(standInCount(scene)).toEqual([50]);

  // Every box is too small.
  culling.view(1, camera(), 2);
  expect(mesh.visible).toBe(false);
  expect(standInCount(scene)).toEqual([]);

  // Below the smallest radius, the mesh draws whole again.
  culling.view(1, camera(), 0.05);
  expect(mesh.visible).toBe(true);
  expect(standInCount(scene)).toEqual([]);

  culling.end();
  expect(mesh.visible).toBe(true);
});

test('a mesh wholly outside the view is dropped, whichever plane it is outside of', () => {
  const scene = new Scene();
  const culling = new ShadowCulling(new ShaderMaterial({ visible: false }));
  const cam = camera();
  // Off every side of the view's ±100 m square, and beyond its far end (along -z, behind it).
  for (const at of [[500, 0, 0], [-500, 0, 0], [0, 500, 0], [0, -500, 0], [0, 0, 500], [0, 0, -500]]) {
    const mesh = field();
    mesh.position.set(...at);
    mesh.updateMatrixWorld();
    scene.add(mesh);
    culling.begin([mesh], scene);
    culling.view(0, cam, 0);
    expect(mesh.visible).toBe(false);
    culling.end();
    scene.remove(mesh);
  }
});
