// Per-view culling of InstancedMesh and BatchedMesh casters (#149, #169). Runs without a GPU.
import { expect, test } from 'bun:test';
import { BatchedMesh, BoxGeometry, InstancedMesh, Matrix4, MeshBasicMaterial, OrthographicCamera, Scene, ShaderMaterial } from 'three';
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

// Twelve boxes in a row, 10 m apart, in one BatchedMesh; the odd ones scaled to a tenth.
function batch() {
  const mesh = new BatchedMesh(16, 64, 128, new MeshBasicMaterial());
  const box = mesh.addGeometry(new BoxGeometry(1, 1, 1));
  for (let i = 0; i < 12; i++) {
    const s = i % 2 ? 0.1 : 1;
    mesh.setMatrixAt(mesh.addInstance(box), new Matrix4().makeScale(s, s, s).setPosition(i * 10, 0, 0));
  }
  mesh.updateMatrixWorld();
  return mesh;
}

// A view of x in `from..to`.
function slice(from, to) {
  const cam = new OrthographicCamera(from, to, 100, -100, -100, 100);
  cam.updateMatrixWorld();
  cam.updateProjectionMatrix();
  return cam;
}

test('a BatchedMesh straddling a view draws only its instances inside, through its own list', () => {
  const mesh = batch();
  mesh.perObjectFrustumCulled = false;
  mesh.sortObjects = false;
  const culling = new ShadowCulling(new ShaderMaterial({ visible: false }));
  culling.begin([], null, [mesh]);
  const drawn = () => culling.standIns.children.map((s) => Array.from(s._indirectTexture.image.data.subarray(0, s._multiDrawCount)));
  const list = [mesh._multiDrawStarts, mesh._multiDrawCounts, mesh._indirectTexture];

  // Wholly inside: the mesh itself.
  culling.view(0, slice(-50, 200));
  expect(mesh.visible).toBe(true);
  expect(drawn()).toEqual([]);

  // Boxes 3 to 6 are inside x 25..65, each tested once.
  culling.view(0, slice(25, 65));
  expect(mesh.visible).toBe(false);
  expect(drawn()).toEqual([[3, 4, 5, 6]]);
  expect(culling.tests).toBe(12);
  const standIn = culling.standIns.children[0];
  expect(standIn.isBatchedMesh).toBe(true);
  expect(standIn.geometry).toBe(mesh.geometry);
  expect(Array.from(standIn._multiDrawCounts.subarray(0, 4))).toEqual([36, 36, 36, 36]);
  const version = standIn._indirectTexture.version;

  // The same view again: nothing tested, nothing uploaded.
  culling.view(0, slice(25, 65));
  expect(culling.tests).toBe(0);
  expect(drawn()).toEqual([[3, 4, 5, 6]]);
  expect(standIn._indirectTexture.version).toBe(version);

  // Another view has its own list; the small boxes fall below a radius of 0.5.
  culling.view(1, slice(25, 65), 0.5);
  expect(drawn()).toEqual([[4, 6]]);
  expect(culling.standIns.children[0]).not.toBe(standIn);

  // Wholly outside: nothing.
  culling.view(0, slice(500, 600));
  expect(mesh.visible).toBe(false);
  expect(drawn()).toEqual([]);
  culling.end();
  expect(mesh.visible).toBe(true);

  // A hidden, a moved and a removed instance are seen by the next frame.
  mesh.setVisibleAt(4, false);
  mesh.setMatrixAt(9, new Matrix4().setPosition(45, 0, 0));
  mesh.deleteInstance(5);
  culling.begin([], null, [mesh]);
  culling.view(0, slice(25, 65));
  expect(drawn()).toEqual([[3, 6, 9]]);
  expect(culling.standIns.children[0]._indirectTexture.version).toBeGreaterThan(version);
  culling.end();

  // The mesh's own draw list is three's.
  expect([mesh._multiDrawStarts, mesh._multiDrawCounts, mesh._indirectTexture]).toEqual(list);
  expect(mesh.perObjectFrustumCulled).toBe(false);
});
