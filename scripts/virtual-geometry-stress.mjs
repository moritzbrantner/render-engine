// Explicit synthetic scaling control from the reference's procedural boulder.
// bun scripts/virtual-geometry-stress.mjs REFERENCE_CHECKOUT OUTPUT_ROOT
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { trimDiscreteLods } from './trim-virtual-geometry-lods.mjs';
const [reference, output] = process.argv.slice(2).map(p => path.resolve(p));
if (!reference || !output) throw new Error('Expected reference and output');
const { makeSample } = await import(pathToFileURL(path.join(reference, 'examples/make-sample.mjs')));
const { buildClod, verifyClod } = await import(pathToFileURL(path.join(reference, 'src/build.js')));
const inputRoot = path.join(output, 'inputs/stress'), sceneDir = path.join(inputRoot, 'scenes/boulder');
const hierarchy = path.join(output, 'stress/boulder');
fs.mkdirSync(sceneDir, { recursive: true }); fs.mkdirSync(hierarchy, { recursive: true });
const source = makeSample('rock');
const length = source.readUInt32LE(12), json = JSON.parse(source.subarray(20, 20 + length));
const binOffset = 28 + length, binary = source.subarray(binOffset, binOffset + source.readUInt32LE(20 + length));
json.buffers[0].uri = 'source.bin';
fs.writeFileSync(path.join(sceneDir, 'source.bin'), binary);
fs.writeFileSync(path.join(sceneDir, 'default.gltf'), JSON.stringify(json));
fs.writeFileSync(path.join(hierarchy, 'input.glb'), source);
fs.writeFileSync(path.join(inputRoot, 'manifest.json'), JSON.stringify({ assets: [{ key: 'boulder', path: 'scenes/boulder/default.gltf' }] }));
const { meta } = await buildClod({ input: path.join(hierarchy, 'input.glb'), outDir: hierarchy, name: 'boulder', lods: true });
const bytes = fs.readFileSync(path.join(hierarchy, 'boulder.bin'));
const types = { Float32Array, Int8Array, Uint32Array, Int32Array };
const section = name => { const s = meta.sections[name]; return new types[s.type](bytes.buffer.slice(bytes.byteOffset + s.offset, bytes.byteOffset + s.offset + s.bytes)); };
const p = section('positions'), n = section('normals'), uv = section('uvs'), idx = section('indices');
for (const [i, lod] of meta.lods.slice(1, 3).entries()) {
  const remap = new Map(), pos = [], normal = [], tex = [], indices = [];
  for (const index of idx.subarray(lod.start, lod.start + lod.count)) {
    if (!remap.has(index)) {
      remap.set(index, remap.size); pos.push(...p.subarray(index * 3, index * 3 + 3));
      normal.push(...Array.from(n.subarray(index * 4, index * 4 + 3), x => x / 127));
      tex.push(...uv.subarray(index * 2, index * 2 + 2));
    }
    indices.push(remap.get(index));
  }
  const arrays = [new Float32Array(pos), new Float32Array(normal), new Float32Array(tex), new Uint32Array(indices)];
  let offset = 0;
  const views = arrays.map(a => { const v = { buffer: 0, byteOffset: offset, byteLength: a.byteLength }; offset += a.byteLength; return v; });
  const min = [0, 1, 2].map(a => pos.filter((_, j) => j % 3 === a).reduce((v, x) => Math.min(v, x), Infinity));
  const max = [0, 1, 2].map(a => pos.filter((_, j) => j % 3 === a).reduce((v, x) => Math.max(v, x), -Infinity));
  const filename = `lod${i + 1}.bin`;
  const g = { asset: { version: '2.0' }, buffers: [{ uri: filename, byteLength: offset }], bufferViews: views,
    accessors: arrays.map((a, j) => ({ bufferView: j, componentType: j === 3 ? 5125 : 5126, count: a.length / (j < 2 ? 3 : j === 2 ? 2 : 1), type: j < 2 ? 'VEC3' : j === 2 ? 'VEC2' : 'SCALAR', ...(j === 0 ? { min, max } : {}) })),
    meshes: [{ primitives: [{ attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2 }, indices: 3 }] }], nodes: [{ mesh: 0 }], scenes: [{ nodes: [0] }], scene: 0 };
  fs.writeFileSync(path.join(sceneDir, filename), Buffer.concat(arrays.map(a => Buffer.from(a.buffer))));
  fs.writeFileSync(path.join(sceneDir, `lod${i + 1}.gltf`), JSON.stringify(g));
}
trimDiscreteLods(hierarchy, 'boulder');
const verification = await verifyClod(hierarchy, 'boulder');
fs.writeFileSync(path.join(hierarchy, 'verification.json'), JSON.stringify(verification, null, 2));
if (!verification.ok) process.exitCode = 1;
