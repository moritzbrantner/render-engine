// Offline geometry experiment only; no runtime dependency on the reference.
// bun scripts/bench-virtual-geometry.mjs REFERENCE_CHECKOUT KIT_DIRECTORY OUTPUT_DIRECTORY
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import os from 'node:os';

const [reference, kit, output] = process.argv.slice(2).map(p => path.resolve(p));
if (!reference || !kit || !output) throw new Error('Expected reference checkout, kit directory, output directory');
const { buildClod, verifyClod, cutStats } = await import(pathToFileURL(path.join(reference, 'src/build.js')));
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const revision = directory => execFileSync('git', ['-C', directory, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
const quiet = () => {};
fs.mkdirSync(output, { recursive: true });

function walk(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name)).flatMap(e => {
    const p = path.join(directory, e.name);
    return e.isDirectory() ? walk(p) : [p];
  });
}

// Preserve vertex attributes, indices and scene transforms. Remove materials explicitly:
// upstream cannot import the kits' KHR_texture_basisu textures. These results must never
// be described as material/lighting parity, download size, or a gameplay benchmark.
function geometryGlb(source) {
  const bytes = fs.readFileSync(source);
  const g = JSON.parse(bytes);
  const hash = createHash('sha256').update(bytes);
  const buffers = [], offsets = [];
  let length = 0;
  for (const b of g.buffers) {
    if (!b.uri || b.uri.startsWith('data:')) throw new Error('Expected external kit buffer');
    const data = fs.readFileSync(path.resolve(path.dirname(source), b.uri));
    hash.update(data);
    offsets.push(length);
    const padded = Buffer.alloc(Math.ceil(data.length / 4) * 4);
    data.copy(padded); buffers.push(padded); length += padded.length;
  }
  for (const v of g.bufferViews) {
    v.byteOffset = (v.byteOffset ?? 0) + offsets[v.buffer]; v.buffer = 0;
  }
  for (const m of g.meshes) for (const p of m.primitives) {
    if (p.extensions?.KHR_draco_mesh_compression) throw new Error('Compressed geometry unsupported');
    delete p.material;
  }
  if (g.bufferViews.some(v => v.extensions?.EXT_meshopt_compression)) throw new Error('Compressed geometry unsupported');
  for (const k of ['materials', 'textures', 'images', 'samplers', 'extensionsRequired', 'extensionsUsed']) delete g[k];
  g.buffers = [{ byteLength: length }];
  const json = Buffer.from(JSON.stringify(g));
  const jsonPad = Buffer.alloc(Math.ceil(json.length / 4) * 4, 32); json.copy(jsonPad);
  const header = Buffer.alloc(20);
  header.writeUInt32LE(0x46546c67); header.writeUInt32LE(2, 4);
  header.writeUInt32LE(28 + jsonPad.length + length, 8);
  header.writeUInt32LE(jsonPad.length, 12); header.writeUInt32LE(0x4e4f534a, 16);
  const binHeader = Buffer.alloc(8); binHeader.writeUInt32LE(length); binHeader.writeUInt32LE(0x004e4942, 4);
  return { bytes: Buffer.concat([header, jsonPad, binHeader, ...buffers]), inputSha256: hash.digest('hex'), geometryBufferBytes: length };
}

const rows = [];
const sources = walk(path.join(kit, 'scenes')).filter(p => path.basename(p) === 'default.gltf');
if (!sources.length) throw new Error('Kit has no default.gltf scenes');
for (const source of sources) {
  const name = path.basename(path.dirname(source));
  const dir = path.join(output, name); fs.mkdirSync(dir, { recursive: true });
  const packed = geometryGlb(source), input = path.join(dir, 'input.glb');
  fs.writeFileSync(input, packed.bytes);
  // One untimed warm build, three measured builds; setup and verification excluded.
  await buildClod({ input, outDir: dir, name, lods: false, log: quiet });
  const buildMs = []; let meta;
  for (let round = 0; round < 3; round++) {
    const start = performance.now();
    ({ meta } = await buildClod({ input, outDir: dir, name, lods: false, log: quiet }));
    buildMs.push(performance.now() - start);
  }
  const verification = await verifyClod(dir, name, quiet);
  const radius = Math.hypot(...meta.bbMax.map((v, i) => v - meta.bbMin[i])) / 2;
  const cuts = cutStats(dir, name, radius, quiet);
  rows.push({ asset: path.relative(kit, source), inputSha256: packed.inputSha256,
    sourceTriangles: meta.sourceTris, geometryBufferBytes: packed.geometryBufferBytes,
    hierarchyBytes: meta.binBytes, hierarchySha256: sha(fs.readFileSync(path.join(dir, `${name}.bin`))),
    clusters: meta.clusters, groups: meta.groups, levels: meta.levels,
    buildMs, verification, cuts,
    existingLods: ['lod1', 'lod2'].flatMap(lod => {
      const p = path.join(path.dirname(source), `${lod}.gltf`);
      if (!fs.existsSync(p)) return [];
      const g = JSON.parse(fs.readFileSync(p));
      return [{ lod, triangles: g.meshes.flatMap(m => m.primitives).reduce((n, pr) => n + g.accessors[pr.indices ?? pr.attributes.POSITION].count / 3, 0),
        geometryBufferBytes: g.buffers.reduce((n, b) => n + b.byteLength, 0) }];
    }) });
  console.log(`${name}: ${meta.sourceTris} triangles, ${meta.clusters} clusters, verify=${verification.ok}`);
}
const report = { scope: 'Offline geometry only; no GPU, frame-time, loading or visual-parity claims',
  referenceRevision: revision(reference), engineRevision: revision(process.cwd()),
  environment: { runtime: process.version, platform: process.platform, arch: process.arch, cpu: os.cpus()[0]?.model },
  sampling: 'One warm build, three timed builds per asset, no texture processing; raw milliseconds',
  cutCamera: 'Reference cutStats: 1080p, 60 degree vertical FOV; original asset size, distances in source units; verifyClod uses seven sphere-relative cameras and five thresholds',
  rows };
fs.writeFileSync(path.join(output, 'report.json'), JSON.stringify(report, null, 2) + '\n');
