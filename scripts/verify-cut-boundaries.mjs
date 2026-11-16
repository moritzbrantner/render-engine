// Independent geometric-boundary diagnostic, supplementing the reference verifier.
// bun scripts/verify-cut-boundaries.mjs HIERARCHY_DIRECTORY ASSET OUTPUT_JSON
import fs from 'node:fs';
import path from 'node:path';
const [directory, name, output] = process.argv.slice(2);
if (!directory || !name || !output) throw new Error('Expected hierarchy directory, asset, output JSON');
const meta = JSON.parse(fs.readFileSync(path.join(directory, name + '.json')));
const bytes = fs.readFileSync(path.join(directory, name + '.bin'));
const types = { Float32Array, Int32Array, Uint32Array };
const section = key => {
  const s = meta.sections[key];
  return new types[s.type](bytes.buffer.slice(bytes.byteOffset + s.offset, bytes.byteOffset + s.offset + s.bytes));
};
const positions = section('positions'), indices = section('indices'), clusters = section('clusterU');
const groups = section('groupF'), offsets = section('memberOff'), members = section('memberIds');
const consumer = new Int32Array(meta.clusters);
for (let g = 0; g < meta.groups; g++) for (let j = offsets[g]; j < offsets[g + 1]; j++) consumer[members[j]] = g;
const tolerance = Math.max(...meta.bbMax.map((v, i) => v - meta.bbMin[i])) * 1e-6;
const vertices = new Map(), canonical = [];
for (let i = 0; i < meta.dagVertices; i++) {
  const key = Array.from(positions.subarray(i * 3, i * 3 + 3), v => Math.round(v / tolerance)).join(',');
  if (!vertices.has(key)) vertices.set(key, vertices.size);
  canonical.push(vertices.get(key));
}
function boundary(cut) {
  const edges = new Map(); let triangles = 0;
  for (const c of cut) for (let j = clusters[c * 4]; j < clusters[c * 4] + clusters[c * 4 + 1]; j += 3) {
    const [a, b, d] = Array.from(indices.subarray(j, j + 3), v => canonical[v]);
    if (a === b || a === d || b === d) continue;
    triangles++;
    for (const [x, y] of [[a, b], [b, d], [d, a]]) {
      const key = x < y ? `${x},${y}` : `${y},${x}`;
      edges.set(key, (edges.get(key) || 0) + 1);
    }
  }
  return { triangles, open: new Set([...edges].filter(([, n]) => n === 1).map(([key]) => key)), nonManifoldEdges: [...edges.values()].filter(n => n > 2).length };
}
const leaves = Array.from({ length: meta.clusters }, (_, c) => c).filter(c => clusters[c * 4 + 2] < 0);
const source = boundary(leaves);
const center = meta.bbMin.map((v, i) => (v + meta.bbMax[i]) / 2);
const diameter = Math.hypot(...meta.bbMax.map((v, i) => v - meta.bbMin[i]));
const cameras = [[.6,.2,.3],[.9,.5,.9],[2,1,2],[6,2,4],[25,8,10],[.2,.05,.31],[-.5,.25,0]];
const rows = [];
for (const offset of cameras) for (const threshold of [.5, 1, 2, 4, 16]) {
  const camera = offset.map((v, i) => center[i] + v * diameter);
  const errors = Array.from({ length: meta.groups }, (_, g) => {
    const o = g * 5, distance = Math.hypot(...camera.map((v, i) => v - groups[o + i])) - groups[o + 3];
    return distance <= 1e-6 ? Infinity : groups[o + 4] * 1080 / (2 * Math.tan(Math.PI / 6) * distance);
  });
  const cut = Array.from({ length: meta.clusters }, (_, c) => c).filter(c => (clusters[c * 4 + 2] < 0 || errors[clusters[c * 4 + 2]] <= threshold) && (consumer[c] === 0 || errors[consumer[c]] > threshold));
  const b = boundary(cut);
  rows.push({ offset, threshold, triangles: b.triangles, openEdges: b.open.size,
    newBoundaryEdges: [...b.open].filter(key => !source.open.has(key)).length, nonManifoldEdges: b.nonManifoldEdges });
}
const closedSourceCutsStayClosed = source.open.size ? null : rows.every(r => r.openEdges === 0);
const result = { asset: name, tolerance, sourceOpenEdges: source.open.size, sourceNonManifoldEdges: source.nonManifoldEdges,
  closedSourceCutsStayClosed, limitation: 'Quantized-position incidence diagnostic; open-source boundary simplification can change exact edges. Does not prove coverage, orientation, shading or runtime traversal.', rows };
fs.writeFileSync(output, JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify({ asset: name, sourceOpenEdges: source.open.size, closedSourceCutsStayClosed, worstOpenEdges: Math.max(...rows.map(r => r.openEdges)) }));
if (closedSourceCutsStayClosed === false) process.exitCode = 1;
