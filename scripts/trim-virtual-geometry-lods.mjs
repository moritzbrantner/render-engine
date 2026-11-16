// Remove reference-only discrete LOD payload from a hierarchy after exporting its baselines.
// bun scripts/trim-virtual-geometry-lods.mjs DIRECTORY NAME
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export function trimDiscreteLods(directory, name) {
  const jsonPath = path.join(directory, name + '.json');
  const binPath = path.join(directory, name + '.bin');
  const meta = JSON.parse(fs.readFileSync(jsonPath));
  if (meta.version !== 3) throw new Error('Only reference hierarchy version 3 is supported');
  if (meta.vertices === meta.dagVertices && meta.lods.length === 1) return;
  const bin = fs.readFileSync(binPath);
  const lengths = { positions: meta.dagVertices * 12, normals: meta.dagVertices * 4,
    uvs: meta.dagVertices * 8, indices: meta.clusterIndexCount * 4 };
  const chunks = []; let offset = 0;
  for (const [key, section] of Object.entries(meta.sections)) {
    const size = lengths[key] ?? section.bytes;
    if (!Number.isSafeInteger(size) || size < 0 || size > section.bytes ||
        !Number.isSafeInteger(section.offset) || section.offset < 0 ||
        section.offset + section.bytes > bin.length) throw new Error('Invalid hierarchy section range');
    chunks.push(bin.subarray(section.offset, section.offset + size));
    section.offset = offset; section.bytes = size; offset += size;
    const padding = (4 - offset % 4) % 4;
    if (padding) { chunks.push(Buffer.alloc(padding)); offset += padding; }
  }
  meta.vertices = meta.dagVertices; meta.lods = meta.lods.slice(0, 1); meta.binBytes = offset;
  fs.writeFileSync(binPath, Buffer.concat(chunks));
  fs.writeFileSync(jsonPath, JSON.stringify(meta));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [directory, name] = process.argv.slice(2);
  if (!directory || !name) throw new Error('Expected hierarchy directory and name');
  trimDiscreteLods(directory, name);
}
