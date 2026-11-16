"""Report file bytes and logical native geometry buffer bytes, not driver residency.

python3 scripts/virtual-geometry-footprint.py KIT HIERARCHY ASSET OUTPUT_JSON
"""
import json
from pathlib import Path
import struct
import sys

kit, hierarchy, name, output = sys.argv[1:]
kit, hierarchy = Path(kit), Path(hierarchy)
scene = kit / "scenes" / name
meta_file = hierarchy / (name + ".json")
meta = json.loads(meta_file.read_text())
binary = (hierarchy / (name + ".bin")).read_bytes()


def native_geometry_bytes(document):
    # World uploads position/normal/surface streams: 12 + 12 + 40 bytes/vertex,
    # and 32-bit indices, even when source glTF indices use uint16.
    return sum(document["accessors"][pr["attributes"]["POSITION"]]["count"] * 64
               + document["accessors"][pr["indices"]]["count"] * 4
               for mesh in document["meshes"] for pr in mesh["primitives"])


def files_for(path):
    g = json.loads(path.read_text())
    files = {path.resolve()}
    for item in [*g["buffers"], *g.get("images", [])]:
        if "uri" in item:
            files.add((path.parent / item["uri"]).resolve())
    return g, files


source, source_files = files_for(scene / "default.gltf")
lod_files = set(source_files)
lods = []
for i in range(1, 5):
    p = scene / f"lod{i}.gltf"
    if p.exists():
        g, files = files_for(p)
        lod_files.update(files)
        lods.append({"level": i, "logicalNativeGeometryBufferBytes": native_geometry_bytes(g)})
section = meta["sections"]["clusterU"]
clusters = list(struct.iter_unpack("<4i", binary[section["offset"]:section["offset"] + section["bytes"]]))
section = meta["sections"]["indices"]
indices = struct.unpack("<" + "I" * (section["bytes"] // 4), binary[section["offset"]:section["offset"] + section["bytes"]])
cluster_buffers = sum(len(set(indices[start:start + count])) * 64 + count * 4 for start, count, _, _ in clusters)
result = {
    "asset": name,
    "metric": "Uncompressed referenced file lengths and logical vertex/index buffer sizes; not RSS, VRAM allocation, compressed HTTP transfer, or measured loading latency",
    "sourceReferencedFileBytes": sum(p.stat().st_size for p in source_files),
    "sourceAndDiscreteLodsReferencedFileBytes": sum(p.stat().st_size for p in lod_files),
    "clusterPrototypeReferencedFileBytes": sum(p.stat().st_size for p in source_files) + meta_file.stat().st_size + len(binary),
    "clusterPrototypeNote": "The experiment also needs source attributes because the reference hierarchy omits tangents/colors and quantizes normals. All hierarchy levels remain resident. Textures are shared.",
    "logicalNativeSourceGeometryBufferBytes": native_geometry_bytes(source),
    "logicalNativeClusterGeometryBufferBytes": cluster_buffers,
    "discreteLods": lods,
    "hierarchyMetadataFileBytes": sum(s["bytes"] for k, s in meta["sections"].items() if k not in ["positions", "normals", "uvs", "indices"]),
}
Path(output).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
