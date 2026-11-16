"""Run the existing 3d-lab LOD processor and package its indices for the GPU experiment.

python3 scripts/bench-discrete-lods.py LOD_CHAIN_BINARY SOURCE_GLTF OUTPUT_KIT
No simplification algorithm lives here; original surface attributes are unchanged.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

processor, source, output = map(lambda p: Path(p).resolve(), sys.argv[1:])
document = json.loads(source.read_text())
primitives = [p for m in document["meshes"] for p in m["primitives"]]
if len(primitives) != 1:
    raise ValueError("This experiment packages single-primitive assets only")
if any(v.get("extensions", {}).get("EXT_meshopt_compression") for v in document["bufferViews"]):
    raise ValueError("Decompress geometry first")
primitive = primitives[0]
buffers = [(source.parent / b["uri"]).read_bytes() for b in document["buffers"]]


def accessor(index):
    a = document["accessors"][index]
    if a.get("sparse"):
        raise ValueError("Sparse input unsupported by this experiment packager")
    v = document["bufferViews"][a["bufferView"]]
    code = {5126: "f", 5125: "I", 5123: "H", 5121: "B"}[a["componentType"]]
    width = {"VEC3": 3, "SCALAR": 1}[a["type"]]
    stride = v.get("byteStride", struct.calcsize(code) * width)
    start = v.get("byteOffset", 0) + a.get("byteOffset", 0)
    return [list(struct.unpack_from("<" + code * width, buffers[v["buffer"]], start + i * stride)) for i in range(a["count"])]


positions = accessor(primitive["attributes"]["POSITION"])
indices = [i[0] for i in accessor(primitive["indices"])]
name = source.parent.name
scene = output / "scenes" / name
resources = output / "resources"
scene.mkdir(parents=True, exist_ok=True)
resources.mkdir(parents=True, exist_ok=True)
base = copy.deepcopy(document)
for entry in [*base["buffers"], *base.get("images", [])]:
    if "uri" not in entry:
        continue
    payload = (source.parent / entry["uri"]).read_bytes()
    extension = Path(entry["uri"]).suffix
    target = resources / (hashlib.sha256(payload).hexdigest() + extension)
    if not target.exists():
        target.write_bytes(payload)
    entry["uri"] = "../../resources/" + target.name
(scene / "default.gltf").write_text(json.dumps(base))
(output / "manifest.json").write_text(json.dumps({"assets": [{"key": name, "path": f"scenes/{name}/default.gltf"}]}))
(output / "source.json").write_text(json.dumps({"schemaVersion": 1, "vertices": positions, "indices": indices}))
ratios = [0.5, 0.125, 0.03125, 0.0078125]
levels = [{"triangleRatio": r, "targetTriangleCount": max(1, math.floor(len(indices) / 3 * r + .5)), "targetError": .01, "lockBorder": False} for r in ratios]
request = {"schemaVersion": 1, "operation": "mesh.lod_chain", "inputPath": "source.json", "parameters": {"sourceTriangleCount": len(indices) // 3, "sourceBased": True, "budgetRounding": "nearest-ties-away-from-zero", "levels": levels}}
(output / "request.json").write_text(json.dumps(request))
subprocess.run([str(processor), "generate", "request.json", "lods.json", "observations.json"], cwd=output, check=True, timeout=180)
result = json.loads((output / "lods.json").read_text())
for level in result["levels"]:
    g = copy.deepcopy(base)
    values = level["indices"]
    data = struct.pack("<" + "I" * len(values), *values)
    filename = f"lod{level['level']}.bin"
    (scene / filename).write_bytes(data)
    buffer_index, view_index, accessor_index = len(g["buffers"]), len(g["bufferViews"]), len(g["accessors"])
    g["buffers"].append({"byteLength": len(data), "uri": filename})
    g["bufferViews"].append({"buffer": buffer_index, "byteLength": len(data), "target": 34963})
    g["accessors"].append({"bufferView": view_index, "componentType": 5125, "count": len(values), "type": "SCALAR"})
    g["meshes"][0]["primitives"][0]["indices"] = accessor_index
    (scene / f"lod{level['level']}.gltf").write_text(json.dumps(g))
print((output / "observations.json").read_text())
