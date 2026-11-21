"""Measure actual loopback HTTP payload bytes/time for the compared asset sets.

python3 scripts/bench-geometry-transfer.py ORIGIN KIT HIERARCHY ASSET OUTPUT_JSON
Serve the repository root first. This is not a WAN or compressed-transfer benchmark.
"""
import json
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request

origin, kit, hierarchy, asset, output = sys.argv[1:]
repo = Path(__file__).resolve().parents[1]
scene = Path(kit).resolve() / "scenes" / asset
hierarchy = Path(hierarchy).resolve()


def referenced(path):
    document = json.loads(path.read_text())
    return {path.resolve(), *((path.parent / v["uri"]).resolve() for v in
                              [*document["buffers"], *document.get("images", [])] if "uri" in v)}


source = referenced(scene / "default.gltf")
discrete = set(source)
for i in range(1, 5):
    p = scene / f"lod{i}.gltf"
    if p.exists():
        discrete.update(referenced(p))
cluster = source | {hierarchy / (asset + ".json"), hierarchy / (asset + ".bin")}
sets = {"source": source, "discrete": discrete, "cluster": cluster}
rows = []
for round_index in range(3):
    order = ["source", "discrete", "cluster"]
    if round_index % 2:
        order.reverse()
    for mode in order:
        started = time.perf_counter()
        files = []
        for file in sorted(sets[mode]):
            relative = file.relative_to(repo).as_posix()
            url = origin.rstrip("/") + "/" + urllib.parse.quote(relative)
            with urllib.request.urlopen(url, timeout=30) as response:
                body = response.read()
                files.append({"path": relative, "payloadBytes": len(body),
                              "contentEncoding": response.headers.get("Content-Encoding", "identity")})
        rows.append({"mode": mode, "round": round_index, "payloadBytes": sum(f["payloadBytes"] for f in files),
                     "loopbackFetchMs": (time.perf_counter() - started) * 1000, "files": files})
result = {"asset": asset, "scope": "Actual HTTP response-body bytes and elapsed serial fetch time on loopback, no HTTP client cache; warm OS cache. Excludes headers, transcode/GPU upload and WAN latency. Cluster set includes source attributes/fallback required by this prototype.", "rows": rows}
Path(output).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps([{k: r[k] for k in ["mode", "round", "payloadBytes", "loopbackFetchMs"]} for r in rows], indent=2))
