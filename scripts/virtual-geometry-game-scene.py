"""Export one asset's real Greyhaven dressing placements and a review-pose camera path.

python3 scripts/virtual-geometry-game-scene.py MMORPG_CHECKOUT ASSET OUTPUT_JSON
Benchmark adapter only: no placement, collision or camera intent is authored here.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

root, asset, output = sys.argv[1:]
root = Path(root).resolve()
files = sorted((root / "content/greyhaven/dressing").glob("*.json"))
transforms = []
identities = []
for file in files:
    document = json.loads(file.read_text())
    for placed in document["placements"]:
        if placed["asset"] != asset:
            continue
        # world_dressing.rs: centimetres, scale in thousandths, 65536 yaw units/turn.
        x, y, z = [v / 100 for v in placed["position"]]
        sx, sy, sz = [v / 1000 for v in placed["scale"]]
        yaw = placed["yaw"] / 65536 * math.tau
        c, s = math.cos(yaw), math.sin(yaw)
        transforms.append([c * sx, 0, -s * sx, 0, 0, sy, 0, 0, s * sz, 0, c * sz, 0, x, y, z, 1])
        identities.append(placed["id"])
if not transforms:
    raise ValueError("Asset has no real dressing placements")
views_file = root / "docs/reviews/greyhaven-reference/views.json"
views = json.loads(views_file.read_text())
shots = {s["name"]: s for s in views["shots"]}
a, b = shots["frame-vale"], shots["frame-lake"]
assert all(abs(s["fieldOfViewDegrees"] - 50) < .001 for s in [a, b])
frames = []
for frame in range(240):
    t = (1 - math.cos(frame / 240 * math.tau)) / 2
    frames.append({key: [x + (y - x) * t for x, y in zip(a[key], b[key])] for key in ["eye", "target"]})
result = {"asset": asset, "scope": "Actual consumer placements, rotations and nonuniform scales; interpolated between two consumer review poses. Scenery subset only, without terrain/units/game loop; not a whole-game A/B.",
          "placementIds": identities, "transforms": transforms, "frames": frames,
          "inputs": [{"path": str(p.relative_to(root)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in [*files, views_file]]}
Path(output).parent.mkdir(parents=True, exist_ok=True)
Path(output).write_text(json.dumps(result, indent=2) + "\n")
print(f"{asset}: {len(transforms)} actual placements, {len(frames)} camera frames")
