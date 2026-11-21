"""Measure fresh-process loading, Linux RSS/HWM and NVIDIA per-process GPU memory.

python3 scripts/bench-geometry-loading.py BINARY KIT HIERARCHY ASSET OUTPUT_JSON
Runs are serialized. The child holds resources until the NVIDIA snapshot is taken.
"""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import xml.etree.ElementTree as ET

binary, kit, hierarchy, asset, output = sys.argv[1:]
binary, kit, hierarchy, output = map(lambda p: Path(p).resolve(), [binary, kit, hierarchy, output])
output.parent.mkdir(parents=True, exist_ok=True)
rows = []
for round_index in range(3):
    modes = ["source", "discrete", "cluster"]
    if round_index % 2:
        modes.reverse()
    for mode in modes:
        command = [str(binary), "--virtual-geometry", str(kit), str(hierarchy), asset, str(output.parent), mode]
        with output.with_suffix(f".{round_index}-{mode}.log").open("w") as errors:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                                     text=True, env={**os.environ, "WGPU_ADAPTER_NAME": "RTX 3060 Ti"})
            lines = queue.Queue()

            def read_lines():
                for line in child.stdout:
                    lines.put(line)
                lines.put(None)

            reader = threading.Thread(target=read_lines, daemon=True)
            reader.start()
            try:
                while True:
                    line = lines.get(timeout=120)
                    if line is None:
                        raise RuntimeError(f"Child exited before ready; see {errors.name}")
                    if line.startswith("VG_LOAD "):
                        result = json.loads(line[len("VG_LOAD "):])
                        break
                xml = ET.fromstring(subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True, timeout=20))
                processes = [p for p in xml.findall("gpu/processes/process_info") if p.findtext("pid") == str(child.pid)]
                result["nvidia_process_used_memory"] = [p.findtext("used_memory") for p in processes]
                result["round"] = round_index
                rows.append(result)
                print(json.dumps(result), flush=True)
                child.stdin.write("release\n")
                child.stdin.flush()
                child.stdin.close()
                child.wait(timeout=30)
                if child.returncode:
                    raise RuntimeError(f"Child exit {child.returncode}; see {errors.name}")
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=10)
                reader.join(timeout=5)
        report = {"scope": "Fresh-process loading and post-warm residency on one NVIDIA adapter. Filesystem caches are not flushed. No network/loading claims for browser or cold OS boot. NVIDIA values are process-level sampled MiB, not isolated mesh allocations.", "rows": rows}
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output)
