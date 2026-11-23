"""Measure fresh-process loading and post-warm process and GPU memory.

python3 scripts/bench-geometry-loading.py BINARY KIT HIERARCHY ASSET OUTPUT_JSON
Runs are serialized. The child holds resources until the memory snapshot is taken.

A Linux BINARY reports Linux RSS/HWM itself; NVIDIA's per-process GPU memory comes from
nvidia-smi. A Windows BINARY (`.exe`, run from WSL through interop) is sampled from Windows:
working set and its peak, and the GPU process memory counters (dedicated and shared; an
integrated GPU allocates from shared memory). VG_ADAPTER picks the adapter (default
`RTX 3060 Ti`); WGPU_BACKEND is passed through.
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
windows = binary.suffix == ".exe"
adapter = os.environ.get("VG_ADAPTER", "RTX 3060 Ti")
environment = {**os.environ, "WGPU_ADAPTER_NAME": adapter}
if windows:
    environment["WSLENV"] = "WGPU_ADAPTER_NAME:WGPU_BACKEND"


def argument(path):
    return subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip() if windows else str(path)


def nvidia_memory(pid):
    xml = ET.fromstring(subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True, timeout=20))
    processes = [p for p in xml.findall("gpu/processes/process_info") if p.findtext("pid") == str(pid)]
    return {"nvidia_process_used_memory": [p.findtext("used_memory") for p in processes]}


def windows_memory(pid):
    # WMI class names are not localized, unlike Get-Counter paths.
    script = (
        f"$p = Get-Process -Id {pid}; "
        "$g = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory "
        f"| Where-Object {{ $_.Name -like 'pid_{pid}_*' }} "
        "| Select-Object Name, DedicatedUsage, SharedUsage, TotalCommitted); "
        "@{ working_set = $p.WorkingSet64; peak_working_set = $p.PeakWorkingSet64; "
        "private_bytes = $p.PrivateMemorySize64; gpu = $g } | ConvertTo-Json -Depth 4 -Compress"
    )
    sample = json.loads(subprocess.check_output(["powershell.exe", "-NoProfile", "-Command", script], text=True, timeout=180))
    gpu = sample["gpu"] if isinstance(sample["gpu"], list) else [sample["gpu"]]
    return {
        "windows_process_bytes": {k: sample[k] for k in ("working_set", "peak_working_set", "private_bytes")},
        "windows_gpu_process_memory": gpu,
    }


rows = []
for round_index in range(3):
    modes = ["source", "discrete", "cluster"]
    if round_index % 2:
        modes.reverse()
    for mode in modes:
        command = [str(binary), "--virtual-geometry", argument(kit), argument(hierarchy), asset, argument(output.parent), mode]
        with output.with_suffix(f".{round_index}-{mode}.log").open("w") as errors:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                                     text=True, env=environment)
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
                result.update(windows_memory(result["pid"]) if windows else nvidia_memory(child.pid))
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
        scope = (
            "Fresh-process loading and post-warm residency on one Windows adapter. Working set and GPU process "
            "memory are sampled from Windows while the child holds its resources; they include driver state and are "
            "not isolated mesh allocations."
            if windows else
            "Fresh-process loading and post-warm residency on one NVIDIA adapter. NVIDIA values are process-level "
            "sampled MiB, not isolated mesh allocations."
        )
        report = {"scope": scope + " Filesystem caches are not flushed. No network/loading claims for browser or cold OS boot.",
                  "adapter": adapter, "rows": rows}
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output)
