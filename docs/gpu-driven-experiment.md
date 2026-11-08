# GPU-driven culling and indirect draws (#71)

A *bounded experiment*, not a production renderer migration. It compares the reference path
(CPU frustum culling per view, instanced draws from the per-frame instance list) with the
smallest opt-in GPU path the renderer can have, on identical geometry, materials, lights and
shadows, on the target hardware.

| Path | What runs where |
| --- | --- |
| Reference (`Culling::Cpu`, default) | `prepare` tests every instance against every view and writes the visible-index buffer; encode issues one `draw_indexed` per non-empty mesh/material/face run and view. |
| GPU (`Culling::GpuIndirect`, opt-in) | A compute pass (`src/shaders/gpu_cull.wgsl`) tests each opaque or alpha-masked unskinned instance against each view with the same box and sphere tests as `src/cull.rs`, compacts the survivors into per-view runs of the visible-index buffer and fills one `DrawIndexedIndirect` per run and view. Blended and skinned instances stay on the CPU path in the same frame. |

The shaders, instance data and sort order are shared; the prototype changes no shading code and
keeps the per-frame instance list. It is the smallest change that replaces the per-frame culling
and visible-index upload with GPU work. It does not make `prepare` incremental, retain instances
on the GPU, batch meshes into a shared buffer or use bindless materials or occlusion (Hi-Z was
already measured in [#59](occlusion-experiment.md)). `Viewport::set_culling` selects the path per
viewport; default frames are unchanged (see "Correctness").

## Reproducing

```sh
bash scripts/bench-gpu-driven.sh --gpu-rounds 3                                                 # full run
WGPU_VALIDATION_INDIRECT_CALL=0 bash scripts/bench-gpu-driven.sh --gpu-rounds 3 --only instances_22k --only groups_22k --only dense_90k --out target/oracle/noval
bash scripts/bench-gpu-driven.sh --only instances_22k --frames 120                              # one scene, quickly
cargo test --locked --test gpu_culling                                                          # the small comparison CI runs
```

Output: `target/oracle/gpu_driven/report.json` (every median, pass span, work counter and per-round
spread), `fingerprint.txt`, and PNG/FLIP heat maps for any capture that is not byte-identical
(there were none). Each run warms up 500 ms like `Headless::gpu_frame_times`, times 300 frames
with one submission per frame and alternates path order across rounds.

| Measured | Where |
| --- | --- |
| CPU `prepare` | `Renderer::prepare`, includes per-frame culling, the instance sort and `queue.write_buffer` of every per-frame table. |
| CPU encode | `Renderer::encode` plus `CommandEncoder::finish` (wgpu records commands at finish). |
| CPU submit | `queue.submit`. |
| GPU cost | Sum of the frame's timestamped pass spans, D3's protocol; the median of each round, the fastest round reported. |
| GPU compute | The `gpu culling` pass's own span. |
| GPU interval | Last pass's end minus first pass's begin, which also holds wgpu's indirect-draw validation dispatches between passes (not inside any span). |

## Results

Measured 2026-11-08 on NVIDIA GeForce RTX 3060 Ti, driver 595.99.02, Vulkan, wgpu 30.0.1, Linux
7.0.0-42-generic, 1280×720, 4× MSAA, rustc 1.99.1, release build, 300 timed frames × 3 rounds per
case on a working tree over commit `228fb9f`. Software adapter timings are not reported.

Scenes (all derive from the oracle's `many_instances`: posts, crates and stones under a sun with
four cascades, so `V = 5` views): `small_1k` 1,020 instances, `instances_22k` 22,401, `dense_90k`
89,601, `dense_202k` 201,601, and `groups_22k`, 22,401 instances over 3 meshes × 256 materials, so
~768 runs in place of about 13 (the draw-call-bound case). Views: `wedge` (the scenario camera:
most of the field is outside the view), `high` (elevated over the field, high visibility), `low`
(at the edge looking away, low visibility), `moving` (an orbit with a new camera every frame) and
`cuts` (a hard camera cut between `high` and `low` every 30 frames).

CPU total: `prepare` + encode + submit. `upload` is bytes per frame; `draws` is summed over views.
All values in ms unless noted.

| Scene | View | CPU prepare (CPU path → GPU path) | CPU encode | CPU total | GPU pass sum | GPU compute | upload (MB) | draws |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| `small_1k` | wedge | 0.099 → 0.078 | 0.070 → 0.132 | 0.190 → 0.239 | 0.198 → 0.198 | 0.005 | 0.138 → 0.134 | 32 → 35 |
| | high | 0.097 → 0.077 | 0.066 → 0.109 | 0.184 → 0.214 | 0.181 → 0.185 | 0.005 | 0.137 → 0.134 | 19 → 35 |
| | low | 0.092 → 0.077 | 0.057 → 0.118 | 0.168 → 0.223 | 0.081 → 0.085 | 0.005 | 0.133 → 0.134 | 13 → 35 |
| | moving | 0.109 → 0.081 | 0.078 → 0.132 | 0.211 → 0.243 | 0.193 → 0.195 | 0.005 | 0.137 → 0.134 | 27 → 35 |
| | cuts | 0.103 → 0.078 | 0.086 → 0.117 | 0.214 → 0.224 | 0.178 → 0.182 | 0.005 | 0.137 → 0.134 | 19 → 35 |
| `instances_22k` | wedge | 2.375 → 1.759 | 0.111 → 0.146 | 2.515 → 1.941 | 0.727 → 0.735 | 0.016 | 2.98 → 2.87 | 65 → 65 |
| | high | 2.314 → 1.756 | 0.092 → 0.145 | 2.432 → 1.937 | 0.701 → 0.714 | 0.016 | 2.97 → 2.87 | 41 → 65 |
| | low | 2.519 → 2.076 | 0.087 → 0.166 | 2.632 → 2.284 | 0.431 → 0.481 | 0.073 | 2.87 → 2.87 | 31 → 65 |
| | moving | 2.385 → 1.746 | 0.100 → 0.144 | 2.512 → 1.925 | 0.669 → 0.693 | 0.016 | 2.97 → 2.87 | 53 → 65 |
| | cuts | 2.370 → 1.786 | 0.115 → 0.164 | 2.521 → 1.990 | 0.953 → 0.957 | 0.021 | 2.87 → 2.87 | 31 → 65 |
| `dense_90k` | wedge | 11.268 → 8.939 | 0.242 → 0.296 | 11.590 → 9.327 | 3.848 → 3.221 | 0.110 | 11.92 → 11.47 | 65 → 65 |
| | high | 12.450 → 9.490 | 0.257 → 0.322 | 12.795 → 9.912 | 3.902 → 3.345 | 0.110 | 11.88 → 11.47 | 41 → 65 |
| | low | 14.239 → 12.149 | 0.235 → 0.315 | 14.559 → 12.564 | 1.155 → 3.207 | 2.265 | 11.48 → 11.47 | 34 → 65 |
| | moving | 11.013 → 9.828 | 0.232 → 0.323 | 11.324 → 10.254 | 3.291 → 3.358 | 0.150 | 11.87 → 11.47 | 41 → 65 |
| | cuts | 15.965 → 9.801 | 0.320 → 0.329 | 16.392 → 10.234 | 2.542 → 3.203 | 0.110 | 11.48 → 11.47 | 34 → 65 |
| `dense_202k` | wedge | 27.301 → 21.805 | 0.256 → 0.326 | 27.640 → 22.236 | 8.224 → 7.880 | 0.326 | 26.82 → 25.81 | 65 → 65 |
| | high | 27.702 → 21.747 | 0.251 → 0.328 | 28.040 → 22.181 | 7.856 → 8.032 | 0.325 | 26.73 → 25.81 | 41 → 65 |
| | low | 32.666 → 27.513 | 0.235 → 0.325 | 32.986 → 27.940 | 1.394 → 6.053 | 4.944 | 25.82 → 25.81 | 34 → 65 |
| | moving | 27.206 → 21.010 | 0.247 → 0.327 | 27.536 → 21.438 | 8.562 → 6.810 | 0.237 | 26.71 → 25.81 | 41 → 65 |
| | cuts | 29.598 → 22.747 | 0.255 → 0.332 | 29.944 → 23.184 | 5.279 → 6.511 | 0.320 | 25.82 → 25.81 | 34 → 65 |
| `groups_22k` | wedge | 2.681 → 2.321 | 1.241 → 2.228 | 3.965 → 4.712 | 0.702 → 1.320 | 0.031 | 2.98 → 2.97 | 2,679 → 3,845 |
| | high | 2.595 → 2.087 | 0.748 → 2.057 | 3.382 → 4.265 | 0.682 → 1.102 | 0.025 | 2.97 → 2.97 | 1,579 → 3,845 |
| | low | 2.840 → 2.505 | 0.320 → 2.618 | 3.196 → 5.268 | 0.535 → 0.889 | 0.097 | 2.87 → 2.97 | 466 → 3,845 |
| | moving | 2.781 → 2.103 | 1.139 → 2.080 | 3.971 → 4.307 | 0.678 → 1.221 | 0.030 | 2.98 → 2.97 | 2,382 → 3,845 |
| | cuts | 2.690 → 2.486 | 0.747 → 2.517 | 3.478 → 5.162 | 0.985 → 1.125 | 0.027 | 2.97 → 2.97 | 1,579 → 3,845 |

Instance work (the `wedge` view; `gpu` columns are the path's counters, the visible count from
`read_gpu_culled`, identical on both paths):

| Scene | `N` | CPU tests (CPU → GPU path) | GPU tests | Visible (camera) | Kept over 4 cascades |
| --- | ---: | --- | ---: | ---: | ---: |
| `small_1k` | 1,020 | 5,100 → 0 | 5,100 | 606 | 683 |
| `instances_22k` | 22,401 | 112,005 → 0 | 112,005 | 13,280 | 14,976 |
| `dense_90k` | 89,601 | 448,005 → 0 | 448,005 | 52,858 | 59,993 |
| `dense_202k` | 201,601 | 1,008,005 → 0 | 1,008,005 | 118,577 | 134,506 |

### Overhead attribution

The GPU path adds work in four places, none of them in the shading or the instance data:

- **Compute** (the `gpu culling` pass): 0.016 ms at 22.4k instances and five views, 0.110 ms at
  89.6k, 0.326 ms at 201.6k, which is 0.14–0.32 ns per instance-view test across parallel
  invocations (`@workgroup_size(64)`, one atomic per run and view).
- **Indirect draws on the GPU** (between passes, not a pass span): the pass sum rises by
  0.004–0.024 ms at 22.4k instances (0.050 ms at `low`). The interval, which holds wgpu's
  indirect-call validation, rises by 0.05–0.07 ms at 22.4k (0.14 ms at `low`) in the default
  configuration and by 0.015–0.07 ms with `WGPU_VALIDATION_INDIRECT_CALL=0`. wgpu's indirect-call
  validation is on by default, including in release builds, so a consumer pays it unless it opts
  out. The `low`-visibility compute spans (2.3 ms at `dense_90k`, 4.9 ms at `dense_202k`, for the
  same number of tests as the other views at 0.1–0.3 ms) are not stable: an earlier 2-round run
  of the same scenes measured 0.107 ms and 0.229 ms for them (see "Limits").
- **Encode, CPU**: one `draw_indexed_indirect` per run and view, empty or not; the CPU path draws
  only the runs a view kept. With few runs this is small (+0.035–0.08 ms at 22.4k); with many
  (`groups_22k`, 3,845 draws against 466–2,679) it dominates: 0.32–1.24 ms becomes 2.06–2.62 ms,
  and the CPU total goes up. Indirect draws cost on the GPU here too: the lit pass goes from
  0.41 to 0.71 ms at `groups_22k` `wedge`, and the pass sum rises by 0.14–0.62 ms across its views.
- **Upload**: at 22.4k `wedge`, 3.7 % fewer bytes per frame (`visible` indices gone, tables
  added), and level or slightly more at `low`; the 128-byte instance entry per instance stays
  (96 % of the upload), as does the sort.
- **Memory**: +393 KB of visible-index capacity at 22.4k (`V · N · 4` bytes, against what a view
  actually keeps; +1.6 MB at 89.6k and +3.1 MB at 201.6k), 3.6 KB of group and view tables and
  indirect arguments at 65 draws, 165 KB at 3,845. CPU scratch is unchanged (4.9 MB at 22.4k, kept
  across frames). No steady-frame allocations on either path.

### Round-to-round spread

The reported numbers are each round's median, the fastest round by GPU pass sum. CPU timings
repeat well in most rounds (`prepare` usually within ±3 % at 22.4k, `small_1k` within ±10 %),
with occasional outlier rounds (for example 2.88 ms against 1.75 ms at `instances_22k` `wedge`).
GPU pass sums vary by 20–40 % between rounds at larger `N` on this adapter, and more on
`dense_90k` (`instances_22k` `wedge`, CPU path: 0.931, 0.727, 0.866 ms; `dense_90k` `moving`, CPU
path: 3.29, 6.83, 7.86 ms), from clock state rather than the path. Across rounds the GPU pass sums
of the two paths at `dense_90k` and `dense_202k` overlap; a pass-sum difference of 0.05 ms or less
at 22.4k, or a few tenths at the dense scenes, is within the noise. Full rounds are in
`report.json`'s `rounds` field.

## Correctness

Every path comparison renders the same frame with the CPU path (the reference, gated against
Bevy by the regular oracle) and the GPU path, FLIP over the pair with D3's thresholds, and
compares the per-view instance counts (`work` on the CPU path against `read_gpu_culled` plus CPU
counts on the GPU path). **Every capture was byte-identical to the reference, FLIP mean 0, p99 0,
kept counts equal.**

- `validation_scenarios`: every lighting scenario in `oracle/src/scenarios.rs` (fog, IBL, sky,
  masked and blended textures, point and spot shadows, soft shadows, skinned figure, 26 captures
  including hard- and soft-shadow variants) renders identically.
- `validation_sequence` at 22.4k instances, one shadowed point light and one spot, stepping:
  default camera; a hard cut into the field with props crossing the near plane; a hard cut high;
  a cut back; a batch of moved instances; removal of 1,000 instances; a mesh's bounds edited.
  Identical at every step. The compute pass reads the current frame's tables, so a cut or a
  moved or removed instance shows on the same frame as on the CPU path.
- `report.json` holds per-view captures (frames 0, 29–32, 60 of each scene and view) with
  byte-identity on every one.
- `tests/gpu_culling.rs` holds the small version (near-plane cut, mutation, removal, mesh bounds,
  point and spot shadows, nothing visible) in CI.

Order is not preserved inside a run (the workgroups append in parallel): opaque
surfaces resolve by depth, so that is invisible unless two instances of a run are exactly coplanar
and overlapping, where the CPU path's fixed order would break the tie. No scene here has that. The
default path's images are unchanged; the regular oracle run in the PR confirms it.

## Native, browser and fallback

| Need | This run | Native wgpu 30 | WebGPU | Fallback |
| --- | --- | --- | --- | --- |
| `INDIRECT_FIRST_INSTANCE` | Requested and available on this adapter; each run is drawn from its offset in the visible-index buffer | adapter-dependent | optional `indirect-first-instance` | `Viewport::set_culling` returns `GpuCullingUnsupported`, the viewport stays on the CPU path |
| Compute, storage read/write | Yes | available | core | The CPU path |
| `max_storage_buffers_per_shader_stage ≥ 5` | 8 on this adapter | adapter/limits-dependent | the `maxStorageBuffersPerShaderStage` default (8) meets it | The CPU path |
| `MULTI_DRAW_INDIRECT_COUNT` | Not used. The adapter reports it, but it can only skip empty runs inside one mesh's draw sequence: the path binds each run's vertex buffers and material separately, so without a shared mesh buffer a multi-draw would only remove the zero-instance draws | adapter-dependent | none | none needed |
| Browser build | Not run: there is no `wasm32` build of the crate (D10), and the three.js client has no compute or indirect draws | n/a | **Unverified**: nothing was run in a browser | The CPU path |

wgpu's indirect-call validation (`WGPU_VALIDATION_INDIRECT_CALL`) is on by default, in release
builds too. It is included in the timings above unless stated. Only this adapter was measured.

## Crossover and decision

Comparing total CPU frame cost (the path's stated gain) and GPU cost on the same frames:

- **Few runs, 22.4k instances and up: the GPU path saves CPU and costs little GPU.** `prepare`
  falls by 0.44–0.64 ms at 22.4k (−18 to −27 %), 1.2–6 ms at 89.6k and 5–7 ms at 201.6k. On the
  GPU, the compute pass plus the interval's increase (which holds wgpu's validation) costs
  0.05–0.14 ms at 22.4k; the pass sum at the dense scenes stays within the noise of the rounds.
- **Small scenes: no gain.** At 1k instances the CPU total goes up 0.01–0.06 ms: encode and its
  overhead outweigh the ~0.02 ms of `prepare` saved.
- **Many runs per scene: the GPU path is worse.** At 768 runs the path records 3,845 indirect
  draws against 466–2,679 CPU draws; encode gets two to eight times longer, and the CPU total 0.3–2.1 ms
  worse, across the views. The GPU pass sum rises by 0.14–0.62 ms on `groups_22k`.
- The CPU work the path leaves in `prepare` (the sort and the full-instance upload, about three
  quarters of it) is untouched, so the gain tops out at about a quarter of `prepare` until
  `prepare` is no longer a full pass (see below).

**Decision: not adopted as the default; the prototype stays opt-in and experimental.** It meets
D10's correctness row but no representative consumer workload has been measured, the benefit is
limited to CPU-bound scenes with few runs and many instances, and it hurts draw-bound scenes (the
current mmorpg shape as described in mmorpg#615, unverified here). `Culling::Cpu` stays the
reference and the default; `Culling::GpuIndirect` remains for a future consumer pilot.

Nothing in the crossover justifies an implementation issue now. Worth revisiting only with a
consumer profile whose `prepare` is dominated by the culling loop at tens of thousands of
instances per view, or together with changes the prototype avoided: a retained, incremental
GPU instance store (removing the 128 B per instance upload) and a shared vertex/index buffer with
`MULTI_DRAW_INDIRECT_COUNT`, which would let empty runs cost nothing. Both are larger than this
experiment. Not measured: skinned-heavy scenes (skinned instances stay on the CPU path by design),
the mmorpg's own frame, and any browser target.

## Limits

- One adapter, one driver, one OS. No AMD, Intel, Apple or browser runs.
- GPU numbers are timestamp pass spans; a validation dispatch between passes is visible only in
  the interval. The driver may time very light frames at low clocks (the `dense_90k`/`dense_202k`
  `low` compute spans), so only the repeatable differences above are claimed.
- Scene geometry is cuboids and spheres. Vertex-heavy scenes change the balance between pass cost
  and culling cost; so would any scene where far more runs than here are empty after culling.
- `prepare` CPU medians are for this single-threaded path; `encode` includes `finish` where wgpu
  records the backend commands. Both are per-frame medians on a desktop CPU, not the mmorpg
  client's frame.
- Nothing in this path was measured under occlusion culling, which remains off (#59).
