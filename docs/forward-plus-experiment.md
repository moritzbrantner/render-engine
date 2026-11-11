# Forward+ clustered lighting (#72, #94)

**Status: the clustered path is the default since #94** (owner decision 2026-11-11, after the
mmorpg's Greyhaven frames measured it; see [Stage 2 and the default](#stage-2-and-the-default)).
The report below is the #72 bounded evaluation that preceded it, written while the path was
opt-in.

A *bounded evaluation*, not a renderer migration. It measures the reference path (every point and
spot light evaluated for every fragment) on a small workload matrix, and, because those
measurements showed the per-fragment loop limiting, compares it with the smallest opt-in
clustered path the renderer can have, on identical content.

| Path | What runs where |
| --- | --- |
| Reference (`LightAssignment::PerFragment`; the default until #94, now opt-in and the fallback) | `lit.wgsl` loops over all `L` point and spot lights in every fragment. |
| Clustered (`LightAssignment::Clustered`; opt-in in #72, the default since #94) | `prepare` assigns the lights to a 16 × 9 × 24 grid of view-space clusters on the CPU (Bevy's grid shape: exponential depth slices, first slice 5 m). The lit pipelines are compiled from `lit.wgsl` with its light loop swapped for a loop over the fragment's cluster (`src/shaders/clustered_lights.wgsl`), with the cluster table and index list as a fourth bind group. |

Prototype only: no depth prepass, no compute light assignment, no change to shadow passes, the
sky, tone mapping or the scene data (D7). `Viewport::set_light_assignment` selects the path per
viewport. The reference module's (`lit.wgsl`) only change is two comments marking the light loop.

## Reproducing

```sh
bash scripts/bench-forward-plus.sh                                   # full run, about 8 minutes
bash scripts/bench-forward-plus.sh --only dungeon_256 --frames 120   # one scene, quickly
cargo test --locked --test clustered_lighting                        # the small comparison CI runs
```

Output: `target/oracle/forward_plus/report.json` (every median, pass span, work counter and
per-round spread), `fingerprint.txt`, the Bevy comparison images in `bevy/`, and PNG/FLIP heat maps
for any capture that is not byte-identical (there were none). Each run warms up at least 500 ms
and 60 frames, times 300 frames with one submission per frame, and alternates path order across
three rounds; the round with the lowest GPU pass sum is reported.

| Measured | Where |
| --- | --- |
| CPU `prepare` | `Renderer::prepare`, including the cluster assignment. |
| CPU encode, submit | `Renderer::encode` plus `CommandEncoder::finish`; `queue.submit`. |
| GPU cost | Sum of the frame's timestamped pass spans (D3's protocol). |
| GPU lit | The main pass's own span. The cluster lookups are inside it; the path adds no pass. |
| Overdraw | `FRAGMENT_SHADER_INVOCATIONS` over the lit pass, one frame, divided by the frame's pixels. Unlike a sample count, this is per fragment shader run under 4× MSAA. |
| Lights per fragment | A diagnostic build of the clustered path counts shaded fragments and the lights they loop over (atomic counters; `ClusterConfig::diagnostic`). On the reference path it is `L` by construction. |
| Memory | `Viewport::memory`: GPU buffers at their high-water size and retained CPU scratch. |

## Inventory: what the renderer has today

| Area | Today |
| --- | --- |
| Directional lights | Sun and optional moon. The sun casts up to 4 cascades (`MAX_CASCADES`); the moon does not cast shadows. |
| Point and spot lights | No count limit on native: a growable storage buffer of 80-byte lights, points then spots, shadow casters first (Bevy's order). |
| Shadow passes | One depth pass per sun cascade, per shadowed spot light and per face of each shadowed point light (six). Spot maps share the sun's array; point maps are a cube array. Every shadow caster renders its maps every frame. |
| Light loop | `lit.wgsl`: every point and spot light, every fragment, opaque, masked and blended alike. Shadow lookups run in the loop, so a shadowed light costs its lookup across the whole screen. |
| Materials | Bevy's `StandardMaterial` core: opaque, alpha-masked, blended; skinned and double-sided variants. 24 lit pipelines. |
| Depth prepass | None. Opaque overdraw is shaded. |
| Web (three.js, D8) | A uniform array sized per material (`LIT_POINT_LIGHTS` + `LIT_SPOT_LIGHTS`), at most 4 point and 4 spot shadow casters. |

## Workload matrix

All values measured 2026-11-08 on **NVIDIA GeForce RTX 3060 Ti, driver 595.99.02, Vulkan**, wgpu
30.0.1, Linux 7.0.0-42-generic, rustc 1.99.1, release build, 4× MSAA, on the working tree over
commit `35fae9a` that became this PR. A report field was renamed afterwards; no measured code
changed. Software-adapter timings are not reported.

| Scene | Content | Local lights | Instances |
| --- | --- | ---: | ---: |
| `outdoor_0` | Greyhaven-style outdoor: sun with four cascades, 160 alpha-masked CC0 bushes (`assets/props`), 8 huts, a blended water plane | 0 (the path's bare cost) | 170 |
| `outdoor_low` | Same | 4 (3 point, 1 spot) | 170 |
| `outdoor_high` | Same | 64 (48 point, 16 spot) | 170 |
| `dungeon_4` … `dungeon_1024` | Underground hall: 80 × 80 m, 121 pillars, 400 props, no sun, overlapping coloured lights (8–14 m range; a quarter are spots) with intensity scaled so the frame stays comparably exposed | 4, 16, 64, 256, 1024 | 527 |
| `dungeon_64_shadowed` | Same, with 2 point and 2 spot shadow casters | 64 | 527 |
| `dungeon_256_1080p` | `dungeon_256` at 1920 × 1080 | 256 | 527 |

No new assets. All scenes render at 1280 × 720 except the 1080p row. The timed camera is static;
assignment is rebuilt every frame regardless of what moved, so a moving camera repeats the same
work (the camera cuts below check its correctness, not its timing).

## Is the per-fragment loop limiting?

Yes. On the reference path the lit pass grows linearly with the light count at about 47 µs per
light at 720p on the dungeon, while overdraw stays near 1.7 invocations per pixel: the loop, not
geometry or overdraw, is the cost.

| Scene | `L` | Lit pass (ms) | GPU pass sum (ms) | Invocations per pixel |
| --- | ---: | ---: | ---: | ---: |
| `dungeon_4` | 4 | 0.282 | 0.307 | 1.68 |
| `dungeon_16` | 16 | 0.830 | 0.858 | 1.68 |
| `dungeon_64` | 64 | 3.032 | 3.064 | 1.68 |
| `dungeon_256` | 256 | 11.938 | 11.971 | 1.68 |
| `dungeon_1024` | 1,024 | 49.147 | 49.180 | 1.68 |
| `outdoor_low` | 4 | 0.710 | 0.966 | 2.07 |
| `outdoor_high` | 64 | 6.281 | 6.536 | 2.07 |
| `dungeon_256_1080p` | 256 | 26.175 | 26.243 | 1.66 |

At 256 lights the loop alone consumes 72 % of a 60 Hz frame at 720p and over 100 % of a 16.7 ms
budget at 1080p, while most of those evaluations contribute exactly zero. That justified step 4.

## Results: reference against clustered

Values are `reference → clustered` in ms unless noted. CPU total is prepare + encode + submit.
Shading work is the mean lights looped over per shaded fragment.

| Scene | `L` | GPU pass sum | Lit pass | CPU prepare | CPU total | CPU + GPU | Lights per fragment |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| `outdoor_0` | 0 | 0.615 → 0.650 | 0.362 → 0.395 | 0.038 → 0.051 | 0.176 → 0.173 | 0.791 → 0.823 | 0 → 0 |
| `outdoor_low` | 4 | 0.966 → 0.701 | 0.710 → 0.445 | 0.035 → 0.071 | 0.156 → 0.172 | 1.122 → 0.873 | 4 → 0.53 |
| `outdoor_high` | 64 | 6.536 → 1.502 | 6.281 → 1.246 | 0.034 → 0.391 | 0.148 → 0.506 | 6.684 → 2.008 | 64 → 6.84 |
| `dungeon_4` | 4 | 0.307 → 0.162 | 0.282 → 0.136 | 0.037 → 0.073 | 0.304 → 0.391 | 0.611 → 0.553 | 4 → 0.34 |
| `dungeon_16` | 16 | 0.858 → 0.207 | 0.830 → 0.180 | 0.044 → 0.111 | 0.342 → 0.421 | 1.200 → 0.628 | 16 → 1.22 |
| `dungeon_64` | 64 | 3.064 → 0.336 | 3.032 → 0.304 | 0.051 → 0.283 | 0.371 → 0.677 | 3.435 → 1.013 | 64 → 3.77 |
| `dungeon_256` | 256 | 11.971 → 1.081 | 11.938 → 1.049 | 0.057 → 0.970 | 0.378 → 1.323 | 12.349 → 2.404 | 256 → 19.09 |
| `dungeon_1024` | 1,024 | 49.180 → 3.674 | 49.147 → 3.641 | 0.074 → 3.457 | 0.394 → 3.852 | 49.574 → 7.526 | 1,024 → 71.35 |
| `dungeon_64_shadowed` | 64 | 3.371 → 0.460 | 3.239 → 0.331 | 0.091 → 0.296 | 0.660 → 0.850 | 4.031 → 1.310 | 64 → 3.77 |
| `dungeon_256_1080p` | 256 | 26.243 → 2.354 | 26.175 → 2.287 | 0.051 → 0.980 | 0.360 → 1.316 | 26.603 → 3.670 | 256 → 19.08 |

GPU pass sums repeat within about 3 % across rounds on these scenes (`dungeon_256` reference 11.97,
12.07, 12.20 ms; clustered 1.081, 1.082, 1.090 ms). Full rounds are in `report.json`.

### Clustering overhead

- **CPU**: assignment is `O(L)` in the clusters each light touches. `prepare` rises by 0.013 ms at
  no lights, 0.04 ms at 4, 0.36 ms at 64 outdoors, 0.91 ms at 256 and 3.38 ms at 1,024. About 10 ns
  per cluster test: 93,004 tests at 256 lights, 334,480 at 1,024.
- **GPU**: the path adds no pass. With no lights to loop over, the lit pass costs 0.033 ms more
  (`outdoor_0`: 0.362 → 0.395 ms) for the cluster lookup in every fragment; this is its bare cost.
- **Upload**: the table and list are rewritten each frame: 25,648 → 67,124 bytes at no lights,
  90,548 → 343,552 at 256 and 151,988 → 977,952 at 1,024.

### Memory

| Scene | `L` | List entries | Cluster buffers added (GPU) | CPU scratch (reference → clustered) |
| --- | ---: | ---: | ---: | --- |
| `outdoor_0` | 0 | 0 | 65,824 B | 40,128 → 81,600 B |
| `outdoor_low` | 4 | 2,213 | 81,952 B | 40,128 → 130,752 B |
| `outdoor_high` | 64 | 21,071 | 196,640 B | 40,128 → 474,816 B |
| `dungeon_16` | 16 | 3,320 | 81,952 B | 168,416 → 259,040 B |
| `dungeon_64` | 64 | 10,222 | 131,104 B | 168,416 → 406,496 B |
| `dungeon_256` | 256 | 52,627 | 327,712 B | 168,416 → 996,320 B |
| `dungeon_1024` | 1,024 | 195,099 | 1,114,144 B | 168,416 → 3,355,616 B |

The table is 12 bytes per cluster (41,472 B at the default grid), the list 4 bytes per entry,
grown by doubling (so the buffer can be up to twice the list). The worst case is bounded by
`x · y · z · max_lights_per_cluster` entries plus the `L`-entry all-lights run: 3.5 MB of list at
the default grid and limit (256).
No steady frame allocated a GPU buffer on either path (`allocations` 0 in every run).

## Correctness

Each comparison renders the same frame on both paths, FLIP over the pair with D3's thresholds.
**Every capture was byte-identical to the reference path, FLIP mean 0, p99 0.**

- `validation_scenarios`: all 26 oracle lighting scenarios (fog, IBL, sky, masked and blended
  textures, point and spot shadows, soft shadows, skinned figure).
- `validation_sequence` on a 64-light hall with 4 shadow casters: a camera cut inside a light's
  volume, a near-plane crossing, a high cut, lights moved (some behind the camera), lights removed
  (shadow indices change), no local lights, overflow at 1 and 2 lights per cluster, and grids of
  32×18×48, 8×4×8 and 1×1×1.
- All ten matrix scenes.
- `tests/clustered_lighting.rs` runs the small version in CI, on lavapipe as well.

Why byte-identical holds: a light outside its range contributes exactly zero under Bevy's
distance attenuation, and a spot light outside its outer cone likewise; the assignment is
conservative (lights padded by 0.1 % + 1 cm, cluster boxes enclose their frusta) and lists keep the
light buffer's order, so the summation is the same with some `+ 0` removed. Shading terms are
unchanged, so `packages/three-lighting` needs no matching change.

Against the pinned Bevy, one frame each (Bevy clusters its lights itself), D3's thresholds:

| Scene | FLIP mean | FLIP p99 |
| --- | ---: | ---: |
| `outdoor_low` | 0.0000324 | 0.00068 |
| `outdoor_high` | 0.0000360 | 0.00081 |
| `dungeon_16` | 0.0000057 | 0.00004 |
| `dungeon_256` | 0.0000322 | 0.00065 |

The clustered path's numbers against Bevy equal the reference path's in every row, as they must
for byte-identical frames. Bevy's GPU cost was not measured on these scenes.

## Overflow, transparency and shadows

- **Overflow.** A cluster whose list would exceed `max_lights_per_cluster` (default 256) points at
  a run of every light, the first entries of the index list. That is the reference loop for that
  cluster alone: the output does not change, the cost in that cluster is the reference's, and the
  decision depends only on the frame's lights and camera. `PrepareWork::cluster_overflows` counts
  them (0 in every matrix scene). Forcing a limit of 1 overflows 2,760 of the 3,456 clusters and 2
  overflows 1,942, both byte-identical. **The frame time of an overflowing frame was not timed.**
- **Transparency.** Clusters are view-space volumes, not depth-derived tiles: masked and blended
  fragments look up the same table as opaque ones, with no depth prepass. Blended water and masked
  foliage are covered by the outdoor scenes. Particles are unlit and unaffected.
- **Shadows.** Shadow maps are not clustered and their passes are unchanged. A shadow caster is a
  list entry like any other, so fragments out of its reach stop paying for its lookup
  (`dungeon_64_shadowed`: lit pass 3.239 → 0.331 ms), but it still renders its maps whether or not
  a visible fragment needs them.

## Native, browser and fallback

| Need | This run | Native wgpu 30 | WebGPU | Fallback |
| --- | --- | --- | --- | --- |
| A fourth bind group | Yes | Core default | The default `maxBindGroups` (4) | `set_light_assignment` returns `ClusteredLightingUnsupported`, the viewport stays per-fragment |
| 4 storage buffers in the fragment stage | 8 on this adapter | adapter/limits-dependent | The default `maxStorageBuffersPerShaderStage` (8) | Same |
| Compute | Not used | n/a | n/a | n/a |
| `PIPELINE_STATISTICS_QUERY`, `SHADER_EARLY_DEPTH_TEST` | Bench diagnostics only | adapter-dependent | Not in WebGPU | The diagnostics degrade; the path does not need them |
| Browser build | Not run: there is no `wasm32` build of the crate (D10) | n/a | **Unverified**: nothing was run in a browser | Per-fragment |
| three.js client | Not changed | n/a | n/a | Its uniform-array loop. WebGL 2 has no fragment storage buffers, so this path does not reach it; a port would need its own lists in textures or uniforms and its own evidence |

The clustered path also ran on lavapipe (`tests/clustered_lighting.rs`), for correctness only.

## Crossover and decision

- **The combined CPU + GPU cost is lower from 4 lights up on both scene types** (outdoor −0.249 ms,
  dungeon −0.058 ms at 4 lights). The dungeon's 4-light margin is within CPU noise: its encode
  median differs by 0.049 ms between paths with the same 295 draws. The GPU side alone is lower
  by 0.145–0.265 ms at 4 lights. At 0 lights the path costs +0.032 ms net. The exact crossover lies
  between 0 and 4 lights and was not located.
- **The gain grows with the light count.** At 64 lights the combined cost falls by 70 % outdoors
  and 71 % in the dungeon; at 256 by 81 %, and at 1,024 by 85 %. On the GPU alone, 11× at 256
  lights.
- **The CPU cost is real**: 0.91 ms more `prepare` at 256 lights and 3.38 ms at 1,024, on one
  thread. A CPU-bound consumer pays for it even when the GPU gains more.
- **Output was unchanged** in every comparison, including forced overflow.

**Decision: Forward+ remains the preferred direction, and this result supports it. The prototype
stays opt-in; `LightAssignment::PerFragment` stays the default and the fallback.** It clears D10's
stage 1 on the measured hardware. D10 adopts a path only on a representative consumer workload,
and none has been measured: the scenes here are synthetic, built to bracket the case. Whether to
take it to stage 2 (an opt-in renderer path, with its own issue) is the owner's decision, best
made with the mmorpg's own light counts. Deferred or hybrid rendering stays unneeded by this
evidence and out of scope.

## Stage 2 and the default

The consumer workload came with #74 ([Greyhaven baseline](greyhaven-baseline.md)): the mmorpg
native client's Greyhaven Vale frame has 13 unshadowed point lights, and switching only the light
assignment took its main pass from 5.2–9.8 to 2.5–3.4 ms and the GPU frame down by 2.7 to 6.4 ms
(32 to 49 %) on the RTX 3060 Ti, with CPU `prepare` unchanged within noise (2.8 to 3.2 ms both
ways, no overflows). **Owner decision 2026-11-11 (#94): `LightAssignment::Clustered` with
`ClusterConfig::default()` is the default.**

- `Viewport::new` starts clustered wherever the device has four bind groups and four fragment
  storage buffers, and on the per-fragment loop otherwise (`Viewport::light_assignment` says
  which). `LightAssignment::default()` is the clustered one.
- `LightAssignment::PerFragment` stays available through `set_light_assignment` as the reference
  this report, the bench and `tests/clustered_lighting.rs` compare against.
- The clustered pipelines compile once per `Renderer`, on the first viewport that uses them,
  instead of once per viewport.
- **Oracle coverage.** Every lighting scenario of the oracle now renders through the clustered
  path and is gated against Bevy under D3's thresholds, which Bevy's own clustering meets with the
  same sums. `many_lights` adds the bench's `dungeon_64_shadowed` hall to the gate: 64
  overlapping point and spot lights, four shadowed, over 527 instances. The bench
  (`--forward-plus-bench`) still compares both paths byte for byte.
- Shading is unchanged, so `packages/three-lighting` keeps its per-fragment loop.

## Limits

- One adapter, one driver, one OS. No AMD, Intel, Apple, mobile or browser runs.
- Synthetic scenes; simple geometry. The mmorpg's frame was not measured.
- Timings use a static camera. Moving-camera behaviour is covered for correctness only.
- GPU numbers are timestamp pass spans; light frames may run at lower clocks.
- `dungeon_*` numbers depend on light range: longer ranges lengthen the lists.
- Lights-per-fragment is from a diagnostic build. Its fragment count matches the invocation
  statistics on the dungeon scenes (1,544,233); on the outdoor scenes it counts 2,088,107 against
  about 1,910,000 because the masked foliage cannot force an early depth test once the diagnostic
  adds its side effect. Those outdoor per-fragment means are therefore approximate.
- Grid sizes other than the default were checked for correctness, not timed.
- CPU assignment was chosen as the smallest step; a compute assignment was not built or compared.
