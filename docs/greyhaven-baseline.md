# Greyhaven baseline (#74)

What render-engine costs on real Greyhaven frames, through the game's own native and browser
profiling paths. This is evidence, not a migration: nothing here changes the renderer, its
defaults or the Bevy oracle gates.

**Result.** At 1280×720 on an RTX 3060 Ti a native Greyhaven frame takes 15.3 to 19.7 ms (median),
and the renderer's share now splits: GPU 8.4 to 13.2 ms, CPU `prepare` 2.8 to 3.1 ms, `encode`
0.3 to 0.5 ms. Two renderer-owned gaps are measured:

1. **The per-fragment light loop is the largest single cost.** Greyhaven has 13 point lights, and
   the main pass evaluates all of them for every fragment. The engine's existing opt-in
   `LightAssignment::Clustered` (#72) cuts the main pass from 5.2–9.8 to 2.5–3.4 ms and the GPU
   frame by 2.7 to 6.6 ms, 32 to 50 %. What looked like the lake's water cost in the first slice
   was this loop. Filed as [#94](https://github.com/moritzbrantner/render-engine/issues/94) (stage 2,
   needs a decision) and, for the consumer's opt-in,
   [mmorpg#725](https://github.com/moenarch-games/mmorpg/issues/725).
2. **`prepare` rewrites every instance every frame**: about 26,000 mostly static instances, 4.1
   to 4.4 MB a frame, about 3 ms of CPU. GPU culling (#71) does not help on this content. Filed as
   [#95](https://github.com/moritzbrantner/render-engine/issues/95).

Game-owned: a 2.8 s stall on the first frame
([mmorpg#726](https://github.com/moenarch-games/mmorpg/issues/726)), the missing starter-area and
crowd cameras ([mmorpg#724](https://github.com/moenarch-games/mmorpg/issues/724)), and browser GPU
spikes that now hold two scenarios at 23 to 32 fps
([mmorpg#727](https://github.com/moenarch-games/mmorpg/issues/727)). Nothing is filed for
asset-tooling. Against the approved native reference captures the browser's FLIP mean is 0.030 to
0.098 per view, recorded without a gate (below); with the reference's night fog fixed
([#102](https://github.com/moritzbrantner/render-engine/issues/102)) the night shots fall to
0.021 to 0.076.

The first slice ([#78](https://github.com/moritzbrantner/render-engine/pull/78), game `6ebc4c64`,
render-engine `bd4708dd`, a heavily loaded machine) had only `render_ms` natively. This revision
replaces its numbers; where they differ, that is said below.

## Sources and setup

| | Native | Browser |
| --- | --- | --- |
| Game | `moenarch-games/mmorpg` `05a1e3eb` (mmorpg#692 merged) | same |
| Renderer | render-engine `df5597b5` (`Cargo.lock`) | three.js 0.186.1 and `@moritzbrantner/three-lighting` at `81752f2` |
| Path | `outpost_frame_time` (`crates/mmorpg-client/src/graphics.rs`), `imported_mesh_buffer_observation`, `greyhaven_lighting_shots` | `scripts/profile-browser.py` (`web/dist`, `?debug&profile`), which now counts multi-draws |
| Build | `cargo test --release --locked`, rustc 1.99.1, wgpu 30.0.1, `TIMESTAMP_QUERY` on | `bun run build`; Chromium 158.0.8127.18 headless (Playwright 1.65.0), ANGLE on OpenGL ES 3.2 |
| Resolution | 1280×720, offscreen, no present | 1440×900, device pixel ratio 1, on-page canvas |
| Scene settings | Zone hour from `ZONE_START_TIME` (08:00). Four sun cascades over 160 m, 2048 per cascade, 4× MSAA, Tony McMapface, Gaussian shadow filter, 13 unshadowed point lights, CPU culling, per-fragment lights | Same hour source and tone mapping, three sun cascades over 160 m (three.js redraws the scene per cascade) |
| Content | Greyhaven Vale, full scenery, NPCs in interest, one player | same |

Machine: AMD Ryzen 7 5700X, 16 logical CPUs, 30 GB RAM, NVIDIA GeForce RTX 3060 Ti (8 GB), driver
595.99.02, Linux 7.0.0-42. Measured 2026-11-10.

**Load.** The machine was nearly idle this time: the 1-minute load average was 2.4 to 7.5 at the
start of the native runs (the higher figures right after a build) and 0.5 to 4.0 during the
browser runs. The six native baseline runs agree within 2 % on every median, so the numbers below
are not marked advisory for load. They remain one machine, one driver.

### Matched cameras

Native and browser use the same viewpoint poses (`DEBUG_VIEWPOINTS` in
`web/src/world/world-view.ts` and the `views` in `outpost_frame_time`), field of view 50°.

| Issue camera | Native view | Browser scenario | Eye / target |
| --- | --- | --- | --- |
| Starter area | none timed ([mmorpg#724](https://github.com/moenarch-games/mmorpg/issues/724)) | `idle` (view after entry, on the spawn plaza) | not a debug viewpoint |
| Hub | `hub` | `dense-hub` | `[5, 8.5, 31]` / `[-14, 3, 4]` |
| Forest / vegetation | `woods` | `dense-woods` | `[-41, 18, 16]` / `[-69, 14, 0]` |
| Water | `lake` | `western-lake` | `[-56, 33, -39]` / `[-87, 11, -53]` |
| Whole Vale | `vale` | `overview` | `[0, 70, 95]` / `[0, 0, -10]` |
| Dense units | none ([mmorpg#724](https://github.com/moenarch-games/mmorpg/issues/724)); see the skinned-unit path below | none | |

New characters spawn on the hub plaza, so `hub` is the closest native view to the starter area.
No existing path puts a crowd of units in a real Greyhaven camera on either renderer; building one
is game work, filed as mmorpg#724.

## Native measurements

`outpost_frame_time`, 6 runs. Each view renders 240 frames; statistics leave out the first 40
(the first frame and the raw lists keep them). `frame_ms` is one composed, rendered and completed
frame; `render_ms` is `FrameRenderer::render` plus the wait for the device. `prepare` and `encode`
are render-engine's CPU calls; GPU is the sum of the frame's timestamped passes. Each cell is the
range of that statistic over the 6 runs.

| View | `frame_ms` median | p95 | p99 | max | `render_ms` median | `prepare` median | `encode` median | GPU median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `hub` | 15.28–15.54 | 16.06–17.56 | 17.16–20.26 | 17.52–21.69 | 13.94–14.18 | 3.00–3.08 | 0.44–0.46 | 8.38–8.47 |
| `woods` | 17.89–18.08 | 18.80–20.47 | 20.02–21.80 | 20.04–22.37 | 16.54–16.76 | 2.88–2.96 | 0.33–0.37 | 11.56–11.67 |
| `vale` | 16.85–17.05 | 17.51–19.16 | 18.38–21.98 | 19.11–24.97 | 15.50–15.73 | 2.82–2.88 | 0.39–0.44 | 10.40–10.51 |
| `lake` | 19.46–19.67 | 20.41–22.52 | 21.45–23.35 | 21.93–25.55 | 18.11–18.32 | 2.91–2.95 | 0.32–0.36 | 13.10–13.23 |

GPU p99 is within 0.4 ms of its median on every view; the CPU side carries the tail.

**Where a frame goes** (`hub`, medians of one run; the others split the same way):

| Part | ms | Owner |
| --- | --- | --- |
| Game composition (`presentation.frame`: units, LOD pick) = `frame_ms` − `render_ms` | 1.34 | game |
| Instance-list build in `FrameRenderer::render` before `prepare` (local timer, one run: 0.26 to 0.28 on all views) | 0.28 | game |
| render-engine `prepare` | 3.02 | renderer |
| render-engine `encode` | 0.44 | renderer |
| GPU passes | 8.43 | renderer (shading cost of game content) |
| Remainder: submission, timestamp resolve, the queued buffer writes and waking from the wait | 1.78 | not split (1.5 to 1.8 on all views) |

The test waits for the GPU each frame, so CPU and GPU time add up here. A windowed client overlaps
them, and its frame would be closer to the larger of the two.

**GPU passes**, median ms, ranges over the 6 runs:

| View | `sky` (LUTs) | `shadow` (4 cascades) | `lit_opaque` | `lit_transparent` | `render_sky` + `tonemap` + `blit` |
| --- | --- | --- | --- | --- | --- |
| `hub` | 0.47–0.48 | 2.56–2.58 | 5.20–5.25 | 0.05 | 0.09 |
| `woods` | 0.48 | 2.46–2.49 | 8.43–8.51 | 0.09 | 0.10 |
| `vale` | 0.45–0.46 | 1.62–1.63 | 8.13–8.22 | 0.09 | 0.11 |
| `lake` | 0.45–0.46 | 2.50–2.53 | 9.78–9.87 | 0.26–0.27 | 0.11 |

**Work per frame** (`Viewport::work()`, the same in every run; static camera):

| View | Instances | Views culled | CPU bounds tests | Visible: camera / shadow views | Draws: main / shadow | Upload | Allocations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `hub` | 26,978 | 5 | 134,890 | 9,826 / 32,093 | 469 / 2,550 | 4.37 MB | 0 |
| `woods` | 26,148 | 5 | 130,740 | 4,638 / 29,253 | 198 / 2,066 | 4.23 MB | 0 |
| `vale` | 25,183 | 5 | 125,915 | 14,070 / 25,131 | 878 / 1,865 | 4.12 MB | 0 |
| `lake` | 26,014 | 5 | 130,070 | 3,402 / 32,049 | 185 / 1,896 | 4.22 MB | 0 |

About 3.4 MB of each upload is the instance array itself (128 B per instance), rewritten whether
or not anything moved; the rest is the visible-index lists, lights, shadow views and the frame
uniform. Triangle counts are not observable natively: `PrepareWork` counts draws and instances,
not indices.

### Engine options on the same frames

The client uses render-engine's defaults. To attribute the main pass and `prepare`, the same build
ran with the engine's two existing opt-in paths switched on in `FrameRenderer::new` (a local,
uncommitted change in the measurement clone; `Culling::GpuIndirect` also needed the device feature
`INDIRECT_FIRST_INSTANCE`, which the client does not request). One run each, medians in ms, GPU /
`render_ms` / `prepare` / `encode`:

| View | Default | `LightAssignment::Clustered` | `Culling::GpuIndirect` | Both |
| --- | --- | --- | --- | --- |
| `hub` | 8.38 / 14.00 / 3.00 / 0.46 | 5.69 / 11.45 / 3.15 / 0.43 | 8.62 / 15.17 / 2.27 / 0.79 | 5.80 / 12.35 / 2.29 / 0.80 |
| `woods` | 11.62 / 16.67 / 2.88 / 0.36 | 6.45 / 11.62 / 3.05 / 0.33 | 11.74 / 18.05 / 2.23 / 0.76 | 6.55 / 12.87 / 2.23 / 0.77 |
| `vale` | 10.47 / 15.64 / 2.82 / 0.42 | 5.38 / 10.90 / 2.94 / 0.49 | 10.51 / 16.68 / 2.12 / 0.76 | 5.45 / 11.66 / 2.13 / 0.75 |
| `lake` | 13.18 / 18.29 / 2.93 / 0.35 | 6.61 / 11.84 / 2.98 / 0.33 | 13.20 / 19.50 / 2.19 / 0.77 | 6.71 / 12.96 / 2.22 / 0.74 |

- **Clustered lights**: `lit_opaque` falls to 2.52 (`hub`), 3.35, 3.14 and 3.44 ms; nothing else
  moves. The cluster assignment costs no measurable `prepare` time (3 to 1,316 light–cluster tests,
  no overflows). The images were not compared here; #72 found the output unchanged on its scenes.
- **GPU culling**: the bounds tests move to a 0.03 ms compute pass and `prepare` drops by about
  0.7 ms, but the indirect path issues 957 to 1,209 main and 4,272 to 4,630 shadow draws (one per
  run), `encode` rises by 0.4 ms and `render_ms` gets worse by 1.0 to 1.4 ms. The upload is
  unchanged. This matches #59/#71: not a win on this hardware, now also on real content.

### Pacing and stutter

The native path renders offscreen, one frame at a time, with no present and no vsync, so this is
the cost of each frame, not display pacing. After warm-up, p99 is at most 1.33× the median and no
frame took twice the median in the run checked. After a camera cut to `woods` and `vale`, the first
40 frames cost 1.5 to 2.7 ms more GPU than the settled frames (13.1 to 13.8 ms against 11.6 and
10.4); this is consistent with level-of-detail cross-fades drawing dissolve copies for 0.35 s, but
that was not verified. Each cut builds 24 to 38 meshes (0.4 to 0.7 MB) and its first frame takes
22 to 30 ms.

**Startup.** The first frame of the process takes **2,781 to 2,992 ms**:

| Part | ms |
| --- | --- |
| Game composition (`presentation.frame`) | about 1,530 |
| `FrameRenderer::render` before `prepare` (`NatureGpu::upload`, the first meshes and textures into `World`) | 1,209 (one run, local timer) |
| render-engine `prepare` / `encode` | 4.1–5.9 / 0.9 |
| GPU | 17–90 |

That frame uploads 379 MB into render-engine's `World`: 1,659 meshes, 77 textures, 1,654
materials. The game decides what is built and uploaded when, so this is game-owned
([mmorpg#726](https://github.com/moenarch-games/mmorpg/issues/726)); the share of the upload calls
themselves is not split from the client's mesh building.

Per process, the test binary run directly (start-up and 960 frames):

| Metric | Value | Source |
| --- | --- | --- |
| Wall time | 22.7 s | `time -v` |
| CPU user + system | 11.1 + 0.9 s | `time -v` |
| Peak resident memory | 1,769 MB | `time -v` |
| GPU memory, process | 718 MB peak (844 MB in one run) | `nvidia-smi pmon` |
| GPU busy, process | 50 % mean, 66 % peak | `nvidia-smi pmon`, 19 samples of 1 s |

### Skinned units (real Ranger asset, not a Greyhaven scene)

Not rerun; from the first slice. `imported_mesh_buffer_observation` draws the pinned Ranger (13
primitives) with no scenery, at 640×360. Later frames create no buffers and upload no mesh bytes,
only one joint palette per unit: 1, 10 and 50 units took 2.6, 3.2 and 5.7 ms from render through
GPU completion (one sample each). A crowd in a real camera is mmorpg#724.

## Browser measurements

`profile-browser.py`, two runs, load average 1.2 at the start of the first and 0.5 of the second
(4.0 at most at the end). Times are per animation frame: `rAF` is the interval between frames
(vsynced to 60 Hz); `CPU` is main-thread time for the whole frame; `GPU` is
`EXT_disjoint_timer_query_webgl2` for the frame's commands. First run:

| Scenario | fps | rAF p50 / p95 / p99 / max ms | CPU p50 / p95 / p99 ms | GPU p50 / p95 / p99 ms |
| --- | --- | --- | --- | --- |
| `idle` | 63.0 | 16.7 / 16.7 / 16.8 / 100.0 | 11.4 / 12.7 / 14.8 | 10.39 / 11.32 / 12.24 |
| `move-rotate` | 58.4 | 16.7 / 50.0 / 66.7 / 799.9 | 12.7 / 18.9 / 50.4 | 10.10 / 25.10 / 54.42 |
| `dense-woods` | 30.0 | 16.7 / 199.9 / 283.3 / 416.6 | 9.4 / 14.1 / 16.4 | 14.62 / 114.80 / 200.79 |
| `dense-hub` | 32.3 | 16.7 / 183.2 / 283.4 / 516.6 | 9.8 / 13.5 / 18.6 | 11.87 / 114.65 / 212.86 |
| `western-lake` | 49.7 | 16.7 / 33.4 / 183.2 / 416.6 | 8.2 / 11.8 / 14.0 | 16.29 / 31.00 / 141.40 |
| `overview` | 41.8 | 16.7 / 100.0 / 233.3 / 500.0 | 10.0 / 14.3 / 17.0 | 13.26 / 75.34 / 176.37 |

The second run (`--no-sweep`) reproduces it: `dense-woods` 28.3 fps with GPU p95 173.08 ms,
`dense-hub` 22.8 fps with 255.54 ms, `western-lake` 45.5 fps with 36.56 ms, `idle` unchanged. At
`6ebc4c64` these GPU p95s were 26.38, 40.25 and 28.03 ms on a loaded machine: the p50s are close,
the tails grew. The CPU frame stays under 16 ms at p95 and no frame creates resources, so the
stalls are GPU work on some frames, unattributed; filed as
[mmorpg#727](https://github.com/moenarch-games/mmorpg/issues/727).

Per frame, first run. *GL draws* counts every draw at the WebGL boundary, now including each
sub-draw of a multi-draw; *calls by pass* is three's own count per pass (`drawsByPass`), where one
`BatchedMesh` multi-draw is one call:

| Scenario | GL draws | GL triangles | Calls by pass: all / sun cascades / main opaque | Triangles by pass: all / sun cascades / main opaque | Instance batches |
| --- | --- | --- | --- | --- | --- |
| `idle` | 8,773 | 8.67 M | 532 / 276 / 177 | 6.91 M / 4.73 M / 2.19 M | 153 |
| `move-rotate` | 6,188 | 5.72 M | 633 / 348 / 210 | 6.66 M / 4.56 M / 2.10 M | 169 |
| `dense-woods` | 6,935 | 7.52 M | 498 / 276 / 144 | 7.52 M / 4.72 M / 2.80 M | 181 |
| `dense-hub` | 7,223 | 7.95 M | 662 / 303 / 280 | 7.91 M / 4.64 M / 3.26 M | 162 |
| `western-lake` | 6,861 | 5.74 M | 453 / 248 / 128 | 5.72 M / 3.73 M / 1.98 M | 167 |
| `overview` | 7,086 | 5.90 M | 641 / 237 / 323 | 5.90 M / 3.44 M / 2.45 M | 168 |

With multi-draws counted, the browser submits 6,200 to 8,800 draws a frame, against 185 to 878 main
and 1,865 to 2,550 shadow draws natively; the native counts are of instanced draws, so neither
side's count is a measure of the other's work. Instance batches fell from 2,493 at `6ebc4c64` to
153 to 181 (the game's tile streaming and batching since).

Main-thread phases, p50 in ms (`?profile`), range over the six scenarios: `render` 4.2 to 7.3,
`simulation` 1.5 to 2.1, `scene` 1.2 to 2.0, `units` 0.6 to 0.9, `projection` 0.3, `hud` 0.2 to
0.4.

Startup: 3.56 and 3.65 s from navigation to an enabled Enter button (`coldSelectionSeconds`), 3.28
and 3.16 s from the Enter World key to a playable view. Not comparable to the native start-up: the
browser downloads and compiles the bundle and streams the scenery.

Viewport sweep (first run), `idle`: GPU p50 10.02 ms at 1280×720 (the native resolution) and 12.36
ms at 1920×1080.

### Not comparable between native and browser

- **Resolution and cascades**: 1280×720 with four cascades, against 1440×900 with three.
- **CPU time**: native `prepare` and `encode` are the renderer's calls alone; browser `render` is
  main-thread three.js and WebGL submission. GPU time is separate on both.
- **Pacing**: native renders offscreen with no present or vsync and waits for each frame; the
  browser is vsynced to 60 Hz and rAF intervals quantize to it. The browser's rAF p50 of 16.7 ms
  is vsync, not cost.
- **Draws**: native counts instanced draws per pass after culling; the browser counts WebGL draws
  and multi-draw sub-draws. Native has no triangle count.
- **Point lights**: the native default evaluates all 13 per fragment; the browser's light path is
  three-lighting's own and was not measured here.
- **Renderer revision**: render-engine `df5597b5` native, `81752f2` for the three.js port.

## Bottlenecks, by owning boundary

Native, largest first, for the `lake` view (the slowest; the others order the same way):

1. **Renderer: per-fragment point lights** in `lit_opaque`. 9.8 ms of the 13.2 ms GPU frame, of
   which 6.4 ms goes away with clustered assignment (2.7 to 6.4 ms over the views). The first
   slice's "water view" bottleneck was this: `lit_transparent`, where the water draws, is 0.27 ms.
   [#94](https://github.com/moritzbrantner/render-engine/issues/94) (needs a decision),
   consumer opt-in [mmorpg#725](https://github.com/moenarch-games/mmorpg/issues/725).
2. **Renderer: `prepare`**, 2.8 to 3.1 ms of CPU, mostly rewriting about 26,000 static instances
   (3.4 MB) and testing them against 5 views. GPU culling moves the tests but costs more than it
   saves. [#95](https://github.com/moritzbrantner/render-engine/issues/95) adds retained instances:
   with the scenery retained, `prepare` takes 1.3 to 1.5 ms on the same views (game adoption
   [mmorpg#728](https://github.com/moenarch-games/mmorpg/issues/728)).
3. **Renderer: sun shadows**, 1.6 to 2.6 ms of GPU over 1,865 to 2,550 draws and 25,000 to 32,000
   shadow-visible instances in four cascades. Measured, not filed: no gap against Bevy's cascade
   defaults is shown, and #59 found Hi-Z no help on this hardware.
4. **Unsplit remainder** inside `render_ms`, 1.5 to 1.8 ms: submission, timestamp resolve, the
   queued writes and the wake-up from the wait. Partly an artefact of waiting each frame.
5. **Game: first-frame stall**, 2.8 s at start-up (composition 1.5 s, asset upload 1.2 s, 379 MB).
   [mmorpg#726](https://github.com/moenarch-games/mmorpg/issues/726).
6. **Game: steady composition**, 1.34 ms, plus 0.26 ms of instance building. Stable and small.

Browser:

1. **GPU spikes** in `dense-woods`, `dense-hub`, `western-lake` and `overview` (GPU p95 31 to 255
   ms), new since `6ebc4c64`. Owner not yet known; [mmorpg#727](https://github.com/moenarch-games/mmorpg/issues/727)
   routes any three-lighting share back here.
2. **Sun cascades**: two thirds of the triangles (3.4 to 4.7 M of 5.7 to 7.9 M) and 237 to 348 of
   the 453 to 662 calls by pass. Game composition and three-lighting; mmorpg#615 (closed) did the
   draw-call work and found triangles alone do not limit the frame.
3. **Main-thread render**, 4.2 to 7.3 ms of a 8.2 to 12.7 ms CPU frame (p50). Down from 7.4 to 9.6
   ms at `6ebc4c64`.

Measured and not a bottleneck: game composition (1.34 ms native), simulation (1.5 to 2.1 ms browser
p50), animation (browser `units` under 1 ms; native skinned units upload one palette per unit),
GPU memory (718 MB of 8 GB), allocations (0 per steady frame).

## Visual comparison

The approved reference (owner decision on #74, 2026-11-11) is the native client's
`greyhaven_lighting_shots` at mmorpg `05a1e3eb`, render-engine `df5597b5`, 1280×720: the Outpost
at noon, dusk and night, a campfire and the lantern post by night, and the four
`outpost_frame_time` views at the zone's starting hour, 08:00. The captures, their poses and
hours (`views.json`) and native | browser | heat-map review images are in mmorpg
`docs/reviews/greyhaven-reference/`
([mmorpg#748](https://github.com/moenarch-games/mmorpg/pull/748)).

The browser at the same revision was captured at each shot's eye, target and hour by
`scripts/capture-browser-reference.py` (world canvas only, 1280×720, pixel ratio 1; Chromium
158.0.8127.18 headless, Playwright 1.65.0, ANGLE on OpenGL ES 3.2, the same RTX 3060 Ti), and
compared with the oracle's FLIP (`oracle --flip-images`, the gate's FLIP without its thresholds).
Recorded, not gated:

| Shot | Matched camera | Hour | FLIP mean | FLIP p99 |
| --- | --- | --- | --- | --- |
| `noon` | hub, orbit camera behind the player | 12.0 | 0.0594 | 0.3211 |
| `dusk` | same | 17.8 | 0.0663 | 0.2657 |
| `night` | same | 23.5 | 0.0984 | 0.2373 |
| `torch-campfire` | near the first point light | 22.0 | 0.0771 | 0.2399 |
| `torch-lantern` | near the last point light | 22.0 | 0.0955 | 0.2755 |
| `frame-hub` | `hub` | 8.0 | 0.0601 | 0.3496 |
| `frame-woods` | `woods` | 8.0 | 0.0537 | 0.5587 |
| `frame-vale` | `vale` | 8.0 | 0.0295 | 0.2084 |
| `frame-lake` | `lake` | 8.0 | 0.0697 | 0.6465 |

For scale, the oracle's parity gate is a mean of 0.005 and a p99 of 0.05. The poses match to
the pixel; what differs, from the heat maps:

- **Content, not shading.** The browser draws the Outpost hall's masonry texture where native
  draws it flat, and its default character wears different gear from the native test's. These dominate `noon` and `frame-hub`.
- **Night and torch shots** carry the highest means: the browser's whole frame sits at a slightly
  different level, and the warm point-light pool on the facades is weaker. The cause was the
  native reference's fog, not three-lighting (#102, below).
- **Foliage edges** in `frame-woods` and `frame-lake` give the highest p99s: alpha-masked
  leaves, tree level of detail and its cross-fades differ per tree. The lake's water and shore
  stones add to `frame-lake`.
- **`frame-vale`**, mostly terrain and sky, is the closest at a mean of 0.030.

No follow-up is filed from these numbers: no threshold was set, and the content differences
are game-owned and visible in the review images. The existing Bevy oracle gates are untouched.

### Night and torch shots (#102)

Both clients feed the renderer the same night: the EV100 curve (2 at the shots' hours), a 2 lx
moon, the sky's environment light at 0.9, TonyMcMapface, and each point light's lumens, colour
and range from `mmorpg_presentation::lights` (render-engine and three-lighting both divide by
4π for candela). The native test did not: `greyhaven_lighting_shots` set the shot's hour but kept
the fog colour of the zone's 08:00. Fog colour is added unexposed, so at night the daytime
grey-blue lifted the whole native frame, 14 % in mean luminance and 40 % in the distant top of
the frame, and the lamp pools looked brighter for it. The browser's `setPresentationTime` sets
the hour's fog. Game-owned and fixed in the test
([mmorpg#757](https://github.com/moenarch-games/mmorpg/issues/757),
[mmorpg#758](https://github.com/moenarch-games/mmorpg/pull/758)), which captures `noon`, `dusk`,
`night` and `torch-*` again at mmorpg `eb58c43b` (render-engine `df5597b5`, three-lighting
`81752f2`, the same machine and browser). three-lighting is unchanged.

| Shot | FLIP mean / p99, #74 | After mmorpg#757 |
| --- | --- | --- |
| `noon` | 0.0594 / 0.3211 | 0.0593 / 0.3211 |
| `dusk` | 0.0663 / 0.2657 | 0.0482 / 0.2629 |
| `night` | 0.0984 / 0.2373 | 0.0355 / 0.1994 |
| `torch-campfire` | 0.0771 / 0.2399 | 0.021 to 0.036 / 0.091 to 0.163 |
| `torch-lantern` | 0.0955 / 0.2755 | 0.037 to 0.076 / 0.078 to 0.202 |

The torch ranges are five browser captures against one native reference: the campfire and the
cave torches flicker, and the clients sample them at different moments (before the fix the same
captures gave 0.068 to 0.078 and 0.071 to 0.109). With the flicker held steady on both sides
(reduced motion) `torch-campfire` measures 0.0134 / 0.1066 and `torch-lantern` 0.0137 / 0.0422,
with mean luminance within 0.2 %; on `night` the lantern's pool on the ground is within 1 %. What
remains on `night` is the hall's masonry and the character's gear, which
[mmorpg#753](https://github.com/moenarch-games/mmorpg/issues/753) removes along with the flicker.
No oracle scenario is added: no shading difference was found.

## Not measured, with the blocker

| Item | Blocker |
| --- | --- |
| Native triangle counts | `PrepareWork` counts draws and instances, not indices. |
| Native display pacing | The path renders offscreen and waits each frame. A windowed run (`--frames`) prints no frame times. |
| A dense-unit Greyhaven camera, a timed native starter-area view | Not in any existing path: [mmorpg#724](https://github.com/moenarch-games/mmorpg/issues/724). |
| Images with clustered lights on Greyhaven | Not captured; part of [mmorpg#725](https://github.com/moenarch-games/mmorpg/issues/725). |
| Browser GPU spike cause | Needs a trace or a bisection: [mmorpg#727](https://github.com/moenarch-games/mmorpg/issues/727). |
| Native `perf`/CPU sampling | `kernel.perf_event_paranoid` is 4 on this machine. |

## Reproducing

```sh
git clone https://github.com/moenarch-games/mmorpg && cd mmorpg && git checkout 05a1e3eb
bash scripts/fetch-bundles.sh && bash scripts/prepare-ranger-native.sh
cargo test --release --locked -p mmorpg-client --lib outpost_frame_time -- --ignored --nocapture
(cd web && bun install --frozen-lockfile && bun run build)
python scripts/profile-browser.py --out artifacts/profile/report.json   # needs Playwright
```

The visual comparison: at `05a1e3eb` with mmorpg#748's test and script applied (their logic is
unchanged on mmorpg `main`), `MMORPG_LIGHTING_SHOTS=<dir> cargo test --release --locked -p
mmorpg-client --lib greyhaven_lighting_shots -- --ignored --nocapture`, then `python
scripts/capture-browser-reference.py docs/reviews/greyhaven-reference/views.json --out <web>`,
then per shot here `cargo run --release --locked --manifest-path oracle/Cargo.toml --
--flip-images <dir>/re9-<shot>.png <web>/<shot>.web.png <shot>.flip.png`.

Run the test binary (`target/release/deps/mmorpg_client-*`, arguments `outpost_frame_time --ignored
--nocapture`) under `/usr/bin/time -v` with `nvidia-smi pmon -s um -d 1` beside it for the process
figures; under `cargo test` they include cargo. The engine-option table needs, in
`FrameRenderer::new` after `Viewport::new`, `viewport.set_light_assignment(&renderer,
LightAssignment::Clustered(Default::default()))` or `viewport.set_culling(&renderer,
Culling::GpuIndirect)` (the latter with `INDIRECT_FIRST_INSTANCE` added to the client's
`required_features`).

Scope of this report: no renderer, shader, default or gate changed, and no new profiler or timing
ratchet was added. The oracle's `--flip-images` compares two images without any threshold.
