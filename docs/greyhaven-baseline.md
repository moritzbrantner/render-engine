# Greyhaven baseline (#74)

What render-engine costs on real Greyhaven frames, through the game's own native and browser
profiling paths. This is evidence, not a migration: nothing here changes the renderer, its
defaults or the Bevy oracle gates.

**Result.** The native client renders Greyhaven on render-engine. At 1280×720 on an RTX 3060 Ti a
frame takes 14 to 19 ms (median over runs), and about 90 % of it is inside the renderer boundary
(`prepare`, `encode`, submission and the wait for the GPU). The existing path cannot split that
further, so **no renderer-owned gap is measured yet and no render-engine issue is filed**. What
is measured: game-side composition is a steady 1.7 ms, GPU-skinned units upload only their joint
palettes after the first frame, the lake is the most expensive view on both renderers, and
browser sun cascades are about two thirds of its triangles. The missing native attribution is
filed as [mmorpg#692](https://github.com/moenarch-games/mmorpg/issues/692).

## Sources and setup

| | Native | Browser |
| --- | --- | --- |
| Game | `moenarch-games/mmorpg` `6ebc4c64` | same |
| Renderer | render-engine `bd4708dd` (`Cargo.lock`), 27 commits behind `dff98d8` (this PR's base). It predates `Viewport::work` (#65), `Culling::GpuIndirect` (#71) and `LightAssignment::Clustered` (#72). | three.js 0.186.1 and `@moritzbrantner/three-lighting` at `81752f2` |
| Path | `outpost_frame_time` (`crates/mmorpg-client/src/graphics.rs`), `imported_mesh_buffer_observation`, `greyhaven_lighting_shots` | `scripts/profile-browser.py` (`web/dist`, `?debug&profile`) |
| Build | `cargo test --release --locked`, rustc 1.99.1, wgpu 30.0.1 | `bun run build`; Chromium 156.0.8078.4 headless (Playwright 1.64.1), ANGLE on OpenGL ES 3.2 |
| Resolution | 1280×720 (skinned-unit path 640×360), offscreen, no present | 1440×900, device pixel ratio 1, on-page canvas |
| Scene settings | Zone hour from `ZONE_START_TIME` (08:00). Four sun cascades over 160 m, 2048 per cascade, 4× MSAA, Tony McMapface, Gaussian shadow filter | Same hour source and tone mapping, three sun cascades over 160 m (three.js redraws the scene per cascade) |
| Content | Greyhaven Vale, full scenery, NPCs in interest, one player | same |

Machine: AMD Ryzen 7 5700X, 16 logical CPUs, 30 GB RAM, NVIDIA GeForce RTX 3060 Ti (8 GB), driver
595.99.02, Linux 7.0.0-42. Measured 2026-11-08/09.

**The machine was shared.** Other agent sessions ran during these runs, with a 1-minute load
average between 5 and 33. The same view's native median varied by up to 1.8× between runs (`vale`:
12.23 to 22.47 ms), and every run's load is recorded below. Comparisons between views within a run
are the sturdier numbers: `lake` was the slowest view in 8 of the 9 runs.

### Matched cameras

Native and browser use the same viewpoint poses (`DEBUG_VIEWPOINTS` in
`web/src/world/world-view.ts` and the `views` in `outpost_frame_time`), field of view 50°.

| Issue camera | Native view | Browser scenario | Eye / target |
| --- | --- | --- | --- |
| Starter area | none of its own | `idle` (view after entry, on the spawn plaza) | not a debug viewpoint |
| Hub | `hub` | `dense-hub` | `[5, 8.5, 31]` / `[-14, 3, 4]` |
| Forest / vegetation | `woods` | `dense-woods` | `[-41, 18, 16]` / `[-69, 14, 0]` |
| Water | `lake` | `western-lake` | `[-56, 33, -39]` / `[-87, 11, -53]` |
| Whole Vale | `vale` | `overview` | `[0, 70, 95]` / `[0, 0, -10]` |
| Dense units | no crowd scene exists; see the skinned-unit path below | none | |

New characters spawn on the hub plaza, so `hub` is the closest native view to the starter area.
No existing path puts a crowd of units in a real Greyhaven camera on either renderer.

## Native measurements

`outpost_frame_time`, 9 runs. In each run each view renders 240 frames and the last 200 are kept.
`frame_ms` is one posed, submitted and completed frame; `render_ms` is the part of it inside
`SceneRenderer::render` plus the wait for the device. The test prints only the median and p95.

| View | `frame_ms` median (min / median / max of 9 runs) | `frame_ms` p95 (same) | `render_ms` median (same) | `render_ms` p95 (same) |
| --- | --- | --- | --- | --- |
| `hub` | 13.11 / 16.03 / 19.76 | 15.63 / 22.40 / 47.44 | 11.69 / 14.36 / 17.89 | 13.86 / 20.21 / 43.79 |
| `woods` | 15.04 / 17.59 / 21.80 | 16.69 / 21.98 / 26.19 | 13.67 / 15.90 / 19.79 | 15.16 / 19.97 / 23.99 |
| `vale` | 12.23 / 14.39 / 22.47 | 14.34 / 17.79 / 44.73 | 10.81 / 12.73 / 19.91 | 12.49 / 15.61 / 39.00 |
| `lake` | 18.24 / 19.10 / 22.29 | 20.13 / 21.38 / 28.23 | 16.74 / 17.41 / 20.66 | 18.44 / 19.20 / 24.73 |

Load average at the start of the runs, 1 to 9: 17.7, 11.2, 8.4, 6.2, 7.1, 5.9, 7.2, 6.6, 6.0.

`frame_ms` minus `render_ms` (median of each) is 1.66 to 1.69 ms on all four views: the game's
scene composition, with unit poses and the level-of-detail pick. It does not depend on the view.

Per process, over a whole run (which includes start-up and 960 frames):

| Metric | Value across runs | Source |
| --- | --- | --- |
| Wall time | 19.9 to 27.9 s | `time -v` |
| CPU user + system | 12.6 to 18.7 s | `time -v` |
| Peak resident memory | 1,539 to 1,567 MB | `time -v` |
| GPU memory, process | 718 MB, peak | `nvidia-smi pmon`, runs 2 to 9 |
| GPU busy, process | 28 to 41 % mean, 50 to 64 % peak | `nvidia-smi pmon`, runs 2 to 9, 1 s samples, including start-up |

The GPU busy samples say the frame loop is not GPU-saturated, so part of `render_ms` is likely CPU
work. They are coarse and include start-up, so that is a hint, not a split.

### Skinned units (real Ranger asset, not a Greyhaven scene)

`imported_mesh_buffer_observation` draws the pinned Ranger (13 primitives) with no scenery, at
640×360. Counts are deterministic; the times are one sample of one run.

| Units | First frame: buffers / bytes uploaded | Later frames: buffers / mesh bytes / palette uploads | Later frame, render through GPU complete |
| --- | --- | --- | --- |
| 1 | 13 / 2,147,256 | 0 / 0 / 1 | 2.6 ms |
| 10 | 13 / 2,147,256 | 0 / 0 / 10 | 3.2 ms |
| 50 | 13 / 2,147,256 | 0 / 0 / 50 | 5.7 ms |

GPU skinning removed the per-frame churn measured earlier
([#338 report](https://github.com/moenarch-games/mmorpg/tree/main/docs/reviews/native-mesh-allocation)):
later frames create no buffers and upload no mesh bytes, only one joint palette per unit. The very
first frame of the process took 342 ms to submit; a first frame on a later renderer in the same
process took about 40 ms. Those are the only startup numbers the native paths give, for this
units-only scene.

## Browser measurements

`profile-browser.py`, primary run with load average 5.4 at start and 7.4 at the end. Times are
per animation frame: `rAF` is the interval between frames (frame pacing, vsynced to 60 Hz); `CPU`
is main-thread time for the whole frame; `GPU` is `EXT_disjoint_timer_query_webgl2` for the frame's
commands.

| Scenario | fps | rAF p50 / p95 / p99 / max ms | CPU p50 / p95 / p99 ms | GPU p50 / p95 / p99 ms |
| --- | --- | --- | --- | --- |
| `idle` | 60.7 | 16.7 / 16.8 / 33.4 / 100.0 | 13.8 / 18.3 / 28.3 | 9.34 / 11.04 / 14.50 |
| `dense-woods` | 44.8 | 16.7 / 50.0 / 183.3 / 383.4 | 10.8 / 14.4 / 17.6 | 12.74 / 26.38 / 112.96 |
| `dense-hub` | 48.0 | 16.7 / 66.6 / 166.7 / 516.7 | 11.8 / 15.3 / 21.9 | 10.18 / 40.25 / 121.59 |
| `western-lake` | 51.5 | 16.7 / 33.4 / 99.9 / 283.4 | 12.8 / 17.4 / 21.5 | 14.86 / 28.03 / 42.55 |
| `overview` | 41.5 | 16.7 / 66.7 / 216.7 / 650.0 | 14.0 / 24.7 / 31.8 | 10.91 / 34.33 / 59.31 |

A second run, started at load average 33 and including the viewport sweep, gives GPU p50 of 9.61,
13.35, 10.59, 14.57 and 12.00 ms and CPU p50 of 14.0, 11.9, 12.8, 11.1 and 15.1 ms on the same
scenarios in the same order. The tails (p95 and up) are dominated by the load, so only the p50s
compare between the two runs.

Per frame, three's own count per pass (`drawsByPass`), mean over the same scenario (these counts
include `BatchedMesh` multi-draws, which the wrapped GL `drawCallsPerFrame` does not; see
[mmorpg#692](https://github.com/moenarch-games/mmorpg/issues/692)):

| Scenario | Calls: all / sun cascades / main opaque | Triangles: all / sun cascades / main opaque | Instance batches |
| --- | --- | --- | --- |
| `idle` | 500 / 254 / 168 | 5.28 M / 3.94 M / 1.34 M | 2,493 |
| `dense-woods` | 395 / 193 / 124 | 5.42 M / 3.72 M / 1.70 M | 2,493 |
| `dense-hub` | 524 / 230 / 217 | 5.95 M / 3.89 M / 2.06 M | 2,493 |
| `western-lake` | 345 / 159 / 109 | 4.63 M / 3.20 M / 1.43 M | 2,493 |
| `overview` | 488 / 153 / 256 | 4.25 M / 2.66 M / 1.59 M | 2,493 |

Main-thread phases, p50 in ms (`?profile`), range over the five scenarios: `render` 7.4 to 9.6,
`simulation` 1.5 to 1.6, `scene` 0.8 to 0.9, `units` 0.5 to 0.9, `projection` 0.3 to 0.4, `hud` 0.3
to 0.4. No frame in these scenarios created resources (`framesCreatingResources` 0).

Startup: 5.5 to 5.7 s from navigation to an enabled Enter button (`coldSelectionSeconds`), 3.4 to
3.6 s for the Enter World key to a playable view (`enterWorldKey.wallSeconds`). Not comparable to
the native start-up: the browser downloads and compiles the bundle.

The 1280×720 row of the viewport sweep (second run) gives `idle` GPU p50 of 9.08 ms, the only
browser number at the native resolution.

### Not comparable between native and browser

- **Resolution and cascades**: 1280×720 with four cascades, against 1440×900 with three.
- **`render_ms` and browser `render`**: native is `prepare`, `encode`, submission and the wait for
  the GPU; browser is main-thread three.js and WebGL submission only, and the GPU time is separate.
- **Pacing**: the native test renders offscreen as fast as it can, with no present and no vsync;
  the browser is vsynced to 60 Hz and rAF intervals quantize to it. The browser's rAF p50 of 16.7
  ms is vsync, not cost.
- **Renderer revision**: the two pin different render-engine commits (`bd4708dd` native, `81752f2`
  for the three.js port).
- **Draws and instances**: native counts are not observable; the browser's come from three.

## Bottlenecks, by owning boundary

1. **Renderer boundary, native (render-engine `prepare`/`encode` and GPU, unsplit).** 12.7 to 17.4 ms
   of the 14.4 to 19.1 ms median frame. The largest measured block. Blocker: the client passes no
   timestamps and the pinned engine has no work counters, so nothing inside it can be attributed
   (`mmorpg#692`). Until that runs, no renderer gap can be named, and neither Forward+ nor
   GPU-driven culling can be judged on this content.
2. **Water view.** The lake is the most expensive view on both: native `render_ms` median 17.41 ms
   against 12.73 to 15.90 ms for the others, browser GPU p50 14.86 ms against 10.18 to 12.74 ms on
   the other three views. The browser draws the fewest calls there (345, against 395 to 524), and
   the overview draws fewer triangles (4.25 M against 4.63 M) yet costs 3.95 ms less GPU, so the
   cost looks like shading or overdraw rather than geometry. The owner cannot be named without
   pass timing (renderer, or game water material).
3. **Browser sun cascades.** About two thirds of all triangles (2.66 to 3.94 M of 4.25 to 5.95 M)
   and 153 to 254 of the 345 to 524 calls in every scenario. Game scene composition and
   `three-lighting`, which redraws the scene per cascade. mmorpg#615 owns the draw-call work here
   and found triangles alone do not limit the frame, so this report files nothing new.
4. **Browser main-thread render.** 7.4 to 9.6 ms of a 10.8 to 14.0 ms CPU frame (p50). Owned by
   mmorpg#615.
5. **Asset preparation and loading.** Native first-frame cost is visible only for the Ranger-only
   scene (342 ms at process start, 40 ms after). The start-up of a full Greyhaven scene is not
   measured (below).

Measured and not a bottleneck: game composition (1.7 ms native, stable), simulation (1.6 ms browser
p50), animation (browser `units` under 1 ms; native skinned units upload one palette per unit).

## Visual comparison

There is no designated approved reference set. The closest are the merged review images in
`docs/reviews/web-lighting/native-vs-web-{noon,dusk,night,woods-mid}.jpg` (mmorpg#610). They are
not formally approved, they were taken from an older game and render-engine, and the Greyhaven
content has moved on since.

`greyhaven_lighting_shots` at the current pin was viewed against them for noon, night and
woods-mid. The camera poses no longer match (the current `noon`/`night` shots are the
`OrbitCamera` default at spawn, the reviewed ones a different spot), and the woods have new tree
kits and ground. Sky, exposure, sun and shadow character look the same as the reviewed native
images; the night shot's lamp pool is present in both. This is a visual review of a few frames
only, not a measurement. No FLIP or numeric image comparison was run. Numeric comparison needs an
approved capture set first.

The existing Bevy oracle gates are untouched: this PR changes docs only.

## Not measured, with the blocker

| Item | Blocker |
| --- | --- |
| Native CPU `prepare`, `encode` | `outpost_frame_time` times the whole `SceneRenderer::render`. |
| Native GPU pass times | The client requests no `TIMESTAMP_QUERY` and passes no timestamps to `encode`. |
| Native instance, draw and triangle counts | `Viewport::work` is not in the pinned engine (`bd4708dd`); the test prints none of them. |
| Native uploads and allocations on a full Greyhaven frame | Only the units-only path above counts them. |
| Native frame p99 | The test prints the median and p95 only. |
| Native frame pacing | The path renders offscreen. A real window run (`--frames`) has no frame-time output. |
| Native startup and loading stutter on Greyhaven | The test drops the first 40 frames of each view and prints no load time. |
| A dense-unit Greyhaven camera (either renderer) | No existing path spawns a crowd in a real scene. |
| Starter-area native view | No timed native path uses the spawn camera (the shots use it, untimed). |
| Browser `move-rotate` Chromium trace | Chromium was OOM-killed at about 7.1 GB with `--trace`; the two reported runs ran without it. |
| Native `perf`/CPU sampling | `kernel.perf_event_paranoid` is 4 on this machine. |
| A quiet machine | Shared with other sessions; see the load figures. |

## Reproducing

```sh
git clone https://github.com/moenarch-games/mmorpg && cd mmorpg && git checkout 6ebc4c64
bash scripts/fetch-bundles.sh && bash scripts/prepare-ranger-native.sh
cargo test --release --locked -p mmorpg-client --lib outpost_frame_time -- --ignored --nocapture
cargo test --release --locked -p mmorpg-client --lib imported_mesh_buffer_observation -- --ignored --nocapture
MMORPG_LIGHTING_SHOTS=/some/dir cargo test --release --locked -p mmorpg-client --lib greyhaven_lighting_shots -- --ignored --nocapture
(cd web && bun install --frozen-lockfile && bun run build)
python scripts/profile-browser.py --out artifacts/profile/report.json   # needs Playwright; omit --trace on a machine that can OOM
```

Wrap the native command in `/usr/bin/time -v` and run `nvidia-smi pmon -s um -d 1` beside it for
the memory and GPU-busy figures.

Scope of this report: no renderer, shader, default or gate changed, and no new profiler or timing
ratchet was added.
