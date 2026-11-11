# WebGPU browser assessment (#75)

A *feasibility assessment*, not a migration. It asks whether render-engine's existing Rust/wgpu
code can render a documented oracle fixture in Chromium WebGPU, and whether a **separate, bounded
prototype** would be worth an issue. It does not replace three.js (D8), does not touch the
mmorpg's web client and does not promise broad browser support.

## Result: GO for a separate prototype issue, nothing beyond it

**GO** for a separate, bounded prototype issue (scope at the end; not filed here). **No** change to
any consumer, default or the D8 web path.

- `cargo check --target wasm32-unknown-unknown` passes on the unmodified crate. The smoke needed
  one 9-line engine change: a WebGPU validation fix (below).
- On the RTX 3060 Ti through Chromium 156 WebGPU, 14 captures (10 lighting scenarios, 4 particle
  captures) rendered by the crate compiled to wasm are within D3's gates against Bevy: FLIP
  mean ≤ 0.0009, p99 ≤ 0.0143 (the gates are 0.005 and 0.05). 10 of the 14 differ from native
  render-engine by at most 11 pixels; the other four (two sky views, the textured particles)
  are also within D3's gates against native.
- GPU time matches native on the same GPU (table below), `prepare` is about 1.7× native on the
  CPU-heavy scenario (`many_instances`, 4.3 vs 2.6 ms).
- Load size is the honest cost: 2.96 MB wasm (2.34 MB gzip, ~2.07 MB of it uncompressible LUT
  and fixture data) against 0.47 MB gzip for three.js plus `three-lighting` today.

What this does **not** show: Firefox or Safari, Chrome on a real window, mobile GPUs, skinned
meshes or `textured_props` drawing in the browser (their glTF fixtures need a file system), the
GPU-driven and clustered opt-in paths, or any real consumer's frame. D10's evidence and adoption
rows are not met by this result.

## Consumer policy and prior decisions

Read before any code, as the issue requires:

- [mmorpg#80](https://github.com/moenarch-games/mmorpg/issues/80) (closed not-planned,
  2026-10-17): MMORPG Pages stays on three.js. "A future browser-renderer migration requires a
  separate actual bottleneck, representative comparison and explicit game-owned approval, not a
  feature-count parity mandate."
- [3d-lab#89](https://github.com/moritzbrantner/3d-lab/issues/89) and
  [#90](https://github.com/moritzbrantner/3d-lab/issues/90) (closed not-planned, 2026-10-17): the
  broad parallel Rust/WASM backend and the competing-backend benchmark are superseded. "No engine
  migration or removal is implied"; "add performance measurements only for an actual candidate
  and representative consumer use".
- D10 and #70 permit a bounded render-engine-owned technical investigation. This assessment stays
  inside that: no game repository, no 3d-lab contract, no scene representation of its own and no
  benchmark corpus. The separate prototype below is what the policy asks a future step to justify,
  and [mmorpg#615](https://github.com/moenarch-games/mmorpg/issues/615) (Greyhaven bound by CPU draw
  submission in the browser) is the bottleneck any consumer pilot would have to test.

## Reproducing

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.130 --locked   # the crate's exact version
bun install                                                  # Playwright's headless Chromium
cargo run --release --locked --manifest-path oracle/Cargo.toml -- --only sun_shadows
bash scripts/webgpu-smoke.sh sun_shadows
```

The smoke is not built by `cargo build` at the repository root or by CI: `oracle/webgpu-smoke/`
has its own manifest and lockfile (the root workspace excludes `oracle/`), and no job runs it. It
compiles the oracle's `content.rs`, `scenarios.rs` and `particles.rs` by path, so the scene is the
oracle's, and it calls the same `Renderer::new`, `prepare` and `encode` a native consumer calls.
Run measured 2026-11-09 on the owner's desktop (Linux 7.0.0-42, RTX 3060 Ti, driver 595.99.02,
rustc 1.99.1), with other GPU clients open (Blender, a browser), on the working tree over commit
`dff98d8`. Timings are advisory for that reason, and each is a single process on one machine.

## Browsers and adapters here

Probed with `navigator.gpu.requestAdapter()` from a page served on `127.0.0.1` (a secure context).

| Browser | How launched | `navigator.gpu` | Adapter |
| --- | --- | --- | --- |
| Playwright headless shell (Chromium 156.0.8078.4) | no flags | present | `null` |
| same | `--enable-unsafe-webgpu --enable-features=Vulkan,WebGPU --use-vulkan=native --ignore-gpu-blocklist --enable-gpu` | present | **hardware**: `nvidia` / `ampere`, `isFallbackAdapter` false. The same adapter without `--enable-unsafe-webgpu` |
| same | above plus `--use-webgpu-adapter=swiftshader`, or `forceFallbackAdapter` | present | **software**: `google` / `swiftshader`, `isFallbackAdapter` true |
| Playwright Chromium 156 (new headless) | no flags, or the flags above | present | `null` |
| Google Chrome 157.0.8103.38 (new headless) | no flags, or the flags above | present | `null` |
| Firefox, Safari | not installed | not tested | not tested |

`adapter.info.description` and `device` are empty here, and wgpu's `AdapterInfo.name` is empty on
the WebGPU backend: a browser build can identify its GPU only as vendor and architecture.
Pointing `VK_ICD_FILENAMES` at lavapipe's ICD still gave SwiftShader in Chromium, so CI's software
adapter does not match Bevy's in a browser.

Headed Chrome on the desktop session was not tried. The "new headless" `null` is what this machine
exposes; it says nothing about Chrome on a real window.

## Inventory: engine use against browser WebGPU

Evidence is what ran on the hardware adapter unless a row says otherwise.

| Area | Engine use | Finding |
| --- | --- | --- |
| Crate build | `wgpu` default features, `pollster` | Compiles for `wasm32-unknown-unknown` unchanged. `Headless` blocks with `pollster` and polled `map_async`, so it is native-only. The consumer path (`Renderer::new`, `prepare`, `encode`) needs no blocking. |
| Surfaces | Consumer's colour target | `SurfaceTarget::Canvas` presented `sun_shadows` byte-identically to the offscreen run. The canvas here offers `rgba8unorm`, not an sRGB format, so the surface needs `view_formats: [Rgba8UnormSrgb]` and an sRGB view. |
| Texture formats | `Rgba16Float` HDR with MSAA, `Depth32Float`, `Rgba8UnormSrgb`, `Rgba32Float` sky LUTs | All validate and render. `float32-filterable` is present on both adapters. |
| Bind groups | Up to 4 groups, dynamic offsets | `maxBindGroups` is 4 on both adapters. All layouts validate. |
| Shaders | `lit.wgsl`, `tonemap.wgsl`, `sky/*`, particles | Dawn/Tint accepts them (24 lit, tonemap, sky and particle pipelines created), `enable dual_source_blending;` included. The one rejection is the sampler below. |
| Compute | Sky LUTs and environment filter, particle simulation | Run, and match native (`sky_day_cycle_*`, `particles_*` below). |
| Depth sampling | The shadow blocker search reads depth through a linear sampler | **WebGPU validation error** on every lit pipeline: `SamplerBindingType::Filtering` on a `Depth` texture. Natively legal and Bevy's choice. Fixed under `cfg!(target_arch = "wasm32")`: `NonFiltering` and nearest. The native build is unaffected. `point_spot_soft` differs from native in 9 pixels. |
| Timestamp queries | Pass-level spans | `timestamp-query` is present on both adapters. Medians with and without `--enable-webgpu-developer-features` agree within noise at these frame times, so Chromium's quantisation did not move them. |
| Limits | Requested `Limits::default()` | Sufficient for every tested path. NVIDIA: `maxStorageBuffersPerShaderStage` 16, `maxStorageTexturesPerShaderStage` 8, `maxTextureDimension2D` 16384. SwiftShader: 10, 4, 8192. |
| Particles | Compute simulation, `draw_indirect`, flipbooks | Run; captures below. `read_alive_particles` blocks on `map_async`, so it cannot work as-is on wasm. The renderer does not need it. |
| Skinning | Skinned pipeline variants | Created without error. **No skinned draw was run**: the oracle's skinned and textured scenarios read glTF from disk. Unverified. |
| Shadows | Cascades, spot, point cube, PCSS | Run; `depth-clip-control` present on both adapters. |
| Fallbacks | `DEPTH_CLIP_CONTROL`, `DUAL_SOURCE_BLENDING` optional | SwiftShader has no `dual-source-blending`; the sky pipeline used its alpha fallback there. |
| `Culling::GpuIndirect`, `LightAssignment::Clustered` | Opt-in at the time (clustered became the default later, #94; the captures here used the per-fragment loop) | `indirect-first-instance` is present on both adapters, 4 fragment storage buffers fit, but **neither path ran in a browser**. Their diagnostic readbacks block (`gpu_culling.rs`, `clustered.rs`) and would need async versions. |

## The smallest smoke

**Estimate before implementing** (written after the probe and the `wasm32` type check): a new
crate of 200–400 lines of Rust and about 100 of JS, **no engine changes**, and about 4–5 wasm→JS
calls per draw because wgpu's WebGPU backend crosses per API call; one wasm call per frame from
JS. Actual: `src/lib.rs` 456 lines, native `compare` 173, `main.js` + `run.mjs` 146; engine change
9 lines (the sampler, which the estimate missed); measured crossings about 6.5 per draw below. The
estimate was close on size and low on engine change and crossings.

It is a narrow change because the engine already separates a device-owned `Renderer` from
headless helpers: the smoke creates the device, calls `Renderer::new`, and reads back with
`map_async` awaited, as D7 already says the consumer does. The oracle sources are reused by path;
the only edit to them is making seven scenario functions `pub`.

## Images

FLIP as the oracle computes it (mean, 99th percentile, D3 gates 0.005 and 0.05), all on the RTX
3060 Ti. `differing` is pixels not byte-identical to native render-engine.

| Scenario | vs native engine (mean / p99 / differing) | WebGPU vs Bevy | WebGPU vs three.js | Native engine vs Bevy |
| --- | --- | --- | --- | --- |
| `sun_shadows` | 0.0000 / 0.0000 / 0 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| `sun_cascades` | 0.0000 / 0.0000 / 2 | 0.0000 / 0.0001 | 0.0000 / 0.0005 | 0.0000 / 0.0001 |
| `many_instances` | 0.0000 / 0.0000 / 1 | 0.0000 / 0.0007 | 0.0000 / 0.0009 | 0.0000 / 0.0007 |
| `tonemap_range_noon_tony` | 0.0000 / 0.0000 / 0 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| `ibl_materials_sun` | 0.0000 / 0.0000 / 11 | 0.0000 / 0.0000 | 0.0000 / 0.0002 | 0.0000 / 0.0000 |
| `fog_atmospheric_sun` | 0.0000 / 0.0000 / 0 | 0.0000 / 0.0004 | 0.0000 / 0.0005 | 0.0000 / 0.0004 |
| `point_spot` | 0.0000 / 0.0000 / 0 | 0.0000 / 0.0003 | 0.0000 / 0.0008 | 0.0000 / 0.0003 |
| `point_spot_soft` | 0.0000 / 0.0000 / 9 | 0.0000 / 0.0009 | 0.0001 / 0.0013 | 0.0000 / 0.0009 |
| `sky_day_cycle_noon` | 0.0008 / 0.0141 / 13813 | 0.0009 / 0.0143 | 0.0008 / 0.0142 | 0.0001 / 0.0025 |
| `sky_day_cycle_golden` | 0.0008 / 0.0097 / 12669 | 0.0008 / 0.0099 | 0.0008 / 0.0099 | 0.0001 / 0.0021 |
| `particles_fountain` t0500 / t2000 | 0.0000 / 0.0000 / 0 (both) | 0.0000 / 0.0000 | no port | 0.0000 / 0.0000 |
| `particles_textured` t0625 | 0.0000 / 0.0003 / 316 | 0.0001 / 0.0020 | no port | 0.0001 / 0.0019 |
| `particles_textured` t2500 | 0.0001 / 0.0045 / 1841 | 0.0001 / 0.0048 | no port | 0.0001 / 0.0048 |

Particle captures replay the engine's seed chain with Bevy's first spawn update (31), so the
native reference is the oracle's own engine capture. The FLIP gates are met everywhere. The sky's
12–14k differing pixels are not explained here; both views stay well inside the gates, and
whether the difference is Tint's arithmetic or something else is not determined.

On the software adapter (SwiftShader, different device from any native side, so advisory only):
`sun_shadows` 0.0048 / 0.0495 and `sky_day_cycle_noon` 0.0139 / 0.0825 against the NVIDIA native
engine. Do not read those as a browser-versus-native result.

## Load size and startup

| Artefact | Bytes | gzip -9 |
| --- | ---: | ---: |
| `webgpu_smoke_bg.wasm` | 2,960,224 | 2,336,570 |
| `webgpu_smoke.js` (glue) | 73,963 | 13,717 |
| Sum | 3,034,187 | 2,350,287 |
| `three.core.js` + `three.module.js` (0.187, unminified, what the oracle serves) | 2,133,699 | 419,212 |
| plus `packages/three-lighting/*.js` | 2,316,921 | 468,660 |

The wasm embeds about 2.07 MB of already-compressed data (the two engine LUTs, 0.37 MB, and the
IBL fixture, 1.69 MB, compiled in by the scenarios), so the engine's own code and its compressed
size are smaller than the table says. A browser build without those fixtures was not produced;
that it would be smaller than 1 MB of code is an estimate, not a measurement. `wasm-opt` is not
installed here and was not applied.

Startup, single process per scenario, hardware adapter, milliseconds from the page script's first
line:

| Stage | Range over the 10 lighting scenarios (last run) |
| --- | ---: |
| `init()` (wasm compile and instantiate) | 25 – 43 |
| Adapter + device | 361 – 446 |
| `Renderer::new` (pipelines) | 19 – 65 |
| Scenario upload | 2 – 63 |
| First frame including readback | 656 – 1352 |
| Time to first frame | 1095 – 1835 |

Earlier runs on the same build reached 150 ms for `init()` and 870 ms for the device, so read
these as magnitudes. Full harness wall time (browser launch to PNG, three runs each): the
three.js oracle page 0.87–1.21 s for `sun_shadows` and 0.98–1.63 s for `many_instances`; the
smoke 1.66–1.93 s and 1.76–1.80 s. The three.js number includes its own scene build and render,
so the difference is a difference of pages, not of renderers alone.

## GPU and CPU time

Advisory: a shared desktop GPU, three process runs per scenario and side. Native render-engine runs the smoke's
`compare native` protocol (the same calls and warm-up, `Headless::gpu_frame_times`); the Bevy and
oracle-engine columns are from the oracle's own GPU protocol (D3). Three.js is the oracle's
timer-query number over `Lighting.render` (D8: advisory). WebGPU is the median of three runs with
`--enable-webgpu-developer-features`, 300 frames each, RTX 3060 Ti.

| Scenario | WebGPU GPU ms | Native GPU ms | Native (oracle) | Bevy GPU ms | three.js GPU ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sun_shadows` | 0.136 | 0.116 | 0.115 | 0.216 | 0.123 |
| `many_instances` | 0.790 | 0.790 | 0.798 | 0.871 | 1.060 |
| `point_spot` | 0.293 | 0.277 | 0.277 | 1.037 | 0.404 |
| `sky_day_cycle_noon` | 0.653 | 0.615 | 0.612 | 1.602 | 1.442 |

The Bevy, oracle-engine and three.js columns are from the full local oracle run (every scenario,
both gates passed); a selected-scenario oracle run minutes earlier gave Bevy `many_instances`
2.73 ms and engine 0.80 ms, which shows how much these numbers move on this desktop.

| CPU, median of runs | `prepare` ms | `encode` ms | `submit` ms |
| --- | ---: | ---: | ---: |
| `sun_shadows` WebGPU / native | 0.015 / 0.019 | 0.045 / 0.075 | 0.010 / 0.025 |
| `many_instances` WebGPU / native | 4.335 / 2.611 | 0.125 / 0.171 | 0.020 / 0.057 |
| `sky_day_cycle_noon` WebGPU / native | 0.020 / 0.023 | 0.080 / 0.163 | 0.010 / 0.025 |
| `point_spot` WebGPU / native | 0.015 / 0.023 | 0.065 / 0.172 | 0.005 / 0.037 |

GPU time is at parity with native on the same GPU; the small `sun_shadows` difference is in the
noise of a 0.1 ms span. CPU `prepare` on the only CPU-heavy scene is about 1.7× native (wasm32
without SIMD or threads). `encode` and `submit` are *not* lower in any meaningful sense: native
`encode` includes wgpu's own recording, the browser's work lands inside Dawn on the GPU process
and is not on this clock. Three.js CPU cost was not measured (the oracle times only its GPU
frame), so no CPU comparison with three.js is claimed.

## JS/WASM crossings

Counted by wrapping the `GPU*` prototypes in the page, on the first frame including its one-time
resource creation; wgpu's WebGPU backend makes one call per API call.

| Scenario | All GPU API calls | Recurring per frame (encode, submit, `writeBuffer`) | Draws | Bytes uploaded (first frame) |
| --- | ---: | ---: | ---: | ---: |
| `sun_shadows` | 236 | 98 | 12 | 1,852 |
| `many_instances` | 586 | 455 | 67 | 2,982,292 |

Per draw that is 6–7 crossings on `many_instances`. The recurring byte count of a *later* frame
was not separated and is unmeasured here. Whether crossings of this order limit a real game frame
in a browser is exactly what [mmorpg#615](https://github.com/moenarch-games/mmorpg/issues/615)
would need to measure; this smoke does not show it either way.

## Where browser adapters live

- **Scene and simulation authority stay in the games** (D7). The engine sees a `World` and a
  `Scene` per frame, in wasm as natively; a browser build changes the device acquisition and
  the readback, not the data. Nothing in this smoke adds scene authority.
- **Browser adapters do not go in the default crate.** Canvas and surface setup, async device
  creation, canvas presentation and any JS boundary belong in a thin, separate crate owned by
  render-engine (`render-engine-web` or similar) with its own manifest, on the same terms as
  `oracle/`: never a default dependency, never a default feature. Today's smoke lives under
  `oracle/webgpu-smoke/`, because it is an oracle-side measurement.
- **`Headless` stays native.** An async wasm counterpart, if it is ever wanted, is the adapter
  crate's. The `Renderer` API already needs only a device and a queue.
- Blocking diagnostic readbacks (`read_alive_particles`, the GPU-culling and clustered diagnostics)
  need an async form before they can run on wasm; they are not on the render path.

## Not verified

- Any browser other than the Playwright headless shell on this machine: Firefox, Safari, Chrome
  stable (which returned no adapter under headless here), a headed window, mobile GPUs, and other
  desktop drivers.
- Skinned meshes and `textured_props` drawing in a browser (pipelines compile; no image).
- `Culling::GpuIndirect` and `LightAssignment::Clustered` in a browser.
- The cause of the sky's 12–14k differing pixels against native.
- CPU time of the three.js renderer, and any real consumer's frame or workload. D10's consumer
  and adoption evidence is not provided by this assessment.
- Device-loss recovery, canvas resize and `wasm-opt`.

## Scope for a separate prototype issue

Not filed: GO means the owner can decide whether to open it. It should cover, and only cover:

1. An adapter crate with its own manifest (above): canvas, device, the wasm entry; `Renderer`
   unchanged; no default dependency for native consumers.
2. Skinned and textured drawing in the browser: a file-free way to deliver glTF bytes to the
   existing `content` loader, then `skinned_figure` and `textured_props` against D3.
3. Asynchronous forms of the blocking diagnostics, and a decision on whether the D8 FLIP gate
   should run in a browser CI job (currently explicitly not).
4. A representative CPU-submission workload, because #615's bottleneck is draw submission and
   one 22 k-instance scene at 67 draws per frame does not settle it; consumer pilot and adoption
   stay subject to D10.
5. A real failure mode: what the build does with no WebGPU, a software adapter or lost device.

Prerequisite for it to be worth more than a demo: a consumer that reports a bottleneck in the
sense of mmorpg#80's closure. Without one, the result of this assessment is still "feasible, but
not needed yet".
