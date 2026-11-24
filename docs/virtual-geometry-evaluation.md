# Virtual geometry evaluation (#119)

## Runtime evaluation

The native comparison also covers all five eligible crag outcrop assets at their
actual Greyhaven dressing placements, rotations and nonuniform scales, following
240 interpolated poses between the consumer's saved Vale and lake review cameras.
See `benchmarks/virtual-geometry/greyhaven-native-summary.json` and the five
`greyhaven-*.json.gz` raw reports. The path is reproducible consumer-data input;
terrain, units and the game loop are excluded, so this is a scenery-subset A/B,
not a candidate whole-game frame-time result. The three actual buttress-1 placements
were visible in 185/240 source frames (181/240 cluster frames, whose bounds cull more
tightly). In round 1 the source CPU/GPU medians were 0.109/0.109 ms versus
0.200/0.101 ms at 1 px; the small GPU reduction does not offset added CPU work.

Generate input with `python3 scripts/virtual-geometry-game-scene.py MMORPG_CHECKOUT
crag.kit.outcrop-buttress-1 target/virtual-geometry/greyhaven-buttress-scene.json`,
then prefix the native benchmark command with
`VG_SCENE=target/virtual-geometry/greyhaven-buttress-scene.json`. The JSON records
the source placement identities, transforms, camera frames and input hashes.
Projected cluster errors use world-space bounds and the largest TRS scale axis,
so nonuniformly scaled placements do not understate their geometric error.

The native CPU-cut prototype is now available through `kit_showcase
--virtual-geometry KIT HIERARCHY ASSET OUTPUT`. It reuses the kit's KTX2 materials,
restores source normals, tangents and vertex colors, and submits off-camera instances
for independent renderer shadow-view culling. The browser prototype under
`oracle/virtual-geometry/` uses public Three.js `BatchedMesh` APIs. Neither is a
production virtual-geometry API. Both keep the full hierarchy resident.

Native evidence is in [the runtime summary](benchmarks/virtual-geometry/native-runtime-summary.json)
and adjacent `native-*.json.gz` raw reports. These are RTX 3060 Ti Vulkan measurements,
1280×720, 240 moving-camera frames after warmup, two reverse-order rounds, release
builds. Repeated asset grids are scaling workloads, not gameplay. CPU spans include
selection, prepare, encode and submission; GPU values sum timestamp pass spans.
The machine was shared, so timing is advisory and raw round variation is retained.

Conventional candidates were generated with the existing 3d-lab `lod_chain` adapter
using `scripts/bench-discrete-lods.py`; no new simplifier was implemented. Its current
checkout failed a locked build because its lockfile needs updating. The benchmark
used the same adapter source from a temporary manifest and recorded its output;
the foundation checkout and lockfile were left unchanged. This runner resolved
meshopt 0.7.0; the older source identifier string is not its dependency version.

At 256 synthetic boulders, round 1 measured the 1 px cluster cut at CPU 1.857 ms /
GPU 0.575 ms and discrete LOD4 at CPU 0.396 ms / GPU 0.538 ms. Four-capture mean
sRGB RMSE against the full source is 0.01992 / 0.01999 respectively. At 256 shrine
arches, discrete LOD2 measured CPU 0.124 ms / GPU 0.666 ms, with RMSE 0.00436;
the 1 px cluster cut measured CPU 0.974 ms / GPU 0.270 ms, with RMSE 0.01370.
RMSE is an image difference diagnostic, not a perceptual acceptance threshold.
Do not add CPU and GPU spans and call that end-to-end gameplay latency.

These comparisons favor conventional LODs for the tested candidates. Actual gameplay
full-frame candidate integration and multi-material validation were not established;
they must not be inferred from these grids. No consumer adoption is justified yet.
The independent `verify-cut-boundaries.mjs` diagnostic found every sampled synthetic
boulder cut closed after positional welding. Crag/arch source meshes have open
boundaries; their diagnostic does not establish watertight transitions.

## Decision and status

Retain conventional geometry and discrete LODs. Do not adopt `three-nanite` as a
runtime dependency. Cluster LOD remains an evaluation candidate, **not a rejected
technique**: evidence now includes preprocessing, sampled cuts and native/browser asset grids.
It does not establish an equal-quality whole-game performance advantage. No production
virtual-geometry API or game migration is shipped: adoption was conditional on a
meaningful benefit that the evaluated candidate did not establish.

Useful candidates are Meshoptimizer meshlet partitioning, locked group boundaries,
monotonic conservative error, and per-view projected-error cuts. Any implementation
belongs with the existing owners below. GPU selection, streaming and HZB require
representative runtime evidence first. No consumer pins or production paths change.

## Capability and ownership inventory

| Capability | Existing owner and evidence | Implication |
| --- | --- | --- |
| Simplification and discrete LOD generation | `3d-lab/crates/three-d-lod/src/lib.rs`: Meshoptimizer through `meshopt` (see the version note above), `simplify_mesh`, `build_lod_chain`, border locking | Extend this owner if cluster preprocessing proves useful; do not create another simplifier |
| Renderer-independent LOD selection | `three-d-lod/src/selection.rs`, `docs/contracts/screen-space-lod.md` | Reuse shared projected-error semantics; cluster DAGs are not currently implemented there |
| Asset operations and packaging | asset-tooling `src/processing-operations.ts`: `mesh.simplify`, `mesh.lod_chain`; glTF-Transform exports with shared buffers and KTX2 | Packaging remains asset-tooling's job; textures and collision must stay independent of render cuts |
| GPU resources, instancing, batching, per-view frustum culling | render-engine `World`, `renderer/prepare.rs`, `renderer/retained.rs` | Keep resources and drawing here; no asset loading or scene ownership |
| GPU culling and indirect draws | Opt-in `Culling::GpuIndirect`, [experiment](gpu-driven-experiment.md) | Existing CPU default; benefit limited to many instances/few runs, regression with many runs |
| Occlusion | [Hi-Z negative experiment](occlusion-experiment.md) | No crossover against no-prepass rendering in tested workloads; do not repeat equivalent work |
| Scenery LOD and lifetime | mmorpg `docs/NATIVE_CLIENT.md`: shared placement LOD, hysteresis, cross-fades, retained instances, terrain/static-group streaming and eviction | Geometry streaming would supplement existing streaming; never replace collision with render LOD |
| Native/browser boundary | mmorpg native uses render-engine; browser uses Three.js. Shooter uses render-engine for native and wasm | A Three.js adapter is an MMORPG requirement, not a reason to add Three.js to native geometry processing |
| Cluster DAG/compressed cluster streaming | Not found in the inspected owned source | New capability, not an existing consumer API to switch on |

Meshoptimizer use does not imply that geometry compression is enabled throughout
the consumers. These sampled kit buffers are ordinary uncompressed vertex/index
buffers; KTX2 compresses textures. No universal compression claim is made.

Inspected open work on 2026-11-16: render-engine #116 (WebGPU small-upload failure),
3d-lab #180 (real-asset screen-space LOD), asset-tooling PRs #660/#666 (animals and
landmarks), MMORPG #683 (art/collision separation), Shooter #53/#55/#60 (arenas and
destruction), PR #84 (robots), PR #98 (CI). None of the listed PRs implements cluster
LOD. Preserve their scope; robot articulation and destruction are outside this
static-mesh experiment. Zoo’s architecture and open performance work (#123) were inspected: primitive batching is established, and no dense-static-geometry bottleneck was demonstrated. No Zoo integration is justified here.

## Reference findings

Inspected [three-nanite](https://github.com/apekshik/three-nanite) at
`b730aac13ab0ea67b75214613eea56c14151f6e5`: `src/build.js`, `src/clod.js`,
`src/inspect.js`, README and [its report](https://github.com/apekshik/three-nanite/blob/b730aac13ab0ea67b75214613eea56c14151f6e5/docs/REPORT.md).

The builder follows `clusterlod.h` in JavaScript using Meshoptimizer WASM, with
custom grouping because the JS interface lacks the native partitioner. It locks
group boundary vertices, simplifies, reclusters, and grows errors/bounds upward.
The runtime supports CPU selection and a scene triangle budget. This inspection
did not independently validate the native upstream `clusterlod.h` implementation.

Direct adoption conflicts with our requirements:

- `ClodBatch` fills private Three.js batching fields; there is no stable adapter boundary.
- Shadows reuse the camera cut, omitting off-camera casters.
- Only one material/UV set survives import; tangents and vertex colors do not.
- Streaming grows whole-level residency, with no eviction or compressed decoder.
- Threshold changes swap geometry without morphing; coarse simplification uses a
  sloppy fallback. Topological crack checks do not prove acceptable shading/popping.
- Stale-depth instance occlusion can reveal objects late. It is not our proposed HZB path.

Its published GPU numbers are **upstream results**, not measurements of these games
or this RTX 3060 Ti. Its report favors decimation/discrete LODs for its target game;
we already have those operations, so that alone justifies no new implementation.

## Measured offline experiment

Raw results: [crag](benchmarks/virtual-geometry/crag.json) and
[arena](benchmarks/virtual-geometry/arena.json). Each records the reference and engine
revision, CPU/runtime, input geometry hashes, three raw build samples, buffer sizes,
existing LOD counts, sampled cut triangle counts and verifier results.

19 crag defaults and 28 arena defaults were tested, including the crag kit's 2-triangle
macro surface. These are real published assets, not synthetic dense replacements.
The largest crag default has 8,999 triangles; the largest arena default has 648.

All 47 assets passed upstream's seven-camera, five-threshold open-edge and error
monotonicity checks. The verifier compares **counts** of open edges against the
source: this is a useful sampled check, not proof of matching boundary locations,
complete DAG coverage, watertight transitions, or runtime selection correctness.

Total geometry buffer bytes across each default asset set:

| Set | Source triangles | Existing default buffers | Cluster hierarchy without discrete LODs |
| --- | ---: | ---: | ---: |
| Crag | 63,621 | 2,867,028 | 3,114,952 |
| Arena | 7,630 | 806,904 | 496,012 |

These are measured serialized buffer lengths, **not** GPU residency or download
sizes. The reference repacks float normals to signed bytes and changes vertex/index
layout. Arena byte reduction therefore does not isolate a cluster-LOD benefit.
The current shipped LOD buffers, textures, manifests, CPU copies and driver
allocations are excluded from this table.

The 8,999-triangle buttress builds in a median 42.65 ms here; its hierarchy is
534,652 bytes against 449,644 default geometry bytes. Existing LOD1/LOD2 have
2,727/755 triangles. Those triangle counts do not establish comparable quality:
no pixel-error equivalence between authored LODs and reference cuts was measured.

## Reproduction

Use a disposable reference checkout and acquire its dependencies separately:

```sh
git clone https://github.com/apekshik/three-nanite.git /tmp/render-engine-three-nanite
git -C /tmp/render-engine-three-nanite checkout b730aac13ab0ea67b75214613eea56c14151f6e5
(cd /tmp/render-engine-three-nanite && npm ci --ignore-scripts --no-audit --no-fund)
mkdir -p target/virtual-geometry/inputs/crag target/virtual-geometry/inputs/arena
gh release download crag-kit -R moritzbrantner/asset-tooling --pattern crag.kit.tar --dir target/virtual-geometry/inputs/crag
gh release download shooter-arena -R moritzbrantner/asset-tooling --pattern shooter-arena.kit.tar --dir target/virtual-geometry/inputs/arena
sha256sum target/virtual-geometry/inputs/*/*.tar
tar xf target/virtual-geometry/inputs/crag/crag.kit.tar -C target/virtual-geometry/inputs/crag
tar xf target/virtual-geometry/inputs/arena/shooter-arena.kit.tar -C target/virtual-geometry/inputs/arena
bun scripts/bench-virtual-geometry.mjs /tmp/render-engine-three-nanite target/virtual-geometry/inputs/crag target/virtual-geometry/crag
bun scripts/bench-virtual-geometry.mjs /tmp/render-engine-three-nanite target/virtual-geometry/inputs/arena target/virtual-geometry/arena
```

Rolling release tar SHA-256 values used:

- Crag: `f3419e2746e8daf753c556eb3143843a0f9844f85f437d2edd46fb58062eab7c`
- Arena: `5c32c82ad5da67527602f5a6ac1a7dfd390a6fb2dc935815b08065bb1ad1bb69`

If releases change, record a new experiment rather than label it a reproduction.
The reference's npm lockfile is used only inside its checkout; no production
dependency or second package-manager lockfile is added here.

The script packs source geometry into GLB without modifying vertices/indices or
scene transforms, removing materials because upstream cannot import KTX2. It warms
each build once then measures three builds, including builder serialization and
its codec-size experiment, excluding input packing and verification. Runs use Bun
with upstream's WASM dependency; they are not Rust release-build runtime timings.
Cut statistics use original asset size, 1080p, 60-degree vertical FOV and the
reference's fixed diagonal distance samples; these are not a gameplay camera path.

## Runtime reproduction commands

After preparing the crag hierarchy above, run the native and browser adapters
serially; parallel GPU workloads would contaminate the timing:

```sh
cargo build --release --locked --example kit_showcase
WGPU_ADAPTER_NAME='RTX 3060 Ti' target/release/examples/kit_showcase --virtual-geometry target/virtual-geometry/inputs/crag target/virtual-geometry/crag/crag.kit.outcrop-buttress-1 crag.kit.outcrop-buttress-1 target/virtual-geometry/reproduce-crag
bun scripts/virtual-geometry-stress.mjs /tmp/render-engine-three-nanite target/virtual-geometry
# LOD_CHAIN_BINARY is the existing 3d-lab asset-tooling-lod-adapter lod_chain binary.
python3 scripts/bench-discrete-lods.py "$LOD_CHAIN_BINARY" target/virtual-geometry/inputs/stress/scenes/boulder/default.gltf target/virtual-geometry/inputs/stress-discrete
WGPU_ADAPTER_NAME='RTX 3060 Ti' target/release/examples/kit_showcase --virtual-geometry target/virtual-geometry/inputs/stress-discrete target/virtual-geometry/stress/boulder boulder target/virtual-geometry/reproduce-stress
bun install
mkdir -p target/virtual-geometry/basis
cp node_modules/three/examples/jsm/libs/basis/basis_transcoder.* target/virtual-geometry/basis/
# In a separate terminal, serve the repository root for actual browser asset loading:
python3 -m http.server 8000 --bind 127.0.0.1
# In the benchmark terminal:
node oracle/virtual-geometry/run.mjs http://127.0.0.1:8000 crag crag.kit.outcrop-buttress-1 target/virtual-geometry/reproduce-browser
```

Build `lod_chain` with `cargo build --release --manifest-path
examples/asset-tooling-lod-adapter/Cargo.toml --bin lod_chain` in a disposable
3d-lab checkout at the recorded revision. The checked-in lockfile did not build
unchanged in this environment; the measured temporary runner lockfile is identified
by hash, but was not retained. Its exact dependency resolution cannot be recovered
from that hash alone. Preserve the new lockfile and observations when rerunning;
compare dependency versions and geometry outputs rather than claiming bit-for-bit
reproduction of that temporary executable.

The exact disposable MMORPG profiler changes are retained in
[the profiler patch](benchmarks/virtual-geometry/mmorpg-browser-profiler.patch).
Apply it to a disposable copy at the recorded consumer revision, update its two
absolute workspace paths, and run its existing Python environment with
`--seconds 2 --no-sweep --out OUTPUT_JSON`. This preserves the hardware flags,
deadlines and omitted input-acceptance journeys used for the recorded baseline.

## Adoption boundary

The current prototype does not meet the adoption criterion. Retain conventional
rendering and discrete LODs for the evaluated games. Do not build GPU selection,
streaming/eviction or another HZB implementation merely to complete a feature list.
A future candidate needs a representative dense consumer workload, complete
material attributes and per-view shadow acceptance, and an equal-quality whole-game
comparison before it can replace any production path. See the technical decision
in [ADR 0001](adr/0001-virtual-geometry-evaluation.md).

This machine exposes only the RTX 3060 Ti graphics adapter. Integrated and mobile
hardware were unavailable; no hardware claim is made for them. Whole-game candidate
integration, arbitrary multi-material assets, crack-free transitions for open meshes,
GPU selection and geometry streaming are not validated. The supplied whole-game
captures/profile runs are conventional-renderer baselines. These limits are reasons
to withhold adoption, not successful feature checks or proof that all future cluster
implementations would lose.

Shared conventions resolved at `15f348b072188255da837578eaafc8a98f12fdf0`.


## Completed browser and gameplay measurements

All 48 arch browser runs finished on the RTX 3060 Ti through ANGLE/Vulkan,
with GPU timers and public Three.js batching. Raw samples are in
`benchmarks/virtual-geometry/browser-arch-discrete.json.gz`. Interrupted invocations
were resumed on the same reported environment; loading entries describe only the
latest page load. GPU variability and differing image errors preclude calling the
cluster GPU reduction an equal-quality gameplay win.

The real MMORPG native Vale profile passed on checkout
`65b5b62f802339bd8cfb53ab008608493eb209ee`, using its pinned renderer, not this
prototype. See `benchmarks/virtual-geometry/mmorpg-vale-native.txt`. The 80-frame run
excludes its warmup in the summary: GPU median/p95 4.04/4.59 ms, prepare 1.23/1.39 ms,
encode 0.39/0.58 ms. Its wall frame includes a GPU wait and must not be compared with
the prototype's CPU submission span. This is a baseline, not a candidate integration.

Reproduce with `MMORPG_PROFILE_VIEW=vale MMORPG_PROFILE_FRAMES=80
WGPU_ADAPTER_NAME='RTX 3060 Ti' cargo test --release --locked -p mmorpg-client --lib
outpost_frame_time -- --ignored --nocapture` in the MMORPG checkout.

`virtual-geometry-footprint.py` reports referenced file lengths and logical native
vertex/index buffer bytes in the adjacent arch/stress footprint JSON files. Those
are not measurements of RSS, driver residency or compressed HTTP transfers. The
prototype retains all levels and needs source attributes as well as the reference
hierarchy. The reports therefore include those extra inputs; no loading or memory
improvement is claimed. The synthetic source is explicitly a stress control, not
an unoptimized substitute for a game asset.


## Gameplay baseline evidence

The completed MMORPG browser profile is
`benchmarks/virtual-geometry/mmorpg-browser-profile.json.gz`. The game page reported
NVIDIA RTX 3060 Ti through ANGLE/Vulkan, 1440×900, with no page errors. Idle CPU frame
median/p95 was 24.5/29.7 ms; the overview was 16.0/18.4 ms. GPU timer sample counts
were zero, so browser gameplay GPU time is unavailable. Submitted work includes
all views: the overview recorded about 16,641 draws and 5.97 million triangles per
frame. These are profiling-baseline counters, not evidence that more detailed
geometry is free or that cluster rendering improves this game.

The disposable profiler copy used ordinary button entry, omitted unrelated keyboard
latency journeys, added frame-wait deadlines and longer startup/action deadlines,
and selected ANGLE/Vulkan. Visual inspection caught an entry Escape action that opened
settings in the earlier capture; the retained report is the replacement run without
that action. The gameplay capture uses the Vale debug camera with settings closed.
The original profiler's OpenGL launch configuration
selected SwiftShader in a matching environment probe. Its incomplete software runs
were discarded. No game source or production configuration changed. The existing
`web/dist` artifact is identified by its index hash in the saved report; it is not
claimed to be rebuilt from the current checkout.

Shooter's actual Assembly Works arena completed 300 native scripted frames on the
RTX 3060 Ti: wall step median/p95 1.63/1.87 ms, including the first frame (no warmup
exclusion). This is not an isolated GPU time. See
`benchmarks/virtual-geometry/shooter-assembly-native.txt`. The existing browser build
also completed its 120-frame Assembly Works journey on an NVIDIA Ampere WebGPU
adapter, with no logged errors: `shooter-assembly-browser.json`. Its reported work
median/p95 is 1.3/2.3 ms; startup frames remain in that distribution. These are
baselines, not equal-quality virtual-geometry A/B results.

Shooter used checkout `eacb061cab622a32ea86120a3b4d68dbad8efbbf`. Its rolling robot
release no longer matches the pin, but all four cached `web/dist/*-kit/kit.tar`
files and their manifests matched the existing consumer pins and were unpacked
into ignored native target directories. No pins were changed.

Gameplay captures: [MMORPG](benchmarks/virtual-geometry/images/mmorpg-gameplay.png),
[Shooter](benchmarks/virtual-geometry/images/shooter-assembly.png),
[Shooter scripted motion](benchmarks/virtual-geometry/images/shooter-assembly.mp4).
The clip plays every 30th simulation frame at 0.5-second intervals; it illustrates
motion, not frame pacing. No candidate gameplay comparison is claimed.


## Measured loading and residency

`bench-geometry-loading.py` launches separate native processes for source geometry,
source plus discrete LODs, and source fallback plus cluster geometry. Each run waits
for uploads, renders 16 warm frames, then samples Linux RSS/HWM and NVIDIA's
per-process GPU memory while the child still holds its resources. Three rounds
alternate order. Filesystem caches are warm; this does not measure cold-boot disk
access or network throughput. Raw rows: `arch-loading.json`, `stress-loading.json`.

Median local load spans (device creation and warm rendering excluded): the arch
source/discrete/cluster paths took 53.91/84.69/75.88 ms. The synthetic boulder paths
took 21.98/79.94/572.78 ms. The cluster path restores missing source attributes and
uploads every cluster once; that repair cost belongs to this prototype, not to
cluster LOD as an algorithm. Median boulder process RSS was 317,772/317,928/407,108
KiB respectively. RSS includes allocator-retained temporary memory and driver state;
it is not live geometry heap size. Arch RSS varied enough that no memory win is
claimed. NVIDIA reported 461 MiB per process for every mode, so its process-level
allocation did not distinguish the mesh buffer differences. Logical buffer sizes
are separately checked against actual renderer upload counters.

Before the final dense loading run, `trim-virtual-geometry-lods.mjs` removed the
reference builder's appended discrete-LOD payload from the cluster binary. The
standalone conventional files remain. Repeating the trim is a no-op, including
file mtimes; logical cluster GPU buffer bytes and the closed-cut diagnostic remain
unchanged. The corrected cluster prototype's referenced-file total is 17,764,075
bytes versus 10,273,562 for source plus conventional LODs. These are file lengths,
not a measured compressed HTTP download. The earlier frame timing compares the same
rendered cluster geometry and excludes loading, so it is not invalidated by trimming
unused file payload.

Reproduce after the existing kit/hierarchy preparation and release build:

```sh
python3 scripts/bench-geometry-loading.py target/release/examples/kit_showcase target/virtual-geometry/inputs/arch-discrete target/virtual-geometry/shrine/shrine.kit.arch shrine.kit.arch target/virtual-geometry/arch-loading.json
bun scripts/trim-virtual-geometry-lods.mjs target/virtual-geometry/stress/boulder boulder
python3 scripts/bench-geometry-loading.py target/release/examples/kit_showcase target/virtual-geometry/inputs/stress-discrete target/virtual-geometry/stress/boulder boulder target/virtual-geometry/stress-loading.json
```

No streaming or eviction is implemented by this prototype; assets stay resident
until their `World` is dropped. These measurements do not support adopting it for
loading or CPU-memory efficiency.


## Browser real-placement cut comparison

`benchmarks/virtual-geometry/greyhaven-browser-buttress.json.gz` contains 12 warmed
runs over the same three real placements and 240 camera frames used by native.
The page reported RTX 3060 Ti through ANGLE/Vulkan. Round-1 source/1 px cluster GPU
medians were 0.531/0.390 ms; CPU medians both rounded to 0.1 ms, with p95 0.1/0.2 ms.
Fixed LOD2 was cheaper on the GPU (0.343 ms), but has a different geometric error;
none of these values establishes an equal-quality whole-game frame improvement.
The browser source uses InstancedMesh; clusters use public BatchedMesh with per-view
cluster frustum culling. That difference is part of the compared mechanics, not a
claim that all production instance-culling policies are equivalent.

Reproduce by setting `VG_SCENE_URL=/target/virtual-geometry/greyhaven-buttress-scene.json`
on `node oracle/virtual-geometry/run.mjs ORIGIN crag crag.kit.outcrop-buttress-1 OUTPUT`.
The runner rejects resumed samples with a different scene or reported environment.

Inspect source/candidate captures directly:
[Greyhaven source](benchmarks/virtual-geometry/images/greyhaven-source.png),
[Greyhaven 1 px clusters](benchmarks/virtual-geometry/images/greyhaven-cluster-1px.png),
[arch source](benchmarks/virtual-geometry/images/arch-source.png),
[arch discrete LOD2](benchmarks/virtual-geometry/images/arch-discrete-lod2.png),
[arch 1 px clusters](benchmarks/virtual-geometry/images/arch-cluster-1px.png).
These scenery-subset images contain no terrain or units; the separately linked
actual gameplay captures are baselines.


## Actual HTTP payload comparison

`bench-geometry-transfer.py` fetched each compared artifact set over loopback HTTP
with no HTTP client cache, three alternating rounds. The response bodies used
identity content encoding. Raw per-file byte counts and serial fetch durations are
in `arch-transfer.json` and `stress-transfer.json`. Measured payload bytes equal the
referenced file totals: arch source/discrete/cluster 1,988,767/2,299,759/2,775,255;
dense control 8,147,219/10,273,562/17,764,075. This tests actual transferred bodies,
not WAN latency, gzip/Brotli size or mobile startup. Local fetch durations are
recorded separately from the native decode/upload spans and must not be presented
as end-to-end browser loading time.

Example: `python3 scripts/bench-geometry-transfer.py ORIGIN
target/virtual-geometry/inputs/arch-discrete
target/virtual-geometry/shrine/shrine.kit.arch shrine.kit.arch
OUTPUT_JSON`, with the repository-root server at `ORIGIN`.

The five crag scene JSON files and every referenced buffer/texture were compared directly with the consumer’s current local crag payload and match. Integrated/mobile hardware follow-up: [#121](https://github.com/moritzbrantner/render-engine/issues/121); the browser comparisons on an integrated GPU are [below](#integrated-gpu-browser-comparison-121).

The current Shooter catalog has Assembly Works and greybox arenas, not a shipped shipyard map. Assembly Works supplies the real factory gameplay baseline; the planned large arena remains in shooter#55. No artificial shipyard scene was substituted.


## Integrated GPU browser comparison (#121)

The browser comparisons were rerun on an integrated GPU on 2026-11-23: an Intel Arc 140V (Lunar
Lake, 16 GB shared). The machine runs WSL2, whose only path to that GPU is D3D12, and Mesa's D3D12
OpenGL driver puts WebGL 2 on it. The runs used headless Chromium 159 with
`GALLIUM_DRIVER=d3d12 VG_ANGLE=gl`, at 1280×720. The page reported
`ANGLE (Microsoft Corporation, D3D12 (Intel(R) Arc(TM) 140V GPU (16GB)), OpenGL ES 3.1)`, with multi-draw
and GPU timers available, three.js 187, the same `bench.js` and two reverse-order rounds. The
tables show round 2 (the reverse order), as the RTX figures above do. Raw reports:
`benchmarks/virtual-geometry/browser-arc-*.json.gz`. RMSE is the four-capture mean sRGB RMSE
against the source capture at the same count, as above. Timing is advisory: it is one laptop, and
the WebGL calls go through ANGLE, then OpenGL, then D3D12. The source and cluster draws also go
through different three.js paths (InstancedMesh against BatchedMesh).

```sh
python3 -c 'import gzip,json;json.dump(json.load(gzip.open("docs/benchmarks/virtual-geometry/greyhaven-browser-buttress.json.gz"))["scene"],open("target/virtual-geometry/greyhaven-buttress-scene.json","w"))'
GALLIUM_DRIVER=d3d12 VG_ANGLE=gl node oracle/virtual-geometry/run.mjs http://127.0.0.1:8000 arch-discrete shrine.kit.arch OUTPUT shrine
GALLIUM_DRIVER=d3d12 VG_ANGLE=gl VG_SCENE_URL=/target/virtual-geometry/greyhaven-buttress-scene.json \
  node oracle/virtual-geometry/run.mjs http://127.0.0.1:8000 crag crag.kit.outcrop-buttress-1 OUTPUT
GALLIUM_DRIVER=d3d12 VG_ANGLE=gl node oracle/virtual-geometry/run.mjs http://127.0.0.1:8000 crag crag.kit.outcrop-buttress-1 OUTPUT
```

**Inputs.**
- **Arch: the same processed inputs as the RTX run.** The current `shrine-kit` release
  (`02ad24aa…`), the 3d-lab `lod_chain` (3d-lab `1d185c6`, meshopt 0.7.0) and the
  three-nanite reference at `b730aac` reproduce `arch-discrete-request.json`,
  `arch-discrete-observations.json` and `arch-footprint.json` exactly: the same LOD triangle
  counts and errors, and the same file and buffer bytes for source, LODs and hierarchy.
- **Greyhaven buttress: the same scene and camera path, but not the same asset.** The scene is
  the recorded one (placements, transforms and the 240 frames, written from
  `greyhaven-browser-buttress.json.gz`). The asset comes from the repository-pinned crag kit
  (then pinned by `scripts/fetch-kits.sh`, `63bb39f4…`), because the evaluated `f3419e27…` is no longer in
  the rolling release. It has the same 8,999 source triangles and kit LOD1/LOD2, but a different
  source file hash and 175 rather than 176 clusters. The current `crag-kit` release
  (`6c35ac10…`) is closer: `bench-virtual-geometry.mjs` builds the evaluated buttress hierarchy
  from it byte for byte (176 clusters, 534,652 bytes, `b7e933cd…`, 449,644 geometry bytes). Only
  the glTF JSON differs (`inputSha256` `a5bdef2b…` against `2c3fa7be…`), so a rerun of this case
  should use that release.
- **Buttress grid: new.** The same asset as a dense grid (1, 64 and 256 instances). Its RTX
  3060 Ti baseline (`browser-rtx-buttress-grid.json.gz`; ANGLE on Vulkan, the same Chromium 159
  and `bench.js`) was run afterwards on the current `crag-kit` release (`6c35ac10…`, 176
  clusters). Source and LOD triangle counts are identical, and the RMSE matches the Arc's to
  within 0.0015.

**Results.**
- **Arch.** With 64 and 256 instances the cluster cuts take the least GPU time on the integrated
  GPU, more clearly than on the RTX. With 256 arches, the 1 px cut takes 3.7 ms against 8.1 ms
  for source and 7.6–8.9 ms for the discrete LODs (the RTX gave 3.2 ms against 5.2 and 3.6–4.9
  ms). It costs 1.5–2 ms of CPU against 0.2 ms, and has 2.5× LOD2's image error (RMSE 0.0071
  against 0.0028). The discrete LODs barely help here because `lod_chain` stops at about 5,900
  triangles from level 2 on. At a single instance, every mode costs about 0.85–0.95 ms, which is
  the frame's fixed cost.
- **Greyhaven buttress, real placements.** Every mode takes 1.05–1.20 ms of GPU time (source
  1.20, 1 px clusters 1.13, LOD2 1.05). That spread is smaller than the round-to-round variation
  (source's first round was 1.47 ms). At three placements, geometry is not the integrated GPU's
  cost, and no candidate is a measurable win.
- **Buttress grid.** The integrated GPU reverses the cluster result. With 256 buttresses, the 1 px
  cut submits 24 % fewer triangles than source (3.48 M against 4.61 M), yet takes more GPU time
  (20.2 ms against 14.0 ms) and 4 ms of CPU. The 4 px cut is level with source on the GPU (15.2
  ms) at RMSE 0.015. The kit's discrete LODs are cheaper (LOD2 6.0 ms), but they are fixed levels
  with no distance selection, at RMSE 0.044–0.056: visibly coarser on the nearby instances, so not
  an equal-quality candidate. On the RTX 3060 Ti, clusters do not win on this asset either: at
  256 the cuts take 3.8–5.0 ms against 4.8 ms for source, which is within the RTX's ±1 ms
  round-to-round variation, plus 1.9–2.7 ms of CPU. At 64 they are slower than source (4.1–4.8
  against 3.0 ms). Its implausibly low medians (64× LOD2 at 0.20 ms against 1.59 ms in round 1) are
  timer-query artefacts, as in the arch table.

**Loading and payload.** Each latest page load (local HTTP, warm cache) took 463 ms for the arch
(481 ms on the RTX) and about 1.5 s for the buttress pages. The arch's payload bytes are the
recorded ones (`arch-transfer.json`), because its files are byte-identical.

**Not covered here.** Native runs on this GPU and process memory are in the [native comparison](#integrated-gpu-native-comparison-121) below (from Windows; WSL2 has no Vulkan driver for the Arc and its D3D12 path reports no per-process GPU allocation). Browser WebGPU on it: Chromium under WSL gives only SwiftShader. Mobile GPUs. The captures were inspected (`OUTPUT/<count>-<mode>-<frame>.png`); none is checked in.

These results do not change the decision. On the integrated GPU, clusters win on the GPU only
for the arch, and there with 2.5× the discrete LODs' image error and seven to ten times their CPU
time (1.5–2 ms against 0.2 ms). On the buttress they lose to full source geometry. The real
placements show no difference.

Arch, 1, 64 and 256 instances (same inputs as `browser-arch-discrete.json.gz`):

| Instances | Mode | Triangles submitted (all views) | CPU ms median / p95 | GPU ms median / p95 | RMSE vs source | RTX 3060 Ti GPU ms median / p95 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | source | 29,144 | 0.20 / 0.40 | 0.93 / 1.04 | 0 | 0.35 / 0.45 |
| 1 | lod1 | 14,570 | 0.20 / 0.40 | 0.94 / 1.01 | 0.0001 | 0.28 / 0.36 |
| 1 | lod2 | 11,824 | 0.30 / 0.40 | 0.87 / 0.93 | 0.0004 | 0.03 / 0.34 |
| 1 | lod3 | 11,840 | 0.20 / 0.40 | 0.86 / 0.92 | 0.0004 | 0.30 / 0.35 |
| 1 | lod4 | 11,840 | 0.20 / 0.30 | 0.84 / 0.92 | 0.0004 | 0.36 / 1.20 |
| 1 | cluster-1 | 7,706 | 0.20 / 0.40 | 0.93 / 0.97 | 0.0008 | 0.29 / 0.31 |
| 1 | cluster-2 | 7,706 | 0.30 / 0.50 | 0.92 / 0.97 | 0.0008 | 0.29 / 0.31 |
| 1 | cluster-4 | 5,220 | 0.30 / 0.50 | 0.92 / 0.96 | 0.0010 | 0.29 / 0.31 |
| 64 | source | 1,865,216 | 0.20 / 0.30 | 5.35 / 6.82 | 0 | 3.40 / 4.80 |
| 64 | lod1 | 932,480 | 0.20 / 0.40 | 3.39 / 3.74 | 0.0010 | 2.51 / 2.82 |
| 64 | lod2 | 756,736 | 0.20 / 0.40 | 3.12 / 3.49 | 0.0023 | 2.23 / 2.45 |
| 64 | lod3 | 757,760 | 0.20 / 0.40 | 3.15 / 3.52 | 0.0023 | 2.24 / 2.45 |
| 64 | lod4 | 757,760 | 0.30 / 0.40 | 3.11 / 3.47 | 0.0023 | 2.23 / 2.45 |
| 64 | cluster-1 | 313,782 | 0.80 / 1.00 | 2.14 / 3.50 | 0.0045 | 1.00 / 1.16 |
| 64 | cluster-2 | 240,305 | 0.70 / 0.90 | 1.88 / 2.33 | 0.0069 | 2.18 / 3.12 |
| 64 | cluster-4 | 129,411 | 0.70 / 0.90 | 1.57 / 1.78 | 0.0187 | 0.63 / 0.72 |
| 256 | source | 7,460,864 | 0.20 / 0.30 | 8.07 / 9.70 | 0 | 5.23 / 5.55 |
| 256 | lod1 | 3,729,920 | 0.20 / 0.40 | 8.88 / 10.17 | 0.0015 | 4.74 / 5.91 |
| 256 | lod2 | 3,026,944 | 0.20 / 0.40 | 7.97 / 9.96 | 0.0028 | 3.75 / 5.04 |
| 256 | lod3 | 3,031,040 | 0.20 / 0.30 | 7.63 / 9.10 | 0.0028 | 3.64 / 4.92 |
| 256 | lod4 | 3,031,040 | 0.30 / 0.40 | 7.94 / 10.00 | 0.0028 | 4.88 / 5.09 |
| 256 | cluster-1 | 926,231 | 2.00 / 2.30 | 3.72 / 4.46 | 0.0071 | 3.25 / 3.63 |
| 256 | cluster-2 | 534,203 | 1.60 / 1.90 | 2.65 / 3.30 | 0.0161 | 2.50 / 3.35 |
| 256 | cluster-4 | 177,667 | 1.50 / 1.70 | 1.83 / 2.30 | 0.0312 | 0.90 / 1.79 |

Greyhaven buttress, three real placements, 240 frames (recorded scene; pinned kit):

| Instances | Mode | Triangles submitted (all views) | CPU ms median / p95 | GPU ms median / p95 | RMSE vs source | RTX 3060 Ti GPU ms median / p95 |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | source | 53,994 | 0.20 / 0.40 | 1.20 / 1.54 | 0 | 0.53 / 1.49 |
| 3 | lod1 | 16,362 | 0.20 / 0.40 | 1.11 / 1.29 | 0.0068 | 0.39 / 0.43 |
| 3 | lod2 | 4,530 | 0.20 / 0.40 | 1.05 / 1.11 | 0.0084 | 0.34 / 0.38 |
| 3 | cluster-1 | 37,714 | 0.30 / 0.50 | 1.13 / 1.39 | 0.0004 | 0.39 / 0.42 |
| 3 | cluster-2 | 37,087 | 0.30 / 0.60 | 1.18 / 1.30 | 0.0006 | 0.39 / 0.42 |
| 3 | cluster-4 | 30,136 | 0.30 / 0.50 | 1.15 / 1.26 | 0.0029 | 0.38 / 0.41 |

Buttress grid (pinned kit; the RTX column ran the current `crag-kit` release, whose cluster cuts
submit 0–3 % different triangle counts):

| Instances | Mode | Triangles submitted (all views) | CPU ms median / p95 | GPU ms median / p95 | RMSE vs source | RTX 3060 Ti GPU ms median / p95 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | source | 17,998 | 0.20 / 0.40 | 0.89 / 1.02 | 0 | 0.50 / 0.52 |
| 1 | lod1 | 5,454 | 0.20 / 0.40 | 0.72 / 0.93 | 0.0218 | 0.40 / 0.41 |
| 1 | lod2 | 1,510 | 0.20 / 0.50 | 0.70 / 0.85 | 0.0289 | 0.32 / 0.33 |
| 1 | cluster-1 | 16,318 | 0.30 / 0.70 | 0.89 / 1.00 | 0.0006 | 0.33 / 0.39 |
| 1 | cluster-2 | 16,318 | 0.30 / 0.80 | 0.90 / 1.01 | 0.0006 | 0.35 / 0.39 |
| 1 | cluster-4 | 16,318 | 0.30 / 0.80 | 0.60 / 0.70 | 0.0006 | 0.37 / 0.91 |
| 64 | source | 1,151,872 | 0.30 / 0.90 | 7.81 / 9.58 | 0 | 3.03 / 3.91 |
| 64 | lod1 | 349,056 | 0.40 / 1.30 | 5.02 / 6.32 | 0.0476 | 1.82 / 2.58 |
| 64 | lod2 | 96,640 | 0.30 / 1.00 | 2.86 / 3.45 | 0.0609 | 0.20 / 1.92 |
| 64 | cluster-1 | 985,955 | 2.00 / 4.00 | 6.19 / 12.95 | 0.0011 | 4.78 / 5.60 |
| 64 | cluster-2 | 924,880 | 2.00 / 4.20 | 5.81 / 12.43 | 0.0034 | 4.64 / 5.31 |
| 64 | cluster-4 | 767,115 | 2.30 / 4.80 | 5.25 / 14.25 | 0.0108 | 4.12 / 4.89 |
| 256 | source | 4,607,488 | 0.20 / 0.30 | 14.02 / 21.19 | 0 | 4.78 / 6.19 |
| 256 | lod1 | 1,396,224 | 0.20 / 0.40 | 8.25 / 17.15 | 0.0435 | 3.25 / 5.11 |
| 256 | lod2 | 386,560 | 0.30 / 0.40 | 5.99 / 7.76 | 0.0556 | 2.13 / 3.12 |
| 256 | cluster-1 | 3,482,522 | 4.00 / 4.50 | 20.22 / 24.12 | 0.0024 | 4.14 / 5.87 |
| 256 | cluster-2 | 2,893,591 | 4.10 / 6.60 | 20.44 / 33.84 | 0.0059 | 5.04 / 7.17 |
| 256 | cluster-4 | 1,911,049 | 4.00 / 9.40 | 15.15 / 29.29 | 0.0152 | 3.82 / 5.56 |

## Integrated GPU native comparison (#121)

The native comparison was run on the same integrated GPU on 2026-11-23: the Intel Arc 140V in a Core
Ultra 7 258V laptop, Windows 11 driver 32.0.101.8991, Vulkan, on AC power with the balanced power
plan. `kit_showcase` was cross-built for Windows (`x86_64-pc-windows-gnullvm`, linked with
[llvm-mingw](https://github.com/mstorsjo/llvm-mingw)) and started from WSL2 through interop, so wgpu
talks to Intel's Windows driver directly; its inputs were copied to a Windows directory. The D3D12
backend also runs, but only Vulkan, which the RTX runs used, was timed.

```sh
rustup target add x86_64-pc-windows-gnullvm
# llvm-mingw's bin/ on PATH:
CARGO_TARGET_X86_64_PC_WINDOWS_GNULLVM_LINKER=x86_64-w64-mingw32-clang \
CC_x86_64_pc_windows_gnullvm=x86_64-w64-mingw32-clang CXX_x86_64_pc_windows_gnullvm=x86_64-w64-mingw32-clang++ \
  cargo build --release --locked --example kit_showcase --target x86_64-pc-windows-gnullvm
# From a directory on the Windows drive holding the .exe, inputs and hierarchies:
export WGPU_BACKEND=vulkan WGPU_ADAPTER_NAME=Arc WSLENV=WGPU_BACKEND:WGPU_ADAPTER_NAME:VG_SCENE
./kit_showcase.exe --virtual-geometry 'inputs\arch-discrete' 'h\shrine.kit.arch' shrine.kit.arch 'out\arch'
./kit_showcase.exe --virtual-geometry 'inputs\crag-current' 'h\crag.kit.outcrop-buttress-1' crag.kit.outcrop-buttress-1 'out\grid'
VG_SCENE=greyhaven-buttress-scene.json ./kit_showcase.exe --virtual-geometry 'inputs\crag-current' 'h\crag.kit.outcrop-buttress-1' crag.kit.outcrop-buttress-1 'out\greyhaven'
WGPU_BACKEND=vulkan VG_ADAPTER=Arc python3 scripts/bench-geometry-loading.py /mnt/c/…/kit_showcase.exe /mnt/c/…/inputs/arch-discrete /mnt/c/…/h/shrine.kit.arch shrine.kit.arch /mnt/c/…/loading/arch-loading.json
```

**Inputs.** The arch uses the same processed inputs as the RTX and browser runs. The buttress uses the
current `crag-kit` release (`6c35ac10…`, now pinned in `scripts/fetch-kits.sh`): it rebuilds the
evaluated hierarchy byte for byte (176 clusters, `b7e933cd…`), so its RTX baseline is
`native-buttress-final.json.gz` and, for the real placements, `greyhaven-outcrop-buttress-1.json.gz`.
The Greyhaven scene is the recorded one.

**Timing is noisy on this laptop.** Each workload ran in two processes a few minutes apart, each with
two reverse-order rounds. The same mode's GPU median moved by up to 3× between the processes (the
single arch took 0.37 ms in the first and about 1.1 ms in the second), presumably as the GPU clock and
power state changed. The tables show the median of the four round medians with their range; read only
differences larger than the ranges. Raw reports: `benchmarks/virtual-geometry/native-arc-*.json.gz`
(rounds 0–1 the first process, 2–3 the second; both sessions' RMSE included). RMSE is the four-capture
mean sRGB RMSE against source, as above; it matches the RTX captures to four decimals.

**Results.**
- **Arch.** At 256 instances the cluster cuts take the least GPU time (1.8–2.4 ms against 3.35 for
  source), but the discrete LODs are close (LOD2 2.44 ms), and the cuts cost 1.4–1.8 ms of CPU against
  0.17 ms, at three to twelve times LOD2's RMSE. At 64, 1 px clusters (0.98 ms) and LOD1/LOD2 (1.17–1.28
  ms) overlap within their ranges.
- **Buttress grid.** Clusters do not beat source reliably: at 64 instances 4.5–5.4 ms against source's
  6.25 ms, with overlapping ranges, and at 256 they are slower (8.0–9.6 against 8.36 ms), at about 3 ms
  of CPU. The kit LODs are the cheapest (LOD2 4.07 ms at 256) but visibly coarser (RMSE 0.10).
- **Greyhaven buttress, real placements.** Every mode is 0.6–0.9 ms with ranges of about 0.45–1.4 ms:
  no measurable difference, as in the browser.

**Memory and loading.** `bench-geometry-loading.py` now also runs a Windows binary and samples Windows:
working set, peak and private bytes from `Get-Process`, and the GPU process memory counters
(`Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory`; an integrated GPU allocates from
shared memory). Raw rows: `arc-arch-loading.json`, `arc-buttress-loading.json`, three rounds each.

| Asset | Mode | Load ms (3 rounds) | Working set MiB | Shared GPU memory MiB | World upload bytes |
| --- | --- | --- | --- | --- | --- |
| arch | source | 70.7–73.0 | 574 | 517 | 18,074,764 |
| arch | discrete | 122.3–127.6 | 574–575 | 517 | 22,865,960 |
| arch | cluster | 118.5–144.1 | 588 | 518 | 21,088,508 |
| buttress | source | 106.5–124.4 | 574–575 | 517 | 51,230,992 |
| buttress | discrete | 115.3–118.3 | 574–575 | 517 | 51,601,736 |
| buttress | cluster | 162.0–169.1 | 584 | 517 | 53,156,728 |

The cluster path holds 10–14 MiB more working set and, for the buttress, loads about 50 ms slower. The
process's GPU allocation is 517–518 MiB in every mode, as NVIDIA's was 461 MiB on the RTX: the process
counter does not resolve a 2–5 MB difference in mesh buffers. The renderer's upload counters remain the
measure of those.

**Browser, current kit.** The Greyhaven buttress browser run (WebGL 2, as in the browser section) was
repeated with the current kit's 176-cluster hierarchy: `browser-arc-greyhaven-buttress-current.json.gz`.
The RMSE values are those of the pinned-kit run to four decimals (0.0004/0.0006/0.0029 for the cuts,
0.0068/0.0084 for LOD1/LOD2), and every mode took 0.57–1.09 ms of GPU time, again with no measurable
difference.

**Not covered.** D3D12 timing; Windows browsers (Chrome's WebGPU and ANGLE on D3D11 reach this GPU,
but the three.js bench uses WebGL 2 and no WebGPU candidate exists); mobile GPUs.

The decision is unchanged. Natively on the integrated GPU, clusters save GPU time only on the dense arch
grid, where conventional LODs come close at a fraction of the CPU time and image error. On the buttress
they do not beat full source geometry, and on the real placements nothing differs. They also cost more
working set and load time.

## Mobile GPU browser comparison (#121)

The browser comparisons were rerun on a mobile GPU on 2026-11-24: a Google Pixel 3a (Snapdragon 670,
Adreno 615), Android 12, Chrome 157.0.8103.78. The page reported
`ANGLE (Qualcomm, Adreno (TM) 615, OpenGL ES 3.2)` with multi-draw, three.js 187 and the same
1280×720 canvas, `bench.js`, camera paths and two reverse-order rounds as the RTX and Arc runs. The
phone was on USB power (battery 46–51 %, 31–35 °C) with its screen kept on. `run.mjs` drove its Chrome
over the DevTools protocol (`VG_CDP`) and reached the repository server through `adb reverse`:

```sh
adb forward tcp:9222 localabstract:chrome_devtools_remote   # Chrome's socket; another Chromium browser may hold this name
adb reverse tcp:8000 tcp:8000
python3 -m http.server 8000 --bind 127.0.0.1
export VG_CDP=http://127.0.0.1:9222
node oracle/virtual-geometry/run.mjs http://127.0.0.1:8000 arch-discrete shrine.kit.arch OUTPUT shrine
VG_SCENE_URL=/target/virtual-geometry/greyhaven-buttress-scene.json \
  node oracle/virtual-geometry/run.mjs http://127.0.0.1:8000 crag-current crag.kit.outcrop-buttress-1 OUTPUT
node oracle/virtual-geometry/run.mjs http://127.0.0.1:8000 crag-current crag.kit.outcrop-buttress-1 OUTPUT
```

**No GPU timer.** Chrome on Android exposes no `EXT_disjoint_timer_query_webgl2` on this phone, so GPU
time is unavailable. Instead `bench.js` records `sync_ms` when timer queries are missing: the time from
cut selection until a 1-pixel `readPixels` returns. That covers the CPU work, the wait for the GPU to
finish the frame, and the readback. It is a frame time, not GPU time, and it is not comparable with the
other sections' GPU columns, which are shown only for orientation. Raw reports:
`benchmarks/virtual-geometry/browser-pixel3a-*.json.gz`. The tables show round 2 (the reverse order).
Round 1's medians differ by at most 2 ms on the arch, 2.5 ms on the real placements and 5.8 ms on the
buttress grid (the 256 1 px cut: 319 against 325 ms). RMSE is the four-capture mean sRGB RMSE against
source, as above. It matches the Arc's to within 0.0013; the buttress uses the current kit here.

**Inputs.** The arch inputs were rebuilt from the `shrine-kit` release (`02ad24aa…`), 3d-lab
`lod_chain` at `1d185c6` (meshopt 0.7.0) and three-nanite at `b730aac`. `arch-discrete-request.json`,
`arch-discrete-observations.json` and `arch-footprint.json` come out byte-identical. The buttress uses
the current `crag-kit` release (`6c35ac10…`, 176 clusters, `b7e933cd…`). The Greyhaven scene is the
recorded one.

**Results.** On this phone, the 1 px and 2 px cuts are the slowest or nearly slowest option in
every dense case:
- **Arch.** At 256 instances, the 1 px and 2 px cuts take 101 and 78 ms a frame, against 83 ms for
  source and 51 ms for LOD2–LOD4. Only the 4 px cut is faster (38 ms), at RMSE 0.031: eleven times
  LOD2's error. The cuts' selection costs 12–16 ms of CPU against 0.5 ms. At 64 instances, LOD2 takes
  19.4 ms, source 27.8 ms and the cuts 25.9–43.6 ms.
- **Buttress grid.** At 256 instances, the cuts take 197–325 ms, against 160 ms for source. LOD2 takes
  35.5 ms but is coarser (RMSE 0.055). At 64 instances the cuts take 92–110 ms, against 65 ms for source.
  This mid-range GPU is far more sensitive to triangle count than the Arc. The kit LODs scale with it,
  but the cuts' 18–25 ms of CPU selection and their BatchedMesh draws cost more than the triangles they
  save.
- **Greyhaven buttress, real placements.** Every mode takes 10.5–14.1 ms. The cuts take 13–14 ms,
  against source's 12.1 ms (10.6 ms in round 1) and LOD2's 10.5 ms. At three placements, geometry is not
  the cost.

**Memory.** Measured with `adb shell dumpsys meminfo` on Chrome's GPU process, after loading each page
fresh and drawing one frame of 256 instances (`browser-pixel3a-memory.json`). With the 1 px cut, the
process holds 10.5 MiB more PSS than with source for the arch (285 against 274 MiB) and 26.4 MiB more
for the buttress (433 against 407 MiB). Its `GL mtrack` is 4.5 and 14.1 MiB higher. These are
single-sample figures. The renderer process's PSS grew with each page load, so it is not reported.

**Loading.** The latest page load (local HTTP over `adb reverse`, with the browser cache disabled)
took 2.65 s for the arch and 1.3 s for each buttress page.

**WebGPU.** Chrome on the phone offers a WebGPU adapter (`qualcomm`, `adreno-6xx`, with
`timestamp-query`). No WebGPU candidate exists, so it was not benchmarked. A future WebGPU candidate
would get GPU timing on this phone, which WebGL 2 does not.

**Not covered.** D3D12 timing on the Arc and browser WebGPU on an integrated GPU. Neither has a
candidate that would change the result: the native Arc runs used Vulkan, as the RTX runs did, and no
WebGPU cluster candidate exists. Other mobile GPUs (Mali, Apple) were not available.

The decision is unchanged. On a mobile GPU, cluster cuts cost more frame time than full source geometry
in both dense grids, except a 4 px arch cut at eleven times the discrete LODs' image error. They also
take 12–25 ms of CPU selection at 256 instances and more GPU-process memory. On the arch, the discrete
LODs beat the 1 px and 2 px cuts on both frame time and image error.

Arch, 1, 64 and 256 instances:

| Instances | Mode | Triangles submitted (all views) | CPU ms median / p95 | Sync frame ms median / p95 | RMSE vs source | Arc 140V GPU ms median |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | source | 29,144 | 0.80 / 0.90 | 11.9 / 13.3 | 0 | 0.93 |
| 1 | lod1 | 14,570 | 1.30 / 1.50 | 9.6 / 10.9 | 0.0001 | 0.94 |
| 1 | lod2 | 11,824 | 1.30 / 1.60 | 9.7 / 11.3 | 0.0004 | 0.87 |
| 1 | lod3 | 11,840 | 0.80 / 1.00 | 11.0 / 12.5 | 0.0004 | 0.86 |
| 1 | lod4 | 11,840 | 1.40 / 1.70 | 10.1 / 11.6 | 0.0004 | 0.84 |
| 1 | cluster-1 | 7,706 | 1.20 / 1.40 | 11.9 / 13.5 | 0.0008 | 0.93 |
| 1 | cluster-2 | 7,706 | 1.10 / 1.50 | 10.7 / 12.9 | 0.0008 | 0.92 |
| 1 | cluster-4 | 5,220 | 1.10 / 1.50 | 10.5 / 12.6 | 0.0010 | 0.92 |
| 64 | source | 1,865,216 | 0.50 / 0.90 | 27.8 / 32.2 | 0 | 5.35 |
| 64 | lod1 | 932,480 | 0.50 / 0.70 | 21.5 / 24.6 | 0.0011 | 3.39 |
| 64 | lod2 | 756,736 | 0.50 / 0.90 | 19.4 / 22.2 | 0.0023 | 3.12 |
| 64 | lod3 | 757,760 | 0.50 / 0.90 | 19.5 / 22.2 | 0.0023 | 3.15 |
| 64 | lod4 | 757,760 | 0.50 / 1.00 | 19.6 / 22.5 | 0.0023 | 3.11 |
| 64 | cluster-1 | 313,782 | 5.50 / 12.40 | 43.6 / 58.3 | 0.0045 | 2.14 |
| 64 | cluster-2 | 240,305 | 5.00 / 10.20 | 37.9 / 49.7 | 0.0069 | 1.88 |
| 64 | cluster-4 | 129,411 | 4.10 / 10.60 | 25.9 / 34.1 | 0.0187 | 1.57 |
| 256 | source | 7,460,864 | 0.50 / 1.10 | 83.0 / 88.2 | 0 | 8.07 |
| 256 | lod1 | 3,729,920 | 0.50 / 1.20 | 57.7 / 61.8 | 0.0016 | 8.88 |
| 256 | lod2 | 3,026,944 | 0.50 / 1.10 | 50.9 / 54.7 | 0.0029 | 7.97 |
| 256 | lod3 | 3,031,040 | 0.60 / 1.20 | 51.1 / 54.7 | 0.0029 | 7.63 |
| 256 | lod4 | 3,031,040 | 0.60 / 1.00 | 51.2 / 54.8 | 0.0029 | 7.94 |
| 256 | cluster-1 | 926,231 | 16.20 / 38.80 | 101.3 / 129.2 | 0.0070 | 3.72 |
| 256 | cluster-2 | 534,203 | 15.00 / 22.80 | 78.4 / 98.2 | 0.0161 | 2.65 |
| 256 | cluster-4 | 177,667 | 12.00 / 13.60 | 37.7 / 49.4 | 0.0313 | 1.83 |

Greyhaven buttress, three real placements, 240 frames (recorded scene; current kit):

| Instances | Mode | Triangles submitted (all views) | CPU ms median / p95 | Sync frame ms median / p95 | RMSE vs source | Arc 140V GPU ms median |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | source | 53,994 | 0.70 / 1.20 | 12.1 / 15.9 | 0 | 1.20 |
| 3 | lod1 | 16,362 | 0.80 / 1.30 | 12.5 / 14.4 | 0.0068 | 1.11 |
| 3 | lod2 | 4,530 | 0.80 / 1.40 | 10.5 / 12.4 | 0.0084 | 1.05 |
| 3 | cluster-1 | 37,714 | 1.40 / 2.40 | 13.1 / 19.3 | 0.0004 | 1.13 |
| 3 | cluster-2 | 37,087 | 1.40 / 2.90 | 13.1 / 19.0 | 0.0006 | 1.18 |
| 3 | cluster-4 | 30,136 | 1.50 / 2.90 | 14.1 / 17.6 | 0.0029 | 1.15 |

Buttress grid (current kit; the Arc column ran the pinned kit):

| Instances | Mode | Triangles submitted (all views) | CPU ms median / p95 | Sync frame ms median / p95 | RMSE vs source | Arc 140V GPU ms median |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | source | 17,998 | 0.70 / 1.00 | 10.8 / 16.3 | 0 | 0.89 |
| 1 | lod1 | 5,454 | 0.70 / 1.00 | 9.9 / 14.7 | 0.0213 | 0.72 |
| 1 | lod2 | 1,510 | 0.90 / 1.50 | 13.8 / 14.9 | 0.0279 | 0.70 |
| 1 | cluster-1 | 16,318 | 1.30 / 2.30 | 12.6 / 18.7 | 0.0006 | 0.89 |
| 1 | cluster-2 | 16,318 | 1.20 / 2.30 | 12.6 / 17.6 | 0.0006 | 0.90 |
| 1 | cluster-4 | 16,318 | 1.00 / 1.60 | 11.2 / 16.6 | 0.0006 | 0.60 |
| 64 | source | 1,151,872 | 0.60 / 0.90 | 64.7 / 73.8 | 0 | 7.81 |
| 64 | lod1 | 349,056 | 0.60 / 1.10 | 39.5 / 45.5 | 0.0473 | 5.02 |
| 64 | lod2 | 96,640 | 0.60 / 1.10 | 23.1 / 25.2 | 0.0599 | 2.86 |
| 64 | cluster-1 | 985,955 | 7.70 / 15.60 | 109.6 / 125.6 | 0.0012 | 6.19 |
| 64 | cluster-2 | 922,946 | 8.00 / 22.70 | 105.3 / 122.7 | 0.0035 | 5.81 |
| 64 | cluster-4 | 766,406 | 6.90 / 22.30 | 92.4 / 110.8 | 0.0108 | 5.25 |
| 256 | source | 4,607,488 | 0.50 / 1.10 | 159.9 / 184.1 | 0 | 14.02 |
| 256 | lod1 | 1,396,224 | 0.60 / 1.10 | 79.2 / 95.5 | 0.0434 | 8.25 |
| 256 | lod2 | 386,560 | 0.60 / 1.10 | 35.5 / 41.3 | 0.0552 | 5.99 |
| 256 | cluster-1 | 3,476,604 | 25.20 / 28.60 | 325.2 / 367.2 | 0.0037 | 20.22 |
| 256 | cluster-2 | 2,892,263 | 24.70 / 32.70 | 278.3 / 357.2 | 0.0064 | 20.44 |
| 256 | cluster-4 | 1,855,608 | 18.40 / 39.30 | 197.3 / 275.1 | 0.0158 | 15.15 |
