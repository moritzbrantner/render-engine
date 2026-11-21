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

The five crag scene JSON files and every referenced buffer/texture were compared directly with the consumer’s current local crag payload and match. Integrated/mobile hardware follow-up: [#121](https://github.com/moritzbrantner/render-engine/issues/121).

The current Shooter catalog has Assembly Works and greybox arenas, not a shipped shipyard map. Assembly Works supplies the real factory gameplay baseline; the planned large arena remains in shooter#55. No artificial shipyard scene was substituted.
