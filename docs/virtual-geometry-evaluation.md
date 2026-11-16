# Virtual geometry evaluation: offline stage (#119)

## Runtime experiment in progress

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

These comparisons favor conventional LODs for the tested candidates. Browser,
actual gameplay, loading/residency and multi-material evaluation remain unfinished;
they must not be inferred from these grids. No consumer adoption is justified yet.
The independent `verify-cut-boundaries.mjs` diagnostic found every sampled synthetic
boulder cut closed after positional welding. Crag/arch source meshes have open
boundaries; their diagnostic does not establish watertight transitions.

## Decision and status

Retain conventional geometry and discrete LODs. Do not adopt `three-nanite` as a
runtime dependency. Cluster LOD remains an evaluation candidate, **not a rejected
technique**: this stage measures preprocessing and sampled cut correctness only.
It does not establish an equal-quality gameplay performance advantage or complete
the requested native/browser benchmark and integration work.

Useful candidates are Meshoptimizer meshlet partitioning, locked group boundaries,
monotonic conservative error, and per-view projected-error cuts. Any implementation
belongs with the existing owners below. GPU selection, streaming and HZB require
representative runtime evidence first. No consumer pins or production paths change.

## Capability and ownership inventory

| Capability | Existing owner and evidence | Implication |
| --- | --- | --- |
| Simplification and discrete LOD generation | `3d-lab/crates/three-d-lod/src/lib.rs`: Meshoptimizer through `meshopt` 0.6.2, `simplify_mesh`, `build_lod_chain`, border locking | Extend this owner if cluster preprocessing proves useful; do not create another simplifier |
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
static-mesh experiment. Zoo was not audited and has no justified integration here.

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

## Remaining evaluation before implementation

Issue #119 remains open. No useful runtime advantage has yet been established.
Next: compare actual Greyhaven camera paths using existing discrete LODs against a
development-only cluster path with equal materials and independent shadow-view cuts.
Measure warmed release CPU/GPU median/p95, draw ranges/submissions, rendered triangles,
CPU/GPU residency, cold/warm loading and bytes. Preserve screenshots and moving-camera
evidence of transitions, shadows and material quality. Add architectural modules,
caves and bounded dense stress scenes only using actual appropriate source assets.

This machine exposes an RTX 3060 Ti (8 GiB, driver 595.99.02); **no GPU timing was
performed in this stage**. Its desktop and Blender were active, so even later GPU
measurements must record contention. Integrated/mobile hardware is unmeasured.
No native/browser gameplay, screenshot comparison, geometry streaming, GPU-driven
selection or consumer integration is claimed. Low asset density alone is not a
negative performance result and must not close the larger investigation.

Shared conventions resolved at `15f348b072188255da837578eaafc8a98f12fdf0`.
