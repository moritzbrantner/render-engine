# Bevy 0.20.1 Hi-Z experiment (#59)

This is a *diagnostic oracle experiment*, not a production renderer feature or a
new parity ratchet. It compares three Bevy configurations on identical cameras,
object handles, materials and geometry:

| ID | Bevy camera components |
| --- | --- |
| `baseline` | Neither depth prepass nor GPU occlusion |
| `depth` | `DepthPrepass` |
| `hiz` | `DepthPrepass`, `OcclusionCulling` (Bevy's two-phase Hi-Z) |

Two scenes: `many_instances` (the existing 22,400-object open-field control),
and `occluded_instances` (the same objects with three rows of opaque building
proxies, gaps, partly protruding geometry and a small near-camera crossing).
The default script also measures 25% and 50% deterministic prop subsets of the
occluded scene (stationary view only), to bracket an object-count crossover.
Each mode runs with a stationary camera, a deterministic oscillating movement,
and a hard cut on frame 31. FLIP checks frames 30, 31, 32, 33, 34, 36 and 44
against the no-occlusion Bevy path. In particular, a false negative in the
first frames after the cut is a correctness failure even when steady-state
timing improves.

## Run on Ubuntu / RTX 3060 Ti

From the repository root, with the pinned Rust toolchain, a working Vulkan
driver and the repository's usual development prerequisites:

```sh
bash scripts/bench-occlusion.sh
```

To debug one scene with fewer samples:

```sh
bash scripts/bench-occlusion.sh --only occluded_instances --frames 120 --occlusion-rounds 1
```

Output: `target/oracle/occlusion/report.json`, the environment
`fingerprint.txt`, each child run's raw GPU samples and pass timings, PNG
captures, and FLIP heat maps. The modes alternate their order across two
rounds; the comparison takes the fastest round's median as in the existing
oracle. Each child starts a new Bevy app, warms up 30 frames, and excludes
captures and the first 20 following frames from timed samples. Software
adapter timings are **advisory only**. Bevy's actual GPU occlusion support is
reported explicitly; unsupported adapters must not be counted as negative
Hi-Z results.

The report also records a separate Render Engine render + median GPU time for
each scene's stationary camera and compares its image to Bevy's baseline.
Neither this nor any Bevy experiment changes the production renderer.

## Interpretation and limits

- Compare `hiz` against `depth` to isolate occlusion's incremental benefit.
  Compare `depth` against `baseline` to expose prepass overhead; also compare
  `hiz` directly with `baseline` to see whether the *whole* optimization pays.
- A crossover occurs only where a tested workload has repeatable
  `hiz/depth < 1` **and** `hiz/baseline < 1`, no failed FLIP captures, and the
  hardware adapter reports GPU occlusion support. Record the relative gain and
  absolute GPU-millisecond difference. Compare the 25%, 50% and full-density
  variants to bracket a crossover; interpolate only as a hypothesis, not an
  observed threshold. Follow up with more occlusion fractions before
  generalizing to Greyhaven.
- The CPU readings are the main-thread **frame interval** and Bevy's summed
  render CPU diagnostic spans (where available). Neither isolates CPU draw
  submission cost, so they must not be labeled as such.
- `logical_objects` is a scene-size input, **not** the number of submitted
  draw calls or visible instances. Bevy's GPU pipeline statistics for vertex,
  clipper, fragment and compute shader work are recorded when available (the
  default adapter features may not expose them). Accurate GPU indirect
  draw/instance counts require separate instrumentation of Bevy's indirect
  buffers. Top-level GPU pass spans show overhead only if Bevy records it.
- The experiment has opaque cuboids and spheres, with cast shadows. It does
  not establish correctness for skinned or alpha-masked geometry or temporal
  anti-aliasing. It also does not show a native Greyhaven win: that requires
  representative camera/asset measurements and comparison with instancing,
  frustum culling and LOD at the consumer's actual bottleneck.

No production Hi-Z implementation should be authorized from a positive
synthetic benchmark alone. If promising, open a separate opt-in implementation
issue with a no-occlusion fallback and native gameplay acceptance.
