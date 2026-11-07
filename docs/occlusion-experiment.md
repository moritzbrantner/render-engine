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

## Results (2026-11-07)

NVIDIA GeForce RTX 3060 Ti, driver 595.99.02, Vulkan, Bevy 0.20.1, 1280×720,
300 timed frames × 2 rounds, fastest round's median. Bevy reported GPU
occlusion culling as supported. Every depth/Hi-Z capture (frames 30–44, all
views, including the frames right after the cut) is byte-identical to the
baseline capture (FLIP mean 0, p99 0): no wrongly culled props, no popping.
render-engine's own frustum-culled control matches Bevy's baseline
(FLIP mean ≤ 0.00005, p99 ≤ 0.0010).

GPU frame time (ms, sum of Bevy's top-level spans):

| Scene (objects) | View | baseline | depth | hiz | hiz/depth | hiz/baseline | render-engine |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `many_instances` (22,401) | stationary | 0.858 | 1.050 | 1.239 | 1.18 | 1.44 | 0.816 |
| | moving | 0.857 | 1.053 | 1.232 | 1.17 | 1.44 | |
| | cut | 0.827 | 0.997 | 1.169 | 1.17 | 1.41 | |
| `occluded_instances` (22,409) | stationary | 0.870 | 1.058 | 0.899 | 0.85 | 1.03 | 0.879 |
| | moving | 0.902 | 1.055 | 0.912 | 0.87 | 1.01 | |
| | cut | 0.903 | 0.998 | 0.921 | 0.92 | 1.02 | |
| `occluded_instances_50` (11,209) | stationary | 0.615 | 0.688 | 0.733 | 1.07 | 1.19 | 0.545 |
| `occluded_instances_25` (5,609) | stationary | 0.506 | 0.538 | 0.657 | 1.22 | 1.30 | 0.400 |

Pass costs, `occluded_instances` stationary (ms), and the camera-view work
from Bevy's pipeline statistics:

| Mode | early prepass | depth pyramid | late preprocess + prepass | main opaque | shadows | vertex invocations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | – | – | – | 0.303 | 0.364 | 1.48 M |
| depth | 0.189 | – | – | 0.278 | 0.366 | 2.95 M |
| hiz | 0.033 | 0.160 | 0.037 | 0.110 | 0.365 | 0.21 M |

- **Crossover: none observed.** Hi-Z beats `DepthPrepass` alone by 8–15 % on
  the occlusion-heavy scene, but it never beats the plain no-prepass path: it
  is 1–3 % slower at full density, 19 % at half and 30 % at quarter density,
  and 41–44 % slower on the open-field control.
- The cost is mostly fixed: the depth-pyramid downsample is ~0.16 ms at
  1280×720 regardless of scene, plus the prepass and late preprocessing. Hi-Z
  removes 93 % of the camera's vertex work and ~0.19 ms of main-pass time, which
  only just pays for that overhead in the densest case.
- Shadow passes (~0.36 ms, 40 % of the frame) are not occlusion-culled by
  Bevy; frustum culling per cascade is all they get.
- CPU: Hi-Z raises the main-thread frame interval (1.65 → 2.28 ms stationary)
  and Bevy's render CPU spans (0.047 → 0.075 ms); neither isolates draw
  submission.
- render-engine's current path (no prepass, frustum culling only) is already
  at or below Bevy's baseline in every case.

**Conclusion: documented negative result.** Bevy's two-phase Hi-Z does not
pay for itself on this native workload on an RTX 3060 Ti; render-engine keeps
frustum culling without occlusion, and no implementation issue is opened. It
would be worth revisiting only for a consumer that already needs a depth
prepass (Hi-Z then wins ~15 %), for far heavier per-object vertex/fragment
cost than these cuboids and spheres, or for occlusion-culled shadow views.
Alpha-masked and skinned geometry, TAA and Masked Software Occlusion Culling
or meshlets were not tested.
