# Virtual geometry: retain conventional LODs for current consumers

Status: technical judgment for the measured workloads, 2026-11-16 (#119).

## Decision

Do not adopt the evaluated cluster runtime or `three-nanite` as a production
dependency. Keep conventional instancing, frustum culling, discrete LODs and the
consumers' existing streaming/collision paths. Keep the native and public-API
Three.js adapters development-only so the experiment can be reproduced.

## Evidence

The [evaluation](../virtual-geometry-evaluation.md) includes actual crag/arena asset
preprocessing, native cave/arch/outcrop comparisons, dense synthetic scaling,
real Greyhaven placement/camera comparisons on native and browser, actual gameplay
baselines for both games, loading/residency samples and captures.

The native real-placement buttress case adds CPU work for a very small GPU gain.
The dense control's conventional LOD has similar image error and lower CPU cost.
The arch's conventional LOD is closer to the source image while avoiding cluster
submission cost. The browser shows some GPU reductions but no demonstrated net
whole-game gain at comparable quality. Cluster loading and CPU RSS regress in the
dense prototype; sampled NVIDIA process allocations do not isolate mesh residency.
Timing is advisory on the shared RTX desktop; image RMSE is not a perceptual gate.

Direct reference adoption also loses source attributes, relies on private Three.js
batching, omits off-camera shadow casters and never evicts streamed levels. The
experiment repairs attributes and uses public batching/per-view culling, but does
not validate a mature multi-material streaming format or general crack-free cuts.

## Consequences

No game integration PR or production runtime switch is warranted by these results.
No new simplifier is needed: the existing 3d-lab processor supplies the conventional
baseline, and asset-tooling retains preprocessing/packaging ownership. There is no
independent offline capability shown useful enough to duplicate those operations.

Do not deepen GPU selection, HZB or streaming without a representative bottleneck.
The earlier negative Hi-Z result and the existing opt-in GPU culling experiment
remain in force. CPU submitted-triangle counters are the small reusable renderer
improvement needed to make this comparison observable; they exclude indirect work.

Revisit with genuinely dense authored scenery and an equal-quality full-game
comparison covering shadows/materials, loading, CPU/GPU residency and the target
browser hardware. Integrated/mobile hardware, arbitrary multi-material assets,
strict conservative error validation and open-mesh transition guarantees remain
unverified. This decision rejects the current candidate for current consumers,
not virtual geometry in general, and does not claim to deliver a mature Nanite API.

Unavailable integrated/mobile hardware validation is tracked in [#121](https://github.com/moritzbrantner/render-engine/issues/121); it does not authorize an adoption switch.
