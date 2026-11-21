#!/usr/bin/env bash
# The browser gate (#129, technical decisions D8): render-engine compiled to wasm32 in a browser's
# WebGPU on every oracle scenario, judged against the oracle's PNGs with D3's gates. Lighting
# scenarios (glTF and skinned content fetched by the page included) against Bevy, three of them
# again with `Culling::GpuIndirect` and the clustered diagnostic (whose readbacks must match native),
# the D9 particle scenarios by their tiers and counts against Bevy, and soft particles against
# native render-engine. CI runs it in Playwright's Firefox on lavapipe, the oracle's own adapter;
# scripts/browser-hardware.sh in Chromium and Firefox on the GPU.
#
#   bash scripts/browser-gate.sh [out dir]      default target/oracle
#
# The out dir holds a full oracle run (`<scenario>.{bevy,engine}.png`, the particle captures and
# `<scenario>.bevy.json`). SMOKE_BROWSER=firefox|chromium (run.mjs) picks the browser. Needs what
# scripts/webgpu-smoke.sh needs. Every step runs; the exit code is 1 if any failed.
set -uo pipefail
cd "$(dirname "$0")/.."
out="$(realpath "${1:-target/oracle}")"
set -e
(cd oracle/webgpu-smoke && cargo build --release --locked --lib --target wasm32-unknown-unknown)
wasm-bindgen --target web --no-typescript --out-dir oracle/webgpu-smoke/pkg \
  oracle/webgpu-smoke/target/wasm32-unknown-unknown/release/webgpu_smoke.wasm
(cd oracle/webgpu-smoke && cargo build --release --locked --bin compare)
set +e
compare=oracle/webgpu-smoke/target/release/compare
mapfile -t lighting < <($compare list lighting -)
mapfile -t particles < <($compare list particles -)
# GPU culling on the culling scenario, the 64-light hall and the glTF props (blended and
# CPU-tested instances beside GPU-culled ones).
gpu_culled=(many_instances many_lights textured_props)
# Soft particles against Bevy are `particles_soft` (a D9 scenario, the particle step); these run
# two more scenarios with every emitter soft, against native render-engine.
soft=(particles_textured smoke_and_sparks)
failed=0

run() {
  if ! node oracle/webgpu-smoke/run.mjs "$out" "$@" >/dev/null; then
    echo "run.mjs $*: failed"
    failed=1
  fi
}

# Fail fast when the browser offers no WebGPU adapter (every scenario would fail the same way).
# Firefox without a display reports "WebGPU is disabled by blocklist": run it under xvfb-run.
echo "== WebGPU adapter (${SMOKE_BROWSER:-chromium})"
if ! probe=$(node oracle/webgpu-smoke/run.mjs "$out" render sun_shadows 2>&1); then
  reason=$(grep -oE 'WebGPU is disabled by blocklist|found no adapters|navigator\.gpu is missing|no WebGPU adapter' <<<"$probe" | head -1)
  echo "browser gate: ${SMOKE_BROWSER:-chromium} gave no working WebGPU adapter (${reason:-see below}); nothing else was run."
  echo "Firefox needs a display for WebGPU (xvfb-run) and a Vulkan driver (lavapipe on CI)."
  head -c 4000 <<<"$probe"
  exit 1
fi
echo "ok"

echo "== Lighting (${SMOKE_BROWSER:-chromium}): engine on WebGPU vs Bevy"
for s in "${lighting[@]}"; do run render "$s"; done
$compare images "$out" "${lighting[@]}" || failed=1

echo "== Culling::GpuIndirect and the clustered diagnostic"
for s in "${gpu_culled[@]}"; do SMOKE_TAG=webgpu-gpu run render "$s" culling=gpu lights=diagnostic; done
$compare images "$out" "${gpu_culled[@]/%/:webgpu-gpu}" || failed=1
$compare diagnostics "$out" "${gpu_culled[@]}" || failed=1

echo "== Particles (D9): tiers and counts vs Bevy"
for s in "${particles[@]}"; do run particles "$s"; done
$compare particles "$out" "${particles[@]}" || failed=1

echo "== Soft particles vs native render-engine"
for s in "${soft[@]}"; do SMOKE_TAG=webgpu-soft run particles "$s" soft=0.5; done
$compare soft "$out" 0.5 "${soft[@]}" || failed=1

exit "$failed"
