#!/usr/bin/env bash
# The browser evidence of docs/webgl2-fallback.md and docs/webgpu-browser-assessment.md on a
# machine with a hardware GPU, in one run: the oracle on the GPU (Bevy, native render-engine,
# three.js, with GPU time), then in Playwright's Chromium and Firefox the engine on WebGPU against
# them, the backend selection (WebGPU, WebGL 2 fallback, error) and the fallback's frames against
# Bevy, browser GPU time, and the showcase (scripts/web-showcase.sh). Everything goes into
# target/browser-hardware/report.md for the PR.
# Not part of CI. Needs what scripts/webgpu-smoke.sh needs, plus
#   npx playwright install chromium firefox
#
#   bash scripts/browser-hardware.sh [scenario...]     default: every oracle scenario (the
#                                                      browser gate); the #75 lighting set for
#                                                      the GPU-time and selection steps
#
# A step that fails is recorded in the report and the run goes on; the exit code is 1 if any did.
set -uo pipefail
cd "$(dirname "$0")/.."
out="$PWD/target/oracle"
report_dir="$PWD/target/browser-hardware"
report="$report_dir/report.md"
mkdir -p "$report_dir"
default=(sun_shadows sun_cascades many_instances tonemap_range_noon_tony ibl_materials_sun
  fog_atmospheric_sun point_spot point_spot_soft sky_day_cycle_noon sky_day_cycle_golden)
scenarios=("${@:-${default[@]}}")
failed=0

# step <title> <command...>: runs it, appends its output to the report as a code block.
step() {
  local title="$1"
  shift
  echo "== $title"
  {
    echo "## $title"
    echo
    echo '```'
  } >>"$report"
  if "$@" 2>&1 | tee -a "$report"; then
    echo '```' >>"$report"
  else
    failed=1
    {
      echo '```'
      echo
      echo "**Failed.**"
    } >>"$report"
  fi
  echo >>"$report"
}

{
  echo "# Browser run on hardware"
  echo
  echo "- Date: $(date -u +%Y-%m-%dT%H:%MZ)"
  echo "- Commit: $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (dirty)')"
  echo "- Machine: $(uname -sr)"
  command -v nvidia-smi >/dev/null && echo "- GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
  echo
} >"$report"

only=()
if (($#)); then
  for s in "${scenarios[@]}" textured_props; do only+=(--only "$s"); done
fi
step "Oracle on the GPU (FLIP and GPU-time gates)" \
  cargo run --release --locked --manifest-path oracle/Cargo.toml -- "${only[@]}"

# Builds the wasm and runs the WebGPU smoke against native and Bevy: with scenarios given, those
# (scripts/webgpu-smoke.sh); otherwise the browser gate on every oracle scenario
# (scripts/browser-gate.sh: lighting, GPU culling, particles, soft particles).
for browser in chromium firefox; do
  if (($#)); then
    step "$browser: engine on WebGPU vs native, Bevy, three.js" \
      env SMOKE_BROWSER="$browser" bash scripts/webgpu-smoke.sh "${scenarios[@]}"
  else
    step "$browser: browser gate (every oracle scenario on WebGPU)" \
      env SMOKE_BROWSER="$browser" bash scripts/browser-gate.sh "$out"
  fi
done

for browser in chromium firefox; do
  step "$browser: backend selection" env SMOKE_BROWSER="$browser" node oracle/webgpu-smoke/check-backends.mjs "$out"
  # The fallback's frames, forced, so both scenarios are compared in this browser.
  for s in sun_shadows textured_props; do
    SMOKE_BROWSER="$browser" node oracle/webgpu-smoke/run.mjs "$out" select "$s" backend=webgl2 >/dev/null || failed=1
  done
  SMOKE_BROWSER="$browser" node oracle/webgpu-smoke/run.mjs "$out" select sun_shadows backend=webgpu >/dev/null || failed=1
  step "$browser: WebGL 2 fallback vs Bevy, native, WebGPU" \
    oracle/webgpu-smoke/target/release/compare select "$out" sun_shadows textured_props
done

# Browser GPU time (#75's protocol; Chromium with unquantised timestamps). Firefox may not
# expose timestamp queries; a failure there is recorded, not fatal to the rest.
gpu_time() {
  local rc=0 result
  for s in sun_shadows many_instances point_spot sky_day_cycle_noon; do
    if result=$(node oracle/webgpu-smoke/run.mjs "$out" time "$s" frames=300 2>&1); then
      node -e 'const r=JSON.parse(process.argv[1]);console.log(`${r.scene}: ${JSON.stringify(r.timing)}`)' "$result"
    else
      echo "$s: failed: $(head -1 <<<"$result")"
      rc=1
    fi
  done
  return $rc
}
step "Chromium: GPU and CPU time" env SMOKE_DEV=1 bash -c "$(declare -f gpu_time); out='$out'; gpu_time"
step "Firefox: GPU and CPU time" env SMOKE_BROWSER=firefox bash -c "$(declare -f gpu_time); out='$out'; gpu_time"
step "Native: the same protocol" oracle/webgpu-smoke/target/release/compare native "$out" sun_shadows many_instances point_spot sky_day_cycle_noon

# The browser showcase (#128): its frames against native, device loss, resize, startup, and GPU,
# prepare, encode and submit times against native, in both browsers.
step "Showcase: shooter frames on WebGPU (Chromium, Firefox) vs native" bash scripts/web-showcase.sh

echo "report: $report"
exit "$failed"
