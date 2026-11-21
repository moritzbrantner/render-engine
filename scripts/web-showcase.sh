#!/usr/bin/env bash
# D10 evidence for the browser showcase (#128, web/showcase/): in Playwright's Chromium and
# Firefox, the showcase's canvas frames against native render-engine (FLIP), device loss and
# recovery, resize at device pixel ratio 2, the page without WebGPU, startup, and GPU, prepare,
# encode and submit times against native on the same views. Not part of CI. Needs what
# scripts/build-web.sh and scripts/webgpu-smoke.sh need, plus
#   npx playwright install chromium firefox
#
#   bash scripts/web-showcase.sh [browser...]     default: chromium firefox
#
# Outputs go to target/showcase/<browser>/; the exit code is 1 if any check failed.
set -uo pipefail
cd "$(dirname "$0")/.."
browsers=("${@:-chromium firefox}")
read -r -a browsers <<<"${browsers[*]}"
frames=(tick1422 tick1782)
views=(eye 0 120 240)
timed=(tick1422:eye tick1422:0 tick1782:eye tick1782:0)
run=oracle/webgpu-smoke/run.mjs
compare=oracle/webgpu-smoke/target/release/compare
failed=0

bash scripts/build-web.sh || exit 1
(cd oracle/webgpu-smoke && cargo build --release --locked --bin compare) || exit 1

# show <json> <fields...>: prints the named fields of run.mjs's JSON result on one line.
show() {
  node -e 'const r = JSON.parse(process.argv[1]); console.log(process.argv.slice(2).map((k) => `${k}=${JSON.stringify(r[k])}`).join(" "))' "$@"
}

for browser in "${browsers[@]}"; do
  out="target/showcase/$browser"
  mkdir -p "$out"
  export SMOKE_BROWSER="$browser"
  echo "== $browser"
  specs=()
  for frame in "${frames[@]}"; do
    for view in "${views[@]}"; do
      node "$run" "$out" showcase "$frame" task=capture "view=$view" >/dev/null || failed=1
      specs+=("$frame:$view")
    done
  done
  # A recovery counts when the frames keep coming: 30 of the page's animation frames and 30 of the
  # loop's within 10 s (Firefox 159 stopped serving them after a destroy, #135).
  recovered='const r=JSON.parse(process.argv[1]); console.log(r.generation === 2 && r.sameImageAfterRecovery && r.animationFramesAfter >= 30 && r.loopFramesAfter >= 30'
  echo "-- device loss (destroy)"
  if r=$(node "$run" "$out" showcase tick1422 task=lose view=eye); then
    show "$r" generation losses errorsAfter sameImageAfterRecovery animationFramesAfter loopFramesAfter framesAfterMs status
    [[ $(node -e "$recovered && r.errorsAfter === 0)" "$r") == true ]] || failed=1
  else
    failed=1
  fi
  if [[ $browser == chromium ]]; then
    echo "-- device loss (GPU process crash)"
    if r=$(node "$run" "$out" showcase tick1422 task=lose view=eye how=crash); then
      show "$r" generation losses errorsAfter sameImageAfterRecovery animationFramesAfter loopFramesAfter framesAfterMs status
      [[ $(node -e "$recovered)" "$r") == true ]] || failed=1
    else
      failed=1
    fi
  fi
  echo "-- resize at devicePixelRatio 2"
  if r=$(SMOKE_DPR=2 node "$run" "$out" showcase tick1422 task=resize); then
    show "$r" devicePixelRatio sizes errors
    [[ $(node -e 'const r=JSON.parse(process.argv[1]); console.log(r.sizes[0].width === 2560 && r.sizes[1].width === 1280 && r.sizes[2].width === 666 && r.sizes[2].height === 1034)' "$r") == true ]] || failed=1
  else
    failed=1
  fi
  echo "-- without WebGPU"
  if r=$(SMOKE_WEBGPU=0 node "$run" "$out" showcase tick1422 task=status); then
    show "$r" backend status
    [[ $(node -e 'const r=JSON.parse(process.argv[1]); console.log(r.backend === null && r.status.startsWith("No supported graphics backend."))' "$r") == true ]] || failed=1
  else
    failed=1
  fi
  echo "-- startup (ms from navigation start; three fresh browser launches)"
  for _ in 1 2 3; do
    r=$(node "$run" "$out" showcase tick1422 task=startup) || failed=1
    node -e 'const s = JSON.parse(process.argv[1]).startup; console.log(Object.entries(s).map(([k, v]) => `${k} ${v.toFixed(1)}`).join(", "))' "$r"
  done
  echo "-- GPU and CPU time, 1280x720 offscreen, 300 frames (medians, ms)"
  for spec in "${timed[@]}"; do
    dev=()
    [[ $browser == chromium ]] && dev=(SMOKE_DEV=1)
    if r=$(env "${dev[@]}" node "$run" "$out" showcase "${spec%%:*}" task=time "view=${spec#*:}" frames=300); then
      node -e 'const t = JSON.parse(process.argv[1]).timing; console.log(`${process.argv[2]}: gpu ${t.gpuMs?.toFixed(3)} prepare ${t.prepareMs.toFixed(3)} encode ${t.encodeMs.toFixed(3)} submit ${t.submitMs.toFixed(3)}`)' "$r" "$spec"
    else
      failed=1
    fi
  done
  echo "-- frames against native render-engine"
  "$compare" showcase "$out" "${specs[@]}" || failed=1
done
unset SMOKE_BROWSER
echo "== native, the same protocol"
"$compare" showcase-native target/showcase "${timed[@]}" || failed=1
exit "$failed"
