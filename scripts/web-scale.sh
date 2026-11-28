#!/usr/bin/env bash
# D10 evidence (#172): the frame cost of a scene of the scale of mmorpg's Greyhaven views
# (examples/scale_scene/scene.rs: 22 000 instances, 5 views, about 2 900 draws) through
# render-engine-web in Playwright's Chromium and Firefox WebGPU, and natively by the same
# protocol: medians over 300 frames at 1280x720 of prepare, encode and submit on the main thread
# and of the passes' GPU time, and the GPU API calls a frame makes. Not part of CI. Needs what
# scripts/build-web.sh needs, a hardware GPU, and for Firefox a display (xvfb-run without one).
#
#   bash scripts/web-scale.sh [browser...]     default: chromium firefox
set -euo pipefail
cd "$(dirname "$0")/.."
browsers=("${@:-chromium firefox}")
read -r -a browsers <<<"${browsers[*]}"
pkg=web/scale/pkg
(cd web && cargo build --release --locked --target wasm32-unknown-unknown --example scale_bench)
wasm-bindgen --target web --no-typescript --out-dir "$pkg" \
  web/target/wasm32-unknown-unknown/release/examples/scale_bench.wasm
wasm-opt -O3 --strip-debug --strip-producers \
  --enable-bulk-memory --enable-multivalue --enable-mutable-globals \
  --enable-nontrapping-float-to-int --enable-reference-types --enable-sign-ext \
  "$pkg/scale_bench_bg.wasm" -o "$pkg/scale_bench_bg.wasm"
echo "== native"
cargo run --release --locked --example scale_frame
for browser in "${browsers[@]}"; do
  echo "== $browser"
  node web/scale/run.mjs "$browser"
done
