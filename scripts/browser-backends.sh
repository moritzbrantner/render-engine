#!/usr/bin/env bash
# Browser backend selection check of docs/webgl2-fallback.md: WebGPU (render-engine compiled to
# wasm), the WebGL 2 fallback (packages/three-lighting) and the error when neither starts, in
# headless Chromium, then the WebGL 2 frames against Bevy with D3's gates. Not part of CI or the
# default build; needs what scripts/webgpu-smoke.sh needs, and the oracle's outputs:
#   cargo run --release --locked --manifest-path oracle/Cargo.toml -- --only sun_shadows --only textured_props
#
#   bash scripts/browser-backends.sh            SKIP_WEBGPU=1 without a WebGPU adapter
set -euo pipefail
cd "$(dirname "$0")/.."
out="$PWD/target/oracle"
(cd oracle/webgpu-smoke && cargo build --release --locked --lib --target wasm32-unknown-unknown)
wasm-bindgen --target web --no-typescript --out-dir oracle/webgpu-smoke/pkg \
  oracle/webgpu-smoke/target/wasm32-unknown-unknown/release/webgpu_smoke.wasm
(cd oracle/webgpu-smoke && cargo build --release --locked --bin compare)
node oracle/webgpu-smoke/check-backends.mjs "$out"
oracle/webgpu-smoke/target/release/compare select "$out" sun_shadows textured_props
