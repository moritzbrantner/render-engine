#!/usr/bin/env bash
# The Chromium WebGPU smoke of docs/webgpu-browser-assessment.md. Not part of CI or the default
# build. Needs `rustup target add wasm32-unknown-unknown`, `bun install` (Playwright's headless
# Chromium) and the wasm-bindgen CLI at the crate's exact version:
#   cargo install wasm-bindgen-cli --version 0.2.130 --locked
# Run the oracle first for the PNGs the comparison reads:
#   cargo run --release --locked --manifest-path oracle/Cargo.toml -- --only <scenario>
#
#   bash scripts/webgpu-smoke.sh <scenario>...    default: sun_shadows
#
# WGSL errors show up as Dawn console warnings ("[Invalid RenderPipeline]") and a white frame:
# run.mjs's JSON output keeps them under `logs`.
set -euo pipefail
cd "$(dirname "$0")/.."
out="$PWD/target/oracle"
(cd oracle/webgpu-smoke && cargo build --release --locked --lib --target wasm32-unknown-unknown)
wasm-bindgen --target web --no-typescript --out-dir oracle/webgpu-smoke/pkg \
  oracle/webgpu-smoke/target/wasm32-unknown-unknown/release/webgpu_smoke.wasm
(cd oracle/webgpu-smoke && cargo build --release --locked --bin compare)
scenarios=("${@:-sun_shadows}")
for name in "${scenarios[@]}"; do
  node oracle/webgpu-smoke/run.mjs "$out" render "$name" >/dev/null
done
oracle/webgpu-smoke/target/release/compare images "$out" "${scenarios[@]}"
