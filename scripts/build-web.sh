#!/usr/bin/env bash
# Builds the browser showcase (#128): render-engine-web's `shooter_showcase` for
# wasm32-unknown-unknown in the release profile, wasm-bindgen into web/showcase/pkg/, then
# wasm-opt, and prints the shipped sizes. Not part of CI or the default build. Needs
#   rustup target add wasm32-unknown-unknown
#   cargo install wasm-bindgen-cli --version 0.2.130 --locked    # the crate's exact version
#   wasm-opt (binaryen) on PATH
# Serve the repository root and open /web/showcase/ (it fetches tests/fixtures/shooter/ and
# packages/three-lighting/backend.js from there), e.g. `python3 -m http.server` and
# http://localhost:8000/web/showcase/ (WebGPU needs a secure context: localhost is one).
set -euo pipefail
cd "$(dirname "$0")/.."
pkg=web/showcase/pkg
(cd web && cargo build --release --locked --target wasm32-unknown-unknown --example shooter_showcase)
wasm-bindgen --target web --no-typescript --out-dir "$pkg" \
  web/target/wasm32-unknown-unknown/release/examples/shooter_showcase.wasm
# The features rustc enables by default for wasm32 (and every current browser has).
wasm-opt -O3 --strip-debug --strip-producers \
  --enable-bulk-memory --enable-multivalue --enable-mutable-globals \
  --enable-nontrapping-float-to-int --enable-reference-types --enable-sign-ext \
  "$pkg/shooter_showcase_bg.wasm" -o "$pkg/shooter_showcase_bg.wasm"
total=0
total_gz=0
for f in "$pkg/shooter_showcase_bg.wasm" "$pkg/shooter_showcase.js" web/showcase/main.js web/showcase/index.html \
  packages/three-lighting/backend.js; do
  bytes=$(stat -c %s "$f")
  gz=$(gzip -9c "$f" | wc -c)
  total=$((total + bytes))
  total_gz=$((total_gz + gz))
  printf '%-45s %10d bytes %10d gzip -9\n' "$f" "$bytes" "$gz"
done
printf '%-45s %10d bytes %10d gzip -9\n' "page total (without the frame JSON)" "$total" "$total_gz"
