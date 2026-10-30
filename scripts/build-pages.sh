#!/usr/bin/env bash
# Assembles pages-dist/: the site plus the latest oracle output (target/oracle).
# Run the oracle first: cargo run --release --locked --manifest-path oracle/Cargo.toml
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/pages-dist"
rm -rf "$out"
mkdir -p "$out/oracle"
cp "$root"/site/* "$out/"
if [[ -d "$root/target/oracle" ]]; then
  cp "$root"/target/oracle/*.png "$root"/target/oracle/report.json "$out/oracle/" 2>/dev/null || true
fi
echo "pages-dist ready: $out"
