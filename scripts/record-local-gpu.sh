#!/usr/bin/env bash
# Runs the oracle with both gates on the local GPU (RTX 3060 Ti) and records its report in
# docs/oracle-local-gpu.json, stamped with the commit and date. Commit that file: the Pages
# site shows it as the GPU-time table (CI has no GPU).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/target/oracle-gpu"
status=0
cargo run --release --locked --manifest-path "$root/oracle/Cargo.toml" -- --out "$out" || status=$?
jq --arg commit "$(git -C "$root" rev-parse HEAD)" --arg date "$(date -u +%Y-%m-%d)" \
  '{commit: $commit, date: $date} + . | .scenarios |= map(del(.bevy_spans))' \
  "$out/report.json" > "$root/docs/oracle-local-gpu.json"
echo "recorded docs/oracle-local-gpu.json (oracle exit $status)"
exit "$status"
