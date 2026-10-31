#!/usr/bin/env bash
# Assembles pages-dist/: the site, the latest oracle output (target/oracle: images and
# report.json) and the latest committed local GPU run (docs/oracle-local-gpu.json).
# Run the oracle first: cargo run --release --locked --manifest-path oracle/Cargo.toml
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/pages-dist"
rm -rf "$out"
mkdir -p "$out/oracle" "$out/gpu"
cp "$root"/site/* "$out/"
if [[ -d "$root/target/oracle" ]]; then
  cp "$root"/target/oracle/*.png "$root"/target/oracle/report.json "$out/oracle/" 2>/dev/null || true
fi
if [[ -f "$root/docs/oracle-local-gpu.json" ]]; then
  cp "$root/docs/oracle-local-gpu.json" "$out/gpu/report.json"
fi
sha="${GITHUB_SHA:-$(git -C "$root" rev-parse HEAD 2>/dev/null || true)}"
run_url=""
if [[ -n "${GITHUB_RUN_ID:-}" ]]; then
  run_url="${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}"
fi
jq -n --arg sha "$sha" --arg run_url "$run_url" --arg date "$(date -u +%Y-%m-%d)" \
  '{sha: $sha, run_url: (if $run_url == "" then null else $run_url end), date: $date}' > "$out/build.json"
echo "pages-dist ready: $out"
