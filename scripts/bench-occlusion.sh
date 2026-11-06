#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
out_dir="target/oracle/occlusion"
mkdir -p "$out_dir"

# Keep the hardware/toolchain fingerprint beside raw GPU samples. These fields
# are essential before comparing different driver or Bevy runs.
{
  git rev-parse HEAD
  rustc -Vv
  cargo -V
  uname -a
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || \
      printf '%s\n' 'nvidia-smi installed but adapter unavailable'
  fi
} > "$out_dir/fingerprint.txt"

# Extra arguments are forwarded (e.g. --only occluded_instances --frames 120).
cargo run --release --locked --manifest-path oracle/Cargo.toml -- \
  --occlusion-bench --occlusion-sweep --no-three --frames 300 --occlusion-rounds 2 "$@"
