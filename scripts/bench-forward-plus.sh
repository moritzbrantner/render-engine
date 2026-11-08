#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
out_dir="target/oracle/forward_plus"
mkdir -p "$out_dir"

# Keep the hardware/toolchain fingerprint beside the numbers: these fields matter before
# comparing runs on different drivers.
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

# Extra arguments are forwarded (e.g. --only dungeon_256 --frames 120).
cargo run --release --locked --manifest-path oracle/Cargo.toml -- \
  --forward-plus-bench --frames 300 --gpu-rounds 3 "$@"
