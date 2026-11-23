#!/usr/bin/env bash
# Downloads the asset-tooling bundles `examples/kit_showcase.rs` renders into target/kits/:
# Greyhaven terrain and nature, the crag kit, and the combat and zoo particle flipbooks. They are
# asset-tooling's rolling releases (its Decision 7.1.5), so each tar is pinned here by SHA-256
# and the script fails when a release has been replaced; update a pin deliberately.
#
#   bash scripts/fetch-kits.sh            # needs gh (authenticated) and sha256sum
set -euo pipefail

repo=moritzbrantner/asset-tooling
dir="$(cd "$(dirname "$0")/.." && pwd)/target/kits"
mkdir -p "$dir"

# release  tar  sha256
pins=(
  "greyhaven-terrain greyhaven.terrain.tar 024f3eede7a346664075bbb7de36fcb4b8fafaee5c1b1720f8d5be6eefe26a92"
  "greyhaven-nature greyhaven.nature.tar 2bfb85b383a366fc07742358da2efd8b77e03ef10263fff185c06b6f795ddde5"
  "crag-kit crag.kit.tar 6c35ac100bde54a02134e528dd74c64011c10fe630c45299b04d35218d217e9a"
  "combat-effects combat-effects.tar e02110b26d2a01a13b2bad3500537511408aa4278492cc226b292cf5babc68f4"
  "zoo-effects zoo-effects.tar c80955de8bde3ccdf4f83e04664310cfa8f9095e87f4f440e08d1d8ec5f86bcf"
)

for pin in "${pins[@]}"; do
  read -r release tar sha <<<"$pin"
  out="$dir/$release"
  if [[ -f "$out/.sha256" && "$(cat "$out/.sha256")" == "$sha" ]]; then
    continue
  fi
  rm -rf "$out" && mkdir -p "$out"
  gh release download "$release" -R "$repo" -D "$out" --pattern "$tar"
  # The effect releases carry their descriptor outside the tar as well; keep it.
  gh release download "$release" -R "$repo" -D "$out" --pattern '*.descriptor.json' 2>/dev/null || true
  echo "$sha  $out/$tar" | sha256sum --check --quiet || {
    echo "$release: $tar no longer matches its pin $sha (the rolling release was replaced)" >&2
    exit 1
  }
  tar xf "$out/$tar" -C "$out"
  echo "$sha" >"$out/.sha256"
  echo "fetched $release"
done
