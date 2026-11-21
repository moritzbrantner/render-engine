#!/usr/bin/env bash
# Native crash evidence for CI (#122): a process that dies on a signal leaves a core dump, and
# the failure step turns each core into an all-threads backtrace. Costs nothing unless a
# process crashes; it neither retries nor hides a failure.
#
#   sudo bash scripts/crash-stacks.sh enable   # core_pattern → $CORE_DIR; then `ulimit -c unlimited`
#   bash scripts/crash-stacks.sh env           # Mesa/Vulkan packages and adapters, for the log
#   bash scripts/crash-stacks.sh report        # backtraces of every core in $CORE_DIR
#
# Cores are named core.<pid>.<executable path, '/' as '!'> (the kernel's %p and %E), so the
# report finds the crashed binary without guessing.
set -euo pipefail

CORE_DIR="${CORE_DIR:-${RUNNER_TEMP:-/tmp}/cores}"

case "${1:-}" in
  enable)
    mkdir -p "$CORE_DIR"
    chmod 1777 "$CORE_DIR"
    sysctl -w "kernel.core_pattern=$CORE_DIR/core.%p.%E"
    # Prove a crash leaves a core here, then clear it (a warning, never a failure).
    (ulimit -c unlimited && exec bash -c 'kill -SEGV $$') || true
    if compgen -G "$CORE_DIR/core.*" >/dev/null; then
      echo "core dumps enabled: $CORE_DIR"
      rm -f "$CORE_DIR"/core.*
    else
      echo "::warning::a crash left no core in $CORE_DIR; crash stacks will be missing"
    fi
    ;;

  env)
    uname -a
    dpkg-query -W 'mesa-vulkan-drivers*' 'libvulkan1*' 'libllvm*' 2>/dev/null || true
    if command -v vulkaninfo >/dev/null; then
      vulkaninfo --summary 2>&1 || true
    fi
    ;;

  report)
    shopt -s nullglob
    cores=("$CORE_DIR"/core.*)
    if (( ${#cores[@]} == 0 )); then
      echo "no core dumps in $CORE_DIR"
      exit 0
    fi
    # Ubuntu's debuginfod serves Mesa's and glibc's symbols, so lavapipe frames get names.
    export DEBUGINFOD_URLS="${DEBUGINFOD_URLS-https://debuginfod.ubuntu.com}"
    for core in "${cores[@]}"; do
      [[ "$core" == *.backtrace.txt ]] && continue
      name="${core##*/core.}"
      exe="${name#*.}"
      exe="${exe//!//}"
      echo "=== $core ($exe)"
      if [[ ! -x "$exe" ]]; then
        echo "executable $exe is gone; backtrace without its symbols"
        exe=""
      else
        # Keep the binary next to its core so the uploaded artifact can be read again.
        cp --update=none "$exe" "$CORE_DIR/" || true
      fi
      gdb -batch -nx \
        -ex 'set debuginfod enabled on' \
        -ex 'set pagination off' \
        -ex 'bt' \
        -ex 'thread apply all bt' \
        ${exe:+"$exe"} -c "$core" 2>&1 | tee "$core.backtrace.txt" || true
    done
    ;;

  *)
    echo "usage: $0 enable|env|report" >&2
    exit 2
    ;;
esac
