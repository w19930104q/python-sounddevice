#!/usr/bin/env bash
# Run the long-run stability regression without touching sounddevice itself.
#
# Quick gate (default):   ./tests/run_regression.sh
# Multi-day soak:         SD_LONGRUN_CYCLES=200000 ./tests/run_regression.sh --soak
#
# The bundled PortAudio shared object ships with the SONAME
# libportaudio.so.2, so on Linux we expose it through a symlink dir via
# LD_LIBRARY_PATH (this only affects how the loader finds the existing
# library; no library file is added to or removed from the package).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
py="${PYTHON:-$repo_root/.venv-regress/bin/python}"
lib_dir="$repo_root/.regress-lib"

if [[ "$(uname -s)" == "Linux" ]]; then
  bundled="$repo_root/src/_sounddevice_data/portaudio-binaries/libportaudio.so"
  if [[ -f "$bundled" ]]; then
    mkdir -p "$lib_dir"
    ln -sf "$bundled" "$lib_dir/libportaudio.so.2"
    ln -sf "$bundled" "$lib_dir/libportaudio.so"
    export LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi

cd "$repo_root"
exec "$py" -I -m pytest tests/ -v "$@"
