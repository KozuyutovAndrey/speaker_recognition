#!/usr/bin/env bash
# Wrapper: sets CUDA libs from venv nvidia packages, then runs any Python script.
# Usage: bash scripts/run.sh scripts/run_baseline.py --data-root data/ ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

NVIDIA_LIBS="$PROJECT_ROOT/.venv/lib/python3.12/site-packages/nvidia"
CUDA_LD=$(find "$NVIDIA_LIBS" -maxdepth 2 -name "lib" -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${CUDA_LD}${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PROJECT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "$PROJECT_ROOT/.venv/bin/activate"
exec python "$@"
