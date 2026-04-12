#!/usr/bin/env bash
# Usage: bash scripts/extract_archives.sh [DATA_ROOT] [OUT_DIR]
# DATA_ROOT: directory with *.tar.gz archives (default: extracted_data/Для участников)
# OUT_DIR:   destination directory for extracted files (default: data)
set -euo pipefail

DATA_ROOT="${1:-extracted_data/Для участников}"
OUT_DIR="${2:-data}"
MAX_PARALLEL=4

mkdir -p "$OUT_DIR"

echo "[INFO] Data root : $DATA_ROOT"
echo "[INFO] Output dir: $OUT_DIR"

# Extract test set
if [ ! -d "$OUT_DIR/test_public" ]; then
    echo "[INFO] Extracting test_public.tar.gz ..."
    tar -xzf "$DATA_ROOT/test_public.tar.gz" -C "$OUT_DIR"
    echo "[INFO] test_public done"
else
    echo "[INFO] test_public already extracted, skipping"
fi

# Extract train archives in parallel (max MAX_PARALLEL at a time)
PIDS=()
for i in $(seq 1 10); do
    ARCHIVE="$DATA_ROOT/train_part_${i}.tar.gz"
    [ -f "$ARCHIVE" ] || continue

    # Check if already extracted (look for any marker dir)
    # We can't easily check per-part, so just always extract if train/ seems incomplete
    echo "[INFO] Extracting train_part_${i}.tar.gz ..."
    tar -xzf "$ARCHIVE" -C "$OUT_DIR" &
    PIDS+=($!)

    if [ ${#PIDS[@]} -ge $MAX_PARALLEL ]; then
        wait "${PIDS[0]}"
        PIDS=("${PIDS[@]:1}")
    fi
done

# Wait for remaining jobs
for pid in "${PIDS[@]}"; do
    wait "$pid"
done

echo "[INFO] All archives extracted to $OUT_DIR"
echo "[INFO] Train speakers: $(ls "$OUT_DIR/train/" 2>/dev/null | wc -l)"
echo "[INFO] Test files:     $(ls "$OUT_DIR/test_public/" 2>/dev/null | wc -l)"
