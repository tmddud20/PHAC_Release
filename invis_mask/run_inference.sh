#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 DATA_ROOT OUT_FOLDER [additional inference arguments]" >&2
    echo "Optional environment: CKPT=/path/best.pt PYTHON_BIN=python" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="$1"
OUT_FOLDER="$2"
shift 2
PYTHON_BIN="${PYTHON_BIN:-python}"
CKPT="${CKPT:-$SCRIPT_DIR/checkpoints_im/best.pt}"

cd "$SCRIPT_DIR"
exec "$PYTHON_BIN" infer_all_invisible.py \
    --data_root "$DATA_ROOT" \
    --ckpt "$CKPT" \
    --img_size 512 \
    --base_ch 32 \
    --amp \
    --out_folder "$OUT_FOLDER" \
    "$@"
