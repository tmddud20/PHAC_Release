#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 DATA_ROOT [additional train arguments]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="$1"
shift
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$SCRIPT_DIR"
exec "$PYTHON_BIN" train_invisible_mask.py \
    --data_root "$DATA_ROOT" \
    --epochs 40 \
    --batch_size 16 \
    --img_size 512 \
    --deocc_variants 4 \
    --base_ch 32 \
    --amp \
    --val_vis_every 10 \
    "$@"
