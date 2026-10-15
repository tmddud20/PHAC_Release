#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 MODEL_DIR DATA_PATH OUTPUT_PATH FOUNDATION_CKPT [additional inference arguments]" >&2
    echo "Optional environment: NUM_GPUS=1 CONTROL_TYPE=po BATCH_SIZE=1 NUM_VALIDATION_IMAGES=16" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRETRAINED_MODEL="$1"
VALIDATION_DATA="$2"
OUTPUT_PATH="$3"
FOUNDATION_CKPT="$4"
shift 4

NUM_GPUS="${NUM_GPUS:-1}"
CONTROL_TYPE="${CONTROL_TYPE:-po}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_VALIDATION_IMAGES="${NUM_VALIDATION_IMAGES:-16}"

cd "$SCRIPT_DIR"
exec torchrun \
    --nproc_per_node="$NUM_GPUS" \
    inference.py \
    --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
    --validation_data_dir "$VALIDATION_DATA" \
    --output_path "$OUTPUT_PATH" \
    --foundation_ckpt "$FOUNDATION_CKPT" \
    --control_type "$CONTROL_TYPE" \
    --batch_size "$BATCH_SIZE" \
    --num_validation_images "$NUM_VALIDATION_IMAGES" \
    "$@"
