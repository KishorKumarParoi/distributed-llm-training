#!/usr/bin/env bash
set -euo pipefail

echo "======================================================================"
echo "⚡ Launching Distributed LLM Training System"
echo "======================================================================"

WORLD_SIZE="${1:-2}"
STEPS="${2:-20}"

PYTHONPATH=. ./.venv/bin/python train.py \
    --world-size "${WORLD_SIZE}" \
    --max-steps "${STEPS}" \
    --hidden-dim 128 \
    --intermediate-dim 384 \
    --num-layers 4 \
    --num-heads 4 \
    --seq-len 64 \
    --batch-size 8 \
    --micro-batch-size 2 \
    --grad-accum 2 \
    --zero-stage 1

echo "======================================================================"
echo "✅ Distributed execution complete."
echo "======================================================================"
