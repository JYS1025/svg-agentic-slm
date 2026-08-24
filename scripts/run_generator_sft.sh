#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/train_lora.yaml}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
NUM_PROCESSES="${NUM_PROCESSES:-3}"

accelerate launch \
  --multi_gpu \
  --num_processes "${NUM_PROCESSES}" \
  --module svg_agentic_slm.train.train_text_to_svg \
  --config "${CONFIG}"
