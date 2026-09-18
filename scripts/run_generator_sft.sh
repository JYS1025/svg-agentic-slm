#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${1:-${REPOSITORY_ROOT}/configs/train_lora.yaml}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if ! GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"; then
    echo "PyTorch is unavailable; install the project training dependencies first." >&2
    exit 2
  fi
  if ! [[ "${GPU_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PyTorch did not detect an NVIDIA GPU." >&2
    exit 2
  fi
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((GPU_COUNT - 1)))"
  export CUDA_VISIBLE_DEVICES
fi

IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
VISIBLE_GPU_COUNT="${#VISIBLE_GPUS[@]}"
for GPU_ID in "${VISIBLE_GPUS[@]}"; do
  if [[ -z "${GPU_ID//[[:space:]]/}" || "${GPU_ID}" == "-1" ]]; then
    echo "CUDA_VISIBLE_DEVICES must contain one or more comma-separated GPU IDs." >&2
    exit 2
  fi
done
NUM_PROCESSES="${NUM_PROCESSES:-${VISIBLE_GPU_COUNT}}"
if ! [[ "${NUM_PROCESSES}" =~ ^[1-9][0-9]*$ ]] || (( NUM_PROCESSES > VISIBLE_GPU_COUNT )); then
  echo "NUM_PROCESSES must be a positive integer no larger than visible GPU count." >&2
  exit 2
fi

python "${SCRIPT_DIR}/check_sft_environment.py" \
  --config "${CONFIG}" \
  --expected-gpus "${NUM_PROCESSES}"

LAUNCH_ARGS=(launch --num_processes "${NUM_PROCESSES}")
if (( NUM_PROCESSES > 1 )); then
  LAUNCH_ARGS+=(--multi_gpu)
fi
LAUNCH_ARGS+=(--module svg_agentic_slm.train.train_text_to_svg --config "${CONFIG}")
accelerate "${LAUNCH_ARGS[@]}"
