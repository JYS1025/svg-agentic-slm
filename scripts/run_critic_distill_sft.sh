#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${REPOSITORY_ROOT}/configs/train_critic_distill.yaml"
if (( $# > 0 )) && [[ "${1}" != --* ]]; then
  CONFIG="${1}"
  shift
fi
cd "${REPOSITORY_ROOT}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_COUNT="$(.venv/bin/python -c 'import torch; print(torch.cuda.device_count())')"
  if ! [[ "${GPU_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PyTorch did not detect an NVIDIA GPU." >&2
    exit 2
  fi
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((GPU_COUNT - 1)))"
  export CUDA_VISIBLE_DEVICES
fi

IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_PROCESSES="${NUM_PROCESSES:-${#VISIBLE_GPUS[@]}}"
if ! [[ "${NUM_PROCESSES}" =~ ^[1-9][0-9]*$ ]] || (( NUM_PROCESSES > ${#VISIBLE_GPUS[@]} )); then
  echo "NUM_PROCESSES must be positive and no larger than visible GPU count." >&2
  exit 2
fi

exec .venv/bin/python scripts/run_critic_distill_pipeline.py \
  --config "${CONFIG}" \
  --num-processes "${NUM_PROCESSES}" \
  "$@"
