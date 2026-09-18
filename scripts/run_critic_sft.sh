#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${1:-${REPOSITORY_ROOT}/configs/train_critic_a100_80gb.yaml}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_EXECUTABLE="${PYTHON_BIN}"
elif [[ -x "${REPOSITORY_ROOT}/.venv/bin/python" ]]; then
  PYTHON_EXECUTABLE="${REPOSITORY_ROOT}/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_EXECUTABLE="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_EXECUTABLE="$(command -v python3)"
else
  echo "Python is unavailable; set PYTHON_BIN to the training interpreter." >&2
  exit 2
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_COUNT="$("${PYTHON_EXECUTABLE}" -c 'import torch; print(torch.cuda.device_count())')"
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
  echo "NUM_PROCESSES must be positive and no larger than visible GPU count." >&2
  exit 2
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

"${PYTHON_EXECUTABLE}" "${SCRIPT_DIR}/check_sft_environment.py" \
  --config "${CONFIG}" \
  --expected-gpus "${NUM_PROCESSES}"

LAUNCH_ARGS=(
  launch
  --num_machines 1
  --num_processes "${NUM_PROCESSES}"
  --mixed_precision no
  --dynamo_backend no
)
if (( NUM_PROCESSES > 1 )); then
  LAUNCH_ARGS+=(--multi_gpu)
fi
LAUNCH_ARGS+=(--module svg_agentic_slm.train.train_critic --config "${CONFIG}")
"${PYTHON_EXECUTABLE}" -m accelerate.commands.accelerate_cli "${LAUNCH_ARGS[@]}"
