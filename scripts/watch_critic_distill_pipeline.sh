#!/usr/bin/env bash
set -u

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-configs/train_critic_distill.yaml}"
PIPELINE_DIR="${2:-outputs/sft/critic_distill_pipeline_main}"
FINAL_EPOCHS="${3:-3}"
POLL_SECONDS="${POLL_SECONDS:-30}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"

if [[ "${PIPELINE_DIR}" = /* ]]; then
  PIPELINE_ROOT="${PIPELINE_DIR}"
else
  PIPELINE_ROOT="${REPOSITORY_ROOT}/${PIPELINE_DIR}"
fi
mkdir -p "${PIPELINE_ROOT}"

WATCHDOG_LOG="${PIPELINE_ROOT}/watchdog.log"
LAUNCH_LOG="${PIPELINE_ROOT}/watchdog_pipeline.log"
LOCK_FILE="${PIPELINE_ROOT}/watchdog.lock"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf '%s watchdog already running\n' "$(date -u --iso-8601=seconds)" >>"${WATCHDOG_LOG}"
  exit 0
fi

log() {
  printf '%s %s\n' "$(date -u --iso-8601=seconds)" "$*" >>"${WATCHDOG_LOG}"
}

pipeline_completed() {
  local manifest="${PIPELINE_ROOT}/pipeline_manifest.json"
  [[ -f "${manifest}" ]] &&
    grep -q '"status": "completed"' "${manifest}" &&
    grep -q '"phase": "completed"' "${manifest}"
}

pipeline_running() {
  local pipeline_name
  pipeline_name="$(basename "${PIPELINE_ROOT}")"
  pgrep -f "[r]un_critic_distill_pipeline.py.*--pipeline-dir [^ ]*${pipeline_name}" >/dev/null
}

log "watchdog_started pipeline_root=${PIPELINE_ROOT} poll_seconds=${POLL_SECONDS}"
while true; do
  if pipeline_completed; then
    log "pipeline_completed watchdog_stopping"
    exit 0
  fi

  if pipeline_running; then
    sleep "${POLL_SECONDS}"
    continue
  fi

  log "pipeline_process_missing relaunching"
  (
    cd "${REPOSITORY_ROOT}" || exit 1
    exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" NUM_PROCESSES="${NUM_PROCESSES}" \
      scripts/run_critic_distill_sft.sh "${CONFIG}" \
      --pipeline-dir "${PIPELINE_DIR}" \
      --final-epochs "${FINAL_EPOCHS}"
  ) >>"${LAUNCH_LOG}" 2>&1 &

  sleep "${POLL_SECONDS}"
done
