#!/usr/bin/env bash

set -uo pipefail

usage() {
  echo "Usage: $0 <run_number>" >&2
  echo "Example: $0 1  # writes outputs/simple_bench_run1/" >&2
}

if [[ $# -ne 1 ]] || [[ ! $1 =~ ^[1-9][0-9]*$ ]]; then
  usage
  exit 2
fi

run_number=$1
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "${script_dir}/.." && pwd)
output_dir="outputs/simple_bench_run${run_number}"

cd "${project_root}"
mkdir -p "${output_dir}"

if [[ -x ".venv/bin/svg-agentic-slm" ]]; then
  cli=".venv/bin/svg-agentic-slm"
elif command -v svg-agentic-slm >/dev/null 2>&1; then
  cli=$(command -v svg-agentic-slm)
else
  echo "svg-agentic-slm was not found. Activate .venv or install the project." >&2
  exit 127
fi

icon_prompts=(
  "Two hands hold together"
  "A calendar with checkmark"
  "A futuristic circuit board chip symbol"
  "A magnifying glass over a document"
  "A pair of 3D glasses"
  "A butterfly symmetrical shape"
)

illustration_prompts=(
  "A minimalist plant growing from a light bulb"
  "A laptop and smartphone connected by glowing lines"
  "A sun and cloud interacting in a friendly, cartoon style"
  "A candle melting beside an open book"
  "A cup of tea with floating lemon slices and herbs"
  "A floating island with a tree and waterfall"
)

failures=()

run_case() {
  local category=$1
  local number=$2
  local prompt=$3
  local output="${output_dir}/text_to_svg_${category}_${number}.svg"

  printf '\n[%s %s/6] %s\n' "${category}" "${number}" "${prompt}"
  if CUDA_VISIBLE_DEVICES=0,1 "${cli}" generate \
    "${prompt}" \
    --config configs/generation.yaml \
    --model-config configs/models/gemma4-gemma4-critic.yaml \
    --output "${output}" \
    --rag \
    --critic \
    --no-render \
    --set generation.orchestration.critic_type=critic_v1 \
    --set generation.orchestration.max_revision_rounds=2 \
    --print-generator-parameters; then
    printf '[saved] %s\n' "${output}"
  else
    local status=$?
    failures+=("${category}_${number} (exit ${status})")
    printf '[failed] %s (exit %s)\n' "${output}" "${status}" >&2
  fi
}

for index in "${!icon_prompts[@]}"; do
  run_case "icon" "$((index + 1))" "${icon_prompts[index]}"
done

for index in "${!illustration_prompts[@]}"; do
  run_case "illustration" "$((index + 1))" "${illustration_prompts[index]}"
done

if (( ${#failures[@]} > 0 )); then
  printf '\nSimple bench run %s completed with %s failed command(s):\n' \
    "${run_number}" "${#failures[@]}" >&2
  printf '  - %s\n' "${failures[@]}" >&2
  exit 1
fi

printf '\nSimple bench run %s completed: 12/12 commands succeeded.\n' "${run_number}"
printf 'Outputs: %s\n' "${output_dir}"
