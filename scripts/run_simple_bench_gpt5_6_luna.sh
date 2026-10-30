#!/usr/bin/env bash

set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "${script_dir}/.." && pwd)
output_dir=${1:-outputs/simple_bench_gpt5.6_luna_upper_bound}
model_config="configs/models/gpt5.6-luna-generator-critic.yaml"

cd "${project_root}"
mkdir -p "${output_dir}"

if [[ -x ".venv/bin/svg-agentic-slm" ]]; then
  cli=".venv/bin/svg-agentic-slm"
else
  echo "svg-agentic-slm was not found in .venv." >&2
  exit 127
fi

if [[ -z ${BIZROUTER_API_KEY:-} ]]; then
  BIZROUTER_API_KEY=$(
    .venv/bin/python -c \
      'from dotenv import dotenv_values; print(dotenv_values(".env").get("BIZROUTER_API_KEY", ""))'
  )
  export BIZROUTER_API_KEY
fi
if [[ -z ${BIZROUTER_API_KEY:-} ]]; then
  echo "BIZROUTER_API_KEY is not set in the environment or .env." >&2
  exit 2
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
    --model-config "${model_config}" \
    --output "${output}" \
    --rag \
    --critic \
    --set generation.max_new_tokens=6000 \
    --set generation.orchestration.critic_type=critic_v1 \
    --set generation.orchestration.critic_score_threshold=3.0 \
    --set generation.orchestration.critic_contract_mode=tolerant \
    --set generation.orchestration.enable_similarity_evidence=true \
    --set generation.orchestration.max_revision_rounds=2; then
    printf '[saved] %s\n' "${output}"
  else
    local status=$?
    failures+=("${category}_${number} (exit ${status})")
    printf '[failed] %s (exit %s)\n' "${output}" "${status}" >&2
  fi
}

printf 'Generator/Critic: openai/gpt-5.6-luna (Responses API, medium reasoning)\n'
printf 'Output directory: %s\n' "${output_dir}"

for index in "${!icon_prompts[@]}"; do
  run_case "icon" "$((index + 1))" "${icon_prompts[index]}"
done

for index in "${!illustration_prompts[@]}"; do
  run_case "illustration" "$((index + 1))" "${illustration_prompts[index]}"
done

if (( ${#failures[@]} > 0 )); then
  printf '\nGPT-5.6 Luna upper-bound bench completed with %s failed command(s):\n' \
    "${#failures[@]}" >&2
  printf '  - %s\n' "${failures[@]}" >&2
  exit 1
fi

printf '\nGPT-5.6 Luna upper-bound bench completed: 12/12 commands succeeded.\n'
printf 'Outputs: %s\n' "${output_dir}"
