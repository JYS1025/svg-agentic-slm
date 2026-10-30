# OmniSVG critic input and one-sample smoke run

The generated data, upstream tokenizer source, and virtual environment live in
the ignored `.work/omnisvg_critic_distill/` directory. Model weights and model
configuration come **only from Hugging Face**, pinned to
`OmniSVG/OmniSVG@40138c74a343eae6c0d1149e0dbe95b163db0665`. The GitHub
checkout supplies only the 2025 SVG tokenizer and its `deepsvg` implementation;
the current OmniSVG GitHub `main` tokenizer is incompatible with this checkpoint.

## Existing environment: rerun the tested input

```bash
cd /home/elicer/svg-agentic-slm
source .work/omnisvg_critic_distill/.venv-omnisvg/bin/activate
CUDA_VISIBLE_DEVICES=0 python scripts/run_omnisvg_critic_smoke.py \
  --input-manifest .work/omnisvg_critic_distill/inputs/records.jsonl \
  --sample-index 1000 \
  --output-dir .work/omnisvg_critic_distill/runs/smoke_1000_repeat
deactivate
```

Choose a new `--output-dir` for each run. The runner refuses to overwrite an
existing result. The output contains `candidate.svg`, `candidate.png`,
`reference.svg`, `reference.png`, generated token IDs, and `status.json`.

## Rebuild from a fresh checkout

```bash
cd /home/elicer/svg-agentic-slm
mkdir -p .work/omnisvg_critic_distill
git clone --filter=blob:none --no-checkout \
  https://github.com/OmniSVG/OmniSVG.git \
  .work/omnisvg_critic_distill/upstream
git -C .work/omnisvg_critic_distill/upstream checkout \
  0d1e5e4e8516c341d432de5f211c0afdbecfb773

uv venv --python /usr/bin/python3.10 \
  .work/omnisvg_critic_distill/.venv-omnisvg
source .work/omnisvg_critic_distill/.venv-omnisvg/bin/activate
uv pip install torch==2.3.0 torchvision==0.18.0 \
  --index-url https://download.pytorch.org/whl/cu121
uv pip install -r configs/requirements_omnisvg_critic_smoke.txt

hf download OmniSVG/OmniSVG config.yaml config.json pytorch_model.bin \
  --revision 40138c74a343eae6c0d1149e0dbe95b163db0665
hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --revision 66285546d2b821cf421d4f5eb2576359d3770cd3

deactivate
.venv/bin/python scripts/prepare_omnisvg_critic_inputs.py \
  --ids-dir data/critic_rollout_sample_ids \
  --output-dir .work/omnisvg_critic_distill/inputs
source .work/omnisvg_critic_distill/.venv-omnisvg/bin/activate
CUDA_VISIBLE_DEVICES=0 python scripts/run_omnisvg_critic_smoke.py \
  --input-manifest .work/omnisvg_critic_distill/inputs/records.jsonl \
  --sample-index 1000 \
  --output-dir .work/omnisvg_critic_distill/runs/smoke_1000
deactivate
```

The input preparation script creates exactly 5,000 icon and 5,000 illustration
records. Within each domain it selects 3,000 Detail and 2,000 Description
instructions by stable SHA-256 rank. It joins the selected IDs to the audit's
Parquet row pointers and verifies source IDs, caption hashes, and canonical SVG
hashes. Details serialized as Python lists of sentences are joined into prose;
the exact source string remains in `detail_raw`.

`records.jsonl` is the input index. The SVG files under `inputs/reference/` are
unmodified source SVGs. Reference PNGs are rendered only when a sample is run,
using the same 512 px CairoSVG settings as its candidate PNG.
