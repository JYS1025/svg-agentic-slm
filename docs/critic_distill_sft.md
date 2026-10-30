# Qwen3.5 critic distillation SFT

The critic student receives exactly the production `critic_v1` evidence: one
candidate render, the original instruction, the labeled candidate SVG, its
allowed target IDs, and the production SigLIP2 pair probability. Reference
images, reference SVG, and teacher-only addenda are excluded. The assistant
target is the validated 18-entry scorecard JSON.

## Precompute SigLIP2 evidence

```bash
.venv/bin/python scripts/precompute_critic_similarity.py --device cuda:0
```

This uses pinned `google/siglip2-base-patch16-224` revision
`75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` and the same
`This is a photo of {instruction}.` template as `simple_bench`. Results are
append-only under `.work/omnisvg_critic_distill/similarity/siglip2/`. Re-running
scores only validated candidate hashes that are not already present.

## Prepare a frozen dataset

```bash
.venv/bin/python scripts/prepare_critic_distill_sft.py
```

This joins the latest generated candidate SHA with the latest validated Luna
teacher result, verifies artifact hashes and the critic contract, excludes blank
candidates, and writes deterministic pass-stratified 90/5/5 splits under
`data/processed/omnisvg_critic_distill/`. Pass records are not duplicated or
oversampled. Re-run this command explicitly when a newer teacher snapshot should
replace the frozen dataset.

## Staged training

```bash
scripts/run_critic_distill_sft.sh
```

The default launcher creates three stages:

1. Current scored snapshot, one epoch from the base model.
2. Wait for `awaiting_teacher=0`, score the remaining records, then train one
   epoch on new validated records only from the stage-1 adapter.
3. Train for `--final-epochs N` on initial train plus all new records from the
   stage-2 adapter.

The initial validation and test sets stay fixed for all stages. Optimizer and
scheduler state are new at each stage; only adapter weights are inherited. The
default config launches one DDP process per visible GPU, loads the pinned
Qwen3.5-4B model with NF4 double-quantized QLoRA, freezes the vision tower, and
attaches LoRA to every linear layer below `language_model`.

By default, stage 1 copies the already-frozen
`data/processed/omnisvg_critic_distill` snapshot. Teacher results arriving after
that snapshot are therefore unambiguously assigned to stage 2. Select another
frozen source with `--initial-snapshot PATH` when needed.

An explicit run location and final epoch count can be selected with:

```bash
scripts/run_critic_distill_sft.sh configs/train_critic_distill.yaml \
  --pipeline-dir outputs/sft/critic_distill_pipeline_main \
  --final-epochs 3
```

One process builds a fingerprinted cache under an exclusive file lock in
`/dev/shm/svg_agentic_slm/critic_sft/`; other ranks wait on a file lock and then
open the same read-only mmap files. The cache stores token IDs and response-mask
boundaries, multimodal token types, image grids, and processor-produced BF16 image patches.
Warm training runs therefore perform no PNG decode or image processing.

Each pipeline contains immutable snapshots, pipeline events, and three stage
directories. Every stage contains `output/adapter`, `output/processor`,
checkpoints, detailed logs, and `training_manifest.json`.

## Live logging and loss graph

Every Trainer log event is appended to `output/logs/metrics.jsonl` with loss,
eval loss, learning rate, gradient norm, estimated samples seen, and CUDA memory.
`output/logs/loss_curve.svg` and `output/logs/status.json` are refreshed live.

Monitor every stage and regenerate a combined graph with:

```bash
.venv/bin/python scripts/monitor_critic_distill_sft.py watch \
  --pipeline-dir outputs/sft/critic_distill_pipeline_main --interval 10

.venv/bin/python scripts/monitor_critic_distill_sft.py plot \
  --pipeline-dir outputs/sft/critic_distill_pipeline_main
```

The combined graph is written to `PIPELINE_DIR/loss_curve.svg`. The reported
`plateau_candidate` state is only a recent training-loss slope diagnostic; eval
loss and critic behavior still determine convergence.

To use the final adapter in production, set these critic model keys:

```yaml
adapter_path: PIPELINE_DIR/stages/03_full_Nepoch/output/adapter
processor_path: PIPELINE_DIR/stages/03_full_Nepoch/output/processor
load_in_4bit: true
```

The present teacher set is overwhelmingly negative. The preparation and training
manifests mark acceptance calibration as insufficient; deployment should be gated
on held-out pass rate and reject-collapse measurements.
