# OmniSVG 10k batch inference

The input corpus is `.work/omnisvg_critic_distill/inputs/records.jsonl`.
`scripts/run_omnisvg_critic_batch.py` uses the pinned Hugging Face
`OmniSVG/OmniSVG` 3B checkpoint and the same SVG reconstruction path as the
successful single-sample smoke run. The two GPU processes each load one model
once, use BF16 and left-padded prompt batches, group examples by reference SVG
size to reduce decode padding, and send SVG token reconstruction and PNG
rendering to CPU processes. These CPU tasks overlap subsequent GPU batches. All
outputs remain under the ignored `.work/omnisvg_critic_distill/`.

## Pilot and batch-size selection

Run a representative pilot first. This example spans 32 icon and 32
illustration inputs starting at zero-based index 4968. It does not affect the
full run.

```bash
cd /home/elicer/svg-agentic-slm
source .work/omnisvg_critic_distill/.venv-omnisvg/bin/activate
python scripts/run_omnisvg_critic_batch.py run \
  --run-dir .work/omnisvg_critic_distill/runs/pilot_64_b16_repeat \
  --gpus 0,1 --start-index 4968 --limit 64 \
  --batch-size 16 --render-workers 2
```

For a batch-size comparison, repeat with `--batch-size 8` and a different
`--run-dir`, such as `pilot_64_b8_repeat`. Compare `summary.json`, worker logs,
and GPU memory use. On the same 32 icon inputs, batch size 16 completed in 135
seconds versus 150 seconds for batch size 8, including model loading and CPU
postprocessing. Batch size 16 used about 10 GB per GPU during this test. If a
batch exhausts GPU memory, the runner splits that batch and records the
result. A smaller production batch size avoids repeated splits.

Existing pilot results are under `.work/omnisvg_critic_distill/runs/`.
The cross-domain batch-size-16 pilot completed 64/64 samples with zero
failures, including hash verification of its generated and reference assets.
Its run directory occupies 3.3 MB, which scales to roughly 0.5 GB at 10k if
the file-size distribution stays similar.

## Full 10k run

Run in a persistent terminal such as `tmux`. The following starts a detached
session; use `tmux attach -t omnisvg10k` to see the coordinator's log and
`Ctrl-b d` to detach again.

```bash
cd /home/elicer/svg-agentic-slm
tmux new-session -d -s omnisvg10k -c /home/elicer/svg-agentic-slm \
  '.work/omnisvg_critic_distill/.venv-omnisvg/bin/python scripts/run_omnisvg_critic_batch.py run --run-dir .work/omnisvg_critic_distill/runs/full_10k --gpus 0,1 --batch-size 16 --render-workers 2'
```

The run is tied to the exact input hash, source commit, model revisions, runner
code hash, GPU list, generation settings, and batch size. Keep the same command
for resumption. To change settings, use a new run directory.

## Monitor and resume

In another terminal, read the current counts, worker phases, and rough ETA:

```bash
cd /home/elicer/svg-agentic-slm
.work/omnisvg_critic_distill/.venv-omnisvg/bin/python \
  scripts/run_omnisvg_critic_batch.py status \
  --run-dir .work/omnisvg_critic_distill/runs/full_10k --watch 15 --compact
```

The coordinator prints aggregate progress every 15 seconds. Per-GPU logs are
`gpu0.log` and `gpu1.log`; machine-readable snapshots are
`progress.gpu0.json` and `progress.gpu1.json`. `status` counts durable records
in `results.gpu0.jsonl` and `results.gpu1.jsonl`. It reports `loading_model`
until the model is ready, `generating` with a batch number while a batch is
running, and `postprocessing` while the last CPU tasks finish. ETA uses
completed work after model loading and is only a rough estimate. Omit
`--compact` to see the full JSON snapshot.

If stopped, run the same full-run command again. Completed samples are skipped.
The runner keeps a fixed batch schedule and regenerates a whole partial batch
to preserve sampling behavior for its unfinished samples. Technical failures
are terminal records. Add `--retry-failures` to retry them explicitly. A
restarted worker also repairs an incomplete final JSONL line and regenerates
any saved candidate whose artifact hash no longer matches.

After both workers finish, the coordinator verifies all selected IDs and
artifact hashes, writes `results.jsonl` in input order, and writes
`summary.json`. You can repeat the verification without model loading:

```bash
.work/omnisvg_critic_distill/.venv-omnisvg/bin/python \
  scripts/run_omnisvg_critic_batch.py audit \
  --run-dir .work/omnisvg_critic_distill/runs/full_10k
```

Each result points to the generated SVG, token IDs, candidate PNG, reference
SVG from the input corpus, and reference PNG. Invalid or unrenderable output is
kept with a failure status and error; the runner does not rank or replace low
quality candidates. Teacher requests can stream `results.jsonl` and join each
record to `inputs/records.jsonl` by `sample_id`.
