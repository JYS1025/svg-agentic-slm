# Live OmniSVG → GPT-5.6 Luna critic distillation

The teacher runner reads each completed `generated` row from the two live
OmniSVG JSONL shards. It never loads OmniSVG or uses a GPU. It joins the row
to the exact selected instruction in `inputs/records.jsonl`, verifies both SVG
hashes, and renders both SVGs again at 512 px on white. Empty `fill=""` and
`stroke=""` attributes are removed before rendering so SVG default paint
semantics apply. Explicit `fill="none"` remains unchanged.

If the freshly rendered OmniSVG candidate is blank white, the runner records
`skipped_blank_candidate` and makes no API request. Otherwise it sends the
candidate render, the `critic_v1` style prompt and labeled candidate SVG, the
rerendered reference, and the reference SVG to BizRouter's
`openai/gpt-5.6-luna` Responses API.

The model receives the repository's current `critic_v1` system contract plus
a short reference-evidence addendum that makes the candidate the only evaluated
image. The teacher addendum does not repeat the explicit 18-item order removed
from the current critic prompt. The response must still contain the 18 ordered
evaluations and up to three actionable issues. Every response is checked against
`critic_output.schema.json` and the candidate's allowed target IDs before it
is marked `validated`.

## Environment and key

The API environment is separate from the OmniSVG GPU environment:

```bash
cd /home/elicer/svg-agentic-slm
uv venv --python /home/elicer/.local/bin/python3.11 \
  .work/omnisvg_critic_distill/.venv-teacher
uv pip install --python .work/omnisvg_critic_distill/.venv-teacher/bin/python \
  -r configs/requirements_omnisvg_teacher.txt
```

The runner reads `BIZROUTER_API_KEY` from the environment first, then from
`/home/elicer/svg-agentic-slm/.env` by default. The `.env` file is ignored by
Git, and the key is never written to the run manifest or result files. Edit
the repository-root `.env` file and replace the empty value with your key:

```dotenv
BIZROUTER_API_KEY=your-key-here
```

An exported environment variable takes precedence over `.env`. To set or
replace it in an interactive shell without putting its value in shell history:

```bash
read -rsp 'BizRouter API key: ' BIZROUTER_API_KEY
echo
export BIZROUTER_API_KEY
```

The quoted string in `read -rsp` is only the displayed prompt. Type or paste
the key after the prompt; `read` does not assign the prompt text to the
variable. If an earlier export is empty, the runner falls back to `.env`.

The existing `.work/omnisvg_critic_distill/test/api_test.py` contains a
hardcoded test key. The new runner does not import it. The existing key can be
replaced by setting the environment variable to a new value before the next
invocation; the output directory can then be resumed.

## One-call test

A nonblank candidate whose reference contained `fill=""` was tested under
`.work/omnisvg_critic_distill/teacher_runs/test_nonblank_v3/`. Both corrected
renders were visible. The API response passed all local checks with 18
evaluations, three issues, 3,390 input tokens, and 1,420 output tokens. A known
blank candidate was also tested separately and recorded
`skipped_blank_candidate` with zero API calls. To test another sample while
bounding the paid requests to exactly one:

```bash
.work/omnisvg_critic_distill/.venv-teacher/bin/python \
  scripts/run_omnisvg_critic_teacher.py run \
  --output-dir .work/omnisvg_critic_distill/teacher_runs/another_test \
  --max-samples 1 --max-api-calls 1 --max-concurrency 1 --max-attempts 1
```

`--max-api-calls` is a physical request cap for one runner invocation, not a
currency cap. The default is **one call** so a new command cannot accidentally
start a 10k paid run. The `status` command reports recorded input and output
tokens; verify the provider's actual billing separately.

## Follow the live OmniSVG run

Start this in a separate terminal or `tmux` session after setting the key.
It processes the existing backlog first, then waits for new OmniSVG results.
The example limits this invocation to 100 paid requests while you inspect the
quality and rate-limit behavior. The default output directory is the new
`luna_critic_v3_nonblank` run, so the shorter command below does not read or
resume the earlier `luna_critic_v1` records:

```bash
.work/omnisvg_critic_distill/.venv-teacher/bin/python \
  scripts/run_omnisvg_critic_teacher.py run \
  --follow --max-api-calls 100 --max-concurrency 2 --min-interval 5
```

Two requests can be active at once, but starts are spaced at least five
seconds apart. HTTP 429 and transient network/server errors cause a shared
cooldown with bounded retries. Authentication, insufficient-credit, and other
nonretryable HTTP errors halt dispatch. The runner stores a durable result
after each sample and skips validated `(sample_id, candidate_svg_sha256)` pairs
on restart. Add `--retry-failures` to retry failed teacher results, or raise
`--max-api-calls` with the replacement key when ready.

Watch the backlog and token usage from another shell:

```bash
.work/omnisvg_critic_distill/.venv-teacher/bin/python \
  scripts/run_omnisvg_critic_teacher.py status --watch 15
```

`source_generated` counts completed OmniSVG samples, `validated` counts accepted
teacher judgments, `skipped_blank_candidate` counts candidates filtered without
an API call, and `awaiting_teacher` is the remaining processing backlog. The run
directory contains `results.jsonl`, `progress.json`, `run_manifest.json`, and
per-sample request metadata and raw API responses under `requests/`. All are
inside the ignored `.work/omnisvg_critic_distill/` tree. The request metadata
records image paths and hashes; it does not duplicate base64 image data.
