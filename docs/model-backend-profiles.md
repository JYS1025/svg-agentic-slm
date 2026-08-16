# Model Backend Profiles

## Purpose

Model profiles let the same generation pipeline test different served models
without source-code changes. The existing in-process Gemma 4 GGUF profile
remains the default and rollback path.

## Selecting a Profile

Pass an explicit profile to `generate`:

```bash
svg-agentic-slm generate "Draw a blue circle" \
  --model-config configs/models/my-model.yaml
```

Each profile must identify one immutable experiment target:

```yaml
model:
  backend_type: "openai_compatible"
  base_url: "http://127.0.0.1:8000/v1"
  model_id: "organization/model@immutable-revision"
  revision: "immutable-revision"
  engine: "vllm"
  api_key_env: "MODEL_API_KEY"
  timeout_seconds: 120
  max_retries: 0
  allow_insecure_http: false
```

Remove `api_key_env` only when the local server intentionally has no API key.
Never put an API key, token, password, or secret directly in YAML. Remote HTTP
is rejected by default; use HTTPS unless an isolated test environment requires
an explicit `allow_insecure_http: true` exception.

`max_retries` applies only to the idempotent `/models` readiness check.
Generation requests are never retried automatically because a timed-out POST
may still have completed on the server.

## Starting a Compatible Server

The model exposed by `/v1/models` must exactly match `model_id`. This prevents
the pipeline from silently testing a different deployment.

For vLLM, use the profile model ID as the served name and prevent hidden model
generation defaults from changing comparison settings:

```bash
vllm serve ORGANIZATION/MODEL \
  --revision IMMUTABLE_REVISION \
  --served-model-name 'organization/model@immutable-revision' \
  --generation-config vllm \
  --api-key "$MODEL_API_KEY"
```

For an existing GGUF model served by llama.cpp:

```bash
llama-server \
  --model /absolute/path/model.gguf \
  --alias 'organization/model@immutable-revision' \
  --host 127.0.0.1 \
  --port 8080
```

Use `engine: "llama_cpp"` and port `8080` in that profile. Keep the current
direct `llama_cpp` profile for the pinned Gemma 4 baseline; moving that GGUF to
vLLM is not required for model comparison.

## Generator and Critic Models

`model` configures the Generator. An LLM Critic shares that backend by default.
To use a different Critic model, add a complete `critic_model` section to the
same profile:

```yaml
critic_model:
  backend_type: "openai_compatible"
  base_url: "http://127.0.0.1:8001/v1"
  model_id: "organization/critic@immutable-revision"
  revision: "immutable-revision"
  engine: "vllm"
  api_key_env: "CRITIC_MODEL_API_KEY"
```

`critic_model` is ignored unless the Critic type is `llm` or `both`. An empty
section is rejected to prevent accidental loading of the default Gemma model.

## Required Verification

Before accepting a model profile:

1. Confirm `/v1/models` exposes the exact configured model ID.
2. Run one deterministic generation with RAG and Critic disabled.
3. Confirm model ID, revision, engine, parameters, token counts, and latency in
   the generation artifact.
4. Run the same evaluation inputs and generation parameters as the baseline.
5. Treat SVG validation failures as model results, not infrastructure success.

The API verifies the served model name, not the weight bytes. The served alias
and profile revision must therefore be tied to the immutable weight revision by
the operator or deployment system.
