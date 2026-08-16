# SVG Agentic SLM

An agentic pipeline utilizing Small Language Models (SLMs) to generate, validate, and refine Scalable Vector Graphics (SVG) from natural language descriptions. The current local baseline is the quantized Gemma 4 12B instruction model; the long-term goal is to evaluate and fine-tune lightweight open-source models for valid, semantically rich SVG output.

## Key Features

- **Agentic Generation & Refinement**: Implements an iterative feedback loop where generator agents produce SVG drafts, and rules-based or LLM-based critic agents analyze and refine the output to fix visual errors or syntax anomalies.
- **Retrieval-Augmented Generation (RAG) Contract**: Defines typed, provenance-preserving retrieval inputs and a metadata whitelist; the concrete ChromaDB retrieval backend remains RAG-owned and in progress.
- **Extensible LLM Backend**: Uses a backend interface with a pinned CUDA llama.cpp/Gemma 4 12B Q4_0 local profile, optional vLLM or llama-server OpenAI-compatible profiles, and injectable fake backends for contract tests.
- **SVG Processing Engine**: Provides secure XML parsing, active/external-content rejection, format normalization, and raster rendering.
- **Artifact Evaluation Framework**: Reports validity, rendering, and latency from generated artifacts; dataset-backed semantic and visual evaluation remains in progress.

---

## Project Documents

- [Generator Development Cycle Roadmap](docs/generator-cycle-roadmap.md):
  concise Cycle 0–8 goals, dependencies, ownership boundaries, experiments,
  and exit criteria.
- [Generator Cycle 0 Cross-Team Handoff](docs/generator-cycle0-team-handoff.md):
  the must-read integration agreements for RAG, Critic, Orchestration,
  Artifact, Validation, and Evaluation teammates.
- [SVG Safety Boundaries](docs/svg-safety-boundaries.md):
  shared validation policy, enforcement points, compatibility rules, and
  vector-collection migration guidance.
- [Model Backend Profiles](docs/model-backend-profiles.md):
  source-compatible model switching through validated OpenAI-compatible
  endpoints, including Generator and Critic profile rules.
- [Generator Cross-Team Contract and Research Assumptions](docs/generator-cross-team-contract.md):
  shared Generator assumptions that affect RAG, Critic, Orchestration, Artifact,
  Validation, and Evaluation workstreams.
- [Generator Implementation Plan](docs/generator-implementation-plan.md):
  dependency-driven implementation and experiment cycles for the Generator.
- [Generator Cycle 0 Status and Experiment Runbook](docs/generator-cycle0-status-and-runbook.md):
  completion gates, agreed contracts, and user-run model/dataset experiment
  commands.
- [Generate Command Workflow](docs/generate-command-workflow.md): current CLI,
  ownership, configuration, and artifact flow.

---

## Design Principles and Architecture

### Layered Architecture

To ensure high maintainability and prevent cyclic import complexities, the system strictly follows a layered design with one-directional dependency flows:

```
[CLI Application Layer]
          │
          ▼
[Agent Orchestration Layer] (Generators, Critics, RAG Agent)
          │
          ▼
[Core Functional Components] (Models, RAG Store, SVG Engine, Data Pipeline, Evaluator)
          │
          ▼
[Data Schemas and Configurations]
```

Higher-level modules depend only on lower-level interfaces and schemas. All components are instantiated and wired together using Dependency Injection, meaning no global mutable state or hardcoded dependencies are utilized.

## Directory Structure Overview

The project is organized into several key directories:

- **configs/**: Houses YAML files defining generation hyper-parameters, model selection details, database paths, evaluation configurations, and training schedules.
- **src/svg_agentic_slm/**: Contains the core library. This is divided into components like agents (generation/critique/orchestration), models (model loaders/backends), rag (vector storage), svg (validation/rendering), and training/evaluation pipelines.
- **data/**: Holds datasets for model fine-tuning, retrieval-augmented generation (RAG) corpora, and example JSONL entries.
- **outputs/**: Receives all dynamic file outputs including generated raw SVGs, rendered PNG raster images, and evaluation performance reports.
- **scripts/**: Contains utility tasks, setup scripts, and independent executable workflows like the environment smoke test.
- **tests/**: Integrates comprehensive unit testing for configurations, CLI endpoints, file formats, and orchestrator loops.

---

## Detailed Directory Structure

Below is an exhaustive breakdown of the project layout, explaining the purpose of each file and folder:

```
svg-agentic-slm/
├── README.md               # Project documentation
├── environment.yml         # Conda environment definition for dependency management
├── pyproject.toml          # Project metadata, build system configuration, and CLI entry points
├── .env.example            # Template for environment variables (e.g., API keys, system paths)
├── .gitignore              # Files and folders to ignore in Git
│
├── configs/                # Configuration profiles (YAML format)
│   ├── eval.yaml           # Parameters for running evaluations
│   ├── generation.yaml     # Inference settings (e.g., temperatures, maximum tokens, critic rounds)
│   ├── model.yaml          # Model identification details (Hugging Face repo IDs, precision, quantization)
│   ├── paths.yaml          # Explicit dataset, checkpoint, output, and log directories
│   ├── rag.yaml            # Retrieval configurations (ChromaDB collection names, embedding models)
│   └── train_lora.yaml     # LoRA/PEFT hyper-parameters (rank, alpha, learning rate, epoch counts)
│
├── src/svg_agentic_slm/    # Main package source directory
│   ├── __init__.py         # Package entry point and version metadata
│   │
│   ├── agents/             # Agentic logical units and control flow
│   │   ├── __init__.py
│   │   ├── base.py         # Abstract interfaces for Agent, Generator, and Critic types
│   │   ├── critic.py       # Interfaces for refinement feedback
│   │   ├── generator.py    # Prompt-based SVG generation agent implementation
│   │   ├── llm_critic.py   # Large language model-based critic for aesthetic and semantic critiques
│   │   ├── rule_critic.py  # Rule-based programmatic critic for deterministic syntax/attribute checks
│   │   ├── rag_agent.py    # Agent interface for executing context retrieval
│   │   ├── orchestrator.py # Core orchestrator managing the generation-critique-refinement loops
│   │   └── schemas.py      # Structured representations for agent interactions
│   │
│   ├── benchmarks/         # One isolated preparation adapter per benchmark candidate
│   │   ├── schemas.py      # Shared prepared-record and result boundary
│   │   └── svgenius.py     # SVGenius-only candidate download/join/manifest logic
│   │
│   ├── cli/                # Command line interface applications and commands
│   │   ├── __init__.py
│   │   ├── app.py          # Main Typer CLI router
│   │   ├── commands_eval.py      # Commands to evaluate generator pipelines
│   │   ├── commands_generate.py  # Commands to run natural-language-to-SVG pipeline runs
│   │   ├── commands_render.py    # Commands to render SVG documents to rasterized formats
│   │   ├── commands_train.py     # Commands to trigger model parameter fine-tuning
│   │   └── commands_validate.py  # Commands to run parsing and syntactic checks on SVG outputs
│   │
│   ├── data/               # Raw, processed, and custom dataset pipelines
│   │   ├── __init__.py
│   │   ├── jsonl.py        # Custom reader/writer for dataset serialization
│   │   ├── preprocess.py   # Placeholder for future dataset-neutral post-processing
│   │   ├── schemas.py      # Standardized python structures representing text-to-SVG data samples
│   │   └── text_to_svg_dataset.py # PyTorch dataset definitions
│   │
│   ├── eval/               # Artifact-backed evaluation framework
│   │   ├── __init__.py
│   │   ├── evaluator.py    # Evaluation interface; dataset execution remains in progress
│   │   ├── metrics.py      # Current artifact metrics and placeholder alignment metric
│   │   ├── report.py       # Report generators outputting markdown or json summaries
│   │   └── schemas.py      # Structured evaluations definitions
│   │
│   ├── models/             # Abstractions for target machine learning models
│   │   ├── __init__.py
│   │   ├── base.py         # Abstract base classes for model backends
│   │   ├── gemma_loader.py # Compatibility alias for the llama.cpp Gemma backend
│   │   ├── llama_cpp_backend.py # Pinned local GGUF inference backend
│   │   ├── openai_compatible_backend.py # Validated external model-server backend
│   │   ├── schemas.py      # Typed model response
│   │   └── generation_config.py # Structured configurations passed down to model backends
│   │
│   ├── prompts/            # Isolated and versionable prompt templates
│   │   ├── __init__.py
│   │   ├── critic_prompts.py    # Refinement and critique prompt variants
│   │   ├── system_prompts.py    # Global constraints and instructions for agents
│   │   └── text_to_svg.py       # Input generation prompt templates
│   │
│   ├── rag/                # Retrieval augmented generation utilities
│   │   ├── __init__.py
│   │   ├── base.py         # Abstract retriever interface definitions
│   │   ├── chroma_store.py # Placeholder ChromaDB implementation owned by RAG
│   │   ├── document_loader.py # Text-to-SVG metadata and code embedding loader
│   │   ├── metadata_policy.py # Shared-boundary metadata whitelist
│   │   └── schemas.py      # Structured schemas for storage elements
│   │
│   ├── svg/                # SVG engine handling file manipulation and rendering
│   │   ├── __init__.py
│   │   ├── base.py         # Abstract interfaces for validation and rendering units
│   │   ├── diff.py         # Code-level and structure-level XML diff implementations
│   │   ├── normalizer.py   # Format and tag standardizer for cleaning AI outputs
│   │   ├── renderer.py     # Interface implementations pointing to rendering libraries
│   │   ├── schemas.py      # Structured representations of SVG files
│   │   ├── utils.py        # XML string sanitization tools
│   │   └── validator.py    # Current lightweight structural validator
│   │
│   └── utils/              # Utility helpers
│       ├── __init__.py
│       ├── config.py       # YAML parser providing typed project parameters
│       ├── logging.py      # Application-wide customized logging setups
│       ├── paths.py        # Absolute path resolution logic based on configuration keys
│       └── seed.py         # Global seed management ensuring deterministic runs
│
├── data/                   # Dynamic runtime storage directories for research data
│   ├── raw/                # Unaltered upstream dataset resources
│   ├── processed/          # Downstream sanitized and tokenized data splits
│   ├── examples/           # Human-readable reference dataset files
│   └── rag_corpus/         # Source code and metadata chunks indexed by RAG stores
│
├── outputs/                # Artifacts produced during run execution
│   ├── generations/        # Output SVG text files
│   ├── renders/            # Exported raster formats (e.g., PNG)
│   └── eval_reports/       # Output metrics summaries and markdown performance reports
│
├── checkpoints/            # Intermediate and fine-tuned model weight checkpoints
├── logs/                   # Continuous execution log output files
├── notebooks/              # Jupyter notebooks for interactive analysis and prototyping
├── scripts/                # Standalone pipelines and execution tasks
│   └── smoke_test.py       # Minimal validation test validating package paths, imports, and setups
│
└── tests/                  # Package unit test suites
    ├── test_cli.py         # Validation for CLI command routers and parsers
    ├── test_config.py      # Verification of YAML schema loaders
    ├── test_jsonl.py       # Serialization and file IO unit tests
    ├── test_orchestrator.py # Integration testing for the main generation loop
    └── test_svg_validator.py # Functional test for SVG code parsing and quality evaluations
```

---

## Setup and Installation

### Prerequisites

- Conda or Mamba package manager
- Python 3.11 or later

### Local Setup

To prepare your environment and install the package locally:

```bash
# Clone this repository
git clone <repository_url>
cd svg-agentic-slm

# Initialize the Conda environment
conda env create -f environment.yml
conda activate svg-agentic-slm

# Install the package in editable development mode
python -m pip install -e ".[dev]"

# Prepare your local configuration files
cp .env.example .env
```

### RTX 4080 Laptop Local GPU Profile

The default Generator model is a pinned LM Studio Community Q4_0 compatibility
quant derived from Google's Gemma 4 12B instruction-tuned QAT upstream. The
previous Google-hosted GGUF is rejected because it aborts during vocabulary
loading. Build the pinned llama.cpp Python binding with CUDA:

```bash
CUDACXX=/usr/local/cuda/bin/nvcc \
CUDAHOSTCXX=/usr/bin/g++-11 \
CMAKE_ARGS="-DGGML_CUDA=on \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11" \
FORCE_CMAKE=1 \
python -m pip install -e ".[local-gpu,dev]"
```

The first real generation downloads the pinned checkpoint configured in
`configs/model.yaml`. The local default requests full GPU offload and does not
silently fall back to CPU. Set a lower `model.n_gpu_layers` explicitly if the
measured workload exceeds available VRAM.

### Verifying Setup

Ensure your local installation and imports work correctly by running the validation suite:

```bash
# Run the light-weight package smoke test
python scripts/smoke_test.py

# Run all unit tests
python -m pytest

# Confirm that the native llama.cpp build supports CUDA offload
python -c "from llama_cpp import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())"
```

---

## CLI Reference

The package registers a unified CLI utility `svg-agentic-slm` to orchestrate pipeline functions:

```bash
# Generate SVG code based on description prompts
svg-agentic-slm generate "A bright blue circle centered on a dark canvas"

# Exercise optional RAG/Critic paths (the concrete RAG and LLM Critic remain in progress)
svg-agentic-slm generate "An intricate golden star" --rag --critic

# Run validations on an existing SVG file
svg-agentic-slm validate path/to/drawing.svg

# Render an SVG document to a high-resolution PNG image
svg-agentic-slm render path/to/drawing.svg --output outputs/renders/drawing.png

# Exercise the current LoRA training scaffold
svg-agentic-slm train --config configs/train_lora.yaml

# Evaluate previously generated artifact bundles (not a dataset runner)
svg-agentic-slm eval --config configs/eval.yaml
```

---

## Data Schemas

### Text-to-SVG Data Instances (JSONL format)

For fine-tuning and evaluation, datasets are saved in JSON Lines format with structured records:

```json
{
  "task": "text_to_svg",
  "instruction": "A simple red triangle with rounded corners",
  "output_svg": "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 100 100\"><polygon points=\"50,15 90,85 10,85\" fill=\"red\"/></svg>"
}
```

### RAG Document Instances (JSONL format)

For RAG retrieval matching, candidate SVG snippets are parsed into structural documents:

```json
{
  "pattern_name": "rounded_triangle",
  "description": "Triangle polygon utilizing points matrix and solid fill colors",
  "svg_snippet": "<polygon points=\"50,15 90,85 10,85\" fill=\"red\"/>",
  "tags": ["polygon", "triangle", "geometry"]
}
```

---

## Unit Testing

Execute tests utilizing the pytest framework. All tests run cleanly without GPU resources or external connections:

```bash
# Run all unit tests
python -m pytest

# Run tests and export code coverage statistics
python -m pytest --cov=svg_agentic_slm

# Target specific test files
python -m pytest tests/test_svg_validator.py
```

---

## Remaining Implementation Roadmap

The shared runtime, strict SVG validation, rendering, artifact-backed
evaluation, and llama.cpp model path are implemented. Remaining owner work is:

1. **RAG retrieval**: complete the production Chroma corpus and retrieval policy.
2. **Critic quality**: calibrate LLM/rule feedback and acceptance thresholds.
3. **Evaluation**: freeze the benchmark and add real semantic quality metrics.
4. **Hardware evidence**: record latency, throughput, and VRAM on the target GPU.
5. **Training and research**: add PEFT/TRL experiments only after baseline gates close.

---

## License

MIT
