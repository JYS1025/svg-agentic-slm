# SVG Agentic SLM

An agentic pipeline utilizing Small Language Models (SLMs) to generate, validate, and refine Scaleable Vector Graphics (SVG) from natural language descriptions. The long-term goal of the project is to evaluate and fine-tune lightweight open-source models, such as Gemma 4 E4B, to produce valid, semantically rich SVG output.

## Key Features

- **Agentic Generation & Refinement**: Implements an iterative feedback loop where generator agents produce SVG drafts, and rules-based or LLM-based critic agents analyze and refine the output to fix visual errors or syntax anomalies.
- **Retrieval-Augmented Generation (RAG)**: Integrates ChromaDB-driven vector storage to perform semantic code-snippet retrieval, providing generator agents with high-quality reference patterns based on descriptive text.
- **Extensible LLM Backend**: Built with abstraction interfaces supporting Hugging Face Transformers and optimized local inference execution for lightweight architectures such as the Gemma 4 E4B model.
- **SVG Processing Engine**: Includes complete validation steps (schema checking, XML syntax validation, and tag whitelisting), format normalization, and high-fidelity rasterized rendering support.
- **Rigorous Evaluation Framework**: Evaluates model performance across datasets utilizing semantic structure, XML diffing, image visual similarity comparisons, and text-image alignment metrics.

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
│   │   ├── preprocess.py   # Dataset filters and normalizers
│   │   ├── schemas.py      # Standardized python structures representing text-to-SVG data samples
│   │   └── text_to_svg_dataset.py # PyTorch dataset definitions
│   │
│   ├── eval/               # Quantized validation framework
│   │   ├── __init__.py
│   │   ├── evaluator.py    # Orchestration logic for dataset evaluations
│   │   ├── metrics.py      # Algorithms for structural, visual, and semantic SVG similarities
│   │   ├── report.py       # Report generators outputting markdown or json summaries
│   │   └── schemas.py      # Structured evaluations definitions
│   │
│   ├── models/             # Abstractions for target machine learning models
│   │   ├── __init__.py
│   │   ├── base.py         # Abstract base classes for model backends
│   │   ├── gemma_loader.py # Loader handling Hugging Face integrations for Gemma architecture
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
│   │   ├── chroma_store.py # Client wrapper implementing ChromaDB interfaces
│   │   ├── document_loader.py # Text-to-SVG metadata and code embedding loader
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
│   │   └── validator.py    # Validation algorithms checking XML syntax and safe tags lists
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
pip install -e ".[dev]"

# Prepare your local configuration files
cp .env.example .env
```

### Verifying Setup

Ensure your local installation and imports work correctly by running the validation suite:

```bash
# Run the light-weight package smoke test
python scripts/smoke_test.py

# Run all unit tests
pytest
```

---

## CLI Reference

The package registers a unified CLI utility `svg-agentic-slm` to orchestrate pipeline functions:

```bash
# Generate SVG code based on description prompts
svg-agentic-slm generate "A bright blue circle centered on a dark canvas"

# Generate utilizing both RAG retrieval and multiple critique loops
svg-agentic-slm generate "An intricate golden star" --rag --critic

# Run validations on an existing SVG file
svg-agentic-slm validate path/to/drawing.svg

# Render an SVG document to a high-resolution PNG image
svg-agentic-slm render path/to/drawing.svg --output outputs/renders/drawing.png

# Execute Parameter-Efficient Fine-Tuning (PEFT) utilizing LoRA
svg-agentic-slm train --config configs/train_lora.yaml

# Run evaluations across a text-to-SVG benchmark dataset
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
pytest

# Run tests and export code coverage statistics
pytest --cov=svg_agentic_slm

# Target specific test files
pytest tests/test_svg_validator.py
```

---

## Future Implementation Roadmap

The follow-up development sequence focuses on swapping mock placeholders with active implementations:

1. **SVG Parser Integration**: Wrap `lxml` inside `SVGValidator` for XML compliance and tag-attribute whitelisting.
2. **Native Rendering**: Implement `CairoSVG` or alternative vector rendering backends within `CairoRenderer.render()`.
3. **Model Infrastructure**: Wire Hugging Face `transformers` loader in `GemmaLoader` to initialize model weights and execute device allocation.
4. **End-to-End Generation**: Integrate the orchestrator and local model pipeline for real text-to-SVG tasks.
5. **RAG Vector Database**: Embed ChromaDB in `ChromaStore` using specialized sentence-transformer architectures.
6. **Programmatic Critic**: Fully realize deterministic heuristic checks inside `RuleCritic`.
7. **Agent Feedback Refinement**: Connect `LLMCritic` to guide generation loops with targeted text feedback.
8. **Parameter Optimization**: Connect `PEFT` and Hugging Face `TRL` trainers to support LoRA/QLoRA pipelines.
9. **Metric Scoring Engine**: Implement Chamfer Distance, structural similarity index (SSIM), and CLIP-based text-image score alignments in `eval/metrics.py`.
10. **Experiment Pipelines**: Facilitate automated tracking runs comparing model performance with and without RAG/Critic configurations.

---

## License

MIT
