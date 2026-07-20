# Generator 구현 계획

## 문서 상태

- 기준 브랜치: `minjun/generator`
- 기준 커밋: `f1a88c3`
- 작성 기준일: 2026-07-20
- 문서 상태: Cycle 0 계약과 Cycle 1 backend의 구현 후보가 현재 working tree에 반영된 상태다.
- 범위: generator를 실제 동작하게 만들고, 아직 placeholder인 RAG와 critic이 나중에 구현되었을 때 generator 내부를 다시 고치지 않고 연결할 수 있도록 준비한다.
- 비범위: RAG index/store, Critic 품질 구현, benchmark runner, memory curation은 구현하지 않는다.
- 소유권 원칙: 이 계획의 실행 주체는 **Generator owner**다. RAG store/index, critic 품질, benchmark framework, memory curation을 대신 구현하지 않는다.
- 공유 계약: 다른 workstream의 구현이나 핵심 연구 가정에 영향을 주는 결정은 [Generator Cross-Team Contract and Research Assumptions](./generator-cross-team-contract.md)를 기준으로 관리한다.
- 실행 현황과 사용자 실행 명령: [Generator Cycle 0 Status and Experiment Runbook](./generator-cycle0-status-and-runbook.md)를 기준으로 관리한다.
- Dataset 예외: dependency를 검토하기 위한 SVGenius candidate adapter만 별도 모듈로 제공하며, 최종 benchmark 선택과 runner 구현으로 범위를 확장하지 않는다.

1절부터 6절까지의 “현재 상태” 분석은 기준 커밋 `f1a88c3`의
gap analysis다. 실제 진행 상태는 7절의 각 Cycle과 공유 계약 문서의
decision register를 기준으로 판단한다.

이 계획은 각 반복을 다음과 같은 고정 형식으로 진행한다.

1. **초기 dependency**: 반복을 시작하기 전에 확보하거나 결정해야 하는 것
2. **무엇을 구현하는가**: 해당 반복에서 변경할 코드와 계약
3. **무엇을 실험하는가**: 구현이 실제 개선인지 판정할 비교 실험
4. **종료 조건**: 다음 반복으로 넘어가기 위한 객관적인 gate

## 0. 역할 범위

### Generator owner가 직접 구현할 것

- 실제 model backend load와 inference
- model별 chat template을 소비할 수 있는 generator-facing backend 계약
- initial generation과 revision 입력 계약
- RAG가 전달한 중립적인 context를 prompt에 안전하게 삽입하는 기능
- raw model output에서 최종 SVG를 추출·정규화하는 기능
- formatting retry와 generator 내부 failure reporting
- prompt/model/output/attempt provenance를 `GenerationResult.metadata`에 제공하는 기능
- fake RAG와 fake critic을 사용한 generator contract test

### 다른 owner의 구현을 dependency로만 받을 것

- ChromaDB, embedding, corpus ingestion, similarity search
- positive/negative feedback memory의 선별, deduplication, 저장, index, retrieval
- rule/LLM critic의 평가 품질과 fine-tuning
- orchestrator의 전체 revision policy와 persistence 정책
- strict SVG validator와 renderer backend
- benchmark dataset 수집, evaluator, metric, dashboard
- LoRA/SFT/DPO training pipeline

### 경계가 중요한 이유

Generator가 RAG index를 직접 쓰거나 critic 결과를 직접 채점·선별하면 다음 문제가 생긴다.

- critic 또는 RAG 구현이 바뀔 때 generator도 바뀐다.
- 잘못된 feedback이 memory를 오염시켰을 때 책임과 rollback 지점이 불명확해진다.
- benchmark sample이 memory에 유입되어 평가 누수가 발생할 수 있다.
- generator 단위 실험과 전체 agentic loop 실험을 분리할 수 없다.

따라서 Generator owner는 **소비 가능한 입력 계약과 재사용 가능한 출력 trace**까지만 책임진다. Memory/RAG owner는 그 trace를 별도 파이프라인에서 선별하고 검색 가능한 memory로 만든다.

## 1. 결론

기준 커밋 `f1a88c3`에서는 `generate` 명령의 조립, artifact 저장,
렌더링, artifact 기반 평가가 연결되어 있었지만 실제 SVG 생성 경로는
완성되지 않았다.

가장 중요한 공백은 다음과 같다.

- `GemmaModelBackend`가 모델을 로드하지 않고 항상 placeholder SVG를 반환한다.
- `GeneratorAgent`는 orchestrator가 전달한 RAG `context`를 사용하지 않는다.
- 모델의 raw text에서 SVG를 추출하는 코드가 존재하지만 generator가 호출하지 않는다.
- critic feedback을 받아 다시 생성하는 generator 계약과 revision loop가 없다.
- RAG backend와 LLM critic은 placeholder다.
- 현재 validator는 `<svg`와 `</svg>` 문자열 유무만 확인하며 XML well-formedness를 검사하지 않는다.
- benchmark가 없고, dataset을 입력으로 실제 generation을 실행하는 evaluator도 없다.
- `simple_instruction_alignment`는 SVG가 비어 있지 않으면 1.0을 주는 placeholder metric이다.
- 기존 generate 통합 테스트는 실제 출력이 아니라 `"Placeholder"` 문자열이 저장되는 것을 성공으로 간주한다.
- critic의 positive/negative feedback과 revision 결과를 다음 요청에서 검색하는 experience memory 경로가 없다.

따라서 권장 순서는 다음과 같다.

```text
계약과 benchmark 후보/평가 경계 정의
        ↓
실제 model backend
        ↓
generator 출력 정제와 실패 처리
        ↓
generator-only 기준선 실험
        ↓
fake RAG로 plug-and-play 계약 검증
        ↓
fake critic으로 revision 계약 검증
        ↓
실제 RAG/critic 도착 시 adapter만 교체하고 ablation
        ↓
별도 memory/RAG owner가 feedback experience retrieval을 연결
```

RAG와 critic의 실제 구현은 generator의 선행 dependency가 아니다. 반면, **RAG context 계약과 critic revision 계약을 fake component로 검증하는 일은 generator 완료의 선행 dependency**다.

benchmark도 부가 작업이 아니라 dependency다. benchmark 없이 실제 모델을 연결하면 “실행된다”는 사실만 알 수 있고, prompt, decoding, retry, RAG, critic이 품질을 개선했는지 또는 회귀를 만들었는지 판단할 수 없다.

Critic feedback memory는 좋은 후속 연구 방향이다. 다만 model weight가 자동으로 갱신되는 것은 아니므로 초기에는 “self-evolving model”보다 **retrieval-conditioned experience reuse** 또는 **self-improving generation loop**라고 기술하는 편이 정확하다. 실제로 weight까지 개선하려면 memory trace를 SFT correction tuple이나 preference pair로 변환하는 별도 training 단계가 필요하다.

## 2. 기준선 런타임 흐름 (`f1a88c3`)

기준선의 `svg-agentic-slm generate ...` 호출은 다음 경로를 따른다.

```text
CLI commands_generate.py
        ↓
factory build_generation_runtime()
        ├── generation/model/rag/paths YAML 로드
        ├── GemmaModelBackend 생성 및 load_model() 호출
        ├── GeneratorAgent 생성
        ├── optional RAGAgent/critic/renderer 생성
        └── SVGGenerationOrchestrator 생성
        ↓
orchestrator.run()
        ├── optional RAG retrieve + format_context
        ├── generator.generate(request, context)
        ├── validator.validate(svg)
        ├── optional renderer.render(svg)
        └── optional critic.critique(...)
        ↓
persist_generation_artifacts()
        ├── .svg
        ├── optional render
        └── .json sidecar
```

조립 구조와 constructor dependency injection은 이미 좋은 출발점이다. 계획에서는 이 흐름을 유지하고, placeholder 구현과 불완전한 경계만 단계적으로 교체한다.

## 3. 기준선 코드별 상태 (`f1a88c3`)

### 3.1 Generator와 model

| 파일 | 현재 구현 | generator 구현에 미치는 영향 |
|---|---|---|
| `agents/base.py` | `BaseGenerator.generate(request, context) -> str` 계약이 존재한다. | 최초 생성은 표현할 수 있지만 raw output, retry 정보, revision을 표현할 수 없다. |
| `agents/generator.py` | system prompt와 user prompt를 문자열로 합친 뒤 model backend를 호출한다. | chat template을 사용하지 않고, 전달된 `context`를 버리며, raw output을 그대로 SVG로 반환한다. |
| `models/base.py` | `load_model`, `generate`, `is_loaded`, `unload_model` 추상 계약이 있다. | backend 교체 지점은 있으나 model request/response provenance와 token usage 계약이 없다. |
| `models/gemma_loader.py` | load는 로그만 남기고, generate는 placeholder SVG를 반환한다. | 현재 CLI가 성공해도 실제 모델 추론은 전혀 실행되지 않는다. |
| `models/generation_config.py` | 기본 sampling 설정과 `from_dict`가 있다. | `num_return_sequences > 1`의 반환 계약이 없고, 알 수 없는 config key가 조용히 무시된다. |
| `prompts/system_prompts.py` | 기본 generator system prompt가 있다. | 버전 식별자와 실험 추적 정보가 없다. |
| `prompts/text_to_svg.py` | RAG example prompt와 revision prompt helper가 있다. | generator는 둘 다 실제 runtime에서 사용하지 않는다. |

### 3.2 RAG 연결 경계

| 파일 | 현재 구현 | generator 구현에 미치는 영향 |
|---|---|---|
| `agents/rag_agent.py` | retriever 결과를 문자열 context로 포맷한다. | orchestrator까지 context가 전달되지만 generator가 무시한다. |
| `rag/base.py` | retriever의 add/retrieve/clear 계약이 존재한다. | future retriever를 주입할 기본 경계는 준비되어 있다. |
| `rag/chroma_store.py` | 모든 메서드가 placeholder이며 retrieve는 항상 빈 리스트다. | 현재 `--rag`는 품질이나 prompt를 바꾸지 않는다. |
| `rag/document_loader.py` | JSONL을 retriever document 형식으로 바꾼다. | runtime factory에서 corpus ingestion을 호출하지 않으므로 index bootstrap 경로가 없다. |
| `rag/schemas.py` | content, description, score, source, metadata가 있다. | retrieval provenance를 담을 수 있지만 artifact에는 item별 정보가 저장되지 않는다. |
| `configs/rag.yaml` | backend, corpus, top-k, threshold 등이 있다. | `similarity_threshold`, corpus path, chunk 설정은 runtime에서 사용되지 않는다. |
| `data/rag_corpus/svg_patterns_sample.jsonl` | 4개의 간단한 snippet만 있다. | 실제 retrieval 품질 실험용 corpus나 benchmark로 사용할 수 없다. |

### 3.3 Critic과 revision 연결 경계

| 파일 | 현재 구현 | generator 구현에 미치는 영향 |
|---|---|---|
| `agents/base.py` | `BaseCritic.critique(...) -> CriticFeedback` 계약이 있다. | future critic 교체 지점은 있다. |
| `agents/rule_critic.py` | validator error 수로 점수를 계산한다. | instruction alignment는 항상 `True`이며 revision에 쓸 actionable suggestion이 없다. |
| `agents/llm_critic.py` | 고정된 5점 placeholder feedback을 반환한다. | 실제 model call과 structured parsing이 없다. |
| `factories/generation.py` | rule/llm/both critic을 조립하며 composite critic도 있다. | 선택 구조는 있으나 composite가 하위 critic 결과를 하나로 평탄화한다. |
| `agents/orchestrator.py` | critic을 한 번 호출해 feedback을 저장한다. | `max_revision_rounds`가 전달되지만 사용되지 않고 `revision_count`는 항상 0이다. |
| `prompts/text_to_svg.py` | `build_revision_prompt`가 존재한다. | 호출자가 없고 revision generator 계약도 없다. |

### 3.4 SVG 처리

| 파일 | 현재 구현 | generator 구현에 미치는 영향 |
|---|---|---|
| `svg/normalizer.py` | whitespace normalize와 `<svg>...</svg>` 추출이 있다. | generator가 호출하지 않으며 code fence, 다중 SVG, truncation 정책이 없다. |
| `svg/validator.py` | opening/closing tag와 xmlns warning만 검사한다. | malformed XML, script, external resource, 위험 attribute도 valid로 통과할 수 있다. |
| `svg/renderer.py` | CairoSVG 렌더링이 실제 구현되어 있다. | 현재 가장 신뢰할 수 있는 실행 가능성 신호지만 render failure를 generation failure로 처리하지 않는다. |
| `svg/diff.py` | text unified diff가 있다. | revision 전후 변화 기록에 사용할 수 있으나 현재 연결되지 않았다. |

### 3.5 Benchmark, evaluation, artifact

| 파일 | 현재 구현 | generator 구현에 미치는 영향 |
|---|---|---|
| `data/examples/text_to_svg_sample.jsonl` | 단순 예시 3개가 있다. | smoke fixture일 뿐 범주, 난이도, held-out split, provenance가 없어 benchmark가 아니다. |
| `eval/evaluator.py` | 저장된 generation artifact를 평가한다. | dataset을 읽어 generator를 실행하는 경로는 `not_implemented`다. |
| `eval/metrics.py` | validity, render success, latency가 있다. | alignment는 non-empty 여부만 보므로 생성 품질 판정에 사용할 수 없다. |
| `eval/run_eval.py` | artifact directory 또는 sidecar를 평가한다. | benchmark prompt batch를 생성하고 variant를 비교하는 runner가 없다. |
| `artifacts/generation.py` | 이동 가능한 sidecar-relative artifact reader가 있다. | 실험 입력으로 재사용하기 좋다. |
| `factories/generation.py` | config와 결과 metadata를 sidecar에 저장한다. | model revision, prompt version, raw model output, retry, token 수, dependency version이 빠져 있다. |

### 3.6 Config, dependency, test

| 영역 | 현재 상태 | 위험 |
|---|---|---|
| `pyproject.toml` | transformers, chromadb, cairosvg, lxml 등이 선언되어 있다. | RAG embedding model에 필요한 runtime dependency가 명시적으로 고정되어 있지 않고, `bitsandbytes`는 `environment.yml`에만 있다. |
| YAML config | 파일 분리와 CLI override가 동작한다. | typed validation이 없고 사용되지 않는 옵션이 많아 config가 적용된 것처럼 보일 수 있다. |
| generate tests | config, artifact, render, override를 잘 검증한다. | 실제로 `"Placeholder"`가 저장되는 것을 assert하므로 real backend 전환 시 반드시 수정해야 한다. |
| orchestrator tests | stub injection으로 기본 run을 검증한다. | RAG context 전달, revision, stop policy, failure path를 검증하지 않는다. |
| RAG/critic/model tests | 전용 테스트가 없다. | plug-and-play 계약의 회귀를 잡을 수 없다. |
| 전체 테스트 | 현재 38 passed, 3 skipped 상태다. | 통합 wiring이 건강하다는 의미이지 model quality나 실제 generation이 구현됐다는 의미는 아니다. |

### 3.7 나머지 레포 영역

| 영역 | 현재 구현 | generator와의 관계 |
|---|---|---|
| `cli/commands_generate.py` | runtime을 조립하고 실행한 뒤 artifact 경로를 출력한다. | CLI는 충분히 얇게 유지되어 있으므로 generator 로직을 이 파일로 옮기지 않는다. |
| `cli/commands_eval.py` | artifact source와 report 위치 override를 지원한다. | 향후 benchmark generation mode를 추가하더라도 config parsing과 실행 service를 분리한다. |
| `cli/commands_render.py`, `commands_validate.py` | standalone render/validate가 동작한다. | generator failure를 재현하는 독립 진단 도구로 재사용할 수 있다. |
| `data/preprocess.py` | placeholder다. | benchmark와 training dataset을 혼합하지 말고 benchmark 전용 loader/schema를 우선 만든다. |
| `data/text_to_svg_dataset.py` | JSONL load는 동작하지만 split/filter 기능은 없다. | dataset-backed evaluator의 입력 기반으로 확장할 수 있다. |
| `train/*` | LoRA config shell은 있으나 PEFT/SFT와 저장은 placeholder다. | generator v1의 blocker는 아니며, 나중에 adapter checkpoint를 backend가 선택적으로 로드할 수 있게만 계약을 열어 둔다. |
| `utils/config.py`, `cli/overrides.py` | YAML load와 recursive CLI override가 동작한다. | typed validation과 “실제로 소비된 key” 검증을 추가할 기반이다. |
| `utils/paths.py`, `utils/seed.py` | project path와 seed helper가 동작한다. | experiment manifest에 resolved path와 seed를 기록하도록 재사용한다. |
| `scripts/smoke_test.py` | import, config, 간단 validator/JSONL만 검사한다. | 실제 model generation이나 RAG/critic 연결 성공을 증명하지 않는다. |
| `docs/generate-command-workflow.md` | ownership, config, artifact 계약을 설명한다. | generator contract가 바뀔 때 함께 갱신해야 하는 기존 기준 문서다. |

README에는 XML validation, agent refinement, semantic evaluation 등이 구현된 기능처럼 설명되어 있으나 현재 코드는 상당 부분 placeholder다. 구현 단계마다 README의 “현재 제공 기능”과 “계획”을 분리해 갱신해야 한다.

## 4. Dependency 분류

### 4.1 P0: 실제 generator 구현 전에 필요한 dependency

#### A. 실행 대상 model 결정과 접근성

- 정확한 Hugging Face model ID와 immutable revision
- model 사용 권한 및 필요한 인증
- tokenizer/chat template 지원 여부
- GPU 종류, VRAM, CPU offload 허용 범위
- `bfloat16`, 4-bit, 8-bit 중 실제 지원 조합
- offline/cache 정책과 최초 weight download 방식

현재 config의 `google/gemma-3-4b-it`는 주석상 placeholder다. model 이름을 추측해 구현하지 말고, 첫 반복에서 실제 사용할 model과 revision을 고정해야 한다.

#### B. Benchmark v0

- prompt schema와 category
- smoke/dev/held-out split
- 라이선스와 provenance
- 자동 판정 가능한 constraint
- generator용 benchmark와 RAG corpus의 누수 방지 규칙

benchmark는 model implementation 뒤로 미루지 않는다. 최소 benchmark와 실행 runner가 있어야 backend, prompt, decoding 변경을 비교할 수 있다.

#### C. 최소 신뢰 가능한 validator와 renderer

- XML well-formedness
- root element와 namespace
- script/event handler/external reference 차단
- 렌더 성공 여부
- timeout과 resource limit

generator가 raw text를 SVG로 정제했는지 판정하려면 현재 문자열 validator보다 강한 oracle이 필요하다.

#### D. 안정된 generator/model 계약

최초 구현 전에 아래 정보를 어느 계층이 소유하는지 결정한다.

- system/user chat message 조립
- chat template 적용
- prompt version
- raw model output과 extracted SVG
- finish reason, token usage, latency
- retryable failure와 terminal failure
- context와 revision feedback 전달 형식

이 계약을 먼저 고정하지 않으면 model backend, future RAG, future critic이 모두 generator 내부 구현에 결합된다.

### 4.2 P1: Generator 품질 실험에 필요한 dependency

- 재현 가능한 seed와 deterministic 설정
- 실제 적용된 config를 저장하는 run manifest
- model/tokenizer/library revision 기록
- prompt version과 context 길이 기록
- sample별 raw output, extracted SVG, validation/render 결과
- variant별 report와 비교 도구

### 4.3 P2: 실제 구현은 나중이어도 되지만 계약은 지금 필요한 dependency

- RAG: query를 받아 generator가 소비할 context와 provenance를 반환하는 adapter
- critic: `CriticFeedback`과 revision stop policy
- fake retriever와 fake critic

RAG/critic의 실제 품질은 generator-only baseline 이후 실험한다. 그러나 fake로 연결되는 contract test는 generator 완료 전에 통과해야 한다.

### 4.4 P3: 현재 generator v1의 직접 dependency가 아닌 것

- LoRA/SFT training 구현
- 대규모 RAG corpus 구축
- production vector DB 운영
- LLM critic 자체의 calibration
- HTML dashboard

이 항목들은 generator v1을 막지 않는다. 다만 fine-tuned checkpoint를 generator에 붙일 계획이라면 model adapter loading 계약은 P0 model contract에서 확장 가능하게 설계한다.

### 4.5 Cycle 0 로컬 환경 확인 결과

2026-07-20에 확인한 현재 로컬 환경은 다음과 같다.

| 항목 | 확인 값 | 의미 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4080 Laptop GPU | CUDA compute capability 8.9 |
| VRAM | 12,282 MiB total, 확인 시 약 11,188 MiB free | desktop 사용량을 고려하면 model runtime budget은 10.5~11 GiB 정도로 잡는 것이 안전하다. |
| GPU power limit | 55 W | 같은 RTX 4080 이름의 desktop benchmark보다 token latency가 느릴 수 있다. |
| System RAM | 30 GiB total, 확인 시 21 GiB available | 제한적인 CPU offload는 가능하지만 큰 model 전체를 자주 왕복시키기에는 여유가 크지 않다. |
| Swap | 8 GiB | model inference용 memory로 의존하면 latency가 급격히 나빠진다. |
| Disk | 약 125 GiB free | 7B BF16와 몇 개 quantized checkpoint는 가능하지만 여러 30B variant 보관은 부담이다. |
| 현재 Python | `torch 2.12.1+cpu` | 현재 environment에서는 GPU inference가 불가능하다. |
| 누락 package | `transformers`, `bitsandbytes` | model을 결정한 뒤 CUDA-enabled PyTorch와 함께 환경을 다시 구성해야 한다. |

따라서 “CPU offload를 적극 사용해 큰 model을 억지로 실행”하는 것보다, 로컬에서는 4-bit 7B~12B를 가능한 한 GPU에 상주시켜 latency를 확보하는 것이 우선이다. CPU offload는 OOM 방지용 fallback으로 측정하며 default로 가정하지 않는다.

### 4.6 Model 후보 조사

아래 memory 수치는 공식 수치가 있는 경우 이를 사용했고, 그렇지 않은 4-bit 값은 `parameter_count × 0.5 byte`에 runtime overhead를 더한 **사전 추정치**다. 최종 선택 전에는 동일한 12개 smoke prompt로 peak VRAM과 token latency를 직접 측정해야 한다.

| 후보 | 공식 정보 | 12GB 로컬 판단 | 역할 |
|---|---|---|---|
| `Qwen/Qwen2.5-Coder-7B-Instruct` | code-specific 7.61B, Apache-2.0, instruction-tuned, chat template 제공. BF16 repository가 약 15.2 GB다. | BF16 GPU 단독은 불가. 4-bit이면 대략 5~6 GiB runtime 범위가 예상되어 GPU 상주가 가장 현실적이다. | code-specialized 비교군과 OOM fallback |
| Gemma 4 E4B | Google 공식 Q4_0 load memory 약 4.5 GB | 넉넉하게 fit. 현재 Gemma backend 방향과 잘 맞지만 SVG code 품질은 직접 확인해야 한다. | 빠른 Gemma-family baseline |
| Gemma 4 12B | Google 공식 Q4_0 load memory 약 6.7 GB. 공식 표는 추가 load overhead 20%를 포함한다. | 12GB에 유력하게 fit하지만 KV cache, Cairo render, desktop 점유량을 포함한 실측이 필요하다. | **선택된 local/cloud 공통 baseline** |
| `microsoft/Phi-4-mini-instruct` | dense 3.8B, 128K context, MIT, BF16 files 약 7.69 GB, code를 포함한 training data | BF16도 실행 가능성이 높고 latency floor로 유용하다. SVG code specialization은 Qwen Coder보다 약할 수 있다. | 저지연 비교군 |
| `Qwen/Qwen3-8B` | 8.2B general/reasoning model, thinking/non-thinking mode 제공 | 4-bit fit 예상. `<think>` output과 latency를 피하려면 SVG generation에서는 `enable_thinking=False`부터 비교한다. | 최신 general reasoning 비교군 |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` | total 30.5B, active 3.3B MoE, Apache-2.0, Transformers 4.51+ 필요 | 4-bit raw weight만 약 15.25 GB라 12GB를 초과한다. CPU offload로 실행은 가능할 수 있으나 interactive latency baseline으로는 부적합하다. | 향후 24GB+ cloud 후보 |

공식 자료:

- [Qwen2.5-Coder-7B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
- [Gemma 4 model overview and inference memory table](https://ai.google.dev/gemma/docs/core)
- [Phi-4-mini-instruct model card](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B)
- [Qwen3-Coder-30B-A3B-Instruct model card](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)

### 4.7 Model 결정과 검증 실험

Cycle 0의 local model profile은 **Gemma 4 12B QAT Q4_0 GGUF +
CUDA llama.cpp**로 고정한다.

선정 이유:

- 12B급을 사용해 향후 cloud GPU로 옮겼을 때도 지나치게 작은 local-only baseline에 묶이지 않는다.
- 공식 Q4_0 load memory 추정치가 약 6.7 GB이므로 12GB VRAM에서 KV cache와 runtime overhead를 위한 공간을 남길 가능성이 있다.
- 현재 repository의 Gemma-oriented backend/config 방향을 일반화하면서 연결하기 좋다.
- 기본은 GPU resident로 두고, 실제 OOM일 때만 일부 CPU offload를 fallback profile로 측정할 수 있다.

고정된 local 설정은 다음과 같다.

- repository: `lmstudio-community/gemma-4-12B-it-QAT-GGUF`
- file: `gemma-4-12B-it-QAT-Q4_0.gguf`
- revision: `291406f49e16eff811c85ad8884d375f34138663`
- upstream model: `google/gemma-4-12B-it-qat-q4_0-unquantized`
- quantization provider: LM Studio Community
- conversion runtime: `llama.cpp b9518`
- runtime: CUDA-enabled `llama-cpp-python==0.3.34`
- placement: `n_gpu_layers=-1`, full GPU offload 우선
- initial context: 8K
- CPU offload: 명시적 OOM fallback profile

이 compatibility distribution은 이전에 선택한 model family와 QAT/Q4_0
조건을 유지한다. Google-hosted GGUF revision
`9d1c295262db03ac01d47caa328b39ecc2fcdf10`은 완전히 다운로드됐지만
vocabulary token 237922에서 native assert로 종료됐고, 최신 standalone
llama.cpp에서도 동일 문제가 보고됐다. 따라서 OOM fallback이나 runtime
재설치 대상이 아니라 rejected distribution으로 기록한다.

`Q4`라는 이름만 artifact에 남기면 서로 다른 quantizer 결과를 같은 model로
오인할 수 있다. 따라서 실제 run에는 upstream model, distribution revision,
quantization, quantization provider, conversion runtime, compute dtype까지
기록한다.

Qwen2.5-Coder-7B-Instruct 4-bit는 code-specialized challenger이자 Gemma가 12GB에서 안정적으로 동작하지 않을 때의 fallback으로 남긴다. Phi-4-mini 또는 Gemma 4 E4B는 latency floor를 확인할 필요가 있을 때만 비교한다. 이 비교는 선택을 다시 무효화하기 위한 선행 조건이 아니라, 선택된 Gemma baseline의 품질·비용 위치를 이해하기 위한 실험이다.

초기 측정 항목:

- model load 성공과 load latency
- idle/load 후 peak VRAM
- 512, 1024, 2048 completion token에서 peak VRAM
- first-token latency
- output tokens/second
- strict SVG validity와 render success
- prompt echo/code fence/설명 문장 발생률
- 동일 seed 재현성

권장 local profile:

- context는 우선 4K 또는 8K로 제한
- `num_return_sequences=1`
- 4-bit model은 우선 GPU resident
- CPU offload variant는 동일 model이 OOM일 때만 별도 측정
- RAG와 critic은 model 선택 실험에서 비활성화
- desktop GPU 점유를 포함해 최소 1 GiB 이상의 headroom 유지

향후 cloud에서도 우선 같은 Gemma 4 12B artifact contract를 유지하고, device/precision/backend config만 바꿔 local 결과와 비교한다. 더 큰 model을 추가할 때는 Qwen3-Coder-30B-A3B 4-bit 같은 후보를 별도 profile로 검증한다.

## 5. 먼저 고정할 계약

### 5.1 Model backend 계약

현재 `generate(prompt: str, **kwargs) -> str`는 최소 동작에는 충분하지만 실험과 오류 처리를 위해 정보가 부족하다. Cycle 0에서 다음 중 하나를 명시적으로 선택한다.

- 최소 변경안: 현재 문자열 반환을 유지하고 generator가 raw output, timing, retry metadata를 별도로 수집한다.
- 권장안: typed `ModelRequest`와 `ModelResponse`를 추가한다.

권장 `ModelResponse`가 표현해야 할 최소 정보는 다음과 같다.

- `text`
- `finish_reason`
- `prompt_tokens`
- `completion_tokens`
- `model_id`
- `model_revision`
- `latency_seconds`

chat template은 model/tokenizer 특성이므로 backend가 책임진다. SVG extraction, normalization, validation은 model backend가 아니라 generator 계층이 책임진다.

### 5.2 Generator output 계약

orchestrator가 raw model output을 유효한 SVG로 오인하지 않도록 generator 결과를 다음 정보로 구분한다.

- 최종 `svg`
- `raw_text`
- prompt version
- attempt count
- attempt별 failure reason
- 사용한 context의 식별 정보

기존 `GenerationResult.generated_svg`와 artifact contract는 유지하되, 추가 정보는 nested metadata에 기록하는 방식으로 호환성을 유지할 수 있다. schema를 바꾼다면 artifact `schema_version`도 동시에 추가한다.

### 5.3 RAG context 계약

generator가 `ChromaRetriever`나 `RetrievedExample` 구현 세부사항을 직접 알게 하지 않는다. 권장 경계는 agent 계층의 중립적인 context object다.

최소 필드는 다음과 같다.

- prompt에 삽입할 `text`
- item count
- source ID 목록
- score 목록
- context version

RAG agent가 retriever-specific 결과를 이 object로 변환하고, generator는 text와 token budget만 소비한다. 실제 retriever가 아직 없어도 fake RAG agent로 prompt 삽입과 metadata를 검증할 수 있다.

현재 `context: str | None` 계약을 유지한다면 적어도 다음을 고정한다.

- 빈 문자열과 `None`의 동일한 처리
- context가 들어갈 prompt 위치
- delimiter
- 최대 token/character budget
- 사용자 instruction이 context보다 우선한다는 규칙
- context provenance는 orchestrator metadata에 별도 저장

### 5.4 Critic revision 계약

최초 생성과 revision을 구분하는 명시적 API가 필요하다. 권장 방식은 `BaseGenerator`에 revision request를 받는 별도 메서드를 두거나, generation request에 mode와 previous attempt를 typed field로 추가하는 것이다.

revision 입력은 최소 다음을 포함한다.

- original instruction
- previous SVG
- structured `CriticFeedback`
- original RAG context 또는 context ID
- current revision index

stop policy는 critic이나 generator가 아니라 orchestrator가 소유한다.

- 최대 revision 수
- 최소 critic score
- validation/render 실패 시 재시도 여부
- 동일 SVG 반복 감지
- score 개선이 없을 때 중단
- critic failure 시 fail-open 또는 fail-closed

이 정책을 config로 노출하고 artifact에 실제 stop reason을 기록한다.

### 5.5 팀 discussion의 단순 인터페이스 평가

제안된 인터페이스는 개념적으로 적절하다.

```text
Generator
  입력: Prompt + optional RAG context
        + optional previous SVG + feedback
  출력: SVG code

Critic
  입력: Prompt + SVG code
  출력: structured feedback

RAG
  입력: Prompt
  출력: related SVG/context items
```

다만 Python contract에서는 다음 두 invariant를 추가하는 것이 안전하다.

- `previous_svg`와 `feedback`은 둘 다 없거나 둘 다 있어야 한다.
- RAG output은 SVG 문자열 하나가 아니라 source와 type을 가진 context item 목록이어야 한다.

Generator API는 검토 결과 다음 두 안 중 **안 B를 채택**한다.

#### 안 A: 하나의 typed input

```text
GenerationInput(
    prompt,
    retrieved_context=[],
    previous_svg=None,
    feedback=None,
)
```

호출부는 단순하지만 initial/revision mode의 잘못된 조합을 runtime validation으로 막아야 한다.

#### 안 B: initial과 revision 분리

```text
generate(prompt, retrieved_context) -> SVG
revise(prompt, retrieved_context, previous_svg, feedback) -> SVG
```

결정은 B다. initial generation benchmark와 correction benchmark를 분리하기 쉽고, 향후 generator fine-tuning example도 명확해진다. 내부에서는 두 메서드가 같은 model 호출, SVG extraction, validation 코드를 재사용한다. 즉 API만 분리하며 구현을 두 벌로 만들지는 않는다.

현재 `GenerationRequest`를 유지하는 최소 형태는 다음과 같다.

```text
generate(request, context=None) -> SVG
revise(request, previous_svg, feedback, context=None) -> SVG
```

향후 typed output을 도입하면 두 메서드 모두 같은 `GenerationOutput`을 반환한다.

현재 코드와의 대응:

- Critic 계약은 이미 `critique(instruction, svg_content) -> CriticFeedback`으로 discussion과 호환된다.
- RAG 계약은 이미 `retrieve(query) -> list[RetrievedExample]`에 가깝지만 backend가 placeholder다.
- Generator initial 계약은 `generate(request, context) -> str`로 존재한다.
- Generator revision 계약만 없다.
- `build_revision_prompt()` helper는 있지만 runtime에서 사용되지 않는다.
- 현재 Generator는 RAG `context`도 실제 prompt에 넣지 않는다.

### 5.6 Feedback experience memory

#### 아이디어

Generator–Critic loop에서 생기는 성공과 실패를 experience로 저장하고, 비슷한 prompt가 들어왔을 때 RAG가 static SVG corpus와 함께 검색한다.

```text
Prompt
  ↓
RAG retrieves
  ├── static SVG examples
  ├── positive experiences
  └── negative lessons / correction pairs
  ↓
Generator draft
  ↓
Critic feedback
  ↓
Generator revision
  ↓
separate memory curator selects reusable experience
```

이 방향은 SVG domain과 잘 맞는다. [IntroSVG](https://arxiv.org/abs/2603.09312)는 rendering feedback을 사용한 generate–critique–refine loop와 failure-to-correction training data의 효과를 보고하며, correction SFT와 preference alignment를 함께 사용한다. 다만 여기서 제안하는 RAG experience memory는 weight update 없이 과거 경험을 context로 재사용한다는 점에서 다른 층의 기능이다.

#### Generator owner의 책임

- prompt, RAG context ID, previous SVG, critic feedback, revised SVG의 연결 관계를 잃지 않는다.
- 각 attempt의 raw output과 final SVG reference를 trace로 제공한다.
- validation/render/critic 결과와 stop reason을 metadata에 제공한다.
- RAG가 반환한 `context_kind`를 구분해 prompt에 안전하게 렌더링한다.
- fake memory context로 positive/negative section이 올바르게 삽입되는지 테스트한다.

#### Generator owner가 구현하지 않을 것

- feedback을 positive/negative로 최종 판정
- memory DB schema와 vector index
- embedding과 hybrid retrieval
- deduplication, confidence decay, eviction
- critic score calibration
- benchmark sample의 memory 유입 차단 파이프라인
- memory를 training dataset으로 변환하는 job

이 기능들은 memory curator/RAG/evaluation owner의 책임이다. Generator가 직접 vector DB에 write하지 않는다.

#### 권장 experience memory item

다른 owner에게 요청할 최소 record는 다음과 같다.

```json
{
  "memory_id": "exp_...",
  "kind": "positive_example | negative_lesson | correction_pair",
  "prompt": "original prompt",
  "source_attempt_id": "run_id:attempt_index",
  "accepted_svg_ref": "optional artifact reference",
  "failed_svg_ref": "optional artifact reference",
  "feedback": {
    "score": 0.0,
    "issues": [],
    "suggestions": []
  },
  "outcome": {
    "validated": true,
    "rendered": true,
    "accepted": true,
    "improvement_delta": null
  },
  "confidence": 0.0,
  "provenance": {
    "critic_type": "rule | llm | human",
    "critic_version": "...",
    "generator_model": "...",
    "prompt_version": "...",
    "created_at_utc": "..."
  }
}
```

negative memory는 실패 SVG만 그대로 few-shot으로 넣지 않는다. 기본 retrieval payload는 “무엇이 실패했고 무엇을 피해야 하는지”라는 lesson이어야 하며, correction pair가 검증된 경우에만 failed→corrected SVG 쌍을 함께 제공한다.

#### Retrieval context type

RAG가 generator에 반환하는 item은 최소 다음 kind를 구분한다.

- `reference_svg`: static corpus의 관련 SVG
- `positive_experience`: 유사 prompt에서 검증된 성공 SVG와 성공 이유
- `negative_lesson`: 피해야 할 issue와 correction guidance
- `correction_pair`: failed SVG, feedback, corrected SVG

Generator는 이를 다음처럼 서로 다른 delimiter로 넣는다.

```text
<reference_examples>...</reference_examples>
<successful_experiences>...</successful_experiences>
<lessons_to_avoid>...</lessons_to_avoid>
<correction_examples>...</correction_examples>
```

instruction은 항상 이 context보다 높은 우선순위를 가진다.

#### “Self-evolving” claim의 검증 조건

다음 ablation 없이 self-evolving 효과를 주장하지 않는다.

- static RAG only
- experience memory only
- static RAG + experience memory
- positive only
- negative lesson only
- positive + negative
- memory 없음

동일 benchmark, model, prompt version, seed에서 round가 진행될수록 held-out prompt 성능이 올라가는지 측정한다. 같은 prompt 재시도만 좋아지거나 benchmark item을 memory에 저장한 경우는 self-evolution evidence로 보지 않는다.

#### 향후 fine-tuning 연결

Generator fine-tuning에는 artifact trace를 다음 형태로 변환할 수 있다.

- direct SFT: `prompt + context -> accepted SVG`
- correction SFT: `prompt + context + previous SVG + feedback -> corrected SVG`
- preference tuning: rejected draft와 accepted correction을 chosen/rejected pair로 구성

Critic fine-tuning은 별도 범위다.

- 입력: `prompt + SVG code`, 가능하면 rendered image 포함
- 출력: calibrated structured feedback
- label: human 또는 검증된 teacher feedback
- 평가: score calibration, issue localization, suggestion usefulness

Generator owner는 이 fine-tuning을 구현하지 않고, 미래 training owner가 사용할 수 있도록 trace의 입력–출력 관계만 보존한다.

### 5.7 현재 artifact 호환성과 최소 확장안

#### 현재 저장 형식

각 run은 다음을 저장한다.

- 최종 `.svg`
- optional render file
- `.json` sidecar

현재 sidecar top-level:

- `instruction`
- `svg_path`
- `is_valid`
- `render_path`
- `revision_count`
- `critic_feedback`
- `runtime`
- `metadata`
- `generated_at_utc`

`critic_feedback`에는 score, validity, instruction match, issues, suggestions, critic type, raw response가 이미 저장된다. `metadata`에는 request, RAG count, validation, render, critic count, timing이 들어간다.

#### 호환되는 부분

- `Critic: Prompt + SVG -> feedback`은 현재 artifact로 저장 가능하다.
- final SVG와 prompt의 연결은 이미 보존된다.
- sidecar-relative path를 사용하므로 artifact bundle 이동이 가능하다.
- loader는 `payload.get(...)`과 default를 사용해 일부 missing field가 있는 과거 artifact도 읽는다.
- loader가 알지 못하는 top-level field는 무시하고 `metadata` 전체는 보존하므로 nested additive metadata는 비교적 안전하다.

#### 아직 호환되지 않는 부분

여기서 “부족하다”는 현재 artifact가 잘못되었거나 단일 생성 결과를 읽지 못한다는 뜻이 아니다. 현재 포맷은 **최종 SVG 하나와 feedback 목록**을 저장하는 데는 충분하다. 그러나 여러 attempt가 있는 generate–critique–revise 실행을 재현하거나, 어느 수정이 어떤 feedback 때문에 생겼는지 추적하거나, 검증된 example만 memory/training data로 선별하는 질문에는 답할 수 없다.

##### 1. `schema_version`: 이 JSON을 어떤 규칙으로 읽어야 하는가

현재 sidecar에는 format version이 없다. reader가 missing field에 default를 쓰므로 과거 정상 artifact와, 새 필드가 저장 도중 빠진 불완전 artifact를 구별하기 어렵다.

- 추가할 값: top-level `"schema_version": 1`
- compatibility: 이 필드가 없으면 legacy version 0으로 간주
- 의미: artifact JSON 형식의 version이며 model version이나 prompt version과는 별개

예를 들어 version 0에서 `attempts`가 없는 것은 정상이지만, version 1에서 `attempts`가 없으면 writer bug나 incomplete artifact로 경고할 수 있다.

##### 2. raw output: model이 실제로 무엇을 출력했는가

현재 `.svg`에는 최종 `generated_svg`만 저장된다. 앞으로 extractor가 code fence, 앞뒤 설명, 여러 SVG 후보를 제거하거나 XML을 normalize하면 model의 원래 응답은 사라진다.

raw output이 없으면 다음을 구별할 수 없다.

- model이 깨진 XML을 생성했는가
- model은 정상 SVG를 생성했지만 extractor가 잘못 잘랐는가
- token limit 때문에 응답이 중간에 끊겼는가
- prompt 변경으로 설명 문장이나 code fence가 늘었는가

각 attempt에 `raw_output_ref`를 두면 나중에 extractor만 개선해 다시 처리할 수 있다. 큰 문자열을 JSON에 inline하기보다 `attempts/a0.raw.txt` 같은 별도 파일 reference가 안전하다. 저장 정책과 민감 정보 처리 책임은 artifact owner와 합의한다.

##### 3. attempt별 SVG: 중간 draft는 무엇이었는가

현재 `svg_path`는 최종 파일 하나만 가리킨다. formatting retry나 critic revision이 생기면 reject된 draft와 중간 correction은 덮이거나 사라진다.

각 시도에 `attempt_id`와 `svg_ref`를 두면 다음이 가능하다.

- `a0.svg`와 `a1.svg` diff
- 어떤 retry에서 문법이 복구되었는지 확인
- critic revision 전후 품질 비교
- 미래 correction SFT의 failed→corrected pair 구성

SVG extraction 자체가 실패한 attempt는 `svg_ref: null`이어도 된다. final top-level `svg_path`는 그대로 유지해 기존 consumer를 깨지 않는다.

##### 4. feedback target attempt ID: 이 평가는 어느 SVG에 대한 것인가

현재 `critic_feedback`은 순서가 있는 목록일 뿐 `feedback_id`와 대상 ID가 없다. feedback 한 개일 때는 문제가 없지만 multiple critic, critic retry, branch candidate가 생기면 “목록의 두 번째 feedback은 두 번째 SVG용”이라는 순서 추론이 깨진다.

각 feedback event에는 최소 다음이 필요하다.

```json
{
  "feedback_id": "f0",
  "target_attempt_id": "a0"
}
```

그러면 `f0`가 final SVG가 아니라 최초 draft `a0`를 평가했다는 사실이 명확하다. 현재 persistence는 `CriticFeedback`의 알려진 필드만 골라 새 dict를 만들기 때문에, 이 두 필드는 Generator metadata만으로 완전히 해결되지 않는다. shared `CriticFeedback` schema와 artifact serializer 변경은 critic/structure owner와 합의해야 한다.

##### 5. previous→revised 연결: 이 revision은 무엇에서 파생되었는가

attempt 배열의 순서만으로 parent를 추정하면 후보를 두 개 생성하거나 이전 attempt로 rollback하는 순간 lineage가 모호해진다. revision attempt에 아래 두 reference를 둔다.

- `parent_attempt_id`: 수정 전 SVG attempt
- `trigger_feedback_id`: 이 수정을 유발한 feedback

예를 들어 `a1.parent_attempt_id = a0`, `a1.trigger_feedback_id = f0`이면 “`a0`에 대한 `f0`를 반영해 `a1`을 만들었다”를 정확히 표현한다. Generator가 이 input provenance를 만들어야 하지만, revision을 몇 번 할지와 어느 branch를 선택할지는 orchestrator 책임이다.

##### 6. accepted/rejected outcome: 유효한 SVG와 채택된 SVG는 같은가

현재 `is_valid`와 critic score는 관측값이지 최종 의사결정이 아니다.

- XML이 유효해도 instruction을 어겨 reject될 수 있다.
- 최대 revision 수에 도달하면 낮은 score의 SVG를 best-effort로 accept할 수 있다.
- 더 나은 revision이 생긴 이전 valid attempt는 superseded될 수 있다.
- extraction 실패 attempt는 critic score 자체가 없을 수 있다.

따라서 attempt에는 `outcome: accepted | rejected | failed | superseded`와 `stop_reason`을 둔다. 이 판정은 Generator가 임의로 하지 않고 validator/critic 결과와 stop policy를 가진 orchestrator 또는 memory curator가 확정한다.

이 값은 positive/negative memory의 label과도 동일하지 않다. `accepted`는 positive candidate가 될 수 있지만 human/quality gate를 거쳐야 하고, `rejected` 하나만 저장하기보다 해당 feedback과 accepted correction이 연결된 경우에 `negative_lesson` 또는 `correction_pair`로 만드는 편이 안전하다.

##### 7. model/prompt/critic version: 같은 실행을 다시 만들 수 있는가

현재 runtime config에 model ID가 들어갈 수는 있지만 immutable model/tokenizer revision은 보장되지 않는다. generator prompt는 Python 함수로 존재해 별도 version이 없고, critic artifact에는 `critic_type`만 있다.

같은 model ID의 upstream `main`이 바뀌거나 prompt 문구가 한 줄 수정되면 결과가 달라져도 원인을 찾기 어렵다. 최소 기록 대상은 다음과 같다.

- Generator: model ID, model revision, tokenizer revision, quantization method/config
- Generator prompt: template ID, version 또는 content hash
- Critic: critic type, model ID/revision, critic prompt version
- 실행 코드: repository commit SHA

공통 Generator version은 run metadata에, feedback마다 달라질 수 있는 Critic version은 각 feedback event에 기록한다.

##### 8. RAG item별 source, score, kind: Generator가 무엇을 참고했는가

현재 orchestrator metadata는 RAG 활성 여부와 retrieved example 수 정도만 보존한다. context가 하나의 문자열로 합쳐지면 그 안의 item 경계와 출처가 사라진다.

각 retrieved item에 최소 다음을 기록한다.

- `item_id`: run 내부 reference
- `source`: corpus, document, memory 또는 SVG 식별자
- `score`: retriever가 반환한 원래 점수
- `score_kind`: cosine similarity처럼 높을수록 좋은지, distance처럼 낮을수록 좋은지
- `rank`: 실제 prompt에 들어간 순서
- `kind`: `reference_svg | positive_experience | negative_lesson | correction_pair`
- 가능하면 corpus/index version

그래야 특정 output을 재현하고, 어떤 source가 도움 또는 악영향을 줬는지 ablation하고, benchmark target이 RAG/memory를 통해 유출되었는지 검사할 수 있다. Generator는 이 item을 검색하거나 score를 해석하지 않는다. RAG가 준 typed item을 prompt에 렌더링하고, 실제로 소비한 provenance를 trace로 돌려주는 역할만 맡는다.

##### 추가 guard: memory ingestion 가능 여부

benchmark 실행에서 나온 feedback을 experience memory에 넣으면 held-out test가 다음 실행의 context로 돌아오는 contamination이 생긴다. run-level `memory_eligible: false` 또는 `data_partition: benchmark_test` 같은 guard가 필요하다. 이 값을 강제하고 ingest하는 것은 evaluation/memory owner의 책임이며, Generator는 전달받은 flag를 보존한다.

##### 필드별 소유권

Generator owner가 이 정보를 모두 직접 판정하거나 저장 시스템까지 구현해야 한다는 뜻은 아니다.

| 정보 | 값을 만드는 주체 | Generator owner의 역할 |
|---|---|---|
| `schema_version` | structure/artifact owner | 필요한 trace shape를 제안하고 versioned writer를 소비 |
| raw output | model backend와 Generator | attempt별 raw text/reference를 output trace로 제공 |
| attempt별 SVG | Generator | extraction 결과와 attempt ID를 제공하고, 실제 file persistence는 artifact owner에게 위임 |
| feedback ID와 target ID | critic/orchestrator contract | `revise()`가 받은 feedback reference를 revision trace에 보존 |
| previous→revised lineage | Generator와 orchestrator | `parent_attempt_id`, `trigger_feedback_id`를 반환; branch 선택은 하지 않음 |
| accepted/rejected | orchestrator/memory curator | Generator 내부 formatting failure만 보고하고 최종 채택 판정은 하지 않음 |
| model/prompt version | 각 model/prompt owner | Generator와 Generator prompt provenance를 제공하고 critic version은 전달받아 보존 |
| RAG source/score/kind | RAG owner | typed item을 안전하게 소비하고 실제 사용한 item provenance를 반환 |

따라서 Cycle 0에서 Generator owner가 직접 완성할 핵심은 `generate()/revise()`의 공통 output trace, raw/extracted 결과, attempt lineage, Generator version, 사용한 RAG item reference다. top-level sidecar migration, 중간 파일 write, memory label 확정은 다른 owner에게 dependency로 요청한다.

#### Generator owner가 사용할 최소 확장

기존 top-level field와 `.svg` final artifact는 바꾸지 않는다. Generator는 우선 다음 nested metadata를 반환한다. 아래 예시는 `a0 → f0 → a1`이라는 한 번의 revision을 표현한다.

```json
{
  "schema_version": 1,
  "instruction": "Draw a blue circle centered on a white canvas.",
  "svg_path": "final.svg",
  "is_valid": true,
  "revision_count": 1,
  "critic_feedback": [
    {
      "feedback_id": "f0",
      "target_attempt_id": "a0",
      "score": 6.0,
      "issues": ["The circle is not centered."],
      "suggestions": ["Move cx and cy to the canvas center."],
      "critic": {
        "critic_type": "llm",
        "model_id": "critic-model",
        "model_revision": "immutable-revision",
        "prompt_version": "critic-v1"
      }
    }
  ],
  "metadata": {
    "generator": {
      "model": {
        "model_id": "gemma-4-12b",
        "model_revision": "immutable-revision",
        "tokenizer_revision": "immutable-revision",
        "quantization_method": "q4-backend-name"
      },
      "prompt_version": "generator-v1",
      "attempts": [
        {
          "attempt_id": "a0",
          "mode": "initial",
          "parent_attempt_id": null,
          "trigger_feedback_id": null,
          "raw_output_ref": "attempts/a0.raw.txt",
          "svg_ref": "attempts/a0.svg",
          "outcome": "rejected",
          "stop_reason": "critic_revision_requested"
        },
        {
          "attempt_id": "a1",
          "mode": "revision",
          "parent_attempt_id": "a0",
          "trigger_feedback_id": "f0",
          "raw_output_ref": "attempts/a1.raw.txt",
          "svg_ref": "final.svg",
          "outcome": "accepted",
          "stop_reason": "acceptance_threshold_met"
        }
      ]
    },
    "rag": {
      "items": [
        {
          "item_id": "r0",
          "source": "svg-corpus:item-42",
          "corpus_version": "corpus-v1",
          "score": 0.82,
          "score_kind": "cosine_similarity",
          "rank": 1,
          "kind": "reference_svg"
        }
      ]
    },
    "memory": {
      "eligible": false,
      "reason": "benchmark_test"
    }
  }
}
```

관계만 줄이면 다음과 같다.

```text
a0 (initial draft)
  └── reviewed by f0 (target_attempt_id=a0)
        └── produces a1 (parent=a0, trigger=f0)
              └── accepted as final.svg
```

실제 sidecar schema version과 attempt file naming/persistence는 structure/integration owner와 합의한다. 특히 Generator가 반환한 `metadata`는 현재 serializer가 그대로 보존하므로 `metadata.generator.attempts`는 additive하게 시작할 수 있다. 반면 top-level `schema_version`, feedback ID, attempt file write는 shared artifact code 변경이 필요하다. 권장 migration은 다음과 같다.

- 새 sidecar에 `schema_version: 1` 추가
- 기존 sidecar는 version 0으로 간주
- 기존 top-level key의 이름과 type 유지
- intermediate artifact는 final `svg_path`를 대체하지 않고 별도 reference로 추가
- current reader에 version-aware additive parsing 추가

즉, 현재 artifact는 discussion의 **단일 generation + 단일 critic feedback**까지는 호환되지만, feedback memory와 fine-tuning dataset을 만들기에는 attempt-level provenance가 부족하다. 이 부족한 trace를 정의하는 것이 Generator owner가 해야 할 artifact 관련 역할이며, artifact store 자체를 재설계하는 것은 역할 밖이다.

## 6. Benchmark v0 설계

현재 예시 3개를 benchmark라고 부르지 않는다. 이 파일들은 smoke fixture로 유지한다.

### 6.1 권장 구조

초기에는 약 60개 규모의 작고 검토 가능한 benchmark를 권장한다.

- smoke: 12개
- development: 24개
- held-out test: 24개

각 sample은 최소 다음 필드를 갖는다.

```json
{
  "id": "primitive_circle_001",
  "instruction": "Create a blue circle centered on a white 256x256 canvas.",
  "category": "primitive",
  "difficulty": "easy",
  "required_constraints": [
    {"type": "element_count", "element": "circle", "min": 1},
    {"type": "color_present", "value": "blue"},
    {"type": "canvas_size", "width": 256, "height": 256}
  ],
  "reference_svg": null,
  "source": "project-authored",
  "license": "MIT",
  "tags": ["circle", "centered", "color"]
}
```

SVG는 정답이 하나가 아니므로 exact string match를 주 metric으로 사용하지 않는다. reference SVG는 visual comparison이 유용한 sample에만 선택적으로 둔다.

### 6.2 최소 category

- 단일 primitive와 색상
- 여러 도형의 상대적 배치
- background, border, opacity
- text와 정렬
- path/polygon
- transform과 group
- gradient 또는 defs 참조
- explicit canvas size/viewBox
- 금지 요소를 요구하는 adversarial prompt
- 한글을 포함한 비영어 instruction
- 모호한 prompt와 긴 prompt

### 6.3 최소 metric

자동 metric:

- strict XML/SVG validity
- render success
- unsafe element/reference rate
- required constraint pass rate
- generation latency
- output length과 token 수
- retry rate

품질 metric:

- development split에 대한 blind human rubric
- optional visual-text metric 또는 judge metric

placeholder인 `simple_instruction_alignment`는 이름을 유지한 채 실제 metric처럼 사용하지 않는다. 구현 전 report에는 `placeholder`로 명시하거나 기본 metric 목록에서 제거한다.

### 6.4 누수 방지

- benchmark test target SVG를 RAG corpus에 넣지 않는다.
- RAG corpus와 benchmark 사이에 normalized SVG hash 및 description 유사도 검사를 둔다.
- prompt tuning은 development split까지만 사용한다.
- held-out test 결과를 보고 prompt를 수정했다면 benchmark version을 올린다.
- benchmark manifest에 version, hash, 작성자, source/license를 기록한다.

### 6.5 공개 benchmark 조사

최종 primary benchmark는 아직 선택하지 않는다. 아래 후보는 서로 다른
upstream schema, split, prompt source, license 조건을 가지므로 공용
preprocessor에 분기문을 쌓지 않고 **dataset별 adapter**로 조사·검증한다.
SVGenius adapter는 이 원칙을 검증하기 위한 첫 candidate 구현이며, 코드가
존재한다는 이유만으로 최종 benchmark가 되지 않는다.

| 후보 | 규모와 특징 | 이 프로젝트에 좋은 점 | 주의점 | 역할 |
|---|---|---|---|---|
| SVGenius | 총 2,377 queries, understanding/editing/generation, 24 domains, easy/medium/hard complexity. 공개 HF base data는 600 rows이며 Apache-2.0 header와 card 본문의 MIT 표기가 불일치한다. | complexity-stratified generator 분석에 유용하다. Text-to-SVG 외에 향후 revision/editing 평가로 확장 가능하다. | 공개 base table은 `svg_code`와 difficulty 중심이라 별도 caption task와 join해야 한다. license 표기 불일치도 확인 필요하다. | **candidate adapter 구현, 최종 선택 아님** |
| VGBench | 4,279 understanding, 5,845 generation samples. SVG, TikZ, Graphviz를 함께 다루며 rasterized output의 CLIP/FID를 사용한다. | 일반 LLM이 low-level SVG code를 얼마나 생성하는지 비교하기 좋고 공개 논문 baseline이 있다. | generation 전체가 SVG 전용은 아니다. caption이 GPT-4V 생성 후 human filtering된 구조이며 CLIP/FID만으로 instruction fidelity를 충분히 보장하지 못한다. | LLM 비교용 SVG subset |
| StarVector SVG-Bench | 10 datasets, image-to-SVG/text-to-SVG/diagram-to-SVG 3 tasks. SVG-Stack 등 대규모 corpus와 표준 test set을 제공한다. | text-to-SVG 분야의 대표적인 비교 기준이고 specialized model과 비교하기 좋다. | component dataset별 license와 source를 각각 확인해야 하며 전체 train data는 너무 크다. generator v1에는 test subset만 고려한다. | 외부 comparability |
| VectorGym / VG-Text | 약 8,000 unique SVG와 7,000+ human annotations, test 293 records, split은 `svg_id` 기준 leakage 방지. Apache-2.0, 약 5.35 GB, gated access. | human-authored instruction, real-world SVG, 명확한 leakage 관리가 강점이다. correction/editing 연구에도 연결 가능하다. | multimodal/multitask dataset이라 text-only generator가 쓸 subset을 정제해야 하고 access 승인과 download 비용이 있다. | **2차 stress test 후보** |
| SArena / InternSVG | icon, illustration, chemistry, animation domain과 text/image-to-SVG evaluation script를 제공한다. | 향후 복잡 domain과 animation까지 확장할 때 강력하다. | 현재 static text-to-SVG generator v1 범위보다 넓고 evaluation dependency가 무겁다. | cloud/후속 연구 |
| SVGauge + SHE | human-aligned reference-based text-to-SVG metric/benchmark를 제안한다. visual fidelity와 semantic consistency를 결합한다. | CLIP/FID 단독보다 human preference와의 상관을 중시한다. | BLIP-2, SigLIP 등 metric runtime이 무겁고 reference SVG가 필요하다. | metric 후보, primary dataset 아님 |

공식 자료:

- [SVGenius paper](https://arxiv.org/abs/2506.03139) / [SVGenius dataset](https://huggingface.co/datasets/xiaoooobai/SVGenius)
- [VGBench project](https://vgbench.github.io/) / [EMNLP 2024 paper](https://aclanthology.org/2024.emnlp-main.213/)
- [StarVector and SVG-Bench](https://starvector.github.io/)
- [VectorGym dataset](https://huggingface.co/datasets/ServiceNow/VectorGym)
- [InternSVG and SArena](https://github.com/hmwang2002/InternSVG)
- [SVGauge paper](https://arxiv.org/abs/2509.07127)

### 6.6 Benchmark 선택에서 중요한 기준

#### 1. 현재 task와 입력 modality가 맞는가

현재 generator는 text-only `Prompt -> SVG code`다. Image-to-SVG, sketch-to-SVG, animation benchmark가 포함되어 있어도 text-only subset이 명확히 분리되지 않으면 초기 benchmark로 부적합하다.

#### 2. output representation이 맞는가

일부 benchmark는 path-only, restricted primitive, tokenized representation을 전제로 한다. 이 프로젝트는 arbitrary SVG code를 출력하므로 root SVG, defs, text, transforms 등 실제 지원 범위가 일치해야 한다.

#### 3. Prompt가 생성 품질을 판정하기에 충분한가

- human-authored인지 synthetic caption인지
- 색, 개수, 위치, style constraint가 명시적인지
- caption이 target SVG의 세부사항을 충분히 설명하는지
- 한글 또는 multilingual coverage가 있는지

너무 짧은 caption과 복잡한 reference SVG의 비교는 model보다 caption ambiguity를 측정할 수 있다.

#### 4. SVG complexity가 층화되어 있는가

단순 icon만 평가하면 `<circle>`, `<rect>` 위주의 model이 과대평가된다. element 수, path command/control point, nesting, defs/gradient, text, transform, output token length 기준으로 easy/medium/hard가 분리되어야 한다.

#### 5. metric이 원하는 품질과 맞는가

최소 세 층을 분리한다.

- code: XML validity, safety, size, editability
- render: render success, visual artifact, reference similarity
- semantics: prompt constraint, text-image alignment, human preference

FID/CLIP 하나로 합치지 않는다. 특히 작은 test set의 FID와 non-empty alignment는 신뢰하지 않는다.

#### 6. Reference SVG 의존성이 적절한가

같은 prompt에 여러 정답 SVG가 가능하다. Exact code match나 pixel match만 사용하면 창의적인 valid output을 벌점 줄 수 있다. Reference-based metric과 prompt-based constraint/human judge를 함께 사용해야 한다.

#### 7. Human annotation과 metric calibration이 있는가

자동 judge가 human ranking과 얼마나 상관되는지, annotator agreement와 rubric이 공개되어 있는지 확인한다. critic model과 같은 계열의 judge만 쓰면 자기 선호 편향이 생길 수 있다.

#### 8. 데이터 누수와 contamination을 통제할 수 있는가

- 공개 benchmark가 base model pretraining에 포함되었을 가능성
- training/RAG corpus와 test SVG 중복
- 같은 `svg_id`의 variant가 train/test에 분리되는 문제
- feedback memory가 held-out sample을 저장하는 문제

Hash뿐 아니라 canonical SVG, rendered perceptual hash, caption similarity를 함께 확인한다.

#### 9. License와 provenance가 명확한가

Dataset repository license와 원본 SVG source license가 모두 명확해야 한다. Repository header와 dataset card의 license가 다르면 채택 전에 upstream에 확인한다.

#### 10. 평가 비용이 감당 가능한가

- test sample 수
- 평균 SVG/output token 길이
- 여러 seed 실행 시간
- CLIP/DINO/VLM judge용 GPU
- render storage

로컬 12GB에서 generator와 heavy judge를 동시에 올리지 못할 수 있으므로 generation과 evaluation을 분리 실행할 수 있어야 한다.

#### 11. 실패 분석이 가능한가

aggregate score뿐 아니라 sample/category/difficulty별 결과와 generated SVG를 볼 수 있어야 한다. 이 조건이 없으면 prompt나 retry를 개선하기 어렵다.

#### 12. Feedback memory 연구와 분리 가능한가

memory on/off를 평가할 held-out split이 있어야 하고, 평가 sample의 critic feedback은 persistent memory에 저장하지 않는 isolation mode가 필요하다.

### 6.7 Benchmark 후보와 선택 절차

Primary external benchmark 선택은 **열려 있다**. Evaluation owner가 후보를
비교하고 task 적합성, provenance, license, metric, 실행 비용을 검토한 뒤
하나 이상의 immutable snapshot을 고정한다. 각 후보는 다음을 명시하는
독립 adapter를 가져야 한다.

- upstream field와 repository 공통 record 사이의 mapping
- text-to-SVG prompt와 evaluation asset의 정확한 split/file
- join, inclusion/exclusion, validation 규칙
- dataset/task revision과 output hash
- license 및 source provenance

현재 SVGenius candidate adapter v2는 HF SVG row와 official caption task를
filename stem으로 join하며, 결과를 `candidate_only`,
`data_partition=candidate_unassigned`, `memory_eligible=false`로 표시한다.
Pinned medium row `page_38_ant_design_48353_icon_95`에는 official caption이
없으므로 dataset/task revision, split, asset key가 모두 일치할 때만 해당
1개를 명시적으로 제외한다. Manifest schema 2는 known/applied exclusion을
보존하며, 다른 mismatch는 계속 strict failure다. 따라서 전체 candidate
shape은 easy 100, medium 99, hard 100의 299개다. 이는 후보의 구조를 읽고
검증하기 위한 준비 코드이지 benchmark 채택, dev/test split 확정,
evaluation runner 구현이 아니다.

조합은 다음과 같다.

1. **Project smoke 12**: schema/validity/render/contract용. 외부 leaderboard 주장은 하지 않는다.
2. **Candidate inspection**: SVGenius adapter의 strict join과 provenance를 소규모로 확인한다.
3. **Primary external — 미정**: Evaluation owner가 아래 기준으로 후보를 비교해 결정한다.
4. **Secondary stress 후보**: VectorGym VG-Text access 후 human-authored complex sample을 검토한다.
5. **Comparison 후보**: VGBench 또는 StarVector SVG-Bench의 text-to-SVG subset을 별도 adapter로 검토한다.
6. **Later metric study**: SVGauge 또는 human rubric을 CLIP/FID 보완용으로 검토한다.

선택 전에 확인할 질문:

- 주요 product domain이 icon인가, illustration인가, diagram인가?
- 한글 prompt 성능이 공식 평가 목표인가?
- reference SVG와 닮는 것이 목표인가, prompt를 만족하는 다양한 SVG가 목표인가?
- 1회 benchmark에 허용할 GPU 시간과 storage는 얼마인가?
- human evaluation 인력을 확보할 수 있는가?
- feedback memory를 논문 핵심 claim으로 둘 것인가, demo feature로 둘 것인가?

위 질문에 답하고 dataset-specific adapter의 실제 output을 검토하기 전에는
최종 benchmark를 선택하지 않는다. 현재 변경에서는 dataset이나 model을
다운로드하지 않았으며, SVGenius 코드는 사용자가 명시적으로 실행할 수 있는
candidate preparation path만 제공한다.

## 7. 반복형 구현 계획

## Cycle 0. 계약, 환경, benchmark의 최소 기반

### 초기 dependency

- 현재 `f1a88c3` 코드와 테스트를 기준선으로 고정
- 실행할 hardware profile: **확인 완료** — RTX 4080 Laptop 12GB, system RAM 30GB
- model 결정: **교체 완료** — Google QAT upstream 기반 LM Studio Community Q4_0 compatibility pin, immutable revision, CUDA llama.cpp local profile
- benchmark 후보 조사: **진행 중** — SVGenius-specific candidate adapter는 구현했으나 최종 benchmark 선택은 Evaluation owner dependency
- artifact 호환성 분석: **완료** — 기존 top-level을 유지하고 nested generator trace를 additive하게 제공
- initial/revision API: **완료** — `generate()`와 `revise()` 분리
- context item 상세 schema와 revision stop policy: **구현 후보 완료** — 관련 owner review 필요
- RAG free-form metadata 정책: **구현 완료** — typed field만 공유하고 Cycle 0 whitelist는 빈 집합
- CUDA-enabled llama.cpp runtime profile: **로컬 native build 검증 완료**
- replacement 6.98 GB GGUF load, VRAM headroom, TTFT/tokens-per-second 실측:
  **대기** — 이전 Google GGUF는 vocabulary assert로 reject되었고 replacement E2가 필요함

### 무엇을 구현하는가

- model request/response와 generator output의 typed contract
- RAG context와 critic revision의 중립 계약
- `GenerationResult.metadata["generator"]`에 넣을 generator trace producer
- fake model backend, fake RAG, fake critic
- generator가 직접 소비하는 config의 fail-fast validation
- initial/revision의 잘못된 입력 조합 validation

Generator owner는 이 Cycle에서 production benchmark runner/evaluator, vector
DB, critic, artifact store migration을 구현하지 않는다. Dependency 검토를
위해 추가한 SVGenius candidate preparation utility는
`benchmarks.svgenius`에 격리하며, 최종 dataset 선택·adapter ownership·batch
evaluation은 Evaluation owner가 검토한다.

다른 owner에게 요청할 dependency:

- structure owner: `schema_version`과 attempt artifact reference persistence
- evaluation owner: 최종 benchmark 선택, dataset별 adapter 승인, batch runner
- RAG owner: 중립 context item schema 검토
- critic/orchestrator owner: `CriticFeedback`과 revision stop policy 검토

변경 예상 파일:

- `agents/base.py`
- `agents/schemas.py`
- `models/base.py`
- 신규 model schema 파일
- `agents/generator.py`
- `prompts/text_to_svg.py`
- 관련 tests

### 무엇을 실험하는가

- fake backend로 initial과 revision input/output contract 확인
- fake RAG context가 generator 호출까지 정확히 전달되는지 확인
- fake critic이 정해진 feedback을 반환했을 때 revision input이 정확히 구성되는지 확인
- generator trace가 JSON serialization 가능한지 확인

이 단계의 placeholder score는 품질 기준선으로 사용하지 않고 infrastructure 기준선으로만 보관한다.

### 종료 조건

- model, prompt, seed, config, context, attempt 정보가 generator metadata로 제공된다.
- fake RAG/critic contract test가 실제 RAG/critic 구현 없이 generator 범위에서 통과한다.
- feedback 없이 previous SVG만 들어오는 잘못된 revision input이 거부된다.
- generator가 소비하는 unknown 또는 미지원 config가 조용히 무시되지 않는다.
- 최종 benchmark 선택과 artifact persistence는 다른 owner의 대기 dependency로 명시되어 있고 Generator PR을 불필요하게 확장하지 않는다.

## Cycle 1. 실제 model backend와 최소 generator baseline

### 초기 dependency

- Cycle 0 계약
- 선택된 Gemma 4 12B GGUF checkpoint ID와 immutable revision: **완료**
- GGUF metadata의 Gemma 4 chat template 사용 정책: **완료**
- local Q4_0 / CUDA full-offload 정책: **완료**
- model license와 Hugging Face cache 위치 확인
- CUDA-enabled `llama-cpp-python` native environment: **현재 장비에서 검증 완료**
- 실제 weight load와 hardware acceptance measurement: **대기**

### 무엇을 구현하는가

- model ID/file/revision 또는 local path를 받는 backend-neutral GGUF backend
- 기존 `GemmaModelBackend`는 `LlamaCppModelBackend`의 호환 wrapper로 유지
- GGUF metadata 기반 backend-owned chat template 적용
- `create_chat_completion()` 결과를 공통 `ModelResponse`로 변환
- generation config merge와 지원 key 검증
- `n_gpu_layers`, context, batch, flash attention, mmap 설정
- load failure 시 placeholder 반환 대신 명시적 failure
- `unload_model()`의 CPU/GPU memory cleanup
- structure owner가 factory에 연결할 수 있는 constructor/config contract 제공

이 단계에서는 RAG와 critic을 비활성화한다.

### 무엇을 실험하는가

- project smoke 12 prompt에 대한 실제 model 출력 확인
- 동일 seed와 deterministic config 반복 실행 비교
- Gemma 4 12B Q4의 GPU-resident 실행과 OOM 시 CPU offload fallback 비교
- 필요할 때만 별도 fallback model profile과 latency, peak memory, SVG validity 비교
- GGUF metadata chat template과 명시적 override 비교
- chat completion response가 prompt echo를 포함하지 않는지 확인
- max token, EOS, truncation failure case 확인

### 종료 조건

- `--no-render`와 RAG/critic 비활성 상태에서 placeholder가 아닌 model completion이 생성된다.
- model을 로드할 수 없으면 명확한 오류와 non-zero CLI exit가 발생한다.
- 동일 deterministic 설정의 결과가 재현된다.
- model backend 단위 테스트가 네트워크와 GPU 없이 fake model/tokenizer로 통과한다.
- 로컬 default가 1 GiB 이상 VRAM headroom을 남기고 측정 latency를 기록한다.

## Cycle 2. SVG 전용 generator 후처리와 실패 정책

### 초기 dependency

- 실제 model completion
- SVG owner가 제공하는 validator contract
- CairoSVG renderer
- raw output fixture: plain SVG, code fence, 설명 포함, truncated XML, 다중 SVG, 빈 출력

### 무엇을 구현하는가

- `GeneratorAgent`의 prompt assembly를 독립 함수로 분리
- raw completion에서 SVG 추출
- code fence와 앞뒤 설명 처리
- SVG normalization
- `max_svg_length` 실제 적용
- injected validator를 호출하는 acceptance/failure 경계
- extraction/validation failure의 typed error
- 제한된 retry와 retry prompt
- raw completion과 attempt별 결과를 metadata에 저장
- prompt version 부여
- 기존 `"Placeholder"` assert를 실제 contract test로 교체

retry는 무한 루프가 되지 않도록 generator 내부 formatting retry와 orchestrator의 critic revision을 구분한다.

- formatting retry: SVG를 추출할 수 없거나 XML이 깨진 경우
- critic revision: 유효한 SVG지만 instruction/quality 개선이 필요한 경우

### 무엇을 실험하는가

- extraction fixture에 대한 단위 실험
- no retry 대비 1회 formatting retry의 validity/render success 개선량
- raw prompt, strict output prompt, chat template prompt 비교
- temperature와 sampling 설정의 작은 grid
- 긴 출력과 truncation에서 `max_new_tokens`, EOS 설정 비교
- unsafe prompt가 script/external reference를 생성하는지 확인

### 종료 조건

- generator가 반환한 최종 문자열은 strict validator를 통과했거나 명시적 failure다.
- raw model text가 그대로 `.svg` artifact로 저장되지 않는다.
- retry 횟수와 stop reason이 artifact에 기록된다.
- code fence, 설명, truncated output, multiple SVG test가 모두 존재한다.
- unsafe SVG가 valid로 보고되지 않는다.

XML parser, whitelist, external reference 검사 자체는 SVG owner의 범위다. Generator owner는 validator 결과를 무시하거나 다시 해석하지 않고 retry/failure 결정에 사용한다.

## Cycle 3. Generator-only benchmark 기준선과 설정 선택

### 초기 dependency

- Cycle 2의 실제 generator
- project smoke split과 evaluation owner가 최종 선택·고정한 benchmark dev subset
- batch generation runner
- 신뢰 가능한 validity, render, constraint metric
- experiment manifest와 report 비교

### 무엇을 구현하는가

- generator-only experiment config profile
- model/prompt/config version을 generator output metadata로 제공
- sample 하나를 재현할 수 있는 generator command/config

다음은 evaluation/structure owner의 dependency이며 Generator owner가 구현하지 않는다.

- variant별 독립 artifact directory와 run ID
- aggregate/category report
- batch runner
- benchmark metric
- run layout과 report comparison

### 무엇을 실험하는가

최소 ablation:

- prompt template A/B
- greedy 또는 deterministic decoding 대 sampling
- temperature/top-p의 작은 범위
- max token 길이
- 기본 정밀도와 quantized backend
- formatting retry 0회/1회

각 stochastic variant는 여러 seed로 실행하고 평균과 분산을 함께 기록한다. held-out test는 설정 선택이 끝난 뒤 한 번만 사용한다.

### 종료 조건

- generator-only baseline `G0`가 versioned report로 저장된다.
- 선택한 기본 config의 근거가 dev metric, latency, memory와 함께 문서화된다.
- placeholder alignment metric 없이도 validity, render, constraint 결과를 비교할 수 있다.
- 실패 sample을 artifact와 command로 재현할 수 있다.
- report framework를 만들기 위해 generator package에 evaluation logic이 유입되지 않는다.

## Cycle 4. RAG plug-and-play 경계 완성

### 초기 dependency

- generator-only baseline `G0`
- Cycle 0의 context contract
- deterministic fake retriever
- context token budget과 precedence 정책
- RAG metadata artifact schema

실제 Chroma 구현과 대규모 corpus는 필요하지 않다.

### 무엇을 구현하는가

- 현재 무시되는 `context`를 generator prompt에 삽입
- 명시적인 context delimiter
- context truncation/token budget
- instruction 우선 규칙
- empty retrieval의 no-op 처리
- generator가 소비한 context kind/source ID를 trace에 기록
- context 포함 prompt의 versioning
- fake context item을 주입하는 generator contract test

generator는 Chroma client를 직접 import하거나 corpus를 직접 로드하지 않는다.
retriever factory와 orchestrator end-to-end integration은 RAG/structure owner가 맡는다.

### 무엇을 실험하는가

- no context와 empty context가 동일 결과를 만드는지 확인
- 정답 패턴이 든 synthetic context가 prompt와 output에 영향을 주는지 확인
- irrelevant context와 conflicting context에 대한 견고성
- context item 순서와 top-k
- context 길이에 따른 latency와 truncation
- prompt injection 형태의 retrieval content 방어

이 실험은 fake/synthetic retrieval로 contract와 generator 반응만 확인한다. RAG retrieval quality를 주장하지 않는다.

### 종료 조건

- RAG 구현 없이 fake context item을 주입해 generator contract test가 통과한다.
- empty context는 generator-only prompt와 동일한 의미를 가진다.
- context 사용 여부와 provenance가 generator trace에서 확인된다.
- future `ChromaRetriever.retrieve()` 구현은 generator 파일 변경 없이 연결 가능하다.

## Cycle 5. Critic revision plug-and-play 경계 완성

### 초기 dependency

- generator-only baseline `G0`
- Cycle 0의 revision 계약
- deterministic fake critic
- revision stop policy
- revision artifact schema

실제 LLM critic은 필요하지 않다.

### 무엇을 구현하는가

- generator revision API 또는 typed revision request
- `build_revision_prompt()`를 generator revision 경로에 연결
- previous SVG와 structured feedback의 prompt rendering
- initial/revision 공통 output extraction과 formatting retry
- revision attempt trace 제공
- fake feedback으로 generator revision contract를 검증하는 test

다음은 critic/orchestrator/structure owner의 dependency다.

- max revision과 score threshold
- validation failure/no-improvement/identical SVG stop
- critic exception 정책
- attempt artifact persistence와 diff
- full revision loop integration test

### 무엇을 실험하는가

- 같은 previous SVG에 positive/negative structured feedback을 넣었을 때 revision prompt 차이
- issue와 suggestion이 empty인 feedback
- 매우 긴 feedback truncation
- feedback prompt injection 방어
- initial과 revision이 동일 extraction/validation contract를 지키는지 확인
- rule/llm/human source의 fake feedback이 동일 generator 계약으로 소비되는지 확인

### 종료 조건

- generator는 previous SVG와 feedback을 함께 소비해 revised SVG를 반환한다.
- previous SVG 또는 feedback 하나만 있는 request를 거부한다.
- revision attempt의 input provenance가 trace에 남는다.
- critic이 없을 때 Cycle 3 baseline 경로에 회귀가 없다.
- future `LLMCritic.critique()` 구현은 generator contract 변경 없이 연결 가능하다.

## Cycle 6. 실제 RAG 도착 후 integration과 ablation

### 초기 dependency

- 구현된 retriever
- versioned RAG corpus
- index build/rebuild command
- embedding model과 revision
- benchmark 누수 검사
- `G0` baseline

### 무엇을 구현하는가

Generator owner의 계획된 추가 구현은 없다.

- RAG owner가 만든 context item이 Cycle 4 계약을 지키는지 확인
- contract mismatch가 있으면 adapter에서 해결
- generator regression test 실행

실제 client, corpus ingestion, similarity threshold, index version, factory, fallback은 RAG/structure owner가 구현한다. 이 단계에서 generator 내부 변경이 필요하면 Cycle 4 계약이 불충분했다는 신호로 본다.

### 무엇을 실험하는가

- `G0`: generator only
- `G0 + RAG`: 동일 model/prompt/seed
- top-k와 threshold ablation
- corpus variant ablation
- retrieval relevance 수동 표본 평가
- category별 개선/악화
- latency와 context token overhead

### 종료 조건

- 실제 RAG가 generator 수정 없이 연결된다.
- benchmark 누수 검사를 통과한다.
- RAG on/off의 quality와 비용 차이가 report로 남는다.
- RAG가 악화시키는 category와 fallback 정책이 문서화된다.

## Cycle 7. 실제 critic 도착 후 integration과 ablation

### 초기 dependency

- 구현된 rule 또는 LLM critic
- structured response parsing
- critic calibration set
- score threshold 근거
- `G0` 및 RAG variant baseline

### 무엇을 구현하는가

Generator owner의 계획된 추가 구현은 없다.

- 실제 `CriticFeedback`이 Cycle 5 revision contract를 지키는지 확인
- source/version metadata가 generator trace로 전달되는지 확인
- generator regression test 실행

critic 연결, parse failure, timeout, composite 결과, 비용 집계는 critic/orchestrator/structure owner가 구현한다. 이 단계에서 generator revision API를 변경하지 않는 것이 목표다.

### 무엇을 실험하는가

- `G0`
- `G0 + critic`
- `G0 + RAG`
- `G0 + RAG + critic`
- revision round 0/1/2
- threshold ablation
- critic score와 human rubric 상관
- critic이 validity는 높이지만 semantic quality를 낮추는 failure 분석
- 총 latency, token, memory 또는 API cost 비교

### 종료 조건

- 실제 critic이 generator 변경 없이 연결된다.
- critic score가 human rubric과 충분히 일치하는지 calibration 결과가 있다.
- 개선 폭보다 비용이 큰 configuration을 기본값으로 채택하지 않는다.
- 최종 default 조합이 held-out report로 결정된다.

## Cycle 8. Feedback experience memory integration

### 초기 dependency

- Cycle 4 context kind 계약
- Cycle 5 attempt/feedback/revision trace
- memory curator가 만든 versioned experience corpus
- RAG owner가 만든 static+experience hybrid retrieval
- benchmark isolation mode와 leakage audit
- critic confidence/calibration 정보

### 무엇을 구현하는가

Generator owner의 구현은 context rendering과 trace 범위로 제한한다.

- `positive_experience`, `negative_lesson`, `correction_pair` context rendering
- kind별 delimiter와 token budget
- untrusted memory content의 prompt injection 방어
- 어떤 memory ID를 소비했는지 generator trace에 기록
- fake memory item contract test

Generator owner는 memory DB write, feedback 선별, index, retrieval ranking을 구현하지 않는다.

### 무엇을 실험하는가

전체 experiment는 memory/RAG/evaluation owner와 공동 수행한다.

- no memory
- static RAG only
- positive memory only
- negative lesson only
- correction pair only
- static RAG + positive/negative memory
- memory round 0/1/2의 held-out 개선
- stale/noisy/adversarial memory에 대한 robustness
- latency와 context token overhead

Generator owner는 동일 model/prompt/config에서 context 소비가 정확한지와 output trace를 제공한다.

### 종료 조건

- experience memory가 generator 파일의 RAG-specific 구현 추가 없이 연결된다.
- memory ID/kind/version이 output trace에 남는다.
- held-out benchmark item이 persistent memory에 들어가지 않는다.
- positive/negative memory가 실제로 품질을 개선하는지 ablation 결과가 있다.
- 개선이 없거나 오염 위험이 크면 memory 기능을 default로 활성화하지 않는다.
- “self-evolving” 표현은 round별 held-out 개선이 재현될 때만 사용한다.

## 8. 실험 공통 규칙

모든 실험은 다음 값을 고정하거나 기록한다.

- git commit
- dirty worktree 여부
- benchmark version/hash와 split
- model ID/revision
- tokenizer revision
- prompt version
- generation config
- seed
- RAG corpus/index/embedding version
- critic model/prompt version
- Python, PyTorch, Transformers, CUDA 버전
- hardware
- 시작/종료 시각

비교는 한 번에 하나의 독립 변수를 바꾸는 ablation을 기본으로 한다. 여러 설정을 동시에 바꿨다면 결과에서 원인을 분리할 수 없다고 표시한다.

각 run은 최소 다음 결과를 남긴다.

- config snapshot
- sample별 raw output
- extracted/final SVG
- render artifact
- validation과 constraint 결과
- retry와 revision trace
- aggregate/category report
- 실패 sample 목록

## 9. 파일별 예상 변경 지도

| 파일 또는 영역 | 계획된 역할 | Generator owner |
|---|---|---|
| `models/base.py` | typed request/response 또는 최소 provenance 계약 | 직접 변경 |
| `models/gemma_loader.py` 또는 신규 HF backend | 실제 model/tokenizer load와 generation | 직접 변경 |
| `agents/generator.py` | prompt 조립, context 소비, SVG extraction, formatting retry, revision | 직접 변경 |
| `agents/base.py` | generator revision/context contract | shared contract로 협의 후 변경 |
| `agents/schemas.py` | output trace, context, revision request schema | shared contract로 협의 후 변경 |
| `prompts/*` | versioned initial/context/revision/memory prompt | 직접 변경 |
| `tests/test_generator*.py` | model, extraction, context, revision contract | 직접 변경/추가 |
| `agents/orchestrator.py` | revision loop와 stop policy | 변경하지 않음 |
| `rag/*` | retrieval, index, experience memory | 변경하지 않음 |
| `agents/llm_critic.py`, `rule_critic.py` | feedback 품질 | 변경하지 않음 |
| `svg/validator.py` | strict XML, safety, structural validation | 변경하지 않음 |
| `factories/generation.py` | backend wiring과 dependency injection | constructor contract만 제공하고 structure owner와 협업 |
| `artifacts/generation.py` | schema version과 trace persistence | metadata shape만 제안하고 structure owner가 변경 |
| `eval/*` | benchmark generation/evaluation/report | 변경하지 않음 |
| `data/preprocess.py` | 공용 데이터 전처리 진입점 | dataset-specific mapping을 넣지 않으며 현재 placeholder 유지 |
| `benchmarks/<dataset>.py` | 후보별 download/join/validation/provenance adapter | SVGenius candidate만 격리 구현; 최종 선택과 runner는 Evaluation owner |
| `configs/` | generator baseline profile | generator-owned key만 제안/변경 |

## 10. 테스트 전략

### 단위 테스트

- model config merge와 unsupported option
- chat template input
- completion-only decode
- SVG extraction/normalization
- injected validator 결과 처리
- context formatting, truncation, precedence
- revision prompt와 input invariant
- positive/negative/correction memory context formatting
- generator trace JSON serialization

### Contract 테스트

- 모든 model backend가 동일 request/response를 준수
- fake RAG context가 generator context contract를 준수
- fake `CriticFeedback`이 generator revision contract로 소비됨
- RAG/critic 구현을 import하지 않고 generator test 실행 가능

### 통합 테스트

- generator only
- generator + empty context
- generator + fake static RAG context
- generator + fake experience memory context
- generator revision + fake feedback

full orchestrator, real RAG/critic, render, artifact 이동 평가는 다른 owner의 integration suite다.

### 느린 실험 테스트

실제 weight와 GPU가 필요한 테스트는 기본 `pytest`에서 분리한다.

- marker 예: `model`, `gpu`, `benchmark`, `slow`
- CI 기본 suite는 fake backend로 실행
- scheduled 또는 수동 job에서 실제 model smoke/dev benchmark 실행

## 11. 주요 위험과 대응

| 위험 | 대응 |
|---|---|
| placeholder가 성공으로 보임 | placeholder fallback 제거, load/generate failure를 명시적 오류로 처리 |
| model이 prompt까지 echo함 | completion token만 decode하고 raw/clean output을 분리 |
| RAG가 구현되어도 context가 무시됨 | Cycle 4 fake RAG contract test를 merge gate로 지정 |
| critic 구현 후 revision API를 다시 설계해야 함 | Cycle 5에서 fake critic으로 generator revision contract를 고정하고 stop policy는 orchestrator owner와 합의 |
| weak validator가 잘못된 SVG를 valid로 판정 | XML, safety, render를 generator 실험 전 P0로 강화 |
| benchmark overfitting 또는 RAG leakage | dev/held-out 분리, manifest, corpus 중복 검사 |
| noisy critic feedback이 memory를 오염 | Generator는 trace만 제공하고, calibrated curator가 confidence/acceptance를 판정 |
| negative SVG를 few-shot으로 모방 | raw failure 대신 negative lesson을 기본 retrieval payload로 사용 |
| benchmark feedback이 experience memory에 저장됨 | evaluation isolation mode와 memory-eligibility flag를 다른 owner의 gate로 요구 |
| “self-evolving”을 과장 | held-out round별 ablation 전에는 experience-augmented로 표현 |
| 역할 범위가 RAG/critic/eval로 확장됨 | 파일별 ownership 표와 contract test까지만 Generator PR에 포함 |
| config가 조용히 무시됨 | typed validation과 실제 적용 config 기록 |
| quantization dependency가 환경마다 다름 | pyproject/environment 차이를 정리하고 hardware profile별 install test |
| multiple return sequence 계약 불명확 | v1에서 1만 지원하거나 typed list 결과를 명시적으로 설계 |
| revision이 무한 반복됨 | max round, identical output, no-improvement stop |
| README와 실제 코드가 불일치 | 각 Cycle 종료 시 current capability 표 갱신 |

## 12. Generator v1 완료 정의

다음 조건을 모두 만족해야 generator v1이 완료된 것으로 본다.

- 실제 model backend가 placeholder 없이 SVG를 생성한다.
- model load 또는 generation 실패가 placeholder SVG로 숨겨지지 않는다.
- generator는 raw model output과 최종 SVG를 구분한다.
- 최종 SVG는 strict validation과 render 결과를 남긴다.
- benchmark v0의 generator-only baseline이 versioned artifact/report로 존재한다.
- prompt와 generation config 선택 근거가 ablation으로 남아 있다.
- fake RAG context가 실제 prompt에 반영되고 provenance가 저장된다.
- fake critic feedback으로 generator revision method가 동작한다.
- full revision loop와 stop policy는 orchestrator contract test에서 별도로 검증된다.
- RAG/critic이 비활성일 때 기존 generator-only 결과 경로가 유지된다.
- future RAG는 retriever/factory adapter 구현만으로 연결 가능하다.
- future critic은 `BaseCritic` 구현만으로 revision loop에 연결 가능하다.
- feedback memory는 kind가 구분된 context item으로 generator 변경 없이 연결 가능하다.
- 기본 test suite는 GPU, network, model weight 없이 통과한다.
- 실제 model smoke/benchmark suite는 별도 명령으로 재현 가능하다.

## 13. 권장 첫 작업 단위

첫 구현 PR은 RAG/critic/evaluator를 건드리지 않고 다음 범위로 제한한다.

1. model/generator/context/revision typed contract
2. initial/revision input invariant
3. generator trace metadata shape
4. fake model, fake RAG context, fake critic feedback contract tests
5. placeholder fallback을 명시적 failure로 바꾸기 위한 error contract
6. structure/RAG/critic owner에게 전달할 contract note

그 다음 PR에서 선택된 real model backend를 구현한다. 최종 benchmark의
dataset adapter, snapshot, subset, license, metric protocol은 Evaluation
owner가 고정하며, Generator owner는 필요한 provenance와 sample 재현
interface만 제공한다. SVGenius-specific adapter는 후보 검토용으로만
유지하고 다른 dataset의 전처리 규칙을 흡수하지 않는다.
