# Generator Cycle 0 Cross-Team Handoff

> RAG·Critic·Orchestration 구현 전 확인할 팀 간 합의

## 문서 관리

| 항목 | 값 |
|---|---|
| 상태 | Implemented candidate for cross-team review |
| 적용 시점 | Generator Cycle 0 change set |
| Primary owner | Generator workstream |
| 필수 reviewer | RAG, Critic, Orchestration/Structure, SVG Validation, Evaluation 담당자 |
| 최종 갱신 | 2026-07-20 |

## 1. 문서 목적

이 문서는 Generator Cycle 0 변경으로 새로 구현되거나 구체화된 팀 간 계약을
RAG, Critic, Orchestration, Artifact, Validation, Evaluation 담당자에게
전달한다. 구현 상세를 모두 설명하는 문서가 아니라, 다른 workstream의
interface나 실험 결과에 영향을 주는 결정만 다룬다.

Normative contract의 전체 내용은
[Generator Cross-Team Contract](./generator-cross-team-contract.md), Cycle별
진행 순서는 [Generator Cycle Roadmap](./generator-cycle-roadmap.md), 실행
상태와 명령은
[Cycle 0 Status and Experiment Runbook](./generator-cycle0-status-and-runbook.md)
을 참조한다.

## 2. 상태 표현

| 상태 | 의미 |
|---|---|
| 구현됨 | 현재 코드와 network/GPU-free test로 확인된다. |
| Cycle 0 baseline | 통합을 위해 현재 사용하는 기본값이나 정책이다. |
| Review 필요 | 구현 후보는 있지만 해당 owner가 수락해야 공유 계약이 된다. |
| 미구현/다른 owner | Generator 변경의 완료로 간주하지 않으며 담당 workstream이 제공한다. |

## 3. 이번 Cycle 0 변경 묶음이 추가한 것

### 3.1 Generator와 model

- `ModelResponse`를 도입해 text뿐 아니라 model revision, token 수, latency,
  finish reason, backend metadata를 전달한다.
- `GeneratorOutput`을 도입해 raw output, extracted SVG, status, error,
  prompt version, attempt ID, model-call trace를 구분한다.
- initial generation과 feedback revision을 `generate()`와 `revise()`로
  분리했다.
- SVG를 추출할 수 없을 때 placeholder를 만들지 않고 failed attempt를
  반환한다.
- real local backend는 config/factory를 통해 교체할 수 있고 Generator는
  llama.cpp 구현을 직접 알지 않는다.

### 3.2 RAG/Critic plug-and-play 기반

- RAG 결과를 한 개의 prompt string이 아니라 `RetrievedExample[]`로 받는다.
- fake RAG item이 실제 Generator prompt에 들어가며 사용/절단된 item ID를
  trace에 남긴다.
- Critic feedback은 target attempt가 명시된 `CriticFeedbackEvent`로
  Generator revision에 전달된다.
- fake Critic을 사용해 attempt → feedback → revised attempt lineage를
  검증한다.

### 3.3 Orchestration과 artifact

- `run_id`, `attempt_id`, `model_call_id`, `feedback_id`를 분리했다.
- artifact schema version 1을 additive하게 기록한다.
- attempt별 SVG, raw output, user/system prompt와 model-call metadata를 별도
  파일로 저장한다.
- feedback target과 previous→revised lineage를 sidecar에서 재구성할 수 있다.
- legacy top-level final SVG/JSON consumer는 유지한다.

### 3.4 Model과 benchmark dependency

- local inference distribution과 immutable revision을 고정했다.
- 실패한 Google-hosted GGUF를 compatibility incident로 reject하고, 동일
  Gemma QAT upstream의 LM Studio Community compatibility distribution을
  선택했다.
- SVGenius를 최종 benchmark가 아닌 candidate로 격리했다.
- SVGenius adapter v2는 pinned upstream의 caption 누락 1건을 명시적으로
  audit하고 299개 candidate snapshot을 만든다.
- dataset-backed accuracy runner와 의미 있는 semantic metric은 아직
  구현하지 않았다.

## 4. Normative data flow

```text
GenerationRequest
  └─ optional RAG.retrieve(instruction)
       └─ RetrievedExample[]
            └─ Generator.generate(request, context)
                 └─ GeneratorOutput attempt
                      ├─ Validator.validate(svg)
                      └─ optional Critic.critique(instruction, svg)
                           └─ CriticFeedbackEvent(target_attempt_id)
                                └─ Generator.revise(
                                     request,
                                     previous_attempt,
                                     feedback_event,
                                     context,
                                   )
```

다음 invariant는 구현을 바꾸지 않는 한 유지한다.

- RAG와 Critic은 optional이다.
- `generate()`는 previous SVG나 feedback을 받지 않는다.
- `revise()`는 previous attempt와 그 attempt를 target으로 하는 feedback을
  함께 요구한다.
- formatting/model-call retry와 Critic revision은 서로 다른 단위다.
- Generator는 retrieval, feedback scoring, global stop policy를 소유하지
  않는다.

## 5. RAG 담당자가 반드시 알아야 할 계약

### 5.1 반환 단위

RAG는 preformatted context string이 아니라 item list를 반환한다.

```text
RetrievedExample
  content
  item_id
  source
  description
  score
  score_kind
  rank
  kind
  corpus_version
  metadata
```

필수 invariant:

- `item_id`는 같은 logical item에 대해 안정적이고 비어 있지 않아야 한다.
- `source`는 비어 있지 않아야 한다.
- `score_kind`는 similarity/distance 등 score 의미를 설명해야 한다.
- `rank`가 있으면 1부터 시작하는 양수다.
- RAG가 deduplication, ranking, score interpretation, corpus version을
  소유한다.

### 5.2 item kind

현재 공유 kind는 다음 네 가지다.

```text
reference_svg
positive_experience
negative_lesson
correction_pair
```

정적 corpus는 보통 `reference_svg`를 사용한다. 나머지 세 kind는 Cycle 8의
experience memory를 위한 계약이며, 지금 당장 RAG가 memory curation을
구현해야 한다는 뜻은 아니다.

### 5.3 metadata whitelist

Cycle 0의 free-form metadata whitelist는 빈 집합이다.

- 필요한 공유 정보는 먼저 typed top-level field로 표현한다.
- vector-store 내부 metadata는 adapter boundary에서 제거한다.
- 새 metadata key는 owner, 의미, artifact retention 정책을 문서화한 뒤
  whitelist에 추가한다.
- `render_ref`와 실제 bitmap은 현재 text-only Generator input이 아니다.

### 5.4 Generator의 context 처리

- user instruction이 retrieved context보다 우선한다.
- 현재 context budget은 12,000 characters다.
- Generator가 deterministic order로 context를 선택/절단한다.
- consumed item과 truncated item ID가 attempt trace에 남는다.
- 현재 Orchestration은 run당 한 번 retrieve하고 revision에서도 같은 context를
  재사용한다.
- feedback-aware re-retrieval은 합의된 기본 동작이 아니라 향후 ablation
  variant다.

### 5.5 RAG 담당자가 구현하지 않아도 되는 것

- Generator prompt template
- SVG extraction과 normalization
- Critic feedback interpretation
- final acceptance와 revision stop

## 6. Critic 담당자가 반드시 알아야 할 계약

### 6.1 입력과 출력

개념적 Critic interface는 다음과 같다.

```text
instruction + SVG attempt -> CriticFeedback
```

현재 structured payload:

```text
score
is_valid
matches_instruction
issues[]
suggestions[]
critic_type
raw_response
critic_version
model_id
model_revision
prompt_version
```

Critic은 actionable issue/suggestion과 자체 provenance를 제공한다.
`score` 하나만 반환하거나 free-form text만 반환하면 revision과 calibration을
재현하기 어렵다.

### 6.2 feedback correlation

Orchestration은 Critic payload를 다음 event로 감싼다.

```text
feedback_id
target_attempt_id
feedback
```

- feedback은 정확히 한 attempt를 target으로 해야 한다.
- Generator는 `target_attempt_id != previous.attempt_id`이면 revision을
  거부한다.
- Critic이 `feedback_id`나 Generator attempt ID를 임의로 생성하지 않는다.

### 6.3 현재 baseline과 열린 결정

- 현재 acceptance score는 `8.0`, 최대 revision은 2회다.
- 이 값은 calibration된 연구 결과가 아니라 Cycle 0의 tunable baseline이다.
- Critic owner는 score calibration, structured parsing, model/prompt version,
  timeout/error semantics를 정해야 한다.
- Orchestration owner는 identical output, no improvement, branch selection,
  rollback, timeout 시 continuation 정책을 정해야 한다.
- Critic은 최종 acceptance나 memory eligibility를 단독으로 결정하지 않는다.

## 7. Orchestration 담당자가 반드시 알아야 할 계약

### 7.1 ID ownership

| ID | 생성 owner |
|---|---|
| `run_id` | Factory/runtime |
| `attempt_id` | Generator |
| `model_call_id` | Generator |
| `feedback_id` | Orchestration |
| RAG `item_id`/source ID | RAG |

```text
run_id
  └─ attempt_id
       └─ model_call_id
```

한 attempt 안의 malformed-output retry와 Critic-driven revision을 동일하게
세면 안 된다.

### 7.2 현재 구현된 loop

- optional RAG retrieval
- initial generation
- validation
- optional Critic feedback
- threshold 미달 시 revision
- 최대 revision 또는 threshold 도달 시 종료
- final accepted/rejected/failed outcome과 stop reason 기록

현재 loop는 baseline일 뿐 다음을 완성하지 않았다.

- identical output/no-improvement stop
- Critic timeout/parse failure fallback
- 여러 candidate branch 선택
- best prior attempt rollback
- calibrated acceptance policy

## 8. Artifact/Structure 담당자가 반드시 알아야 할 계약

### 8.1 하위 호환

기존 final artifact의 top-level field는 유지한다.

```text
instruction
svg_path
is_valid
render_path
revision_count
critic_feedback
runtime
metadata
generated_at_utc
```

Schema version 1은 additive extension이다. `schema_version`이 없으면 legacy
version 0으로 해석한다.

### 8.2 version 1 trace

- `run_id`
- attempt별 `attempt_id`, mode, SVG reference, raw output reference
- model call별 prompt/system prompt/raw output와 generation parameters
- model/backend/revision/token/latency metadata
- `parent_attempt_id`
- `trigger_feedback_id`
- feedback의 `feedback_id`와 `target_attempt_id`
- status/error/outcome/stop reason
- consumed/truncated RAG item ID

Generator와 Orchestration은 trace 데이터를 생산한다. Structure는 atomic
write, path layout, migration, retention, incomplete artifact detection을
소유한다.

## 9. Model/Platform 담당자가 반드시 알아야 할 계약

현재 local baseline:

| 설정 | 값 |
|---|---|
| Distribution | `lmstudio-community/gemma-4-12B-it-QAT-GGUF` |
| File | `gemma-4-12B-it-QAT-Q4_0.gguf` |
| Revision | `291406f49e16eff811c85ad8884d375f34138663` |
| Upstream | `google/gemma-4-12B-it-qat-q4_0-unquantized` |
| Runtime | CUDA `llama-cpp-python==0.3.34` |
| Context | 8192 |
| GPU layers | `-1` |
| Batch | 512 |

- GGUF metadata가 chat template을 소유한다.
- full GPU offload를 우선하고 CPU offload는 explicit measured fallback이다.
- model artifact에는 distribution revision뿐 아니라 upstream, quantization
  provider, conversion runtime을 함께 남긴다.
- custom GGUF는 이 기본 provenance를 잘못 상속하지 않는다.
- cloud backend로 교체할 때 Generator interface를 변경하지 않는다.

## 10. Evaluation 담당자가 반드시 알아야 할 계약

### 10.1 현재 가능한 것

- 저장된 generation artifact의 SVG validity
- render success
- generation latency
- artifact/report plumbing
- SVGenius candidate snapshot preparation과 provenance audit

### 10.2 현재 불가능한 것

- dataset prompt batch generation
- SVGenius 299개에 대한 model accuracy
- 신뢰 가능한 semantic instruction alignment
- reference visual similarity
- held-out aggregate/category report

현재 `simple_instruction_alignment`는 non-empty SVG를 확인하는 placeholder이므로
품질 metric으로 사용하면 안 된다.

### 10.3 SVGenius 상태

- 최종 benchmark가 아닌 `candidate_only`
- adapter `svgenius-text-to-svg-v2`
- manifest schema 2
- 299 joined records: easy 100, medium 99, hard 100
- pinned medium caption 누락 1건을 revision-bound exclusion으로 기록
- `data_partition=candidate_unassigned`
- `memory_eligible=false`
- license inconsistency와 final metric protocol은 미결정

Evaluation owner가 benchmark를 채택하기 전 dataset revision, task subset,
license, inclusion/exclusion, dev/test partition, metrics, thresholds를 승인해야
한다.

## 11. Feedback experience memory에 대한 팀 합의

핵심 연구 방향은 다음 trace를 future RAG context로 재사용하는 것이다.

```text
failed attempt -> Critic feedback -> accepted correction
accepted attempt
```

소유권:

| 역할 | 책임 |
|---|---|
| Generator | consumed context와 attempt/model-call trace 생산 |
| Critic | structured/versioned feedback 생산 |
| Orchestration | attempt-feedback lineage와 outcome 기록 |
| Memory Curator | eligibility, confidence, deduplication, retention |
| RAG | curated memory indexing/retrieval |
| Evaluation | held-out leakage 차단과 round별 ablation |

중요한 제한:

- rejected SVG 하나만으로 negative memory를 만들지 않는다.
- negative memory는 lesson 또는 failed→feedback→correction tuple이 우선이다.
- benchmark artifact는 `memory_eligible=false`다.
- held-out round별 개선과 ablation 전에는 “self-evolving”이라고 주장하지
  않는다.

## 12. 이번 변경에서 합의되지 않은 것

다음은 Cycle 0 구현이 대신 결정하지 않았다.

- RAG embedding model, vector DB, top-k, threshold
- Critic model, prompt, 학습법, calibrated score threshold
- final benchmark와 metric
- SVG strict safety policy
- artifact retention 기간과 storage migration
- memory curation algorithm
- cloud serving backend
- Generator/Critic fine-tuning recipe

각 owner는 이 항목을 결정할 수 있지만, shared schema나 다른 workstream에
영향을 주면 같은 변경에서 cross-team contract를 갱신해야 한다.

## 13. 구현 시작 전 체크리스트

### RAG 담당자

- [ ] `RetrievedExample`의 typed field를 모두 보존한다.
- [ ] stable `item_id`, source, score semantics, rank, corpus version을 정의한다.
- [ ] free-form metadata가 기본적으로 Generator로 넘어가지 않게 한다.
- [ ] benchmark와 RAG corpus의 leakage 검사를 설계한다.
- [ ] empty retrieval이 정상 결과가 되게 한다.

### Critic 담당자

- [ ] structured feedback과 version provenance를 제공한다.
- [ ] 어떤 attempt를 평가했는지 Orchestration이 연결할 수 있게 한다.
- [ ] score calibration 계획과 human rubric을 정의한다.
- [ ] timeout, parse failure, empty feedback을 명시한다.
- [ ] acceptance와 memory eligibility를 Critic 단독 책임으로 두지 않는다.

### Orchestration/Structure 담당자

- [ ] producer-owned ID를 덮어쓰지 않는다.
- [ ] feedback target과 revision lineage를 보존한다.
- [ ] schema version 1 compatibility를 검토한다.
- [ ] timeout/no-improvement/identical-output 정책을 결정한다.
- [ ] raw/intermediate artifact retention과 atomic write를 결정한다.

### Evaluation 담당자

- [ ] adapter test와 model accuracy를 구분한다.
- [ ] final benchmark와 dev/held-out split을 고정한다.
- [ ] placeholder alignment metric을 사용하지 않는다.
- [ ] batch runner와 versioned report를 제공한다.
- [ ] benchmark result가 memory로 ingest되지 않게 한다.

## 14. 권장 변경 제목

이 변경 묶음을 공유할 때는 다음과 같은 제목이 적합하다.

```text
feat(generator): establish Cycle 0 contracts and local GGUF baseline
```

또는 회의/문서 제목으로는 다음을 권장한다.

```text
Generator Cycle 0 Cross-Team Handoff:
RAG, Critic, Orchestration, and Evaluation Agreements
```
