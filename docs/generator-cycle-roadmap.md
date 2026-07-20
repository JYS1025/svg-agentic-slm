# Generator Development Cycle Roadmap

## 문서 관리

| 항목 | 값 |
|---|---|
| 상태 | Working roadmap |
| 범위 | Generator Cycle 0–8 |
| Primary owner | Generator workstream |
| 독자 | Generator, RAG, Critic, Orchestration, Artifact, Validation, Evaluation, Memory 담당자 |
| 최종 갱신 | 2026-07-20 |

## 1. 문서 목적

이 문서는 Generator 개발을 Cycle 0부터 Cycle 8까지 어떤 순서로 진행하는지
팀원이 빠르게 이해할 수 있도록 정리한 실행 지도다. 각 Cycle마다 다음을
구분한다.

- 먼저 충족되어야 하는 dependency
- Generator 담당자가 직접 구현하는 범위
- 다른 workstream이 제공해야 하는 범위
- 해당 Cycle에서 수행할 실험
- 다음 Cycle로 넘어가기 위한 종료 조건

세부 설계와 근거는
[Generator Implementation Plan](./generator-implementation-plan.md), 현재
공유 계약은
[Generator Cross-Team Contract](./generator-cross-team-contract.md), Cycle 0
변경에서 팀원이 알아야 할 사항은
[Generator Cycle 0 Cross-Team Handoff](./generator-cycle0-team-handoff.md)를
참조한다.

## 2. Cycle을 해석하는 방법

Cycle은 git commit이나 한 번의 pull request와 일대일로 대응하지 않는다.
Cycle은 다음 단계의 실험을 시작하기 전에 닫혀야 하는 **dependency gate**다.

따라서 한 변경 묶음에서 이후 Cycle의 기반 코드를 미리 구현할 수 있다.
예를 들어 이번 Cycle 0 변경에는 real model backend, RAG context contract,
Critic revision contract의 일부가 함께 들어갔다. 이는 이후 통합을 위한
기반이 선행 구현됐다는 뜻이며, 실제 RAG retrieval 품질, Critic calibration,
benchmark accuracy가 완료됐다는 뜻은 아니다.

상태 표기는 다음과 같다.

| 상태 | 의미 |
|---|---|
| 완료 | 구현과 해당 Cycle의 필수 증거가 모두 존재한다. |
| 구현 후보 | 코드와 contract test는 존재하지만 owner review 또는 실측 gate가 남았다. |
| 부분 구현 | 이후 Cycle에 필요한 기반 일부만 구현됐다. |
| 대기 | 선행 dependency 또는 다른 owner의 결과가 필요하다. |

## 3. 전체 Cycle 요약

| Cycle | 핵심 질문 | 주 결과물 | 현재 상태 |
|---|---|---|---|
| 0 | 무엇을 서로 약속하고 무엇을 각자 소유하는가? | typed contract, model/benchmark pin, artifact trace, fake integration tests | Generator 범위 구현 후보 |
| 1 | 실제 local model이 안정적으로 SVG를 생성하는가? | real backend, load/generation evidence, hardware profile | 부분 구현 |
| 2 | model의 raw text를 안전한 SVG attempt로 바꿀 수 있는가? | extraction, normalization, strict failure/retry policy | 부분 구현 |
| 3 | RAG/Critic 없이 Generator 자체 성능을 측정할 수 있는가? | `G0` benchmark baseline과 config 선택 근거 | 대기 |
| 4 | 실제 RAG가 없어도 Generator–RAG 경계를 고정할 수 있는가? | fake RAG contract, context budget/provenance | 부분 구현 |
| 5 | 실제 Critic이 없어도 feedback revision 경계를 고정할 수 있는가? | fake Critic contract, revision lineage | 부분 구현 |
| 6 | 실제 RAG가 Generator 변경 없이 연결되고 도움이 되는가? | RAG on/off ablation | 대기 |
| 7 | 실제 Critic이 Generator 변경 없이 연결되고 도움이 되는가? | Critic/revision ablation과 calibration | 대기 |
| 8 | 과거 feedback experience를 안전하게 재사용하면 개선되는가? | experience-memory ablation과 leakage audit | 대기 |

## 4. Cycle 0 — 계약, 환경, benchmark dependency

### 목적

실제 RAG와 Critic이 아직 없어도 Generator를 구현할 수 있도록 공유 경계와
소유권을 먼저 고정한다. 이 단계에서 “실행된다”와 “품질이 좋다”를 분리한다.

### 선행 dependency

- 기준 branch/commit과 현재 구현 상태
- local hardware와 runtime 제약
- 사용할 model family, distribution, file, immutable revision
- RAG item과 Critic feedback의 최소 typed schema
- artifact 하위 호환 정책
- benchmark 후보와 최종 benchmark 선택 책임

### Generator가 구현하는 범위

- backend-neutral `ModelResponse`
- initial `generate()`와 feedback 기반 `revise()` 분리
- attempt/model-call/feedback correlation에 필요한 trace
- fake model, fake RAG, fake Critic을 사용한 contract tests
- Generator가 소비하는 config의 fail-fast validation
- RAG metadata whitelist 경계
- local model backend를 교체할 수 있는 factory

### 다른 owner의 dependency

- RAG: stable item/source ID, ranking, score semantics, corpus version
- Critic: feedback 품질, parsing, calibration
- Orchestration: continuation, branch, acceptance, stop policy
- Structure: artifact migration, atomic persistence, retention
- SVG Validation: strict XML/safety/render semantics
- Evaluation: 최종 benchmark, metric, batch runner, report

### 실험

- fake component contract tests
- real model load 및 한두 개 Generator-only smoke
- model distribution compatibility 확인
- SVGenius candidate의 pinned join/provenance 확인

### 종료 조건

- RAG/Critic 담당자가 Generator 내부 구현을 import하지 않고 각 모듈을 만들 수
  있다.
- placeholder SVG fallback 없이 success/failure가 구분된다.
- model, prompt, context, attempt, feedback lineage가 artifact로 재현 가능하다.
- 미결정 사항이 owner와 함께 명시되어 있다.

### 현재 상태

Generator 소유 코드와 contract tests는 구현 후보 상태다. 실제 model load와
non-placeholder SVG 생성은 확인됐다. 다만 VRAM/latency 실측, cross-team schema
review, strict validator, 최종 evaluation protocol은 열려 있다. SVGenius 299개
snapshot은 candidate preparation 결과이며 model accuracy가 아니다.

## 5. Cycle 1 — 실제 model backend와 최소 baseline

### 목적

선택한 checkpoint가 target hardware에서 실제로 로드되고, RAG/Critic 없이
non-placeholder SVG를 생성하는지 확인한다.

### 선행 dependency

- Cycle 0 model/runtime 계약
- immutable checkpoint와 GGUF file
- CUDA-enabled llama.cpp 환경
- chat template 소유권
- local memory/latency acceptance 기준

### Generator가 구현하는 범위

- GGUF download/local-path resolution
- config-selectable llama.cpp backend
- GGUF metadata 기반 chat completion
- token/latency/backend/model provenance
- 명시적 load/generation failure와 unload

### 실험

- real load와 full GPU offload
- VRAM headroom, load latency, TTFT, tokens/second
- deterministic 설정 반복
- context/output length와 truncation
- 필요할 때만 explicit partial-offload fallback

### 종료 조건

- 실제 completion에서 SVG가 생성된다.
- failure가 non-zero exit와 명시적 오류로 노출된다.
- model distribution과 runtime provenance가 artifact에 남는다.
- 승인된 hardware budget을 충족한다.

### 현재 상태

Backend와 real generation은 동작한다. Google-hosted GGUF는 vocabulary-load
incident로 reject됐고 LM Studio Community compatibility distribution을
사용한다. 정식 VRAM/latency 기록과 deterministic 반복 증거는 남아 있다.

## 6. Cycle 2 — SVG 후처리와 실패 정책

### 목적

모델의 자유 형식 text를 그대로 성공으로 취급하지 않고, 추출 가능한 SVG 또는
typed failure로 변환한다.

### 선행 dependency

- 실제 model completion fixture
- strict validator contract
- renderer
- malformed/truncated/multiple-SVG fixture

### Generator가 구현하는 범위

- SVG extraction과 normalization
- code fence/설명 문장 처리
- output length 제한
- formatting failure와 Critic revision의 분리
- bounded formatting retry와 trace
- raw output과 extracted SVG의 분리 보존

### 다른 owner의 dependency

- SVG Validation이 unsafe element, external reference, XML, render semantics를
  정의한다.

### 실험

- raw fixture별 extraction
- retry 0/1회 validity와 render success
- token truncation과 multiple SVG
- unsafe prompt와 external reference

### 종료 조건

- 최종 출력은 strict validation을 통과하거나 typed failure다.
- raw response가 그대로 final SVG로 저장되지 않는다.
- retry 횟수와 failure/stop reason이 남는다.

### 현재 상태

기본 extraction, normalization, length failure와 raw trace는 구현됐다. strict
safety validator와 formatting retry 정책은 아직 완료되지 않았다.

## 7. Cycle 3 — Generator-only benchmark baseline

### 목적

RAG와 Critic을 끈 상태에서 Generator 자체의 품질·비용 기준선 `G0`를 만든다.
이 Cycle부터 처음으로 benchmark-backed 품질 결과를 보고한다.

### 선행 dependency

- Cycle 2 Generator
- Evaluation owner가 고정한 dev/held-out split
- dataset batch runner
- 의미 있는 validity/render/constraint/semantic metric
- experiment report format

### Generator가 구현하는 범위

- 재현 가능한 Generator-only config profile
- model/prompt/config provenance
- 단일 sample 재현 command

### Evaluation/Structure dependency

- batch execution
- variant별 artifact directory와 run ID
- aggregate/category metrics와 report
- held-out isolation

### 실험

- prompt A/B
- deterministic decoding 대 sampling
- temperature/top-p/max token
- formatting retry 0/1
- 여러 seed의 평균과 분산

### 종료 조건

- versioned `G0` report가 존재한다.
- placeholder metric 없이 기본 config 선택 근거가 있다.
- 실패 sample을 command와 artifact로 재현할 수 있다.

### 현재 상태

대기 상태다. SVGenius snapshot preparation은 끝났지만 dataset-backed runner와
의미 있는 metric이 없으므로 “SVGenius 정확도”는 아직 계산하지 않는다.

## 8. Cycle 4 — RAG plug-and-play 계약

### 목적

실제 vector DB가 없어도 Generator가 neutral typed context를 올바르게 소비하는
계약을 고정한다.

### 선행 dependency

- `G0`
- `RetrievedExample` contract
- deterministic fake retriever
- context budget과 instruction precedence

### Generator가 구현하는 범위

- item boundary를 보존한 context rendering
- deterministic context truncation
- empty context no-op
- consumed/truncated item ID trace
- kind/source/version provenance

### RAG owner 범위

- vector DB, embedding, corpus ingestion, deduplication
- ranking, top-k, threshold, score interpretation
- stable item/source ID와 corpus version

### 실험

- no context 대 empty context
- relevant/irrelevant/conflicting synthetic context
- top-k/order/length
- retrieval prompt injection

### 종료 조건

- fake item이 실제 prompt에 들어가고 provenance가 남는다.
- 실제 RAG는 Generator 수정 없이 adapter/factory로 연결 가능하다.

### 현재 상태

typed item, four item kinds, 12,000-character budget, trace, 빈 free-form
metadata whitelist와 fake tests가 구현됐다. 실제 retrieval 품질은 Cycle 6
범위다.

## 9. Cycle 5 — Critic revision plug-and-play 계약

### 목적

실제 LLM Critic이 없어도 `attempt → feedback → revised attempt` 경계를 고정한다.

### 선행 dependency

- `G0`
- structured feedback
- deterministic fake Critic
- artifact lineage와 stop-policy proposal

### Generator가 구현하는 범위

- `revise(request, previous, feedback, context)`
- previous SVG와 feedback의 prompt rendering
- `parent_attempt_id`와 `trigger_feedback_id`
- initial/revision 공통 model-call trace
- feedback target mismatch 거부

### Critic/Orchestration 범위

- feedback parsing과 quality
- threshold calibration
- max revision, timeout, identical/no-improvement stop
- final accepted/rejected/superseded decision

### 실험

- positive/negative/empty/long feedback
- mismatched target ID
- initial/revision extraction contract
- rule/LLM/human feedback source 호환

### 종료 조건

- fake feedback으로 revision이 동작한다.
- feedback이나 previous attempt가 잘못 연결되면 실패한다.
- 실제 Critic은 Generator API 변경 없이 연결 가능하다.

### 현재 상태

typed feedback event, revision API, lineage, 최대 2회/score 8.0의 baseline loop가
구현됐다. threshold는 calibration 결과가 아니라 tunable default다.
identical/no-improvement/timeout 정책과 실제 Critic 품질은 열려 있다.

## 10. Cycle 6 — 실제 RAG integration과 ablation

### 목적

실제 retriever가 plug-and-play로 연결되는지, 그리고 비용 대비 품질을
개선하는지 검증한다.

### 선행 dependency

- Cycle 4 contract
- versioned corpus/index/embedding
- benchmark leakage audit
- `G0` report

### Generator가 구현하는 범위

계획된 새 RAG 구현은 없다. contract mismatch가 있다면 RAG-side adapter에서
해결하고 Generator regression만 확인한다.

### 실험

- `G0` 대 `G0 + RAG`
- top-k/threshold/corpus ablation
- category별 개선과 악화
- relevance 수동 표본
- latency와 context-token overhead

### 종료 조건

- 실제 RAG가 Generator 수정 없이 연결된다.
- leakage audit을 통과한다.
- on/off quality/cost report와 fallback 근거가 있다.

## 11. Cycle 7 — 실제 Critic integration과 ablation

### 목적

실제 Critic이 revision 품질을 개선하는지, score를 신뢰할 수 있는지 검증한다.

### 선행 dependency

- Cycle 5 contract
- rule 또는 LLM Critic
- structured parsing
- calibration set
- `G0`와 RAG variants

### Generator가 구현하는 범위

계획된 새 Critic 구현은 없다. 실제 feedback이 기존 contract와 provenance를
지키는지만 확인한다.

### 실험

- `G0`, `G0 + Critic`, `G0 + RAG`, `G0 + RAG + Critic`
- revision round 0/1/2
- threshold ablation
- Critic score와 human rubric 상관
- latency/token/API cost
- semantic quality가 오히려 낮아지는 failure

### 종료 조건

- 실제 Critic이 Generator 변경 없이 연결된다.
- calibration과 비용 대비 개선 근거가 있다.
- held-out report로 default 조합을 결정한다.

## 12. Cycle 8 — Feedback experience memory

### 목적

Generator–Critic trace에서 선별한 경험을 RAG가 다시 검색했을 때 held-out
품질이 round별로 개선되는지 검증한다.

### 선행 dependency

- Cycle 4 item kinds와 Cycle 5 lineage
- curated/versioned experience corpus
- static+experience hybrid retrieval
- benchmark isolation과 leakage audit
- Critic confidence/calibration

### Generator가 구현하는 범위

- `positive_experience`, `negative_lesson`, `correction_pair` rendering
- kind별 delimiter와 budget
- untrusted memory prompt-injection 방어
- consumed memory ID/kind/version trace

### 다른 owner 범위

- Memory Curator: eligibility, confidence, deduplication, retention
- RAG: storage, indexing, hybrid retrieval와 ranking
- Evaluation: round별 held-out protocol과 leakage prevention

### 실험

- no memory
- static RAG only
- positive only
- negative lesson only
- correction pair only
- static+experience
- round 0/1/2
- stale/noisy/adversarial memory

### 종료 조건

- held-out item이 persistent memory에 들어가지 않는다.
- memory on/off와 kind별 ablation이 존재한다.
- 반복 개선이 재현될 때만 “self-evolving” 표현을 사용한다.

## 13. 공통 실험 원칙

모든 품질 비교는 최소 다음을 기록한다.

- git commit과 dirty-worktree 여부
- model/distribution/runtime revision
- prompt와 generation config version
- seed와 hardware
- benchmark manifest/hash/partition
- RAG corpus/index/embedding version
- Critic model/prompt version
- raw output, extracted SVG, render, validation
- attempt/feedback/revision lineage
- latency, token, memory 또는 API cost

한 번에 하나의 독립 변수를 바꾸는 ablation을 기본으로 하며, benchmark test
feedback을 persistent memory나 다음 config 선택에 재사용하지 않는다.
