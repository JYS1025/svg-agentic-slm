# `simple_bench_run1` RAG–Generator–Critic failure analysis

분석일: 2026-08-19  
대상 실행: 2026-08-18 12:58–13:11 UTC에 생성된 12개 text-to-SVG 케이스  
대상 폴더: `simple_bench_run1/`

## 1. 결론

이 실행은 파이프라인의 연결성과 추적 가능성은 보여 주지만, 생성 품질이나 RAG/critic의 개선 효과를 입증하는 benchmark로는 사용할 수 없다.

- 시스템 판정은 12개 중 11개 accepted(91.7%), 최종 SVG validity는 12/12다.
- 그러나 accepted는 같은 VLM critic의 자기 판정이다. 독립 평가, 사람 라벨, reference metric, RAG/critic ablation이 없다.
- RAG는 모든 케이스에서 정답 개념을 거의 그대로 설명하는 예시를 반환했다. 특히 calendar 케이스는 동일 설명의 예시 3개를 반환했다. 이는 retrieval 성공 사례인 동시에 benchmark leakage/난이도 붕괴다.
- 12개 중 10개의 generator prompt가 raw reference SVG를 문자 예산 끝에서 path 중간에 잘랐다. 검색 컨텍스트가 길고 불완전하며, 실제 user instruction보다 압도적으로 크다.
- critic은 시각 품질을 과대평가한다. `floating island`처럼 핵심 공간 관계가 약한 결과, glow가 거의 보이지 않는 연결선, 단순하고 어색한 형태도 10점으로 통과했다.
- revision loop는 best candidate를 보존하지 않는다. hands 케이스는 7.5 → 6.0 → 6.0으로 악화됐는데 마지막 6점 결과가 최종물이다. 기록된 `max_no_improvement_rounds=1`도 실제 중단에 반영되지 않았다.
- critic 호출의 3/22(13.6%)가 응답 계약을 위반했다. 3회의 retry 중 2회만 복구됐고, butterfly에서는 서로 반대되는 색상 지시가 연속으로 나온 뒤 계약 실패가 generator revision을 유발했다.
- 실행 artifact에는 code Git SHA와 실제 실행 명령이 없다. 더구나 기록된 `text-to-svg-v2-conservative`, `svg-revision-v2-conservative`, `critic_v1`, `configs/models/gemma4-gemma4-critic.yaml`은 현재 checkout에서 찾을 수 없어 이 실행을 현재 저장소만으로 재현할 수 없다.

따라서 다음 run 전 최우선 과제는 **held-out benchmark와 ablation 확립**, **critic calibration**, **atomic/token-aware RAG context**, **best-candidate rollback 및 no-improvement stop 구현**, **run provenance 고정**이다.

## 2. 분석 범위와 방법

다음을 서로 대조했다.

- 최종 sidecar JSON 12개
- top-level 최종 SVG 12개
- immutable artifact bundle의 `final.svg` 12개
- generator attempt SVG 21개와 raw response/prompt/system prompt
- critic render PNG 19개, labeled SVG/manifest 19개
- critic raw response 22개와 response schema/validation 결과
- 현재 checkout의 orchestrator, generator prompt, RAG, critic, artifact/eval 구현 및 설정

시각 평가는 critic PNG를 attempt 순서로 contact sheet로 만들어 직접 확인했다. 이 문서의 수동 평가는 정식 human study가 아니라 failure triage이며, 시스템의 10점 판정과 실제 보이는 결과가 충돌하는 지점을 찾기 위한 것이다.

## 3. 실행 요약

| 항목 | 결과 |
|---|---:|
| 케이스 | 12 (icon 6, illustration 6) |
| 시스템 accepted / rejected | 11 / 1 |
| 초기 SVG valid | 10/12 (83.3%) |
| 최종 SVG valid | 12/12 (100%) |
| revision 발생 케이스 | 7/12 |
| 총 revision | 9 |
| generator 호출 | 21 |
| critic model 호출 | 22 |
| critic contract-invalid 호출 | 3/22 (13.6%) |
| generator prompt / completion tokens | 243,665 / 16,411 |
| generator 평균 prompt / completion tokens | 약 11,603 / 782 |
| generator / critic model latency 합 | 307.6초 / 219.8초 |
| end-to-end latency 합 / 케이스 평균 | 687.4초 / 57.3초 |
| retrieval 결과 | 36개 (모든 케이스 top-k=3) |
| 초기 prompt에 사용된 context item 표기 | 22개 |
| truncated item 표기 | 24개 |

`context_item_ids`와 `truncated_context_item_ids`는 10개 케이스에서 같은 item ID를 동시에 포함한다. 실제 prompt를 보면 해당 item을 일부 사용한 뒤 SVG path 중간에서 잘랐기 때문이다. 현재 metadata만 보면 “사용됨”, “부분 사용됨”, “완전 제외됨”을 명확히 구분할 수 없다.

최종 전달 일관성은 양호하다. 12개 모두 top-level `.svg`, artifact bundle의 `final.svg`, 마지막 attempt SVG가 SHA-256 기준 동일하다. 다만 이것은 마지막 attempt를 정확히 전달했다는 뜻이지, 가장 좋은 attempt를 전달했다는 뜻은 아니다.

## 4. 케이스별 검토

| 케이스 | 점수 흐름 / 결과 | 수동 검토 | 주요 관찰 |
|---|---|---|---|
| icon 1 — Two hands hold together | 7.5 → 6.0 → 6.0, rejected | 실패 | 세 attempt 모두 손가락/손바닥 구조가 없고 활, 구름, 말풍선 같은 blob으로 보인다. critic은 이를 감지했지만 revision이 악화됐고 마지막 결과를 보존했다. |
| icon 2 — Calendar with checkmark | 10, accepted | 개념 충족 | 달력과 체크가 명확하다. 그러나 RAG 3개가 모두 사실상 동일한 “calendar icon with a checkmark” 예시라 독창 생성 능력의 증거가 아니다. 첫 critic 응답은 preserve 4개로 계약 위반 후 retry됐다. |
| icon 3 — Futuristic circuit board chip | 10, accepted | 부분 충족 | chip/board 상징은 인지되지만 “futuristic” 표현은 약하고 단순하다. exact-concept RAG의 도움을 분리할 수 없다. |
| icon 4 — Magnifying glass over document | 0 → 10, accepted | 개념 충족 | 초기 SVG는 중복 `x` attribute로 XML invalid. evidence gate가 수정시켜 최종물은 명확해졌다. |
| icon 5 — Pair of 3D glasses | 7.5 → 10, accepted | 개념 충족 | 초기 irregular frame을 critic이 잘 지적했고 최종 red/blue lens는 알아볼 수 있다. 최종 critic 첫 응답은 preserve 4개로 계약 위반 후 retry됐다. |
| icon 6 — Symmetrical butterfly | 5(invalid) → 10, accepted | 최종은 단순 충족, loop는 실패 | 첫 시도는 오히려 전형적인 4-wing butterfly다. critic retry가 “더 진하게”와 “더 옅게”라는 반대 지시를 내고 두 번째 응답이 잘려 계약 실패했다. 일반적인 contract-error feedback으로 재생성한 뒤 2-wing 형태를 10점 통과시켰다. |
| illustration 1 — Plant growing from light bulb | 8.5 → 10, accepted | 개념 충족, 불필요 revision | critic이 user가 요구하지 않은 “잎과 꽃을 각각 두 개 더 추가”를 요구했다. 이는 instruction grounding보다 주관적 naturalness를 요구한 것이다. |
| illustration 2 — Laptop and smartphone connected by glowing lines | 7.5 → 7.5 → 10, accepted | 경계/과대평가 | 3차 결과도 연결이 얇은 흰색/청색 선 몇 개 수준이며 glow가 강하지 않다. 두 번의 revision 비용 뒤 10점으로 급상승할 근거가 약하다. |
| illustration 3 — Friendly sun and cloud | 10, accepted | 개념 충족 | 표정과 heart로 상호작용을 표현한다. heart는 “friendly”의 장식적 해석이지만 엄격한 only-supported-object 규칙과는 긴장이 있다. |
| illustration 4 — Melting candle beside open book | 0 → 10, accepted | 대체로 충족 | 초기 결과는 `<animate>` 때문에 static-SVG policy 위반. 실제 generator system prompt에는 animation 금지가 명시되지 않아 validator와 prompt 정책이 어긋난다. 최종 candle은 책 앞/위에 겹쳐 “beside”는 약하다. |
| illustration 5 — Tea with floating lemon slices and herbs | 10, accepted | 부분 충족 | cup, tea, lemon, herb, steam은 보이나 구성이 매우 단순하고 herb 표현이 약하다. 10점은 품질 차이를 표현하지 못한다. |
| illustration 6 — Floating island with tree and waterfall | 10, accepted | 핵심 관계 미흡 | 평평한 초록 mound가 화면 하단에 놓여 있어 “floating”이 명확하지 않다. 노출된 섬 하부/공중 간격/낙하감이 없고 waterfall도 짧은 파란 홈처럼 보인다. 대표적인 critic false positive다. |

## 5. Failure modes

### P0 — benchmark 판정이 실제 품질을 대표하지 못함

`accepted=91.7%`는 외부 품질 지표가 아니라 같은 pipeline critic의 종료 상태다. 점수도 사실상 이산적이다. 통과한 11개 feedback은 모두 10점이고, revise는 6.0–8.5, evidence/contract failure는 0 또는 5다. 미세한 visual quality 차이를 측정하지 못한다.

특히 다음 false positive가 있다.

- illustration 6: floating 관계가 약한데 10점
- illustration 2: glow가 약한데 revision 후 10점
- illustration 4: candle의 beside 관계가 애매한데 10점
- illustration 5: 매우 단순한 구성과 약한 herb 표현에도 10점

rule critic은 valid SVG에 항상 10점을 주며 instruction match를 항상 true로 둔다. composite 평균은 이 10점을 VLM 결과에 섞으므로 score의 의미도 불명확해진다. acceptance는 score뿐 아니라 `matches_instruction`을 확인해 즉시 오통과하지는 않지만, 보고되는 숫자는 calibrated quality score가 아니다.

### P0 — RAG leakage와 정답 복제형 난이도

모든 top-1 retrieved description이 user instruction의 핵심 객체/관계를 직접 재진술한다. calendar는 동일 설명 3개, chip/glasses/document도 사실상 동일 icon의 변형 3개다. illustration도 대부분 정확한 구도 하나가 top-1이다.

이 설정에서는 결과가 좋아도 다음을 구분할 수 없다.

1. generator가 instruction을 이해해 새 SVG를 구성했는가
2. retrieved SVG의 구도/부품을 변형했는가
3. RAG 없이도 같은 결과를 냈는가

“Do not copy” 문구만으로 leakage를 통제할 수 없다. 실제 plant 결과는 retrieved example의 “bulb outline 안의 plant” 구도를 그대로 따른다. benchmark query와 RAG corpus를 split/entity/template 수준에서 격리하고, RAG-off baseline과 구조 유사도 검사를 같이 두어야 한다.

### P0 — revision이 악화돼도 마지막 결과를 반환

hands는 7.5 → 6.0 → 6.0으로 내려갔다. 그럼에도:

- `max_no_improvement_rounds=1`
- `min_critic_score_improvement=0.1`

가 runtime metadata에 기록된 상태에서 두 번 모두 revision했고, stop reason은 `max_revisions_reached`다. 최종 artifact는 가장 높은 7.5점 후보가 아니라 마지막 6.0점 후보다.

현재 checkout의 orchestrator도 best-so-far, no-improvement counter, rollback을 구현하지 않고 마지막 `current.svg`를 결과로 둔다. 다음 run 전 최소한 다음이 필요하다.

- hard validity와 semantic score를 분리한 candidate ordering
- best-so-far 저장
- score 하락 또는 충분한 개선 부재 시 rollback/early-stop
- critic infrastructure failure를 generator 수정 요구로 전달하지 않는 별도 error path

### P0 — 실험 설계상 RAG와 critic 효과를 추정할 수 없음

현재 run은 `RAG on + critic on` 한 조건뿐이다. 최소한 같은 seed와 prompt set으로 아래 4-way ablation이 필요하다.

| 조건 | 목적 |
|---|---|
| Generator only | SLM 자체 baseline |
| Generator + RAG | retrieval의 순효과 및 copy/leakage 측정 |
| Generator + critic | revision의 순효과 및 regression 측정 |
| Generator + RAG + critic | 전체 pipeline의 상호작용 측정 |

평가는 pipeline critic과 분리된 blind judge 또는 사람 평가로 해야 한다. 최소 rubric은 object presence, attribute, count, spatial relation, style, legibility, SVG validity, originality/reference overlap이다.

### P1 — RAG context가 과도하고 mid-SVG로 잘림

generator 21회에 prompt token 243,665개, completion token 16,411개가 쓰였다. 평균 prompt는 약 11.6k tokens로 completion의 약 14.8배다. 짧은 user instruction 대신 raw path 숫자가 context 대부분을 차지한다.

12개 중 10개 초기 prompt에서 마지막 포함 item이 `d` path 숫자 중간에서 끊긴다. 이는:

- well-formed SVG 예시라는 few-shot 신호를 훼손하고
- generator가 불완전 문법을 모방할 위험을 만들며
- item provenance를 모호하게 하고
- top-k 대부분을 검색하고도 실제로는 1개 일부만 쓰게 만든다.

문자 수가 아니라 backend tokenizer 기준 token budget을 사용해야 한다. item은 원자적으로 포함/제외하고, raw SVG 전체 대신 caption + viewBox + element inventory + 단순화된/정규화된 구조 snippet을 사용해야 한다. partial item을 허용한다면 SVG element 경계에서만 자르고 `fully_used`, `partially_used`, `dropped` 및 실제 사용 token 수를 기록해야 한다.

### P1 — critic 응답 계약 불안정과 상충하는 feedback

critic model call 22개 중 3개가 invalid였다.

- icon 2: `preserve` 4개 > 최대 3개, retry 복구
- icon 5 final: `preserve` 4개 > 최대 3개, retry 복구
- icon 6 initial: `target_ids` 7개 > 최대 4개; retry 응답은 384-token 한도에서 잘려 JSON failure, 복구 실패

icon 6의 두 응답은 같은 이미지를 두고 첫 번째에는 “너무 옅으니 saturation/contrast를 높이라”고 했고, 두 번째에는 “이미 saturated blue이니 pastel로 낮추라”고 했다. retry가 단순 형식 복구가 아니라 판단 자체를 뒤집었다.

또한 critic prompt는 `status=pass requires issues=[] and preserve=[]`라고 명시하지만, 성공으로 검증된 pass raw response 11개 모두 non-empty preserve를 출력했다. JSON schema에는 status별 조건이 없고 semantic validator도 이를 막지 않는다. 즉 문서화된 contract와 실제 검증 contract가 다르다.

개선 방향:

- JSON Schema에 `if/then` 또는 `oneOf`로 status별 invariants 강제
- retry 시 원래 판단을 재생성하지 말고 format-repair만 수행하거나 constrained decoding 사용
- contract/timeout failure는 visual defect와 분리하고 generator에 generic revision을 요구하지 않음
- content checklist를 instruction에서 deterministic하게 추출해 object/count/attribute/relation별 판정
- score를 validity, instruction alignment, visual quality로 분리하고 사람 라벨로 calibration

### P1 — generator prompt와 validator policy가 일치하지 않음

초기 SVG 2/12가 invalid였다.

- icon 4: duplicate `x` attribute로 XML parse failure
- illustration 4: static policy에서 금지한 `<animate>` 사용

artifact의 generator system prompt는 script/event/external reference를 금지하지만 `<animate>`/`<animateTransform>` 등 animation 금지를 명시하지 않는다. validator가 요구하는 static subset을 generator가 알 수 없으므로 두 번째 실패는 예측 가능하다.

validator allowlist/denylist에서 generator constraint를 생성하거나 동일 policy module을 공유해야 한다. 출력은 가능하면 XML AST로 parse 후 canonical serialization하고, critic에 보내기 전에 schema/static-policy/render 세 단계를 hard gate로 유지해야 한다.

### P1 — 재현성 provenance 부족 및 checkout drift

artifact에는 model/revision, backend version, generation parameters, RAG corpus version은 잘 남아 있다. 반면 다음이 없다.

- source Git commit/dirty state
- 실행 명령과 CLI version
- Python/package lock 또는 environment fingerprint
- config file content hash
- benchmark prompt-set version/hash
- machine/runtime identifier(민감 정보 제거 가능)

현재 checkout에서는 artifact가 기록한 다음 항목을 찾을 수 없다.

- `text-to-svg-v2-conservative`
- `svg-revision-v2-conservative`
- `critic_v1` runtime implementation
- `configs/models/gemma4-gemma4-critic.yaml`

현재 source의 `prompts/text_to_svg.py`는 v1 template이고 factory는 critic type으로 `rule`, `llm`, `both`만 허용한다. 따라서 이 폴더만으로는 실행 코드를 복원할 수 없다. 원격/다른 worktree 또는 uncommitted code에서 생성됐을 가능성이 있으므로, 다음부터 artifact root에 immutable `run_manifest.json`을 추가해야 한다.

### P2 — 관측 지표와 artifact 의미가 모호함

- `metadata.timing.generation_latency_seconds`는 이름과 달리 retrieval, 모든 generator/critic revision, artifact overhead를 포함한 end-to-end 시간에 가깝다. RAG latency, validation/render latency가 따로 없다.
- runtime `enable_render=false`라 `metadata.render.success=false`지만 critic용 PNG는 생성된다. “사용자 render 비활성”과 “critic evidence render 성공”을 별도 필드로 명확히 해야 한다.
- `.json.lock` 12개가 성공 후에도 0-byte로 남는다. 기능 문제는 아니지만 완료된 benchmark bundle을 지저분하게 하고 consumer가 파일로 오인할 수 있다.
- top-level SVG, artifact `final.svg`, last attempt SVG가 중복 저장된다. 현재 hash는 모두 일치해 무결성은 좋지만 canonical artifact와 convenience copy의 역할을 명시할 필요가 있다.

## 6. 잘 작동한 부분

- 최종 12개 SVG가 모두 well-formed/static-policy valid다.
- invalid 초기 SVG 2개는 evidence gate와 revision으로 복구됐다.
- prompt, raw output, attempt SVG, critic PNG, labeled SVG, manifest, critic schema/validation이 attempt 단위로 남아 원인 추적성이 좋다.
- critic grounding용 synthetic element ID와 manifest가 보존된다.
- retrieval item ID/source/score/corpus revision, generator/critic model revision과 per-call token/latency가 기록된다.
- top-level/final/last-attempt artifact hash가 일치한다.

이 기반은 유지하되, 현재 trace를 “성공률 증명”이 아니라 “실패 분석 데이터”로 사용해야 한다.

## 7. 다음 run 전 권장 조치

### 즉시(P0)

1. 4-way ablation과 held-out corpus split을 만든다.
2. benchmark query와 retrieval corpus 간 exact/near-duplicate caption, same-template SVG를 제거한다.
3. best-so-far selection, score regression rollback, `max_no_improvement_rounds`를 실제 loop에 연결한다.
4. independent evaluator/human rubric을 추가하고 pipeline acceptance rate와 외부 품질 score를 분리한다.
5. run manifest에 Git SHA/dirty state, command, config hashes, prompt versions, prompt-set hash를 저장한다.

### 단기(P1)

1. RAG를 token-aware atomic packing으로 변경하고 raw SVG를 구조적으로 단순화한다.
2. `fully_used`/`partially_used`/`dropped` context provenance와 item별 token 수를 기록한다.
3. critic response를 constrained JSON으로 만들고 status-dependent schema를 실제로 강제한다.
4. critic retry를 format repair와 semantic re-evaluation로 분리한다.
5. validity/semantic/visual score를 분리하고 acceptance threshold를 calibration set으로 정한다.
6. static SVG policy를 generator system prompt와 공유한다.
7. critic failure에는 generator revision을 수행하지 않는 정책을 추가한다.

### 후속(P2)

1. phase별 latency/token/cost 지표를 남긴다.
2. revision마다 instruction checklist 충족 변화와 structural/image diff를 계산한다.
3. output complexity, clipping, contrast, small-size legibility, reference similarity를 자동 평가한다.
4. lock file cleanup과 artifact canonicalization을 정리한다.

## 8. 다음 benchmark의 최소 합격 기준 제안

- 최소 100개 이상의 held-out prompt, icon/illustration 및 관계 난이도 층화
- initial validity ≥ 98%, final validity = 100%
- critic contract success ≥ 99.5%
- 외부 judge 기준 instruction pass rate와 pipeline self-pass rate 차이 ≤ 5%p
- revision 후 외부 score non-regression ≥ 95%; 악화 시 best candidate rollback 100%
- RAG on이 RAG off 대비 외부 score를 유의하게 개선하되 reference-copy similarity gate 통과
- context item mid-element truncation 0건
- 모든 run에 source/config/prompt/dataset provenance 100% 기록

이 기준 전에는 `accepted` 비율을 모델 품질 KPI로 사용하지 않는 것이 안전하다.
