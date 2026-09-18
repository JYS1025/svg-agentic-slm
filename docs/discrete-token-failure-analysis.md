# Discrete-token SFT failure analysis

작성일: 2026-09-03
대상: `c1_description_discrete_full20k_seed42`, canonical adapter SHA-256
`b88bad5cbfd1a563b80ed0e53c1cb632402830141cc091c2ef4373e36a9b94f3`

## 결론

이 run은 **학습 경로가 끊긴 실패가 아니다**. 모델은 BOS, opcode, 짧은 지역 문법과
training sequence를 학습했다. 실패한 부분은 held-out caption을 구체적인 좌표, 색상,
path 종료와 EOS로 변환하는 조건부 일반화다.

가장 큰 원인은 official OmniSVG의 token ID/serialization만 가져오고, 그 token을 이미
학습한 checkpoint나 embedding은 가져오지 않은 데 있다. Gemma에는 의미 구조가 없는
45,062개 새 row가 생겼고, 그중 40,000개가 독립적인 `(x, y)` pair class다. 18,000개의
짧은 description으로 새 codebook과 caption-to-geometry mapping을 동시에 학습하게 했다.
Teacher forcing에서는 정답 SVG prefix가 다음 token을 강하게 예측하므로 caption을 거의
쓰지 않아도 loss가 내려간다.

## 무엇이 정상인가

| 점검 | 결과 | 해석 |
|---|---:|---|
| Cached ID = live official encode | 256/256 exact | cache corruption 아님 |
| Strict decode/render | 256/256 | codec/decoder contract 정상 |
| Source vs decoded RGBA MAE | mean 0.00766, p95 0.01828 | codec 손실은 작음 |
| Label/mask/BOS/EOS/truncation | 전부 통과 | response framing 오류 아님 |
| 새 row gradient 및 save/reload | 전부 통과 | head가 동결되거나 유실된 것 아님 |
| 1-sample memorization | step 30 exact | optimizer 경로 정상 |
| 32-sample memorization | step 900에서 32/32 exact | 모델 용량과 decode 경로 정상 |

근거 artifact:

- `g0_codec_contract/g0_audit.json`
- `g1_token_head_labels/g1_audit.json`
- `g2_micro_overfit/thirty_two_fresh_max1200_exposure_corrected_resume600/audit.json`

## 무엇을 배웠고 무엇을 못 배웠나

Canonical validation loss는 epoch 1~5에서
`2.8428 → 2.2678 → 2.2146 → 2.3758 → 2.4681`이었다. Best는 epoch 3이고 이후
train loss가 계속 낮아지는 동안 validation은 악화했다. 따라서 단순히 같은 설정으로 epoch를
늘리는 것은 해결책이 아니다.

Held-out 256개 teacher-forced audit은 학습 내용의 차이를 직접 보여 준다.

| Token role | Mean NLL | Top-1 |
|---|---:|---:|
| BOS | 0.0004 | 100.0% |
| Opcode | 0.2576 | 90.0% |
| Coordinate/arc | 3.0479 | 47.6% |
| Color/path terminator | 3.7413 | 37.4% |
| EOS | 1.1686 | 37.9% |

올바른 caption이 matched wrong caption보다 주는 token NLL 이득은 `0.0221`, null prompt보다
주는 이득은 `0.0072`뿐이다. 올바른 prompt가 모든 negative보다 낮은 NLL을 얻은 sample도
51.95%에 불과하다. Output length가 길어질수록 이 conditioning gap은 거의 0으로 줄어든다.
즉 낮은 CE의 대부분은 caption 이해가 아니라 gold SVG history의 local continuation에서 온다.

## 원인 우선순위

### 1. 의미 없는 대형 codebook의 cold start

- 45,062개 named token은 upstream integer provenance를 보존할 뿐 Qwen/OmniSVG weight를
  옮기지 않는다.
- Hugging Face resize는 기존 embedding의 통계로 새 row를 초기화하지만, 좌표·색 의미나
  인접성은 주지 않는다.
- 새 tied embedding/head row는 173.0M parameter이고 LoRA는 65.6M이다. 새 row가 전체
  trainable parameter의 약 72.5%다.
- 등록한 contiguous namespace 45,062개 중 실제 encoder-producible class는 44,205개다.
  Reserved gap 857개도 trainable row로 만들었지만 target에는 한 번도 나타나지 않는다.
- 40,000개 joint-coordinate class에서 `(x, y)`와 `(x+1, y)`는 parameter를 공유하지 않는다.
- 공식 training config는 full-model, LR `1e-5`, detail/brief 및 text/image 혼합, 150 epochs를
  전제로 한다. 현재 run은 description-only 18K, QLoRA, 5 epochs, LR `2e-4`다. 같은 token
  번호를 쓴다고 같은 representation prior를 얻는 것은 아니다.

### 2. Teacher-forcing shortcut과 underspecified caption

Description 중앙값은 약 17~18 words인데 illustration target은 평균 약 910 tokens다. 하나의
짧은 caption은 정확한 path ordering, control point, 색상까지 유일하게 결정하지 못한다.
그 상황에서 next-token CE는 caption보다 정답 prefix를 사용하는 쉬운 해를 제공한다.

### 3. 긴 autoregressive horizon과 종료 신호 부족

기존 prefix audit에서 natural EOS는 11/32였고 가장 긴 quartile은 0/8이었다. 그러나 이
audit은 `repetition_penalty=1.05`를 사용했다. 같은 32개를 true greedy
(`repetition_penalty=1.0`)와 exact token range로 다시 실행하자 natural EOS와 strict render는
각각 **1/32**로 감소했고, exact full/suffix는 여전히 0/32였다. Gold target의 75%를 prefix로
준 8개도 natural EOS와 exact suffix가 모두 0이었다. 초기 한두 token의 exposure-bias만이
아니라 좌표·색·종료 transition 자체가 held-out에서 불안정하다.

True-greedy 32개 중 31개는 oracle remaining length + 32의 generation limit까지 도달했다.
종료하지 않은 31개 중 14개는 마지막 최대 256 tokens가 64-token 이하의 정확한 주기로
반복됐고, 대표적으로 `curve + 같은 coordinate 3개`가 되풀이됐다. 즉 주된 free-running
failure는 문법 밖 token을 고르는 것이 아니라 **문법상 가능한 command/coordinate cycle에
들어가 path color와 EOS로 전이하지 못하는 것**이다.

EOS×8 pilot은 EOS top-1을 약 `0.35 → 0.91`로 바꿀 수 있음을 보였지만 global NLL은
개선하지 못했다. EOS는 실제 failure mode지만 caption-to-geometry 실패의 근본 원인은 아니다.

### 4. 과적합과 optimizer 조건

Best가 3 exposure/sample 부근이고 4~5 epoch에서 validation이 악화했다. 반면 32개 sample을
완전히 외우는 데는 약 28 exposure/sample이 필요했다. 현재 model은 더 많은 반복으로 training
sequence는 외울 수 있지만 held-out mapping은 더 나빠지는 regime에 있다. Warm-start adapter에
fresh optimizer와 LR `2e-4`를 사용한 60-step G5 control도 바로 회귀했다.

## 발견된 버그와 실험 confound

1. **Historical length sampler**: canonical run은 raw-XML
   `metadata.full_chat_token_length`로 discrete sample 순서를 만들었다. Label이나 truncation은
   바꾸지 않았지만 exact provenance와 optimizer trajectory를 오염시켰다. 현재 runtime
   discrete serialization length를 사용하도록 수정됐다.
2. **G5 loss reduction mismatch**: control C는 batch=1의 per-sample token mean이었지만 E/P는
   accumulation-window token mean이었다. 기존 C/E/P 결과는 structural weight의 단독 효과가
   아니다. `structural_response_loss_reduction: sample`을 추가했고 다음 matched pilot은 이를
   사용해야 한다. 실제 `Gemma4UnifiedForConditionalGeneration`은
   `accepts_loss_kwargs=False`라 Trainer가 C에 window token count를 전달하지 않는다. 길이
   1/5인 두 microbatch의 LoRA update probe도 per-sample 식과 max diff 0, token-global 식과
   `1.331e-3` 차이로 이를 확인했다.
3. **Over-broad grammar ranges**: 기존 grammar/path-class audit은 encoder가 만들 수 없는
   coordinate gap 3개, color 뒤 reserved gap 392개를 class에 포함했고 arc boundary 하나는
   반대로 누락했다. Exact producible range로 수정한 G6B에서도 global grammar gain 변화는
   `1e-9` 미만이고 결론은 동일했다.
4. **Repetition penalty**: 기존 free-generation audit은 greedy라고 기록했지만
   `repetition_penalty=1.05`를 썼다. SVG는 command와 좌표 반복이 정상이라 true greedy
   baseline은 `1.0`이어야 한다. 재실행 결과 이 penalty는 실패 원인이 아니라 자연 EOS를
   1/32에서 11/32로 올려 failure를 일부 가리던 confound였다.
5. **MMSVG detail representation**: 일부 raw `detail`은 Python list repr 형태의 string이다.
   Train 4,761 / validation 271 / test 239개가 해당한다. Description-only canonical에는 영향이
   없지만 mixed/detail 실험 전에 strict parse + sentence join이 필요하다. Opt-in
   `detail_text_normalization: mmsvg_list_repr_v1`을 추가했고, 선택된 detail branch에서만
   fail-closed parse하며 split별 결과 hash/count와 raw-record 불변성을 manifest에 기록한다.
6. **Inference loaded-state typo**: legacy discrete backend의 `is_loaded()`가 존재하지 않는
   `_discrete_grammar`를 읽었다. `_codec_grammar`로 수정했다.
7. **Dependency contract**: selected-row training은 PEFT 0.20 API에 의존하지만 package lower
   bound가 0.10이었다. `peft>=0.20,<1`로 맞췄다.

## 배제된 원인

- Cache/live encoder mismatch
- Decoder 또는 renderer의 systematic corruption
- Response mask, BOS/EOS framing, padding label 오류
- Context truncation
- Tied input/output row가 학습되지 않거나 adapter save/reload에서 사라지는 문제
- Validation OOV: train-unseen row의 validation mass는 0.0142%뿐이다
- 문법 밖 probability가 핵심이라는 가설: exact-range G6B grammar-renormalized NLL gain은
  약 0.003이고 illegal argmax는 0.0072%다
- Warmup bug: Transformers 5.15의 float `warmup_steps=0.03`은 ratio로 해석되어 실제 57
  warmup steps가 적용됐다

## 다음 실험의 go/no-go 순서

1. Canonical b88을 diagnostic artifact로 동결한다. 현재 설정으로 장기 resume하지 않는다.
2. True greedy(`repetition_penalty=1.0`) + exact grammar range로 G3를 재평가한다.
3. G5를 재사용한다면 C/E/P 모두 per-sample semantics, 같은 sampler/LR/optimizer 조건으로
   맞춘다. 이 조건의 G5B contract와 C/E/P config는 CPU preflight까지 준비됐으며 세 arm 모두
   launch `GO`다. 예상 GPU 시간은 평가 포함 약 96분이므로 EOS/path weighting은 보조
   실험으로만 두고 factorized/prompt-conditioning pilot보다 우선하지 않는다.
4. Full retrain 전에 2K/8K clean pilot에서 다음을 gate로 쓴다: correct-vs-null/shuffle prompt
   gap, coordinate/color role accuracy, natural EOS by length/domain, strict render, perceptual metric.
5. Primary representation arm은 joint XY를 `X[0..199] + Y[0..199]`로 factorize한다. 20K
   전수 offline audit에서 cache-ID round-trip은 20,000/20,000이었고, target은 평균
   551→883, 최대 2048→3528 tokens였다. 2,304 target-token budget은 7.15%가 넘지만 4,096은
   전부 통과했다. Coordinate vocabulary는 40,000→400으로 줄고 모든 X/Y component가
   train에서 최소 5,435/6,484회 관측된다. RGB channel과 explicit `PATH_END`까지 factorize하면
   전체 exact vocabulary는 44,205→558, target 평균/최대는 904/3587이다. 단, full-chat
   context와 production decoder는 별도 pilot에서 다시 검증해야 한다.
6. 현 dialect를 유지해야 한다면 official checkpoint teacher distillation 또는
   geometry-aware row initialization/auxiliary loss를 사용한다. 단순 CE 장기 학습보다 우선한다.
7. Richer detail arm은 runtime normalization을 manifest에 기록한 뒤 열고, short-description
   평가를 별도로 유지한다.

Factorization audit이 추가로 드러낸 upstream 표현 한계도 있다. Color offset 4097은 quantized
white와 gradient placeholder가 충돌하며 20K cache에서 13,816회 등장한다. 따라서 cache token
ID 수준 factorization은 lossless지만 원본 SVG의 white/gradient 의미를 복원하는 것은 불가능하다.
이는 전체 학습 실패의 주원인은 아니지만 새 dialect에서는 분리해야 한다.

추가 artifact:

- `g3b_prefix_greedy_rp1_exact_ranges/audit.json` (SHA-256
  `9d05f52a5ae8785271cb53ef698b4304287ca7664b162b466d78824b88a21874`)
- `g6b_exact_producible_grammar_mass/grammar_mass_audit.json` (SHA-256
  `92c5b9e07a5ee8916b12a68d49b263f88b73c39b758b2e262d5ce0b5aa3aba59`)
- `artifacts/official_discrete_factorization_feasibility.json` (SHA-256
  `d05a8df1cd716ac26f6bb27de08e844edf5d7752756f6b73310c05114d19a44c`)
- `g5b_structural_pilot_sample_mean_full18k/contract.json` (GPU 미실행, SHA-256
  `dbeef775a415b1ace24a615dea2700e8d8e2be83c36f35be5546fde7c2b6b3b3`)

핵심 성공 조건은 validation CE의 소폭 하락이 아니다. Held-out prompt를 null/shuffled prompt와
구분하고, 긴 sequence에서도 좌표·색·EOS를 자연스럽게 생성하며, decode/render된 결과가 caption과
정렬되는지가 성공 조건이다.
