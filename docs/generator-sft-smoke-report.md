# Generator SFT 실행 검증 보고서

검증일: 2026-08-24

전체 학습 및 ablation 설계는
[`generator-sft-ablation-plan.md`](generator-sft-ablation-plan.md)에 정리되어 있다. 이
문서는 sample data, tiny Gemma 4 및 공식 Gemma 4 12B로 실제 실행한 결과만 기록한다.

## 1. 실행 환경

| 항목 | 값 |
|---|---|
| Conda | `svg`, Python 3.11 |
| GPU | NVIDIA RTX PRO 6000 Blackwell, GPU 0/1/2 |
| PyTorch | 2.13.0+cu130 |
| Transformers | 5.15.1 |
| PEFT | 0.20.0 |
| bitsandbytes | 0.50.1 |
| Accelerate | 1.14.0 |
| 공식 base | `google/gemma-4-12B-it-qat-q4_0-unquantized` |
| 고정 revision | `b6ed86275a6a5735884e208bfed95b445a684ca2` |

공식 checkpoint는 `/data/minjun_dev/hf_home`에 저장했다. 9개 파일의 합계는
23,951,777,679 bytes이며 `model.safetensors`는 23,919,549,408 bytes이다.

## 2. Sample MMSVG 준비

운영 기본 split gate는 Icon/Illustration 각각 train 9,000, validation 500, test 500을
강제한다. smoke config에서만 `allow_nonstandard_split: true`를 사용했다.

| 검증 항목 | 결과 |
|---|---|
| 입력 | Icon 4건 + Illustration 4건 |
| benchmark exclusion | domain별 1건 제외 |
| 최종 train/validation/test | 각 2건, domain별 1건 |
| SVG validation/render 실패 | 0건 |
| tokenizer length 실패 | 0건 |
| RAG top-3 metadata | 최종 6건 모두 보존 |

산출물은 `/tmp/svg_sft_smoke/output`에 있다.

Discrete smoke는 direct path-only Icon 3건과 Illustration 3건으로 별도 수행했다.
6건 모두 codec encode/decode, SVG validation 및 CairoSVG render에 성공했다. sample별 token
수는 12, 14, 18, 36, 38, 61이며 결과는 `/tmp/svg_sft_smoke_discrete`에 있다.

## 3. QLoRA 학습 경로

36.9MB의 `tiny-random/gemma-4-dense`를 사용해 네 실험 경로를 각각 1 optimizer step
실행했다.

| 실험 | Instruction | Target | 결과 | Loss |
|---|---|---|---|---:|
| R0 | description-only | raw XML | 성공 | 12.47045 |
| R1 | detail 60% + description 40% | raw XML | 성공 | 12.47097 |
| R2 | detail-only | raw XML | 성공 | 12.47097 |
| C2 | description-only | discrete SVG | 성공 | 12.64548 |

Discrete run은 codec token 44,468개를 추가했다. 전체 171 tokens 중 SOP부터 EOS까지 36개만
active label이고, system/user/model marker/closing/padding은 `-100`이었다. Adapter에는 추가
embedding과 LM head가 포함됐으며 tokenizer, codec manifest 및 training manifest가 함께
저장됐다.

실행 과정에서 다음 Transformers 5.15 호환 문제를 찾아 수정했다.

1. `warmup_ratio` 대신 float `warmup_steps` API를 사용한다.
2. chat template의 `BatchEncoding["input_ids"]` 반환을 처리한다.
3. Gemma 4 generation prompt의 thought-channel token 때문에 깨지는 prefix masking을 제거한다.
4. rendered chat offset으로 정확한 SVG 문자 범위만 response loss에 포함한다.
5. mixed instruction metadata는 `detail` 또는 `description` enum으로 고정한다.

## 4. GPU 0/1/2 DDP

6개 sample로 3-process Accelerate 실행을 수행했다.

| 항목 | 결과 |
|---|---|
| Launcher exit | 0 |
| Global optimizer step | 1 |
| Loss | 12.47304 |
| Rank | 3개 모두 load/train/barrier/종료 성공 |
| Final artifact | rank 0에서 adapter/tokenizer/manifest 각 1개 저장 |

PyTorch가 종료 시 `destroy_process_group()` 미호출 NCCL cleanup warning을 한 번 출력했지만
세 rank 및 launcher 종료 코드는 모두 0이었다.

## 5. 공식 Gemma 4 12B 학습

공식 exact-revision checkpoint를 NF4 QLoRA로 재양자화해 raw description-only sample을
1 step 학습했다.

| 항목 | 결과 |
|---|---|
| Global step | 1 |
| Loss | 1.659376 |
| Train runtime | 2.0143초 |
| GPU 0 peak usage | 15,839 MiB |
| Adapter size | 262,373,216 bytes |
| Tokenizer size | 32,169,626 bytes |

산출물은 `/data/minjun_dev/svg_sft_smoke/gemma4_12b_raw_1step`에 있다.

## 6. Inference 및 Full Pipeline

Tiny 1-step adapter는 production `TransformersTextBackend`에서 offline load, CUDA BF16
generation, provenance 기록 및 unload까지 성공했다.

공식 12B adapter는 toy Chroma RAG와 cached Qwen2.5-VL-3B Critic을 연결한 production CLI로
실행했다.

```text
12B SFT adapter
-> toy Chroma top-3 retrieval
-> token-aware RAG context
-> Generator system prompt
-> SVG generation
-> static SVG validation
-> Critic evidence PNG render
-> Qwen VLM Critic
-> final SVG/JSON/PNG artifact
```

생성 SVG는 ivory background, centered teal circle 및 2px navy outline을 정확히 포함했고
validator와 renderer를 통과했다. 전체 latency는 15.851초이며 RAG 0.666초, Generator
6.024초, Critic 8.856초였다.

최종 연구 outcome은 `critic_contract_failure`였다. Qwen Critic이 두 시도 모두 정확히 하나의
JSON object만 반환해야 하는 contract를 위반했다. 이 실패는 SVG 결함이나 Generator revision
feedback으로 전달되지 않고 격리됐다.

주요 결과:

```text
/data/minjun_dev/svg_sft_smoke/gemma4_12b_full_pipeline.svg
/data/minjun_dev/svg_sft_smoke/gemma4_12b_full_pipeline.json
/data/minjun_dev/svg_sft_smoke/gemma4_12b_full_pipeline.run_bc476fa9.png
```

## 7. 검증 결과와 남은 작업

| 검증 | 결과 |
|---|---|
| 전체 pytest | 230 passed |
| compileall | 통과 |
| git diff --check | 통과 |
| MyPy | 실패, 12개 파일 36건 |

MyPy 오류에는 기존 artifact/RAG/SVG typing 오류, third-party stub 부재 및 일부 신규 annotation
오류가 섞여 있다. 런타임과 별도로 정적 타입 정리가 필요하다.

실제 20K 학습 전에 필요한 작업은 다음과 같다.

1. Elice MMSVG 원본과 benchmark exclusion manifest 경로를 운영 config에 주입한다.
2. 20K immutable manifest와 18K/1K/1K split을 생성한다.
3. R0 10K pilot의 optimizer step budget과 checkpoint selection 규칙을 동결한다.
4. parse/validator/render 지표와 validation loss를 함께 모니터링한다.
5. R0 안정화 후 R1, R2 및 codec-compatible C1/C2를 같은 budget으로 실행한다.
6. Critic JSON contract failure를 별도 failure mode로 집계하고 prompt/schema calibration을 한다.
7. prompt/config/checkpoint를 동결한 뒤 MMSVGBench와 VectorGym을 최종 평가한다.
