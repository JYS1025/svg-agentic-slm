# Generator SFT code review

검토일: 2026-09-09. 기준 커밋: `9836752`.
초기 결과: 운영 재개 경로의 P2 결함 1건, chunked helper 재사용 경로의 조건부 P2 결함 1건.
수정 상태: 2026-09-10 두 결함을 보완하고 회귀 테스트를 추가했다. 아래 결과는 전체 12B GPU 학습의 무결성을 보증하지 않는다.

## 범위와 검증 환경

- `train_text_to_svg.py`, `sft_trainer.py`, `lora_config.py`, `chunked_causal_lm_loss.py`, `paged_optimizer_resume.py`를 추적했다.
- 데이터 준비, response-only serialization/collation, 공식 discrete cache, token 등록, selected-row adapter 저장/로드 및 기본 학습 YAML을 함께 확인했다.
- 기존 사용자 작업과 전역 Python 환경을 변경하지 않고 `/private/tmp/svg-sft-review-env`에 별도 환경을 구성했다.
- Python 3.12.9, torch 2.14.0, transformers 5.15.0, peft 0.20.0, accelerate 1.15.0, pytest 9.1.1. macOS CPU 실행이다. 실제 학습 서버의 설치 버전은 확인하지 않았다.
- 프로젝트에 고정된 모델 revision의 공개 config는 `gemma4_unified` / `Gemma4UnifiedForConditionalGeneration`이다. 기존 수치 테스트의 `gemma4` 모델과 구분해 축소 unified 모델을 추가 검증했다.

모델 설정 출처: https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized/raw/b6ed86275a6a5735884e208bfed95b445a684ca2/config.json

## F1. P2: resume 시 early-stopping 이력 누락

위치: `src/svg_agentic_slm/train/sft_trainer.py:1655`의 TrainingArguments 생성과 `:1704`의 EarlyStoppingCallback 구성, `:1752`의 resume 호출.

현재 코드는 EarlyStoppingCallback을 새로 생성하지만 `restore_callback_states_from_checkpoint`를 활성화하지 않는다. transformers 5.15.0의 기본값은 false이며, `_load_callback_state()`는 이 값이 false이면 저장된 callback state를 복원하지 않는다. SFTConfig에도 해당 옵션이 없어 YAML에 추가하는 것만으로 해결되지 않는다.

그 결과 optimizer/trainer state가 복원되어도 연속 무개선 횟수인 `early_stopping_patience_counter`는 0부터 시작한다. 기본 학습 YAML은 early stopping을 활성화하므로, 해당 설정의 checkpoint resume에서 실제 종료 동작이 달라진다.

재현: 동일한 작은 모델/데이터, learning_rate=0, eval/save 매 step, patience=2, threshold=0, max_steps=6. step 2 checkpoint에는 patience counter=1이 저장되어 있다.

| 실행 | 종료 global step |
| --- | --- |
| 연속 학습 | 3 |
| 현재 기본값으로 checkpoint-2 재개 | 4 |
| callback state 복원을 활성화한 대조군 | 3 |

영향: 중단/재개에 따라 불필요한 학습과 평가가 추가되고 조기 종료의 재현성이 깨진다. 이것만으로 가중치 손상이나 기존 완료 모델의 품질 저하를 단정하지 않는다.

수정 방향: resume 계약에 early-stopping state 복원을 포함하고, checkpoint의 callback 설정과 현재 YAML 설정이 다를 때의 정책도 명시한다. 전체 callback 복원은 저장된 설정도 복원하므로 단순히 옵션만 켜고 끝내지 말고 변경 설정의 허용/거부 기준을 검증한다.

## F2. P2, 조건부: chunked helper의 accumulation normalization 계약 불일치

위치: `src/svg_agentic_slm/train/chunked_causal_lm_loss.py:67`, `:167`.

helper는 `num_items_in_batch`를 제거하고 각 microbatch의 active-token 평균을 반환한다. 하지만 `accepts_loss_kwargs=True`로 인식되는 causal-LM에서는 Trainer가 이미 전체 accumulation-window 분모로 정규화됐다고 판단해 별도 GAS 나눗셈을 하지 않는다.

재현: 기존 테스트와 같은 작은 `Gemma4ForCausalLM` + PEFT에 helper를 적용한다. 유효 batch=4, 같은 길이/동일 응답 token 수, dropout=0, SGD, clipping=0, 단일 update를 GAS=1/2로 비교했다.

| 항목 | GAS=1 | GAS=2 |
| --- | --- | --- |
| 보고 training loss | 4.2214928 | 8.4429855 |
| parameter update norm 비율 | 1.0 | 2.0000017 |

이는 helper의 재현 가능한 결함이다. 다만 **현재 기본 `Gemma4UnifiedForConditionalGeneration`은 `accepts_loss_kwargs=False`이므로 이 배율 오류가 적용되지 않는다.** 이 결과를 현재 raw XML 실험의 gradient가 16배 잘못됐다는 주장으로 확대하면 안 된다.

수정 방향: 현재 모델의 reduction 의미를 보존하면서 helper가 분모를 처리하는 계약과 Trainer의 loss-kwargs 판정을 일치시킨다. GAS=1/2/16, 불균등 응답 길이, 마지막 불완전 window를 비교하고 기본 multimodal 경로의 의미가 바뀌지 않는지 검증한다.

## 실행 결과

| 검증 | 결과 |
| --- | --- |
| 학습 관련 기존 테스트 9개 파일 | 109 passed |
| 전체 기존 테스트 | 428 passed, 8 warnings, 41.14s |
| early-stopping 연속/재개 비교 | F1 재현, 복원 대조군은 연속 실행과 같은 step에 종료 |
| causal-LM chunked GAS 비교 | F2 재현 |
| 기존 multimodal 축소 모델 GAS 비교 | GAS=2/1 update norm 비율 0.9999999, 배율 오류 없음 |
| unified 축소 모델 chunked/reference 비교 | loss/gradient 허용 오차 내 일치; 최대 gradient 절대 차이 약 2.21e-6 |
| unified selected-row 학습/저장/재로드 | selected row 변경, 기존 row 유지, 재로드 logits 비교 통과 |

전체 테스트 명령:

```bash
/private/tmp/svg-sft-review-env/bin/python -m pytest -q
```

추가 재현 스크립트는 임시 검증 자료이며 저장소 회귀 테스트로 추가하지 않았다:

- `/private/tmp/svg_sft_review_probes.py`: F1/F2와 multimodal 대조군.
- `/private/tmp/svg_sft_unified_probe.py`: unified loss/gradient 및 selected-row reload.
- `/private/tmp/svg-sft-review-all.log`: 전체 테스트 로그.
- `/private/tmp/svg-sft-review-probes.log`, `/private/tmp/svg-sft-unified-probe.log`: 추가 검증 로그.

초기 환경 구성에서는 전역 구형 torchvision/NumPy와 충돌했고, 이후 system-site-packages 상속을 끊어 해결했다. 전체 테스트 첫 수집은 pyarrow 누락으로 중단됐으나, 격리 환경에 필요한 의존성을 추가한 뒤 428개가 모두 실행됐다. 이 환경 오류를 프로젝트 결함으로 분류하지 않았다.

전체 테스트의 8개 warning은 synthetic 모델의 base config 조회 실패 및 PEFT tying 경고였다. selected-row 물리적 공유와 저장/재로드 수치 테스트는 통과했다. 추가 unified probe에도 축소 config에 대한 Transformers 보정 경고가 있었으므로 해당 probe는 실제 12B config 전체를 그대로 재현한 실험은 아니다.

## 남은 검증 한계

- 실제 12B 가중치, 실제 tokenizer 전체 데이터, CUDA bf16/4bit, bitsandbytes paged managed-memory 복원, multi-GPU DDP를 실행하지 않았다.
- paged optimizer unit test의 통과는 실제 CUDA allocation 복원의 증명이 아니다.
- 기존 structural loss 테스트는 실제 Trainer를 사용하는 수치 비교를 포함하며 통과했다. 코드도 weighted-loss multi-process를 명시적으로 차단한다. DDP 미지원 자체를 새 결함으로 보고하지 않는다.
- 기존 CPU 테스트 대부분은 작은 legacy Gemma 모델을 사용한다. unified 축소 모델을 추가 확인했지만 production 아키텍처와 precision/optimizer를 사용하는 GPU 회귀 테스트가 여전히 필요하다.
- raw XML 학습의 일반적인 수렴이나 discrete 표현의 품질은 이번 코드 정확성 검토로 평가하지 않았다.

우선 수정 대상은 F1이다. F2는 helper의 재사용 안전성을 위해 보완할 수 있지만 현재 기본 모델의 학습이 잘못됐다는 근거는 아니다. 이 외에 이번 범위에서 새롭게 확정한 실질적 결함은 없다.

## 2026-09-10 수정 및 재검증

- early-stopping이 활성화된 resume는 checkpoint의 callback 설정과 상태를 사전 검증한다. 현재 YAML의 patience/threshold가 다르거나 state가 없으면 잘못 재개하지 않고 중단한다.
- 검증된 checkpoint에만 `restore_callback_states_from_checkpoint=true`를 적용하며, 복원할 counter와 경로를 training manifest에 기록한다.
- chunked loss는 Trainer가 `num_items_in_batch`를 제공하면 전체 accumulation-window active-token 분모를 사용하고, 제공하지 않는 현재 unified 모델 경로는 기존 local mean 동작을 유지한다.
- 회귀 테스트는 불균등 응답 길이의 GAS 1/2/4/16 update 동등성, callback 설정 불일치·상태 누락 거부, 연속/재개 종료 step 동등성을 포함한다.
- 전체 테스트: 447 passed, 8 warnings. Ruff 및 `git diff --check` 통과.
- unified 축소 모델의 reference/chunked loss·gradient 및 selected-row 저장·재로드 검증도 다시 통과했다.
