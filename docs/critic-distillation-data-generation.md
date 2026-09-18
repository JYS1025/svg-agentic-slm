# Critic distillation 데이터 생성 경로

## 확정 범위

- Generator SFT와 겹치지 않는 별도 MMSVG 10K를 입력으로 사용한다.
- SFT, RAG 원본, benchmark/eval, Critic 10K의 `sample_id`는 생성 전에 pairwise disjoint 검사를 통과해야 한다.
- Generator-only로 실행하며 RAG와 Critic은 끈다.
- SVG validity, 생성 성공 여부, 품질 점수로 데이터를 선별하지 않는다.
- 각 입력은 artifact bundle과 `results.jsonl`의 terminal record를 남긴다. 예외도 `failed` record로 보존한다.
- 같은 출력 디렉터리에서 재실행하면 완료된 ID를 건너뛰고, artifact만 기록된 중단 지점은 복구한다.

## ID manifest 계약

권장 JSONL 레코드는 다음과 같다.

```json
{"sample_id":"OmniSVG/MMSVG-Icon::1234","instruction":"A flat icon of ...","metadata":{"dataset_id":"OmniSVG/MMSVG-Icon","record_id":"1234","dataset_type":"icon"}}
```

`sample_id`가 없으면 `metadata.dataset_id`와 `metadata.record_id` 조합으로 생성한다. bare `id`만 있는 레코드는 namespace 충돌을 방지하기 위해 거부한다. SFT 준비 결과의 `manifest.json`은 `splits.*.file`을 따라가며 읽을 수 있다.

## Dry run

Dry run은 모델을 로드하지 않고 네 데이터 역할의 ID와 설정 provenance만 검증한다.

```bash
python scripts/run_critic_distillation_batch.py \
  --critic-manifest data/sft_sample_ids/critic_distill_10k.jsonl \
  --sft-manifest data/processed/mmsvg_sft_20k/manifest.json \
  --rag-manifest data/sft_sample_ids/rag_source_ids.jsonl \
  --eval-manifest data/sft_sample_ids/eval_ids.jsonl \
  --config configs/generation.yaml \
  --model-config configs/models/gemma4-sft-raw.yaml \
  --output-dir outputs/critic_distillation_10k \
  --dry-run
```

## 전체 생성

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_critic_distillation_batch.py \
  --critic-manifest data/sft_sample_ids/critic_distill_10k.jsonl \
  --sft-manifest data/processed/mmsvg_sft_20k/manifest.json \
  --rag-manifest data/sft_sample_ids/rag_source_ids.jsonl \
  --eval-manifest data/sft_sample_ids/eval_ids.jsonl \
  --config configs/generation.yaml \
  --model-config configs/models/gemma4-sft-raw.yaml \
  --output-dir outputs/critic_distillation_10k
```

실패를 명시적으로 재시도할 때만 `--retry-failures`를 추가한다. 기본값은 실패도 학습 후보 corpus의 관측값으로 유지한다.

## 미확정 경계

Critic label schema와 distillation trainer는 Critic 작업에서 확정한다. 초기 SFT 이후 3-5회 multi-turn 학습의 state/transition 형식과 GRPO reward 조합도 합의 전이므로 이 경로에는 구현하지 않는다. 현재 출력은 이후 정책과 무관한 원시 Generator 관측 corpus이다.
