# A100 SFT 최적화 및 서버 검증 보고서

기준일: 2026-09-18 UTC

## 1. 기준 코드와 결론

원격 `origin/main`의 최신 커밋
`ff3197349ea0e9cb1e7ee67000ec5549937b7592`(PR #9, Generator SFT loss/data
pipeline 최적화)를 기준으로 했다. 그 커밋의 tokenized cache, non-blocking loader,
active-token streamed cross entropy, 자동 GPU 수 감지는 그대로 유지하면서 이 서버의
2×A100 80GB에 맞는 실행 프로필과 Critic SFT 경로를 추가했다.

권장 프로필은 다음과 같다.

| 목적 | 설정 | 정밀도/메모리 전략 | 용도 |
|---|---|---|---|
| Generator 안전한 처리량 | `configs/train_lora_a100_80gb.yaml` | BF16 LoRA, checkpointing 켬 | 정확한 길이가 8K 이하인 빠른 실험/학습 |
| Generator 데이터 커버리지 | `configs/train_lora_a100_80gb_long.yaml` | NF4 QLoRA, checkpointing 켬 | 32K 이하 raw SVG 97.22% 보존 |
| Critic | `configs/train_critic_a100_80gb.yaml` | BF16 VLM LoRA, vision tower 고정 | 현재 운영 Critic의 18항목 JSON 계약 SFT |

8K cohort의 실제 중앙 길이인 3,168-token sample로 유효 batch 16을 동일하게 맞춘
단일-GPU 비교에서, 권장 BF16+checkpointing 프로필은 micro-batch를 1→4로 키우고
accumulation을 16→4로 줄였다. 같은 50,688 input token/한 optimizer update의 처리량은
기존 일반 QLoRA의 **823.1 → 1,010.9 tokens/s(1.228×)**로 증가했고 runtime은
**61.581 → 50.139초(18.6% 감소)**였다. 두 GPU의 effective batch는 이전과 동일한
32를 유지한다. checkpointing을 끄는 방식은 중앙 길이 batch 1만으로도 74.0 GiB를
사용하므로 운영 프로필에서는 제외했다.

## 2. 서버 하드웨어

### GPU

| 항목 | GPU 0 | GPU 1 |
|---|---|---|
| 모델 | NVIDIA A100 80GB PCIe | NVIDIA A100 80GB PCIe |
| UUID | `GPU-31e06360-a22e-3456-cd2e-348dbb3c8f3a` | `GPU-8cabf246-c155-9c9d-a207-cfd5dfe8d440` |
| PCI 주소 | `0000:17:00.0` | `0000:E3:00.0` |
| VRAM | 81,920 MiB | 81,920 MiB |
| Compute capability / SM | 8.0 / 108 SM | 8.0 / 108 SM |
| ECC / persistence / MIG | enabled / enabled / disabled | enabled / enabled / disabled |
| 전력 한도 | 300 W | 300 W |
| 최대 SM / HBM clock | 1,410 / 1,512 MHz | 1,410 / 1,512 MHz |
| PCIe | Gen4 ×16 current/max | Gen4 ×16 current/max |

- Driver는 `535.183.06`, driver가 보고하는 CUDA capability는 12.2이다.
- PyTorch에서 BF16과 native SDPA Flash attention을 실제로 사용할 수 있다.
- `nvidia-smi topo -m` 결과 GPU 간 경로는 `SYS`이고 NVLink가 없다. 즉 두 GPU의
  DDP gradient 통신은 PCIe와 CPU interconnect를 통과한다.
- NVIDIA 공식 A100 80GB PCIe 사양은 BF16 Tensor 312 TFLOPS, TF32 Tensor
  156 TFLOPS, HBM2e 1,935 GB/s, PCIe Gen4 64 GB/s, 300 W이다.

### CPU, RAM, 저장장치

- CPU: 2× Intel Xeon Gold 6338, socket당 32 core/64 thread. 컨테이너에는 32개
  logical CPU만 online으로 보이고 CPU quota도 32 core이다.
- NUMA: 2 node. 현재 허용 CPU 집합은 주로 NUMA node 0의 짝수 CPU이므로
  DataLoader worker를 과도하게 늘리지 않고 프로필당 4개로 제한했다.
- RAM: 384 GiB, swap 없음.
- 루트 overlay: 128 GB 중 약 101 GB 사용, 약 28 GB 가용. Torch 교체 뒤 재다운로드
  가능한 `uv` cache 23.3 GiB를 정식 cache 명령으로 비웠다. 모델 cache와 checkpoint를
  함께 복제하면 여전히 저장공간이 먼저 병목이 될 수 있다.
- GPU 간 P2P read/write와 peer access는 가능하지만 NVLink P2P가 아니다.

### 소프트웨어

| 구성요소 | 버전 |
|---|---|
| Python | 3.11.13 |
| PyTorch / torchvision | 2.6.0+cu118 / 0.21.0+cu118 |
| cuDNN / NCCL | 9.1.0 / 2.21.5 |
| Transformers / TRL | 5.16.1 / 1.12.0 |
| PEFT / Accelerate | 0.20.0 / 1.14.0 |
| bitsandbytes | 0.50.2 |

`bitsandbytes`는 이 서버의 프로젝트 `.venv`에 설치했고, 두 GPU에서 NF4와 8-bit
optimizer smoke를 통과했다. 별도 `flash-attn` 패키지 대신 PyTorch native SDPA의
Flash backend를 사용한다. 기존 PyTorch 2.5.1은 Transformers가 CVE-2025-32434 때문에
optimizer checkpoint 로드를 차단하므로, driver 535와 호환되는 공식 CUDA 11.8 build의
PyTorch 2.6.0으로 올렸다. 실제 resume 회귀 테스트도 통과했다.

이 서버에서 사용한 설치 명령은 다음과 같다.

```bash
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu118 \
  'torch==2.6.0+cu118' 'torchvision==0.21.0+cu118'
uv pip install --python .venv/bin/python 'bitsandbytes>=0.46,<1'
```

## 3. 공통 최적화

- BF16 Tensor Core와 TF32 matmul을 활성화하고 matmul precision을 `high`로 고정했다.
- 8K Generator는 quantization을 끈 BF16 LoRA와 non-reentrant gradient checkpointing을
  사용한다. micro-batch 4에서도 최대 8,191-token 실샘플 네 개를 peak
  allocated/reserved 52.76/56.38 GiB로 완료해 80GB에서 충분한 여유를 확인했다.
- 두 GPU 기준 global batch 32를 바꾸지 않고 device batch/accumulation만 1×16에서
  4×4로 바꿨다. 더 큰 실제 batch가 Tensor Core 활용률과 kernel 효율을 높이고 같은
  update를 위한 Python/forward/backward 반복 횟수를 1/4로 줄인다.
- 8K Generator에서 checkpointing까지 끄면 중앙 길이만으로 73.97 GiB가 할당되어 긴
  샘플의 OOM 여유가 없다. 3B Critic은 실제 smoke에서 메모리 여유가 커 BF16 LoRA와
  checkpointing off를 유지한다.
- 32K Generator는 NF4와 non-reentrant gradient checkpointing을 함께 사용해 긴 샘플의
  activation 메모리를 제한한다. checkpointing은 속도와 메모리의 명시적 trade-off다.
- `adamw_torch_fused`, pinned/non-blocking loader, persistent worker, prefetch 4,
  length grouping을 적용했다.
- streamed/chunked LM-head loss는 전체 vocabulary logits를 한 번에 materialize하지 않고
  loss가 있는 active response token만 projection한다. 8K는 chunk 512, 32K는 256이다.
- DDP는 NCCL, bucket 100 MiB, broadcast buffers off를 사용한다. PyTorch 2.6에서
  static graph와 gradient accumulation/no-sync를 함께 사용하면 reducer internal assert가
  재현되어 세 프로필 모두 `ddp_static_graph: false`로 고정했다.
- 런처는 repo의 `.venv/bin/python`을 우선 사용하고, 현재 shell처럼 `python` 명령이
  없는 환경에서도 동작한다. rank 0만 최종 manifest를 출력하고 process group을
  정상 종료한다.
- 매 실행 manifest에 world size, TF32/SDPA 상태, package version, 실제 peak CUDA
  allocated/reserved bytes 및 처리 token 수가 남는다.

이 플랫폼에서는 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`가 지원되지 않는다는
CUDA allocator 경고가 발생했으므로 강제하지 않았다. 사용자가 다른 allocator 설정을
명시하면 그대로 상속한다.

## 4. Generator 데이터 길이와 프로필 선택

원본 18,000개 train 샘플을 Gemma processor의 실제 chat template로, truncation 없이
측정했다. train 길이는 min 617, p50 6,360, p90 21,963, p95 28,405, p99 39,127,
max 54,700 token이다. 따라서 기존 기본값 8,192로 전체 데이터를 바로 넣는 것은
안전하지 않다.

`scripts/prepare_generator_length_cohort.py`가 source/output SHA-256, retained/dropped ID,
정확한 token 길이를 기록하면서 디렉터리를 atomic하게 게시한다. 이 서버에는 아래 두
cohort를 이미 생성했다.

| cohort | train | validation | test | 해석 |
|---|---:|---:|---:|---|
| 8K | 10,453/18,000 (58.07%) | 575/1,000 | 566/1,000 | 속도 우선; 전체 20K 품질 비교용이 아님 |
| 32K | 17,499/18,000 (97.22%) | 966/1,000 | 978/1,000 | 커버리지 우선; 어떤 target도 자르지 않음 |

재생성 명령:

```bash
.venv/bin/python scripts/prepare_generator_length_cohort.py \
  --input-dir data/processed/mmsvg_sft_20k \
  --output-dir data/processed/mmsvg_sft_20k_gemma4_8k \
  --max-seq-length 8192 --local-files-only

.venv/bin/python scripts/prepare_generator_length_cohort.py \
  --input-dir data/processed/mmsvg_sft_20k \
  --output-dir data/processed/mmsvg_sft_20k_gemma4_32k \
  --max-seq-length 32768 --local-files-only
```

기존 디렉터리는 의도치 않은 overwrite를 막기 위해 자동으로 거부한다.

## 5. Critic SFT 계약

기존 repository에는 Critic 관측 corpus만 있고 label schema/trainer는 미확정 상태였다.
새 경로는 임의의 label을 만들지 않고 현재 실제 `VLMScoreCritic`이 사용하는 계약을
그대로 학습 대상으로 삼는다.

- JSONL 필수 필드: `sample_id`, `instruction`, `image_path`, `labeled_svg`,
  `allowed_target_ids`, `critic_output`.
- `critic_output`은 18개 category/type pair를 정확히 한 번씩 평가하고, production
  parser가 허용하는 issue만 포함해야 한다.
- image-grounded prompt/completion 포맷, assistant completion-only loss를 사용한다.
- Qwen2.5-VL-3B vision tower는 고정하고 language model의 attention/MLP projection에만
  LoRA를 적용한다.
- collator는 silent truncation을 하지 않는다. image token을 포함한 실제 batch 길이가
  8,192를 넘으면 해당 sample ID와 함께 즉시 실패한다.
- train/validation sample ID 중복, 누락 이미지, 잘못된 target ID 또는 불완전한
  scorecard를 학습 전에 거부한다.

실제 라벨을 배치하기 전에 다음 검증을 통과해야 한다.

```bash
.venv/bin/python -m svg_agentic_slm.train.train_critic \
  --config configs/train_critic_a100_80gb.yaml --validate-only
```

현재 `data/processed/critic_sft/{train,validation}.jsonl`에는 production label이 아직
없으므로 실제 Critic 전체 학습은 시작하지 않았다. schema-valid 합성 label과 실제 PNG를
사용한 GPU 스모크는 성공했지만, 이는 실행 경로 검증이지 학습 품질 검증이 아니다.

## 6. 실제 GPU 검증 결과

주 비교는 8K cohort의 실제 중앙값과 정확히 같은 3,168-token row를 반복하여,
cache/평가/저장/DataLoader worker를 끈 뒤 유효 batch 16과 한 optimizer update로
통제했다. QLoRA와 BF16 batch 1은 1×16, 최종 권장 설정은 4×4로 모두 같은 50,688
input token을 처리했다. 최대 길이 안전성은 실제 8,191-token row 네 개를 한 batch로
별도 확인했다.

| 실험 | GPU | step / input token | runtime | token/s | peak allocated / reserved |
|---|---:|---:|---:|---:|---:|
| Generator 기존 NF4 QLoRA+GC, batch 1×accum 16 | 1 | 1 / 50,688 | 61.581 s | 823.1 | 14.51 / 15.00 GiB |
| Generator BF16+GC, batch 1×accum 16 | 1 | 1 / 50,688 | 55.237 s | 917.7 | 26.23 / 27.71 GiB |
| **Generator 권장 BF16+GC, batch 4×accum 4** | 1 | 1 / 50,688 | **50.139 s** | **1,010.9** | 34.29 / 35.75 GiB |
| Generator 권장, 실제 최대 8,191 token × batch 4 | 1 | 1 / 32,764 | 48.999 s | 668.7 | 52.76 / 56.38 GiB |
| Generator BF16, GC off 실험 상한, batch 1 | 1 | 5 / 15,840 | 13.786 s | 1,149.0 | 73.97 / 76.66 GiB |
| **Generator 권장 2-GPU DDP, batch 4×accum 4** | 2 | 1 / 101,376 | **51.056 s** | **1,985.6** | 34.52 / 48.97 GiB on rank 0 |
| Critic Qwen2.5-VL BF16, batch 2×accum 8 | 1 | 1 / 58,784 | 9.675 s | 6,076.3 | 49.28 / 51.50 GiB |
| Critic Qwen2.5-VL BF16, 2-GPU DDP | 2 | 1 / 117,568 | 9.939 s | 11,829.4 | 49.39 / 51.54 GiB on rank 0 |

첫 세 Generator 행은 동일 입력과 동일 유효 batch의 직접 상대 비교다. BF16 전환만으로
처리량이 11.5% 증가했고, 남은 VRAM으로 실제 batch를 4까지 키워 추가 10.2%, 기존 대비
총 22.8% 증가했다. loss chunk 512/1,024/2,048 비교는 1,010.9/1,009.0/1,011.4
tokens/s로 유의미한 차이가 없어 메모리가 가장 적은 512를 유지했다. GC off 행은 빠르지만
가용 VRAM의 대부분을 점유하므로 권장하지 않는다. 2-GPU 행은 같은 workload를 rank당
16개씩 처리한 실제 global-throughput 비교로 단일 GPU 대비 1.964×, scaling efficiency
98.2%였다. LoRA gradient 통신량이 작아 `SYS` 경로의 NCCL/PCIe 비용이 이 측정에서는
작았다. 다만 반복 sample 한 update 결과이므로 전체 데이터의 장시간 scaling 보장은 아니다.

Critic은 3,674-token 합성 sample 16개로 현재 batch 2×accumulation 8을 측정했을 때
6,076 tokens/s, peak allocated/reserved 49.28/51.50 GiB였다. Batch 4/no-checkpointing은
OOM이었고 batch 4/checkpointing은 4,168 tokens/s로 31% 느려 현재 설정을 유지했다.
2-GPU에서는 11,829 tokens/s로 1.947× scaling했다. Production label이 아직 없으므로 이
결과는 학습 품질이나 실제 길이 분포의 보장이 아니다.

8K train cohort의 한 epoch 입력 합계 38,290,028 token에 위 처리량이 그대로 유지된다고
단순 환산하면 기존 QLoRA는 epoch당 약 12.92시간, 권장 설정은 약 10.52시간이다. 5 epoch는
**64.61 → 52.60시간**, 약 **12.00시간 절감**이다. 이는 단일 GPU 계산 처리량을 사용한
추정치다. 새 2-GPU 실측 1,985.6 tokens/s를 그대로 적용하면 약 5.36시간/epoch,
26.78시간/5 epochs이나, 길이 혼합, 평가와 checkpoint I/O가 포함된 wall time은 100-step
이상의 production benchmark로 다시 확인해야 한다.

최종 검증에서는 전체 pytest **684개가 모두 통과**했고, 변경 파일 Ruff, Python
compile, shell syntax, dependency compatibility, 세 A100 설정의 2-GPU 환경 검사도
통과했다. 2-GPU 실행 종료 시 NCCL process-group 미정리 경고도 발생하지 않았다.

## 7. 실행 방법

환경 사전 점검과 2-GPU DDP를 포함한 권장 실행:

```bash
# 빠른 8K Generator
./scripts/run_generator_sft.sh configs/train_lora_a100_80gb.yaml

# 32K coverage Generator
./scripts/run_generator_sft.sh configs/train_lora_a100_80gb_long.yaml

# production label 검증 후 Critic
.venv/bin/python -m svg_agentic_slm.train.train_critic \
  --config configs/train_critic_a100_80gb.yaml --validate-only
./scripts/run_critic_sft.sh configs/train_critic_a100_80gb.yaml
```

NVLink가 없으므로 두 독립 실험의 총 처리량이 중요할 때는 한 작업당 한 GPU를 주는 것이
유리할 수 있다. 출력 디렉터리는 반드시 서로 달라야 한다.

```bash
CUDA_VISIBLE_DEVICES=0 NUM_PROCESSES=1 \
  ./scripts/run_generator_sft.sh configs/train_lora_a100_80gb.yaml &
CUDA_VISIBLE_DEVICES=1 NUM_PROCESSES=1 \
  ./scripts/run_critic_sft.sh configs/train_critic_a100_80gb.yaml &
wait
```

반대로 한 학습의 wall time이 우선이면 기본 런처가 두 GPU를 모두 사용한다. production
결정 전에는 실제 길이 분포로 100 step 이상 측정하고 manifest의
`num_input_tokens_seen / train_runtime`과 peak VRAM을 비교해야 한다.

## 8. 참고 자료

- GitHub 기준 커밋: <https://github.com/JYS1025/svg-agentic-slm/commit/ff3197349ea0e9cb1e7ee67000ec5549937b7592>
- NVIDIA A100 공식 사양: <https://www.nvidia.com/en-gb/data-center/a100/>
- PyTorch SDPA: <https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial>
- Transformers mixed precision: <https://huggingface.co/docs/transformers/v5.10.1/mixed_precision_training>
- Transformers gradient checkpointing: <https://huggingface.co/docs/transformers/v5.7.0/en/grad_checkpointing>
