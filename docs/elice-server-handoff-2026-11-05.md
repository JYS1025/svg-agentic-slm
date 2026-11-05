# Elice 서버 반납과 Critic 학습 재개 인수인계

2026년 11월 5일 기준으로 현재 Critic distillation 작업의 코드, 데이터, 학습 상태를
보존했다. 새 인스턴스에서 실행 환경과 기본 모델을 준비하면 Stage 2의
`checkpoint-150`부터 재개할 수 있다. 새 인스턴스의 실제 GPU 학습 실행은 별도 확인이
필요하며, ZIP을 풀기만 하면 기존 서버 환경까지 복원되는 것은 아니다.

## 인계 파일과 코드 기준점

| 항목 | 위치 또는 값 |
| --- | --- |
| GitHub 저장소 | <https://github.com/JYS1025/svg-agentic-slm> |
| 작업 브랜치 | `elice_backup` |
| ZIP 내부 실행 코드 | `6de4583fb24884b78077327e356dfd68308094e8` |
| 다운로드할 ZIP | `/home/elicer/elice-backup-20261105.zip` |
| ZIP 검증 파일 | `/home/elicer/elice-backup-20261105.zip.sha256` |
| ZIP 크기 | 2,002,637,029 bytes, 약 2.00 GB |
| ZIP SHA-256 | `7424cf60f42181a58d8b6e9314fdaa621d454909200724c3ad4fba8a2bf4f929` |
| 상세 복원 검증 결과와 관측 환경 | [검증 JSON](elice-backup-verification-2026-11-05.json) |

GitHub에는 `backup/mmsvg-rag-before-origin-sync`의 `5306d1b`와
`backup/pr10-before-cleanup-20260918`의 `716ee3b`도 보존했다. 이 handoff 문서는
ZIP 생성 이후 추가했으므로 ZIP 내부 bundle의 브랜치는 위 코드 기준점에 머문다.
GitHub의 최신 문서와 ZIP의 실행 코드를 함께 사용한다.

서버에서 다운로드 파일을 만들고 검증한 상태다. 사용자 로컬 장비에 전송됐는지와
그 장비에서의 체크섬 일치는 별도로 확인해야 한다.

## 백업 내용

ZIP에는 다음 8개 파일이 들어 있다. 용량은 십진수 MB 기준이다.

| 파일 | 압축 후 용량 | 내용 |
| --- | ---: | --- |
| `01-omnisvg-generation.tar.gz` | 721.16 MB | `inputs`, `runs`, `teacher_runs`, `similarity`, `inspections`, `smoke_input` |
| `02-critic-distill-sft.tar.gz` | 920.82 MB | `outputs/sft/critic_distill_pipeline_main/` 전체 |
| `03-derived-datasets.tar.gz` | 328.03 MB | `data/processed/`와 sample ID 압축 파일 |
| `04-evaluations-and-logs.tar.gz` | 6.83 MB | `outputs/sft`를 제외한 `outputs/`, `logs/`, `artifacts/` |
| `05-svg-agentic-slm.bundle` | 3.88 MB | Git refs와 이력의 오프라인 사본 |
| `INVENTORY.json` | 21.93 MB | 원본 파일별 경로, 크기, SHA-256 |
| `RESTORE.txt` | 소량 | 기본 복원 안내 |
| `SHA256SUMS` | 소량 | ZIP 내부 파일의 SHA-256 목록 |

원본 데이터는 총 88,787개 파일, 3,052,369,888 bytes다. 데이터와 checkpoint를
선별하거나 중복 제거하지 않았고, teacher JSONL의 재시도 이력도 유지했다.

## 현재 작업 상태

| 항목 | 상태 |
| --- | --- |
| OmniSVG 생성 | 10,000건 중 9,985건 생성, 15건 렌더 실패 |
| Teacher 완료 조건 | `source_run_finished=true`, `awaiting_teacher=0` |
| 유효 teacher 라벨 | 7,920건 |
| Stage 1 | 완료, step 88, checkpoint 50/88 및 최종 adapter와 processor 보존 |
| Stage 2 | 완료 manifest 없음, 마지막 로그 step 151, 저장된 checkpoint 100/150 |
| Stage 3 | 실행 결과 없음 |
| 재개 위치 | Stage 2 `checkpoint-150`, scheduler step 150, 해당 단계 max step 151 |
| 실행 프로세스 | 점검 당시 Critic 학습 및 OmniSVG 생성 프로세스 없음 |

pipeline과 Stage 2의 `running` 표시는 2026년 9월 29일에 남은 기록이다.
Stage 2의 마지막 한 step과 완료 처리, 평가, 최종 adapter 저장을 다시 수행한 뒤
Stage 3으로 넘어가야 한다.

| 스냅샷 | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| `01_initial` | 2,789 | 155 | 155 |
| `02_new_only` | 4,821 | 155 | 155 |
| `03_full_stable` | 7,610 | 155 | 155 |
| `final_raw` | 7,128 | 396 | 396 |

후속 단계는 최초 validation/test를 고정해 사용한다. 전체 고정 스냅샷의 통과 라벨은
train 10, validation 1, test 1건이다. 배포 전 통과 예제를 보강하고 과도한 거절 여부를
평가해야 한다.

## 복원 검증 결과

ZIP을 별도 임시 디렉터리에 풀고, 그 안의 bundle에서 `elice_backup` 코드를 clone한
다음 네 archive를 실제 파일로 복원해 다음을 검사했다.

- 원본 파일 88,787개의 크기와 SHA-256 전부 일치.
- 스냅샷 12개 분할의 manifest 해시 전부 일치.
- 이미지 참조 24,070건 모두 존재하며 이미지 해시 일치. 여러 스냅샷의 반복 참조를 포함한 수다.
- 복원된 코드의 checkpoint 탐색 함수가 `checkpoint-150`을 선택.
- adapter tensor 496개, optimizer state 496개 정상 로드.
- scheduler `last_epoch=150`, trainer `global_step=150` 확인.
- `rng_state_0.pth`, `rng_state_1.pth`, adapter 설정, `training_args.bin` 보존.

이는 파일 복원과 학습 상태 역직렬화 검증이다. 새 인스턴스에서의 모델 로드,
GPU 학습 step, 다른 GPU 수에서의 동일한 결과까지 보장하는 검증은 아니다.

## 새 인스턴스 복원 순서

### 다운로드와 체크섬 확인

ZIP과 `.zip.sha256`을 함께 가져온다. 로컬의 SSH 주소와 포트에 맞춰 실행한다.

```bash
scp -P SSH_PORT 'elicer@SSH_HOST:/home/elicer/elice-backup-20261105.zip*' .
sha256sum -c elice-backup-20261105.zip.sha256
```

macOS에서는 `shasum -a 256 -c elice-backup-20261105.zip.sha256`을 사용할 수 있다.
Windows에서는 `Get-FileHash -Algorithm SHA256` 결과를 위 ZIP SHA-256과 대조한다.

### 코드와 데이터를 같은 저장소 경로에 복원

새 Linux 인스턴스에 ZIP을 전송한 뒤 실행한다. `/home/elicer`를 만들거나 쓰기 위한
권한과 ZIP 해제 도구가 필요하다. 대상 저장소 디렉터리는 비어 있어야 한다.

```bash
mkdir -p /home/elicer/restore-20261105
unzip /path/to/elice-backup-20261105.zip -d /home/elicer/restore-20261105
cd /home/elicer/restore-20261105
sha256sum -c SHA256SUMS
gzip -t ./*.tar.gz

git clone --branch elice_backup \
  /home/elicer/restore-20261105/05-svg-agentic-slm.bundle \
  /home/elicer/svg-agentic-slm

for archive in 01-omnisvg-generation.tar.gz 02-critic-distill-sft.tar.gz \
  03-derived-datasets.tar.gz 04-evaluations-and-logs.tar.gz; do
  tar -xzf "$archive" -C /home/elicer/svg-agentic-slm
done

cd /home/elicer/svg-agentic-slm
git rev-parse HEAD
```

기대 커밋은 `6de4583fb24884b78077327e356dfd68308094e8`이다. 최신 handoff 문서는
GitHub `elice_backup`에서 확인한다.

스냅샷의 `candidate_image_path`와 trainer의 `best_model_checkpoint` 등에
`/home/elicer/svg-agentic-slm` 절대경로가 저장돼 있다. 다른 위치로 옮기려면 해당
참조와 설정 및 manifest 해시까지 함께 정리해야 한다. 단순히 현재 디렉터리만 바꿔
실행하지 않는다. ZIP 내부 `RESTORE.txt`의 일반적인 별도 `code`/`restored` 디렉터리
예시 대신, 학습 재개에는 위의 같은 저장소 경로 복원을 따른다.

### Python과 CUDA 환경 설치

관측 환경은 Python 3.11.13, PyTorch 2.6.0+cu118, CUDA runtime 11.8,
NVIDIA A100 80GB PCIe 2개다. GPU 수나 유효 batch를 바꾸면 학습 진행과 RNG의 의미가
달라지므로 우선 같은 2-GPU 구성으로 재개한다. 다른 구성의 메모리 적합성은 따로
측정해야 한다.

| 패키지 | 관측 버전 |
| --- | --- |
| torchvision | 0.21.0+cu118 |
| transformers | 5.16.1 |
| peft | 0.20.0 |
| accelerate | 1.14.0 |
| bitsandbytes | 0.50.2 |
| numpy | 2.4.6 |
| safetensors | 0.8.0 |
| huggingface-hub | 1.29.0 |

다음은 해당 버전을 기준으로 한 설치 절차다. 새 인스턴스에서는 NVIDIA driver와
PyTorch CUDA 호환성, CairoSVG가 사용할 시스템 Cairo 라이브러리를 먼저 준비한다.
깨끗한 새 환경에서 이 설치 절차 전체를 실행한 검증은 아직 필요하다.

```bash
cd /home/elicer/svg-agentic-slm
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install \
  'torch==2.6.0' 'torchvision==0.21.0' \
  --index-url https://download.pytorch.org/whl/cu118
.venv/bin/python -m pip install -e '.[train,vlm,dev]' \
  'transformers==5.16.1' 'peft==0.20.0' 'accelerate==1.14.0' \
  'bitsandbytes==0.50.2' 'numpy==2.4.6' 'safetensors==0.8.0' \
  'huggingface-hub==1.29.0'
.venv/bin/python -m pip check
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())'
```

전체 관측 패키지 버전은 [검증 JSON](elice-backup-verification-2026-11-05.json)의
`observed_environment`에 있다. 이는 설치 상태 기록이며 새 환경에서 검증한 lockfile은
아니다. OmniSVG 생성과 API teacher용 별도 환경은 각각
`configs/requirements_omnisvg_critic_smoke.txt`,
`configs/requirements_omnisvg_teacher.txt`를 사용한다.

### 기본 모델 다운로드와 캐시 준비

Critic adapter는 기본 모델 가중치를 포함하지 않는다. 현재 설정은
`local_files_only=true`이므로 먼저 정확한 revision을 캐시에 내려받아야 한다.

```bash
.venv/bin/hf download Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
```

| 모델 | Revision | 필요한 작업 |
| --- | --- | --- |
| `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | Stage 2/3 학습과 Critic 추론 |
| `google/siglip2-base-patch16-224` | `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` | similarity 재계산과 benchmark |
| `lmstudio-community/gemma-4-12B-it-QAT-GGUF` | `291406f49e16eff811c85ad8884d375f34138663` | Gemma generator 추론 |

위 세 revision은 2026년 11월 5일 Hub 조회에서 public/non-gated 상태로 확인했다.
이미 frozen 데이터와 최종 snapshot이 있으므로 현재 Critic 학습 재개를 위해 teacher를
다시 호출하거나 similarity를 다시 계산할 필요는 없다.

멀티모달 학습 캐시는 백업에서 제외했고 재생성한다. 기본 경로는
`/dev/shm/svg_agentic_slm/critic_sft`다. 이미지 patch만 Stage 2 약 16.14 GB,
Stage 3 약 24.91 GB이며 token 배열과 기타 파일이 추가된다. 두 캐시를 보존하면
누적 41 GB 이상이므로 충분한 tmpfs를 준비한다. 작은 `/dev/shm` 환경에서는
`configs/train_critic_distill.yaml`의 `train.sft.multimodal_cache_dir`를 여유 있는
디스크 경로로 바꾼다. 디스크에서도 mmap 캐시를 사용하며 처리 성능은 별도 확인한다.

### 기존 pipeline 재개

```bash
cd /home/elicer/svg-agentic-slm
CUDA_VISIBLE_DEVICES=0,1 NUM_PROCESSES=2 \
  scripts/run_critic_distill_sft.sh \
  --pipeline-dir outputs/sft/critic_distill_pipeline_main \
  --final-epochs 3
```

기존 `--pipeline-dir`을 반드시 지정한다. 생략하면 새 timestamp 디렉터리를 만들며
기존 Stage 2를 재개하지 않는다. launcher는 완료한 Stage 1을 재사용하고 Stage 2에서
`checkpoint-150`을 선택하며, optimizer를 호환되는 `adamw_bnb_8bit`로 사용한다.
재개 시 `pipeline_events.jsonl`의 `stage_checkpoint_resume_selected` 이벤트와
checkpoint 경로를 확인한다.

백업에 있는 Stage 2 `stage_config.yaml`의 `resume_from_checkpoint`는 이전 실행의
`checkpoint-100`이다. 그 파일을 trainer에 직접 넘기지 않고 위 pipeline launcher를
통해 최신 완전 checkpoint를 선택한다. 모델, LoRA 설정, 데이터, epoch를 변경한
실험은 새 pipeline 디렉터리에서 수행한다. 현재 완료 단계 재사용은 manifest 존재만
검사하므로 설정을 바꾸고 같은 pipeline에 덮어 실행하지 않는다.

## 복원 범위의 제한

- **Generator SFT 가중치:** `outputs/sft/gemma4_raw/adapter`와
  `outputs/sft/gemma4_discrete/adapter`는 원래 이 서버에 없었고 ZIP에도 없다.
  R0 보고서가 가리키는 `/data/minjun_dev/svg_sft_experiments/...` checkpoint도
  이 서버에 없다. 과거 Generator SFT 모델의 복원에는 원래 실험 장비의 가중치가 필요하다.
- **원본 데이터와 벡터 DB:** `data/raw/`, Chroma DB, 모델 캐시와 가상환경은
  제외했다. 현재 frozen Critic 학습에는 가공 데이터가 충분하며, 새 RAG 색인이나
  원본 데이터 실험에는 재다운로드와 재색인이 필요하다.
- **인증:** `.env`, `test/api_keys.txt`, GitHub/Hugging Face 인증 정보는 제외했다.
  API teacher나 유료 generator를 다시 사용할 때 별도 인증이 필요하다.

## 후속 작업과 서버 반납 조건

1. 사용자 로컬 또는 다른 저장소에서 ZIP과 SHA-256을 검증한다.
2. 새 인스턴스에 환경과 기본 모델을 준비하고 checkpoint-150부터 Stage 2를 완료한다.
3. Stage 3 완료 뒤 통과율과 과도한 거절 여부를 측정한다. test의 통과 라벨이 1건인
   현재 상태로 배포 성능을 판단하지 않는다.
4. 완료 단계 재사용에 설정·산출물 일치 검증을 추가하고, `origin/main`의 별도
   critic SFT 경로와 현재 distillation 경로의 역할을 대조한다.

로컬 수신 검증이 끝나기 전에는 서버의 원본이나 백업을 삭제하지 않는다.
다운로드 완료와 인스턴스 삭제는 이 문서 작성 시점에 확인되지 않았으며,
새 인스턴스의 첫 GPU step 성공까지 확인하면 실행 환경을 포함한 이전 확인이 된다.

관련 문서: [백업 생성 절차](elice_artifact_backup.md),
[Critic distillation 구현과 학습](critic_distill_sft.md).
