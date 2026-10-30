# Elice 서버 산출물 압축 백업 지침

이 문서는 Elice 서버 사용 종료 전에 Git으로 보존하지 않는 학습·생성 산출물을
다른 장비로 이전하기 위한 작업 지침이다. 모든 경로는 저장소 루트
`/home/elicer/svg-agentic-slm` 기준이다.

코드는 이미 GitHub의 `elice_backup` 브랜치에 백업되어 있다.

- 원격 브랜치: `origin/elice_backup`
- 백업 커밋: `f4c35507c4409553d986879ed1c670a49be27e81`
- 관련 테스트: 70 passed

이 문서에서 다루는 대상은 Git에 들어가지 않은 생성 결과, teacher 결과, 가공
데이터, 학습 checkpoint와 실행 로그다.

## 백업 원칙

1. 아래의 **필수 백업 묶음 1~4**를 모두 다른 장비로 전송한다.
2. 각 압축 파일과 Git bundle의 SHA-256을 생성하고, 수신 장비에서 다시 검증한다.
3. `.env`와 API key는 압축 파일에 넣지 않는다.
4. 가상환경, Hugging Face cache, 원본 MMSVG parquet, Chroma DB는 백업하지 않는다.
5. JSONL 파일을 정리하거나 중복 제거하지 않는다. 재시도 이력과 append-only 로그도
   실험 provenance의 일부이므로 원본 그대로 보존한다.
6. 학습 또는 데이터 생성 프로세스가 파일을 쓰는 중에는 압축하지 않는다. 실행 중인
   작업이 있다면 정상 checkpoint 저장 후 중지하거나, 작업이 끝난 뒤 압축한다.

## 현재 확인된 중요 상태

- OmniSVG 10k 실행: 10,000건 완료, 9,985건 생성 성공, 15건 렌더 실패
- Critic SFT stage 1: 완료, best checkpoint는 `checkpoint-88`
- Critic SFT stage 2: `checkpoint-150`까지 존재하며 약 1 epoch 지점
- Critic SFT stage 3: 결과 없음
- Pipeline manifest는 아직 `running/new_only_stage` 상태

Stage 2가 완결 처리되지 않았으므로 adapter만 복사해서는 안 된다. 재개에 필요한
optimizer, scheduler, RNG state와 trainer state를 포함하도록 pipeline 디렉터리 전체를
백업해야 한다.

## 백업 대상 요약

| 묶음 | 내용 | 압축 전 논리 크기(대략) | 우선순위 |
|---|---|---:|---:|
| 01 | OmniSVG 입력, 생성 결과, teacher 결과, similarity | 1.12 GB | 필수 |
| 02 | Critic SFT checkpoint, optimizer, snapshot, 로그 | 1.22 GB | 필수 |
| 03 | 전체 가공 데이터 | 701 MB | 필수 |
| 04 | Benchmark·generation 결과와 일반 로그 | 약 12 MB | 필수 |
| 05 | Git bundle과 복구 메타데이터 | 수 MB | 권장 |

압축 후 전체 전송량은 데이터 특성에 따라 달라지지만 수 GB 수준으로 예상한다.

## 필수 묶음 1: OmniSVG 생성 및 teacher 결과

다음을 모두 보존한다.

- `.work/omnisvg_critic_distill/inputs`
  - 10,000개 입력 record와 reference 이미지
  - 약 142 MB, 약 10,002개 파일
- `.work/omnisvg_critic_distill/runs`
  - OmniSVG candidate, render, token ID, run manifest
  - 약 414 MB
- `.work/omnisvg_critic_distill/teacher_runs`
  - teacher 요청·응답, 검증 결과, run manifest
  - 약 554 MB
- `.work/omnisvg_critic_distill/similarity`
  - SigLIP2 score와 manifest
  - 약 8 MB
- `.work/omnisvg_critic_distill/inspections`
  - 수동 검사 자료와 snapshot

`teacher_runs/luna_critic_v3_nonblank/results.jsonl`에는 재시도 이력이 포함되어 있어
입력 수보다 행 수가 많을 수 있다. 정상 상태이므로 수정하지 않는다.

## 필수 묶음 2: Critic distillation SFT

다음 디렉터리를 통째로 보존한다.

```text
outputs/sft/critic_distill_pipeline_main/
```

특히 다음 파일을 누락하지 않는다.

- Stage 1 `checkpoint-50`, `checkpoint-88`, 최종 adapter
- Stage 2 `checkpoint-100`, `checkpoint-150`
- 모든 `adapter_model.safetensors`
- 모든 `optimizer.pt`, `scheduler.pt`, `rng_state_*.pth`
- `trainer_state.json`, `training_args.bin`, `adapter_config.json`
- stage별 `stage_config.yaml`
- `metrics.jsonl`, `status.json`, `loss_curve.svg`, `console.log`
- `snapshots/01_initial`, `02_new_only`, `03_full_stable`, `final_raw`
- `pipeline_manifest.json`, `watchdog_events.log`

재개 가능성을 보존하기 위해 중간 checkpoint를 삭제하거나 adapter끼리 중복 제거하지
않는다.

## 필수 묶음 3: 가공 데이터

다음을 백업한다.

```text
data/processed/
```

여기에는 다음이 포함된다.

- OmniSVG critic distillation train/validation/test
- Generator SFT용 MMSVG 가공 데이터
- RAG retrieval hit, candidate score, reference 결과
- benchmark exclusion 데이터
- 데이터 manifest와 split SHA-256

원본 데이터에서 다시 만들 수 있는 자료도 있지만, retrieval·filtering·split 작업의
재현 비용을 피하기 위해 약 701 MB인 가공 데이터는 모두 보존한다.

## 필수 묶음 4: 평가 결과와 일반 로그

다음을 보존한다.

- `outputs/`에서 `outputs/sft/`를 제외한 전체
- `logs/`
- `artifacts/`

여기에는 simple benchmark, GPT-5.6 Luna upper-bound 결과, checkpoint-100/150 critic
평가 결과, SVG·JSON·render·보고서가 포함된다.

## 백업하지 않을 대상

다음은 용량이 크거나 다시 내려받을 수 있으므로 압축에서 제외한다.

| 경로 | 대략적 크기 | 복구 방법 |
|---|---:|---|
| `data/raw/mmsvg_icon_v2` | 16.4 GB | 원본 dataset 재다운로드 |
| `data/raw/mmsvg_illustration_v2` | 10.2 GB | 원본 dataset 재다운로드 |
| `data/chroma_db` | 6.8 GB | 원본 데이터에서 재색인 |
| `data/chroma_db_detail` | 9.2 GB | 원본 데이터에서 재색인 |
| `.venv` | 수 GB | requirements로 재설치 |
| `.work/omnisvg_critic_distill/.venv-*` | 수 GB | requirements로 재설치 |
| Hugging Face cache | 모델별 수 GB | pinned revision 재다운로드 |
| `.work/omnisvg_critic_distill/upstream` | 약 23 MB | 고정 commit 재-clone |

다음 모델·소스 revision은 Git의 manifest와 문서에 기록되어 있다.

- `OmniSVG/OmniSVG`: `40138c74a343eae6c0d1149e0dbe95b163db0665`
- OmniSVG upstream code: `0d1e5e4e8516c341d432de5f211c0afdbecfb773`
- `Qwen/Qwen2.5-VL-3B-Instruct`: `66285546d2b821cf421d4f5eb2576359d3770cd3`
- `Qwen/Qwen3.5-4B`: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- `google/siglip2-base-patch16-224`: `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`

## 절대 포함하면 안 되는 파일

- `.env`
- API key, access token, SSH private key
- `.git-credentials`
- Hugging Face 인증 token

현재 `.env`에는 `BIZROUTER_API_KEY`가 있으므로 일반 압축 파일에 포함하면 안 된다.
새 서버에서는 별도로 재발급하거나 안전한 비밀 저장소를 통해 전달한다.

## 권장 압축 절차

저장소 내부에 압축 파일을 만들면 다음 압축에 다시 포함될 수 있으므로, 저장소의 형제
디렉터리를 사용한다.

```bash
cd /home/elicer/svg-agentic-slm
backup_dir=/home/elicer/elice_backup_archives
mkdir -p "$backup_dir"
```

먼저 학습·생성 프로세스가 실행 중이지 않은지 확인한다.

```bash
ps -ef | rg 'run_critic|train_critic|omnisvg|accelerate launch' || true
```

### 01 OmniSVG 산출물

```bash
tar --zstd -cf "$backup_dir/01-omnisvg-generation.tar.zst" \
  .work/omnisvg_critic_distill/inputs \
  .work/omnisvg_critic_distill/runs \
  .work/omnisvg_critic_distill/teacher_runs \
  .work/omnisvg_critic_distill/similarity \
  .work/omnisvg_critic_distill/inspections
```

### 02 Critic SFT

```bash
tar --zstd -cf "$backup_dir/02-critic-distill-sft.tar.zst" \
  outputs/sft/critic_distill_pipeline_main
```

### 03 가공 데이터

```bash
tar --zstd -cf "$backup_dir/03-derived-datasets.tar.zst" \
  data/processed
```

### 04 평가 결과와 로그

```bash
tar --zstd -cf "$backup_dir/04-evaluations-and-logs.tar.zst" \
  --exclude='outputs/sft' \
  outputs logs artifacts
```

### 05 Git bundle

GitHub 브랜치와 별개로 전체 Git history의 오프라인 사본을 만든다.

```bash
git bundle create "$backup_dir/05-svg-agentic-slm.bundle" --all
git bundle verify "$backup_dir/05-svg-agentic-slm.bundle"
```

## 압축 직후 검증

압축 파일이 완전히 읽히는지 검사한다.

```bash
for archive in "$backup_dir"/*.tar.zst; do
  zstd -t "$archive"
  tar --zstd -tf "$archive" >/dev/null
done
```

SHA-256 목록을 생성한다.

```bash
(
  cd "$backup_dir"
  sha256sum *.tar.zst *.bundle > SHA256SUMS
)
```

최종 파일과 크기를 확인한다.

```bash
ls -lh "$backup_dir"
```

전송 도구가 큰 파일을 안정적으로 처리하지 못하면 각 archive를 1.9 GB 단위로
분할한다. 원본 archive는 수신 장비에서 결합·검증이 끝날 때까지 삭제하지 않는다.

```bash
split -b 1900M -d -a 3 \
  "$backup_dir/ARCHIVE_NAME.tar.zst" \
  "$backup_dir/ARCHIVE_NAME.tar.zst.part-"
```

수신 장비에서 결합할 때는 다음과 같이 처리한다.

```bash
cat ARCHIVE_NAME.tar.zst.part-* > ARCHIVE_NAME.tar.zst
```

## 수신 장비 검증

모든 파일 전송 후 수신 장비에서 다음을 수행한다.

```bash
sha256sum -c SHA256SUMS

for archive in *.tar.zst; do
  zstd -t "$archive"
  tar --zstd -tf "$archive" >/dev/null
done

git bundle verify 05-svg-agentic-slm.bundle
```

검증이 모두 성공한 뒤 별도 임시 디렉터리에 최소 복원 시험을 한다.

```bash
mkdir restore-test
tar --zstd -xf 02-critic-distill-sft.tar.zst -C restore-test

test -f restore-test/outputs/sft/critic_distill_pipeline_main/\
stages/02_new_only_1epoch/output/checkpoints/checkpoint-150/adapter_model.safetensors
test -f restore-test/outputs/sft/critic_distill_pipeline_main/\
stages/02_new_only_1epoch/output/checkpoints/checkpoint-150/optimizer.pt
test -f restore-test/outputs/sft/critic_distill_pipeline_main/pipeline_manifest.json
```

## 완료 조건

백업은 다음 조건을 모두 만족해야 완료로 본다.

- `01`~`05` 파일이 수신 장비에 존재한다.
- 수신 장비의 `sha256sum -c SHA256SUMS`가 모두 성공한다.
- 모든 `.tar.zst`가 `zstd -t` 및 archive listing 검사를 통과한다.
- Git bundle 검증이 성공한다.
- `checkpoint-150`의 adapter, optimizer, trainer state를 복원할 수 있다.
- OmniSVG `full_10k/results.jsonl`, teacher `results.jsonl`, similarity
  `scores.jsonl`을 복원할 수 있다.
- 압축 파일에 `.env`나 인증 token이 포함되지 않았음을 확인한다.

검증 완료 전에는 Elice 서버의 원본 파일이나 압축 파일을 삭제하지 않는다.
