# Qwen3.5 critic distillation SFT

For restoring the 2026-11-05 server backup and resuming checkpoint-150, follow
the [server handoff](elice-server-handoff-2026-11-05.md). It records the required
environment, model downloads, absolute paths, and the existing pipeline directory.

The critic student receives exactly the production `critic_v1` evidence: one
candidate render, the original instruction, the labeled candidate SVG, its
allowed target IDs, and the production SigLIP2 pair probability. Reference
images, reference SVG, and teacher-only addenda are excluded. The assistant
target is the validated 18-entry scorecard JSON.

## Precompute SigLIP2 evidence

```bash
.venv/bin/python scripts/precompute_critic_similarity.py --device cuda:0
```

This uses pinned `google/siglip2-base-patch16-224` revision
`75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` and the same
`This is a photo of {instruction}.` template as `simple_bench`. Results are
append-only under `.work/omnisvg_critic_distill/similarity/siglip2/`. Re-running
scores only validated candidate hashes that are not already present.

## Prepare a frozen dataset

```bash
.venv/bin/python scripts/prepare_critic_distill_sft.py
```

This joins the latest generated candidate SHA with the latest validated Luna
teacher result, verifies artifact hashes and the critic contract, excludes blank
candidates, and writes deterministic pass-stratified 90/5/5 splits under
`data/processed/omnisvg_critic_distill/`. Pass records are not duplicated or
oversampled. Re-run this command explicitly when a newer teacher snapshot should
replace the frozen dataset.

## Staged training

```bash
scripts/run_critic_distill_sft.sh
```

The default launcher creates three stages:

1. Current scored snapshot, one epoch from the base model.
2. Wait for `awaiting_teacher=0`, score the remaining records, then train one
   epoch on new validated records only from the stage-1 adapter.
3. Train for `--final-epochs N` on initial train plus all new records from the
   stage-2 adapter.

The initial validation and test sets stay fixed for all stages. Optimizer and
scheduler state are new at each stage; only adapter weights are inherited. The
default config launches one DDP process per visible GPU, loads the pinned
Qwen3.5-4B model with NF4 double-quantized QLoRA, freezes the vision tower, and
attaches LoRA to every linear layer below `language_model`.

By default, stage 1 copies the already-frozen
`data/processed/omnisvg_critic_distill` snapshot. Teacher results arriving after
that snapshot are therefore unambiguously assigned to stage 2. Select another
frozen source with `--initial-snapshot PATH` when needed.

An explicit run location and final epoch count can be selected with:

```bash
scripts/run_critic_distill_sft.sh configs/train_critic_distill.yaml \
  --pipeline-dir outputs/sft/critic_distill_pipeline_main \
  --final-epochs 3
```

One process builds a fingerprinted cache under an exclusive file lock in
`/dev/shm/svg_agentic_slm/critic_sft/`; other ranks wait on a file lock and then
open the same read-only mmap files. The cache stores token IDs and response-mask
boundaries, multimodal token types, image grids, and processor-produced BF16 image patches.
Warm training runs therefore perform no PNG decode or image processing.

Each pipeline contains immutable snapshots, pipeline events, and three stage
directories. Every stage contains `output/adapter`, `output/processor`,
checkpoints, detailed logs, and `training_manifest.json`.

## Live logging and loss graph

Every Trainer log event is appended to `output/logs/metrics.jsonl` with loss,
eval loss, learning rate, gradient norm, estimated samples seen, and CUDA memory.
`output/logs/loss_curve.svg` and `output/logs/status.json` are refreshed live.

Monitor every stage and regenerate a combined graph with:

```bash
.venv/bin/python scripts/monitor_critic_distill_sft.py watch \
  --pipeline-dir outputs/sft/critic_distill_pipeline_main --interval 10

.venv/bin/python scripts/monitor_critic_distill_sft.py plot \
  --pipeline-dir outputs/sft/critic_distill_pipeline_main
```

The combined graph is written to `PIPELINE_DIR/loss_curve.svg`. The reported
`plateau_candidate` state is only a recent training-loss slope diagnostic; eval
loss and critic behavior still determine convergence.

To use the final adapter in production, set these critic model keys:

```yaml
adapter_path: PIPELINE_DIR/stages/03_full_Nepoch/output/adapter
processor_path: PIPELINE_DIR/stages/03_full_Nepoch/output/processor
load_in_4bit: true
```

The present teacher set is overwhelmingly negative. The preparation and training
manifests mark acceptance calibration as insufficient; deployment should be gated
on held-out pass rate and reject-collapse measurements.

## 2026-11-05 구현·Git 이력 후속 검토

검토 기준은 `elice_backup`의 `c0ead38`이다. 작업 시작 시 tracked/untracked
변경은 없었다. `f4c3550`은 critic teacher, 증류 데이터 준비, 공유 메모리 캐시,
3단계 학습 및 추론 연동을 포함한 58개 파일의 백업 커밋이며, `c0ead38`은
산출물 보존 절차를 추가했다. `git log`, `reflog`, 브랜치 차이와 구현·테스트를
대조했다. 네트워크 fetch는 하지 않았으므로 원격 상태는 로컬 추적 ref 기준이다.

### 수정한 결함

- **후속 학습 단계의 평가셋 누수 방지 누락:** `_assemble_stable_snapshots`는
  `(sample_id, candidate_svg_sha256)`가 새 조합이면 모두 학습에 넣었다. 이 때문에
  평가셋 입력의 새 SVG 또는 다른 입력에서 생성된 동일 평가 SVG가 학습에 들어갈
  수 있었다. 이제 delta/full 구성 모두에서 입력 ID와 SVG 해시가 분할을 넘는지
  파일 저장 전에 검사한다. 학습셋 내부의 반복 SVG는 계속 허용한다. 평가셋과
  충돌하면 명시적으로 실패하며, 자동으로 평가셋을 이동하거나 행을 제거하지 않는다.
- **최초 스냅샷 복사의 잘못된 완료 처리:** 기존에는 manifest를 복사한 다음
  해시를 검증하므로, 검증 실패 후 재시작이 해당 스냅샷을 완료 상태로 재사용할
  수 있었다. 이제 임시 파일에 복사·해시 검증 후 각 split을 교체하고, 마지막에
  manifest를 원자적으로 저장한다. 복사 중단으로 디렉터리만 남아도 재시도한다.
- **깨끗한 체크아웃에서 테스트 실패:** generate CLI 테스트가 실제 작업 트리의
  `dirty=True`를 전제했다. Git provenance 입력을 주입하여 clean/dirty 상태 모두
  저장되는지 검사하도록 변경했다.

원래 HEAD 함수를 메모리에 로드한 대조 실행에서 새 누수 테스트 2개는 모두
실패했다. 수정 후 관련 pipeline/CLI 테스트 29개가 통과했다. 중단된 복사의
재시도와 해시 불일치 후 복구도 회귀 테스트에 포함했다.
최종 전체 테스트는 **708개 통과**했으며, 변경 Python 파일의 Ruff 검사와
`git diff --check`도 통과했다. 테스트에는 기존 PEFT 경고 6건이 있었다.

### 기존 산출물 상태

`outputs/sft/critic_distill_pipeline_main`을 읽기 전용으로 확인했다.

| 항목 | 확인 결과 |
| --- | --- |
| Stage 1 | 완료, global step 88, checkpoint 50/88 및 training manifest 존재 |
| Stage 2 | 마지막 기록 step 151, checkpoint 100/150, 완료 manifest 없음 |
| 실행 여부 | 확인 시 critic 증류 학습·pipeline Python 프로세스 없음 |
| 상태 파일 | `running` 표기는 2026-09-29 기록이므로 현재 실행을 뜻하지 않음 |
| 최초 스냅샷 | train 2,789 / validation 155 / test 155 |
| 추가 학습 데이터 | 4,821건 |
| 전체 고정 스냅샷 | train 7,610 / validation 155 / test 155 |
| 전체 통과 라벨 | train 10 / validation 1 / test 1 (전체 7,920건 중 12건) |
| 현재 전체 스냅샷 분할 중복 | sample ID와 SVG 해시 모두 pairwise 중복 0건 |

### 후속 작업 우선순위

1. **학습 재개:** 현재 실행 중이 아니며 Stage 2가 미완료다. 동일 설정으로 명시적으로
   재개할 때 launcher가 checkpoint-150을 선택하고 optimizer를 호환되는 비페이징
   방식으로 바꾸는 경로가 있다. GPU 학습이나 유료 teacher 호출은 이번 검토에서
   실행하지 않았다. 기존 체크포인트·데이터·로그는 수정하지 않았다.
2. **완료 단계 재사용 계약:** `_launch_stage`는 training manifest 존재만으로 단계를
   재사용하며 현재 설정과의 동일성을 검증하지 않는다. 모델·데이터·epoch 설정을
   바꿀 때에는 새 pipeline 디렉터리를 사용하고, 향후 재사용 시 설정 및 산출물
   해시 검증을 추가해야 한다.
3. **배포 전 평가:** 고정 test의 통과 예제가 1건이므로 acceptance 성능 추정이
   어렵다. pass/reject 평가 예제를 보강하고 과도한 거절 여부를 측정해야 한다.
4. **브랜치 통합:** 로컬 `origin/main`에는 `5b71e8f`/`ff8aca9`의 별도 critic SFT와
   A100 설정 변경이 있다. 현재 브랜치는 다른 증류 경로를 추가한 상태이므로,
   `critic_sft_trainer`와 `train_critic_distill`의 역할 및 설정을 대조한 뒤 통합한다.
   이번 검토에서 merge/rebase는 실행하지 않았다.
