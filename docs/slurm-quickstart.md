# Slurm 빠른 사용 안내

Slurm에서는 로그인 노드에서 작업을 직접 오래 실행하지 않고, 필요한 자원을 요청해 계산 노드에서 실행합니다. 가장 자주 쓰는 흐름은 다음 네 단계입니다.

```text
자원 확인 → 배치 스크립트 작성 → sbatch 제출 → squeue/sacct 확인
```

## 1. 클러스터 정보 확인

파티션 이름, 계정, QoS, GPU 표기는 클러스터마다 다릅니다. 먼저 운영 문서를 읽고 아래 명령으로 사용 가능한 값을 확인합니다.

```bash
sinfo -o '%P %a %l %D %G'
squeue -u "$USER"
```

- `PARTITION`: 실행할 큐입니다. 예: `gpu`, `batch`
- `ACCOUNT`, `QOS`: 기관이 요구할 때만 지정합니다.
- `GRES`: GPU 같은 일반 자원 표기입니다. 예: `gpu:1`, `gpu:a100:1`
- `TIME`, `MEM`, `CPU`: 실제 필요한 양에 가깝게 요청해야 대기 시간이 불필요하게 길어지지 않습니다.

## 2. 짧게 대화형으로 확인하기

환경이나 GPU를 잠깐 확인할 때 사용합니다. 아래의 `SITE_GPU_PARTITION`은 실제 파티션 이름으로 바꿉니다.

```bash
salloc --partition=SITE_GPU_PARTITION \
  --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=16G --time=00:30:00 --gres=gpu:1
srun --pty bash -l
nvidia-smi
exit
```

마지막 `exit`로 셸을 닫고 할당을 해제합니다. 학습이나 긴 생성 작업은 아래의 배치 방식으로 실행합니다.

## 3. 배치 작업 작성하기

예를 들어 프로젝트 루트에서 `jobs/generate.sbatch`를 다음처럼 작성합니다. `SITE_GPU_PARTITION`과 환경 활성화 줄은 사용하는 클러스터에 맞게 수정합니다.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=svg-generate
#SBATCH --partition=SITE_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?프로젝트 루트에서 sbatch를 실행하세요}"
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "job=${SLURM_JOB_ID} host=$(hostname) start=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun --ntasks=1 svg-agentic-slm generate \
  "A blue circle centered inside a 256 by 256 canvas" \
  --config configs/generation.yaml \
  --no-render
```

핵심 지시어는 다음과 같습니다.

| 지시어 | 의미 |
|---|---|
| `--ntasks=1` | 독립 프로세스 한 개 실행 |
| `--cpus-per-task=4` | 그 프로세스가 사용할 CPU 수 |
| `--mem=32G` | 노드에서 요청할 메모리 |
| `--time=02:00:00` | 최대 실행 시간 |
| `--gres=gpu:1` | GPU 한 개 요청(표기는 사이트별로 다를 수 있음) |
| `%x`, `%j` | 작업 이름과 작업 ID를 로그 파일명에 삽입 |

`#SBATCH` 줄에서는 셸 변수나 `$(...)`가 확장되지 않습니다. 모든 지시어는 스크립트의 첫 실행 명령보다 앞에 두어야 합니다.

## 4. 제출하고 상태 확인하기

로그 디렉터리는 제출 전에 만들어야 합니다. Slurm이 스크립트 본문보다 먼저 로그 파일을 열기 때문입니다.

```bash
mkdir -p logs jobs
bash -n jobs/generate.sbatch
sbatch jobs/generate.sbatch
```

제출에 성공하면 `Submitted batch job 12345`처럼 작업 ID가 나옵니다.

```bash
JOB_ID=12345

# 대기·실행 상태
squeue -j "$JOB_ID" -o '%.18i %.9P %.30j %.2t %.10M %.10l %R'

# 상세 요청 자원과 대기 사유
scontrol show job "$JOB_ID"

# 종료 결과와 사용량
sacct -j "$JOB_ID" \
  --format=JobID,JobName%30,State,ExitCode,Elapsed,AllocTRES%50,MaxRSS

# 필요할 때 취소
scancel "$JOB_ID"
```

자주 보는 상태는 `PD`(대기), `R`(실행), `CG`(종료 처리), `COMPLETED`, `FAILED`, `OUT_OF_MEMORY`, `TIMEOUT`입니다. `PD`일 때는 `squeue`의 마지막 열에 표시되는 사유를 먼저 확인합니다.

## 여러 입력을 독립적으로 실행하기

입력 100개를 최대 4개씩 병렬 실행하려면 job array를 사용합니다.

```bash
#SBATCH --array=0-99%4
#SBATCH --output=logs/%x-%A_%a.out

mapfile -t inputs < inputs.txt
input=${inputs[SLURM_ARRAY_TASK_ID]}
srun svg-agentic-slm validate "$input"
```

배열 범위(`0-99`)와 `inputs.txt` 행 수가 정확히 맞는지 제출 전에 확인합니다. `%4`는 동시에 실행되는 배열 작업 수를 4개로 제한합니다.

## 이 프로젝트에서 주의할 점

- 기본 생성 모델은 `configs/model.yaml`에 revision과 파일명이 고정되어 있습니다. 계산 노드의 외부 네트워크가 차단되어 있다면, 사이트가 허용하는 공유 캐시에 먼저 내려받습니다.

  ```bash
  export HF_HOME=/SITE_SHARED_PATH/huggingface
  hf download lmstudio-community/gemma-4-12B-it-QAT-GGUF \
    --revision 291406f49e16eff811c85ad8884d375f34138663 \
    --include gemma-4-12B-it-QAT-Q4_0.gguf
  ```

  같은 `HF_HOME`을 배치 작업에도 설정해야 합니다. 토큰은 스크립트에 직접 적지 않습니다.
- 생성 작업은 CUDA 지원 `llama-cpp-python` 설치가 필요합니다. 제출 전에 `python -c "from llama_cpp import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())"`가 `True`인지 확인합니다.
- `svg-agentic-slm train`은 현재 LoRA 학습 골격이며 `configs/train_lora.yaml`의 모델과 데이터 경로가 준비되어야 합니다. 준비되지 않은 상태에서 장시간 GPU 작업으로 제출하지 않습니다.
- GPU가 보이지 않으면 `CUDA_VISIBLE_DEVICES`를 덮어쓰지 말고, 먼저 GPU 자원 표기와 실제 할당 여부를 확인합니다.

문제가 생겼을 때는 자원을 무조건 늘리기보다 로그의 첫 오류, `State`, `ExitCode`, `MaxRSS`, 경과 시간을 함께 봅니다. Slurm 공식 명령 설명은 [sbatch](https://slurm.schedmd.com/sbatch.html), [squeue](https://slurm.schedmd.com/squeue.html), [sacct](https://slurm.schedmd.com/sacct.html), [job array](https://slurm.schedmd.com/job_array.html), [GRES](https://slurm.schedmd.com/gres.html)에서 확인할 수 있습니다.
