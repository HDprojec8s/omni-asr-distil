#!/bin/bash
#SBATCH --job-name=distil-s2
#SBATCH --output=logs/distil-s2_%j.out
#SBATCH --error=logs/distil-s2_%j.err
#SBATCH --nodes=1
#SBATCH --partition=p4
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=10G
#SBATCH --qos=gpuultimate
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

# --- Arguments ---
# Usage: sbatch [--gpus-per-node=N --ntasks-per-node=N] slurm/stage2.sh <config.yaml> <stage1_config> [--resume]
# Example: sbatch --gpus-per-node=4 --ntasks-per-node=4 slurm/stage2.sh configs/stage2/stream_dct_large.yaml s_large_512
CONFIG_FILE=${1:?Usage: sbatch slurm/stage2.sh <config.yaml> <stage1_config> [--resume]}
STAGE1_CONFIG=${2:?Usage: sbatch slurm/stage2.sh <config.yaml> <stage1_config> [--resume]}
RESUME=${3:-""}
CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
OUTPUT_DIR="/nfs1/scratch/students/witzlch88229/output/distil-stage2/${CONFIG_NAME}"
STAGE1_OUTPUT="/nfs1/scratch/students/witzlch88229/output/distil-stage1/${STAGE1_CONFIG}"

# Detect GPU count from SLURM allocation
NUM_GPUS=${SLURM_NTASKS_PER_NODE:-1}

# --- Find latest Stage 1 checkpoint ---
STAGE1_CHECKPOINT=$(ls -d "${STAGE1_OUTPUT}"/ws_*/checkpoints/step_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
if [ -z "$STAGE1_CHECKPOINT" ]; then
    echo "ERROR: No Stage 1 checkpoint found in ${STAGE1_OUTPUT}/ws_*/checkpoints/step_*"
    exit 1
fi
STAGE1_MODEL="${STAGE1_CHECKPOINT}/model"
echo "Stage 1 checkpoint: ${STAGE1_MODEL}"

# --- Resume check ---
# Auto-resume on Slurm requeue (preemption) or explicit --resume flag.
if [ "$RESUME" = "--resume" ] || [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
    echo "Resume mode (restart_count=${SLURM_RESTART_COUNT:-0}): continuing from last checkpoint in ${OUTPUT_DIR}"
elif [ -d "${OUTPUT_DIR}" ] && ls "${OUTPUT_DIR}"/ws_*/checkpoints/step_* &>/dev/null; then
    echo "ERROR: Output directory ${OUTPUT_DIR} already contains checkpoints."
    echo "Use '--resume' as third argument to continue training, or remove the directory."
    exit 1
fi

# --- GPU-aware batch sizing ---
# Scale max_num_elements to GPU VRAM; adjust grad accumulation to keep
# effective batch size constant (~61.4M elements per update).
# Effective = max_num_elements × num_gpus × num_batches
case "${SLURM_JOB_PARTITION}" in
    p4)  # H200 141GB
        MAX_NUM_ELEMENTS=15360000
        NUM_BATCHES=$(( 4 / NUM_GPUS > 0 ? 4 / NUM_GPUS : 1 ))
        ;;
    p2)  # A100 80GB
        MAX_NUM_ELEMENTS=8000000
        NUM_BATCHES=$(( 8 / NUM_GPUS > 0 ? 8 / NUM_GPUS : 1 ))
        ;;
    p1)  # A100 40GB
        MAX_NUM_ELEMENTS=3840000
        NUM_BATCHES=$(( 16 / NUM_GPUS > 0 ? 16 / NUM_GPUS : 1 ))
        ;;
    *)
        echo "WARNING: Unknown partition '${SLURM_JOB_PARTITION}', using p4 defaults"
        MAX_NUM_ELEMENTS=15360000
        NUM_BATCHES=$(( 4 / NUM_GPUS > 0 ? 4 / NUM_GPUS : 1 ))
        ;;
esac

# --- Trap SIGUSR1 → forward to training process for graceful checkpoint ---
cleanup() {
    echo "$(date): Caught USR1 signal, forwarding to training process..."
    if [ -n "$TRAIN_PID" ]; then
        kill -USR1 "$TRAIN_PID"
        wait "$TRAIN_PID"
    fi
}
trap cleanup USR1

# --- Setup ---
cd /nfs1/scratch/students/witzlch88229/projects/omni-asr-distil || { echo "Directory not found"; exit 1; }
mkdir -p logs

export TMPDIR="/nfs1/scratch/students/witzlch88229/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

# --- Environment activation ---
# Multi-GPU requires conda (venv has anaconda/stdlib conflicts that cause GIL crashes).
# Single-GPU uses venv.
if [ "$NUM_GPUS" -gt 1 ]; then
    eval "$(conda shell.bash hook 2>/dev/null)"
    conda activate omni-distil
else
    unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE CONDA_SHLVL
    unset PYTHONPATH PYTHONHOME
    PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '/conda\|/anaconda' | paste -sd ':')"
    LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/conda\|/anaconda' | paste -sd ':')"
    source .venv/bin/activate
fi

echo "=================================================================="
echo "Starting Stage 2 Streaming Distillation at $(date)"
echo "Job submitted to partition ${SLURM_JOB_PARTITION} on ${SLURM_CLUSTER_NAME}"
echo "Config: ${CONFIG_FILE}"
echo "Stage 1: ${STAGE1_MODEL}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${NUM_GPUS} | max_num_elements: ${MAX_NUM_ELEMENTS} | grad_accum: ${NUM_BATCHES}"
echo "Python: $(which python)"
echo "=================================================================="

# --- Launch training (background, so trap can fire) ---
if [ "$NUM_GPUS" -gt 1 ]; then
    # torchrun for multi-GPU. common.cluster=none disables fairseq2's SlurmHandler
    # (which restricts CUDA_VISIBLE_DEVICES and breaks NCCL).
    torchrun --nproc_per_node="${NUM_GPUS}" \
        scripts/run_stage2.py "$OUTPUT_DIR" \
        --config-file "${CONFIG_FILE}" \
        --config common.cluster=none \
                  model.path="${STAGE1_MODEL}" \
                  teacher.path="${STAGE1_MODEL}" \
                  dataset.asr_task_config.max_num_elements="${MAX_NUM_ELEMENTS}" \
                  trainer.grad_accumulation.num_batches="${NUM_BATCHES}" &
else
    python scripts/run_stage2.py "$OUTPUT_DIR" \
        --config-file "${CONFIG_FILE}" \
        --config model.path="${STAGE1_MODEL}" \
                  teacher.path="${STAGE1_MODEL}" \
                  dataset.asr_task_config.max_num_elements="${MAX_NUM_ELEMENTS}" \
                  trainer.grad_accumulation.num_batches="${NUM_BATCHES}" &
fi
TRAIN_PID=$!
wait "$TRAIN_PID"
