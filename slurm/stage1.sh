#!/bin/bash
#SBATCH --job-name=distil-s1
#SBATCH --output=logs/distil-s1_%j.out
#SBATCH --error=logs/distil-s1_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=p4,p2,p1
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=40G
#SBATCH --gres=gpu:1
#SBATCH --qos=gpuultimate
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

# --- Arguments ---
CONFIG_FILE=${1:?Usage: sbatch stage1.sh <config.yaml> [--resume]}
RESUME=${2:-""}
CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
OUTPUT_DIR="/nfs1/scratch/students/witzlch88229/output/distil-stage1/${CONFIG_NAME}"

# --- Resume check ---
# Auto-resume on Slurm requeue (preemption) or explicit --resume flag.
if [ "$RESUME" = "--resume" ] || [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
    echo "Resume mode (restart_count=${SLURM_RESTART_COUNT:-0}): continuing from last checkpoint in ${OUTPUT_DIR}"
elif [ -d "${OUTPUT_DIR}" ] && ls "${OUTPUT_DIR}"/step_* &>/dev/null; then
    echo "ERROR: Output directory ${OUTPUT_DIR} already contains checkpoints."
    echo "Use '--resume' as second argument to continue training, or remove the directory."
    exit 1
fi

# --- GPU & model-aware batch sizing ---
# Scale max_num_elements to GPU VRAM and model size; adjust grad accumulation
# to keep effective batch size constant (~61.4M elements).
#                    | p4 (H200 141GB)  | p2 (A100 80GB)   | p1 (A100 40GB)
# small  (256)       | 15.360.000 / 4   | 12.000.000 / 6   | 11.520.000 / 6
# medium (384)       | 15.360.000 / 4   |  8.000.000 / 8   |  3.840.000 / 16
# large  (512)       | 15.360.000 / 4   |  8.000.000 / 8   |  3.840.000 / 16
case "${SLURM_JOB_PARTITION}" in
    p4)  # H200 141GB
        MAX_NUM_ELEMENTS=15360000
        NUM_BATCHES=4
        ;;
    p2)  # A100 80GB
        case "${CONFIG_NAME}" in
            s_small*)  MAX_NUM_ELEMENTS=12000000; NUM_BATCHES=6 ;;
            *)         MAX_NUM_ELEMENTS=8000000;  NUM_BATCHES=8 ;;
        esac
        ;;
    p1)  # A100 40GB
        case "${CONFIG_NAME}" in
            s_small*)  MAX_NUM_ELEMENTS=11520000; NUM_BATCHES=6 ;;
            *)         MAX_NUM_ELEMENTS=3840000;  NUM_BATCHES=16 ;;
        esac
        ;;
    *)
        echo "WARNING: Unknown partition '${SLURM_JOB_PARTITION}', using p4 defaults"
        MAX_NUM_ELEMENTS=15360000
        NUM_BATCHES=4
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

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE CONDA_SHLVL
unset PYTHONPATH PYTHONHOME
PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '/conda\|/anaconda' | paste -sd ':')"
LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/conda\|/anaconda' | paste -sd ':')"

source .venv/bin/activate

echo "=================================================================="
echo "Starting Stage 1 Distillation at $(date)"
echo "Job submitted to partition ${SLURM_JOB_PARTITION} on ${SLURM_CLUSTER_NAME}"
echo "Config: ${CONFIG_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo "Partition: ${SLURM_JOB_PARTITION} | max_num_elements: ${MAX_NUM_ELEMENTS} | grad_accum: ${NUM_BATCHES}"
echo "=================================================================="

# --- Launch training (background, so trap can fire) ---
python scripts/run_stage1.py "$OUTPUT_DIR" \
    --config-file "${CONFIG_FILE}" \
    --config dataset.asr_task_config.max_num_elements="${MAX_NUM_ELEMENTS}" \
              trainer.grad_accumulation.num_batches="${NUM_BATCHES}" &
TRAIN_PID=$!
wait "$TRAIN_PID"
