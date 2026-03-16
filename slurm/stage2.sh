#!/bin/bash
#SBATCH --job-name=distil-s2
#SBATCH --output=distil-s2_%j.out
#SBATCH --error=distil-s2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=p4
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1
#SBATCH --qos=preemptible
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

# --- Arguments ---
CONFIG_FILE=${1:?Usage: sbatch stage2.sh <config.yaml>}
CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
OUTPUT_DIR="/nfs1/scratch/students/witzlch88229/output/distil-stage2/${CONFIG_NAME}"

# --- Trap SIGUSR1 → forward to training process for graceful checkpoint ---
cleanup() {
    echo "$(date): Caught USR1 signal, forwarding to training process..."
    if [ -n "$TRAIN_PID" ]; then
        kill -USR1 "$TRAIN_PID"
        wait "$TRAIN_PID"
    fi
    echo "$(date): Requeueing job $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
}
trap cleanup USR1

# --- Setup ---
cd /nfs1/scratch/students/witzlch88229/projects/omni-asr-distil || { echo "Directory not found"; exit 1; }

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE CONDA_SHLVL
PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '/conda' | paste -sd ':')"

source .venv/bin/activate

echo "=================================================================="
echo "Starting Stage 2 Streaming Distillation at $(date)"
echo "Job submitted to partition ${SLURM_JOB_PARTITION} on ${SLURM_CLUSTER_NAME}"
echo "Config: ${CONFIG_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo "=================================================================="

# --- Launch training (background, so trap can fire) ---
python scripts/run_stage2.py "$OUTPUT_DIR" \
    --config-file "${CONFIG_FILE}" &
TRAIN_PID=$!
wait "$TRAIN_PID"
