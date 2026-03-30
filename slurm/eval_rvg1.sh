#!/bin/bash
#SBATCH --job-name=eval-rvg1
#SBATCH --output=logs/eval-rvg1_%j.out
#SBATCH --error=logs/eval-rvg1_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=p6,p4,p2,p1
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1
#SBATCH --qos=basic

# --- Arguments ---
ARCH=${1:?Usage: sbatch eval_rvg1.sh <arch> [split] [dataset_path] [output_dir]}
SPLIT=${2:-"test"}
DATASET=${3:-""}
OUTPUT_DIR_OVERRIDE=${4:-""}

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

EXTRA_ARGS=""
if [ -n "${DATASET}" ]; then
    EXTRA_ARGS="--dataset ${DATASET}"
fi
if [ -n "${OUTPUT_DIR_OVERRIDE}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --output-dir ${OUTPUT_DIR_OVERRIDE}"
fi

echo "=================================================================="
echo "ASR Evaluation at $(date)"
echo "Arch:      ${ARCH}"
echo "Split:     ${SPLIT}"
echo "Dataset:   ${DATASET:-rvg1_de (default)}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "=================================================================="

python scripts/eval_rvg1.py --arch "${ARCH}" --split "${SPLIT}" ${EXTRA_ARGS}
