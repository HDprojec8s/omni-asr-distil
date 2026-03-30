#!/bin/bash
# Submit all Stage 2 evaluation jobs (streaming configs × 2 datasets).
#
# Usage: bash scripts/eval_stage2_all.sh

DATA_BASE="/nfs1/scratch/students/witzlch88229/data/omni-asr-ft"
ORT_DATASET="${DATA_BASE}/rvg1_de/version=0"
TR2_DATASET="${DATA_BASE}/rvg1_de_tr2/version=0"
STAGE2_OUTPUT="/nfs1/scratch/students/witzlch88229/output/distil-stage2"

# Stage 2 configs map to architectures:
#   stream_dct       → distill_s_medium
#   stream_dct_large → distill_s_large
CONFIGS=(
    "stream_dct_large:distill_s_large"
    "stream_dct:distill_s_medium"
)

COUNT=0
for entry in "${CONFIGS[@]}"; do
    config="${entry%%:*}"
    arch="${entry##*:}"
    output_dir="${STAGE2_OUTPUT}/${config}"

    # Check if checkpoint exists
    if ! ls "${output_dir}"/ws_*/checkpoints/step_* &>/dev/null; then
        echo "SKIP ${config} (${arch}): no checkpoints found"
        continue
    fi

    echo "Submitting ${config} (${arch}) × ORT..."
    sbatch slurm/eval_rvg1.sh "${arch}" test "${ORT_DATASET}" "${output_dir}"

    echo "Submitting ${config} (${arch}) × TR2..."
    sbatch slurm/eval_rvg1.sh "${arch}" test "${TR2_DATASET}" "${output_dir}"

    COUNT=$((COUNT + 2))
done

echo "Submitted ${COUNT} eval jobs."
