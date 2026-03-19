# omni-asr-distil

Two-stage knowledge distillation of [omnilingual-asr](https://github.com/ChipCracker/omnilingual-asr) CTC models (300M / 1B) into small, streaming-capable student models for German ASR.

## Method: Two-Stage Distillation

### Distillation Objectives

Three loss components covering both representation-level and output-level knowledge transfer:

```
L_total = λ_ctc · L_ctc_gt + λ_kd · L_kd_logit + λ_hid · L_hid_cosine
```

| Loss | Description |
|---|---|
| **L_ctc_gt** | Standard CTC loss on ground-truth labels — anchor against teacher errors |
| **L_kd_logit** | KL-Divergence on temperature-scaled CTC logits (soft targets) |
| **L_hid_cosine** | Cosine similarity on hidden representations with linear projection |

**Why KL instead of LF-MMI:** KL-Divergence on softened logits captures the same "dark knowledge" (relative token probabilities) without requiring lattice generation, Kaldi dependencies, or denominator graph construction. Temperature scaling (T=2.0–3.0) exposes non-top-token probabilities.

**Why Cosine instead of MSE:** Cosine similarity is dimension-invariant and focuses on representation direction (what matters for downstream prediction). MSE is scale-sensitive when hidden dimensions differ between teacher and student.

### Stage 1: T → S_N (Size Reduction)

- **Teacher**: `omniASR_CTC_300M_v2` or `omniASR_CTC_1B_v2` (frozen, eval mode)
- **Student**: Smaller Conformer architecture
- **CNN Encoder**: Shared from teacher and frozen
- **Prediction Layer**: Freshly initialized (model_dim changes)
- **Layer Mapping**: `g(i) = 2i + 1` (24→12 layers) with final layer always mapped
- **Loss Weights**: `λ_ctc=0.3, λ_kd=0.3, λ_hid=0.4`
- **Data**: Labeled data (cv_swc_rvg1_de, ~1500h)

### Stage 2: S_N → S_S (Streaming Conversion)

- **Teacher**: S_N from Stage 1 (frozen) — not the large teacher
- **Student**: S_S initialized from S_N, with chunked/causal attention
- **Dynamic Chunk Training**: Random chunk sizes {16, 24, 32, 48, 64, ∞} per forward pass
- **Loss Weights**: `λ_ctc=0.3, λ_kd=0.5, λ_hid=0.2` (more logit KD, less hidden)
- **Benefit**: Only Stage 2 needs to be repeated for different latency targets

### Student Architecture: Conformer

Conformer blocks with depthwise convolution compensate for limited attention range in streaming mode. Rotary position encoding (RoPE) replaces the teacher's conv-based position encoder for streaming compatibility.

| Config | Enc dim | FFN dim | Heads | Layers | Conv Kernel | ~Params | Compression |
|---|---|---|---|---|---|---|---|
| **S-Large** | 512 | 2048 | 8 | 12 | 31 | ~45M | ~7x |
| **S-Medium** | 384 | 1536 | 8 | 12 | 31 | ~28M | ~11.5x |
| **S-Small** | 256 | 1024 | 8 | 10 | 31 | ~15M | ~21.5x |

### Training Hyperparameters

| Parameter | Stage 1 | Stage 2 |
|---|---|---|
| Learning Rate | 3e-4 | 1e-4 |
| Training Steps | 50,000 | 20,000 |
| λ_ctc / λ_kd / λ_hid | 0.3 / 0.3 / 0.4 | 0.3 / 0.5 / 0.2 |
| Temperature | 3.0 | 2.0 |
| Teacher | omniASR_CTC_300M_v2 | S_N (Stage 1 output) |
| Chunk Sizes | N/A (full context) | {16,24,32,48,64,∞} |
| Optimizer | AdamW (β=0.9,0.98, wd=0.01) | AdamW |
| LR Schedule | Tri-stage (0.1/0.4/0.5) | Tri-stage |
| Mixed Precision | bfloat16 | bfloat16 |

## Project Structure

```
omni-asr-distil/
├── README.md
├── .gitignore
├── pyproject.toml
├── configs/
│   ├── stage1/
│   │   ├── s_medium_384.yaml    # Primary target (~28M params)
│   │   ├── s_large_512.yaml     # (~45M params)
│   │   └── s_small_256.yaml     # (~15M params)
│   └── stage2/
│       ├── stream_dct.yaml      # Dynamic Chunk Training
│       ├── stream_480ms.yaml    # Fixed 480ms latency
│       └── stream_320ms.yaml    # Fixed 320ms latency
├── src/omni_asr_distil/
│   ├── __init__.py
│   ├── losses.py                # KL-div, cosine sim, combined objective
│   ├── student_config.py        # Student archs (fairseq2 ConfigRegistrar)
│   ├── student_factory.py       # Student creation + streaming modifications
│   ├── distill_criterion.py     # Teacher + student forward, combined loss
│   ├── distill_recipe.py        # DistillRecipe (extends TrainRecipe)
│   ├── distill_train_unit.py    # DistillTrainUnit (extends TrainUnit)
│   ├── hidden_extractor.py      # Hook-based hidden state extraction
│   ├── streaming.py             # DynamicChunkBias (AttentionBias)
│   ├── wer_utils.py             # WER/CER metrics + greedy CTC decode
│   └── data.py                  # Reuse MIXTURE_PARQUET datasets
├── scripts/
│   ├── run_stage1.py            # Stage 1 entry point
│   ├── run_stage2.py            # Stage 2 entry point
│   ├── evaluate.py              # WER evaluation (fairseq2 recipe)
│   └── eval_rvg1.py             # BAS-RVG1 eval with per-sample CSV
└── slurm/
    ├── stage1.sh                # Training (preemptible, auto-resume)
    ├── stage2.sh
    └── eval_rvg1.sh             # BAS-RVG1 evaluation
```

## Setup

```bash
cd omni-asr-distil
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Usage

### Stage 1: Size Reduction

```bash
# Fresh start (aborts if checkpoints exist)
sbatch slurm/stage1.sh configs/stage1/s_medium_384.yaml

# Resume from last checkpoint
sbatch slurm/stage1.sh configs/stage1/s_medium_384.yaml --resume
```

Batch sizing is GPU & model-aware — the script auto-detects the partition (p4/H200, p2/A100-80, p1/A100-40) and adjusts `max_num_elements` and gradient accumulation to keep effective batch size constant. Smaller models get larger batches on constrained GPUs.

Preemption is handled automatically: USR1 signal triggers a checkpoint save, `--requeue` restarts the job, and `SLURM_RESTART_COUNT` bypasses the checkpoint guard. wandb runs persist across restarts via `run_id: persistent`.

### Stage 2: Streaming Conversion

```bash
sbatch slurm/stage2.sh configs/stage2/stream_dct.yaml

# Resume
sbatch slurm/stage2.sh configs/stage2/stream_dct.yaml --resume
```

### Evaluation: BAS-RVG1

```bash
# Auto-finds latest checkpoint for the architecture
sbatch slurm/eval_rvg1.sh distill_s_small test

# Output: <checkpoint_dir>/eval_rvg1_test.csv
# Columns: reference, hypothesis, wer, cer
```

## Key Design Decisions

### fairseq2 Integration

The implementation extends fairseq2's recipe system:
- `DistillRecipe(TrainRecipe)` — manages teacher + student lifecycle
- `DistillTrainUnit(TrainUnit)` — joint forward pass per batch
- `DistillCriterion` — computes combined KL + cosine + CTC loss
- `HiddenStateExtractor` — uses `TransformerEncoder.register_layer_hook()`
- `DynamicChunkBias(AttentionBias)` — integrates with SDPA attention

### Streaming Architecture

- `DynamicChunkBias`: Randomly samples chunk sizes during training; fixed at eval
- `ConformerConvolution(causal_depthwise_conv=True)`: Left-only padding for causal conv
- `RotaryEncoder` (RoPE): Position encoding compatible with chunked attention
- Both modifications applied in-place to an existing model via `apply_streaming_bias()` and `enable_causal_conv()`

## References

- **DistillW2V2**: Fu et al., *"DistillW2V2: A Small and Streaming Wav2Vec 2.0 Based ASR Model"*, arXiv:2303.09278, 2023
- **Knowledge Distillation**: Hinton et al., *"Distilling the Knowledge in a Neural Network"*, NeurIPS Workshop 2015
- **Conformer**: Gulati et al., *"Conformer: Convolution-augmented Transformer for Speech Recognition"*, Interspeech 2020
- **Dynamic Chunk Training**: Unified Streaming and Non-streaming Two-Pass End-to-End Model (WeNet)
