# omni-asr-distil

Two-stage knowledge distillation of [omnilingual-asr](https://github.com/ChipCracker/omnilingual-asr) CTC models (300M / 1B) into small, streaming-capable student models for German ASR.

Based on the **DistillW2V2** method ([Fu et al., 2023](https://arxiv.org/abs/2303.09278)).

## Motivation

The omni-asr CTC models (300M–7B parameters) achieve strong multilingual ASR performance but require full-sequence attention, making them unsuitable for streaming (real-time) inference. On-device or low-latency applications need models that are both **smaller** and **streaming-capable**.

## Method: Two-Stage Distillation

Following DistillW2V2, the distillation is split into two stages to decouple model compression from the streaming conversion:

### Stage 1: T → S_N (Size Reduction)

Distill the large non-streaming teacher into a **small non-streaming** student.

```
Teacher (300M / 1B)          Student S_N
┌──────────────────┐         ┌──────────────────┐
│  Prediction Layer │ ·····> │  Prediction Layer │  (shared weights)
│  24 Encoder Layers│ ─L_h─> │  12 Encoder Layers│  (layer mapping g(i)=2i)
│  CNN Encoder      │ ·····> │  CNN Encoder      │  (shared weights)
└──────────────────┘         └──────────────────┘
```

- **Shared parameters**: CNN encoder and prediction layer are initialized from the teacher
- **Compressed transformer**: fewer blocks, smaller hidden dimension
- **Distillation objective**: weighted combination of hidden-layer MSE and prediction-layer (CTC/LF-MMI + MSE) losses

### Stage 2: S_N → S_S (Streaming Conversion)

Convert the small non-streaming student into a **small streaming** student.

```
Student S_N (non-streaming)     Student S_S (streaming)
┌──────────────────────┐        ┌──────────────────────┐
│  Full-context Attn   │ ────>  │  Chunk-Transformer   │
│  (sees entire seq)   │        │  (chunk + history)   │
└──────────────────────┘        └──────────────────────┘
```

- **Architecture**: Chunk-based Transformer (Transformer-XL style) with limited look-ahead
- **Initialization**: S_S is initialized from S_N weights
- **Latency control**: configurable chunk size (e.g., 480ms with chunk=48, history=600 frames)
- **Benefit**: only Stage 2 needs to be re-run for different latency targets

### Distillation Objective

```
L_distill = (1 - α) · L_hidn  +  α · L_pred

L_hidn = Σ_i MSE(H_i^S · W_i, H_{g(i)}^T)       # hidden layer alignment

L_pred = β · MMI_distill + (1 - β) · MSE(O^S, O^T) # prediction layer
```

| Hyperparameter | Description | Recommended |
|---|---|---|
| `α` | Weight of prediction loss vs hidden loss | 0.8 |
| `β` | Weight of LF-MMI vs MSE in prediction loss | 0.8 |
| `g(i) = 2i` | Layer mapping (student layer i ← teacher layer 2i) | — |

## Teacher Models

| Model | Params | Encoder dim | FFN dim | Blocks | Tokenizer |
|---|---|---|---|---|---|
| `omniASR_CTC_300M_v2` | 325M | — | — | — | v2 (10288 tokens) |
| `omniASR_CTC_1B_v2` | 975M | — | — | — | v2 (10288 tokens) |

## Planned Student Configurations

Based on the DistillW2V2 paper's ablation (Table 1), adapted for omni-asr:

| Student | Encoder dim | FFN dim | Blocks | Target params | Compression |
|---|---|---|---|---|---|
| S1 | 768 | 3072 | 12 | ~95M | ~3.4x (from 300M teacher) |
| S2 | 512 | 2048 | 12 | ~44M | ~7.4x |
| S3 | 384 | 1536 | 12 | ~27M | ~12x |
| S4 | 384 | 1536 | 10 | ~23M | ~14x |
| S5 | 384 | 1536 | 10 | ~22M | ~14.8x (reduced CNN) |

## Datasets

Reusing the data pipeline from [omni-asr-ft](https://github.com/ChipCracker/omni-asr-ft):

| Corpus | Hours | Usage |
|---|---|---|
| **Common Voice 24 DE** | ~1500h | Labeled fine-tuning + distillation |
| **SWC German** | variable | Labeled fine-tuning + distillation |
| **BAS RVG1** | ~10h | Evaluation (regional variants) |

Unlabeled data (for pre-training/distillation stages) TBD.

## Project Structure

```
omni-asr-distil/
├── README.md
├── .gitignore
├── configs/
│   ├── stage1/              # Stage 1: size reduction configs
│   │   ├── s1_768.yaml
│   │   ├── s2_512.yaml
│   │   ├── s3_384.yaml
│   │   └── ...
│   └── stage2/              # Stage 2: streaming conversion configs
│       ├── chunk_480ms.yaml
│       └── ...
├── src/
│   └── omni_asr_distil/
│       ├── __init__.py
│       ├── distill.py       # Distillation training loop
│       ├── losses.py        # L_hidn (MSE), L_pred (CTC + MSE), combined objective
│       ├── student.py       # Student model definitions (S1–S5)
│       ├── streaming.py     # Chunk-Transformer for Stage 2
│       └── data.py          # Data loading (reuses omni-asr-ft parquet format)
├── scripts/
│   ├── run_stage1.py        # Launch Stage 1 distillation
│   ├── run_stage2.py        # Launch Stage 2 streaming distillation
│   └── evaluate.py          # WER evaluation
├── slurm/
│   ├── stage1.sh            # SLURM job for Stage 1
│   └── stage2.sh            # SLURM job for Stage 2
└── pyproject.toml
```

## Setup

```bash
git clone git@github.com:ChipCracker/omni-asr-distil.git
cd omni-asr-distil

python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Usage

### Stage 1: Size Reduction

```bash
python scripts/run_stage1.py \
  --teacher omniASR_CTC_300M_v2 \
  --student-config configs/stage1/s3_384.yaml \
  --data-dir /path/to/omni-asr-ft/data \
  --output-dir checkpoints/stage1_s3 \
  --alpha 0.8 --beta 0.8 \
  --epochs 30 --lr 5e-4
```

### Stage 2: Streaming Conversion

```bash
python scripts/run_stage2.py \
  --init-from checkpoints/stage1_s3/best.pt \
  --streaming-config configs/stage2/chunk_480ms.yaml \
  --data-dir /path/to/omni-asr-ft/data \
  --output-dir checkpoints/stage2_s3_480ms \
  --alpha 0.8 --beta 0.8 \
  --epochs 15 --lr 1e-4
```

### Evaluation

```bash
python scripts/evaluate.py \
  --model checkpoints/stage2_s3_480ms/best.pt \
  --test-set /path/to/rvg1/test.parquet \
  --streaming --chunk-size 48 --history-size 600
```

## References

- **DistillW2V2**: Fu et al., *"DistillW2V2: A Small and Streaming Wav2Vec 2.0 Based ASR Model"*, arXiv:2303.09278, 2023
- **Wav2Vec 2.0**: Baevski et al., *"wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"*, NeurIPS 2020
- **Chunk-Transformer**: Chen et al., *"Developing Real-Time Streaming Transformer Transducer for Speech Recognition on Large-Scale Dataset"*, ICASSP 2021
- **LF-MMI**: Povey et al., *"Purely Sequence-Trained Neural Networks for ASR Based on Lattice-Free MMI"*, Interspeech 2016

## License

TBD
