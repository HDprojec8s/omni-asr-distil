#!/usr/bin/env python
"""Evaluate a distilled Stage 1 model on BAS-RVG1.

Automatically finds the latest checkpoint for the given architecture,
runs greedy CTC decoding on the specified split, and writes a CSV with
per-sample reference, hypothesis, WER, and CER.

Usage:
    python scripts/eval_rvg1.py --arch distill_s_small --split test
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import editdistance
import torch

from fairseq2.data.tokenizers import load_tokenizer
from fairseq2.models.wav2vec2.asr.config import Wav2Vec2AsrConfig
from fairseq2.models.wav2vec2.asr.factory import Wav2Vec2AsrFactory
from fairseq2.models.wav2vec2.asr.model import Wav2Vec2AsrModel
from fairseq2.nn import BatchLayout

from omni_asr_distil.student_config import _base_student
from omni_asr_distil.wer_utils import greedy_ctc_decode

OUTPUT_BASE = Path("/nfs1/scratch/students/witzlch88229/output/distil-stage1")

DATASET_PATH = Path(
    "/nfs1/scratch/students/witzlch88229/data/omni-asr-ft/rvg1_de/version=0"
)

TOKENIZER_NAME = "omniASR_tokenizer_written_v2"

ARCH_MAP: dict[str, tuple[str, Wav2Vec2AsrConfig]] = {
    "distill_s_small": ("s_small_256", _base_student(256, 1024, 8, 10)),
    "distill_s_medium": ("s_medium_384", _base_student(384, 1536, 8, 12)),
    "distill_s_large": ("s_large_512", _base_student(512, 2048, 8, 12)),
}


def find_latest_checkpoint(arch: str) -> Path:
    """Find the highest step_* checkpoint across all ws_* workspaces."""
    config_name = ARCH_MAP[arch][0]
    output_dir = OUTPUT_BASE / config_name

    # fairseq2 stores checkpoints in ws_*/checkpoints/step_*/
    step_dirs = sorted(
        output_dir.glob("ws_*/checkpoints/step_*"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not step_dirs:
        raise FileNotFoundError(
            f"No step checkpoints found in {output_dir}/ws_*/checkpoints/. "
            f"Training may not have reached a checkpoint step yet."
        )
    return step_dirs[-1]


def load_model(
    arch: str, checkpoint_dir: Path, device: torch.device
) -> Wav2Vec2AsrModel:
    """Create model from architecture config and load checkpoint weights."""
    config = ARCH_MAP[arch][1]

    factory = Wav2Vec2AsrFactory(config)
    model = factory.create_model()

    # fairseq2 distributed checkpoint format: model/pp_00/tp_00/sdp_00.pt
    model_path = checkpoint_dir / "model" / "pp_00" / "tp_00" / "sdp_00.pt"
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model


def sample_wer(ref: str, hyp: str) -> float:
    ref_words = ref.split()
    hyp_words = hyp.split()
    if not ref_words:
        return 0.0 if not hyp_words else 100.0
    return editdistance.eval(hyp_words, ref_words) / len(ref_words) * 100.0


def sample_cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 100.0
    return editdistance.eval(list(hyp), list(ref)) / len(ref) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate distilled model on BAS-RVG1")
    parser.add_argument("--arch", required=True, choices=ARCH_MAP.keys())
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # --- Find and load checkpoint ---
    checkpoint_dir = find_latest_checkpoint(args.arch)
    print(f"Checkpoint: {checkpoint_dir}")

    model = load_model(args.arch, checkpoint_dir, device)
    print(f"Model loaded: {args.arch}")

    # --- Load tokenizer ---
    tokenizer = load_tokenizer(TOKENIZER_NAME)
    text_decoder = tokenizer.create_decoder(skip_special_tokens=True)
    pad_idx = tokenizer.vocab_info.pad_idx or 1

    # --- Load dataset ---
    from omnilingual_asr.datasets.impl.mixture_parquet_asr_dataset import (
        MixtureParquetAsrDataset,
    )
    from omnilingual_asr.datasets.storage.mixture_parquet_storage import (
        MixtureParquetStorageConfig,
    )
    from omnilingual_asr.datasets.tasks.asr_task import AsrTaskConfig
    from fairseq2.gang import create_fake_gangs

    gangs = create_fake_gangs(device)

    storage_config = MixtureParquetStorageConfig(
        dataset_summary_path=str(
            DATASET_PATH.parent / "language_distribution_0.tsv"
        ),
    )
    task_config = AsrTaskConfig(
        min_audio_len=32_000,
        max_audio_len=960_000,
        max_num_elements=15_360_000,
        normalize_audio=True,
        batch_shuffle_window=1,
        example_shuffle_window=1,  # 1 = no shuffling (eval)
    )

    dataset = MixtureParquetAsrDataset(path=DATASET_PATH)

    data_reader = dataset.create_reader(
        split=args.split,
        tokenizer=tokenizer,
        gangs=gangs,
        dtype=torch.bfloat16,
        num_accumulate=1,
        storage_config=storage_config,
        task_config=task_config,
    )

    # --- Evaluate ---
    output_csv = checkpoint_dir / f"eval_rvg1_{args.split}.csv"

    total_word_err = 0
    total_word_len = 0
    total_char_err = 0
    total_char_len = 0
    num_samples = 0

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["reference", "hypothesis", "wer", "cer"])

        for batches in data_reader:
            for batch in batches:
                source_seqs, source_layout = batch.as_source_input()
                source_seqs = source_seqs.to(device)

                with torch.no_grad():
                    logits, logits_layout = model(source_seqs, source_layout)

                hyp_seqs, hyp_layout = greedy_ctc_decode(
                    logits, logits_layout, pad_idx=pad_idx
                )

                target_seqs, _ = batch.as_target_input()

                for i in range(batch.batch_size):
                    ref = text_decoder(target_seqs[i])
                    hyp = text_decoder(hyp_seqs[i])

                    wer = sample_wer(ref, hyp)
                    cer = sample_cer(ref, hyp)

                    writer.writerow([ref, hyp, f"{wer:.2f}", f"{cer:.2f}"])

                    ref_words = ref.split()
                    total_word_err += editdistance.eval(hyp.split(), ref_words)
                    total_word_len += len(ref_words)
                    total_char_err += editdistance.eval(list(hyp), list(ref))
                    total_char_len += len(ref)
                    num_samples += 1

    overall_wer = total_word_err / total_word_len * 100 if total_word_len else 0
    overall_cer = total_char_err / total_char_len * 100 if total_char_len else 0

    print(f"\n{'='*60}")
    print(f"Results: {output_csv}")
    print(f"Samples: {num_samples}")
    print(f"WER:     {overall_wer:.2f}%")
    print(f"CER:     {overall_cer:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
