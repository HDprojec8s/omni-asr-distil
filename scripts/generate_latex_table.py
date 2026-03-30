#!/usr/bin/env python
"""Generate LaTeX tables for Stage 1 and Stage 2 distillation results.

Reads per-sample CSV files from the latest checkpoints and computes
corpus-level WER/CER for the thesis tables.

Usage:
    python scripts/generate_latex_table.py          # both tables
    python scripts/generate_latex_table.py --stage1  # stage 1 only
    python scripts/generate_latex_table.py --stage2  # stage 2 only
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import editdistance

STAGE1_OUTPUT = Path("/nfs1/scratch/students/witzlch88229/output/distil-stage1")
STAGE2_OUTPUT = Path("/nfs1/scratch/students/witzlch88229/output/distil-stage2")

STAGE1_MODELS = [
    ("S-Large", "s_large_512", r"$\sim$45M"),
    ("S-Medium", "s_medium_384", r"$\sim$28M"),
    ("S-Small", "s_small_256", r"$\sim$15M"),
]

# (display_name, stage2_config, stage1_config_for_baseline)
STAGE2_CONFIGS = [
    ("S-Large", [
        ("Non-streaming (stage 1)", None, "s_large_512"),
        ("DCT", "stream_dct_large", None),
    ]),
    ("S-Medium", [
        ("Non-streaming (stage 1)", None, "s_medium_384"),
        ("DCT", "stream_dct", None),
    ]),
]

DATASETS = {
    "ort": "rvg1_de",
    "tr2": "rvg1_de_tr2",
}


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Find the highest step_* checkpoint across all ws_* workspaces."""
    if not output_dir.exists():
        return None
    step_dirs = sorted(
        output_dir.glob("ws_*/checkpoints/step_*"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    return step_dirs[-1] if step_dirs else None


def compute_corpus_metrics(csv_path: Path) -> tuple[float, float] | None:
    """Compute corpus-level WER and CER from a per-sample CSV."""
    if not csv_path.exists():
        return None

    total_word_err = 0
    total_word_len = 0
    total_char_err = 0
    total_char_len = 0

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref = row["reference"]
            hyp = row["hypothesis"]

            ref_words = ref.split()
            total_word_err += editdistance.eval(hyp.split(), ref_words)
            total_word_len += len(ref_words)
            total_char_err += editdistance.eval(list(hyp), list(ref))
            total_char_len += len(ref)

    if total_word_len == 0:
        return None

    wer = total_word_err / total_word_len * 100
    cer = total_char_err / total_char_len * 100
    return wer, cer


def fmt(val: float | None, is_best: bool = False) -> str:
    if val is None:
        return "--"
    s = f"{val:.1f}"
    return rf"\textbf{{{s}}}" if is_best else s


def get_metrics(checkpoint: Path | None, dataset_name: str) -> tuple[float, float] | None:
    if checkpoint is None:
        return None
    csv_path = checkpoint / f"eval_{dataset_name}_test.csv"
    return compute_corpus_metrics(csv_path)


def generate_stage1():
    """Generate tab:distil_stage1 table."""
    results: dict[str, dict[str, tuple[float, float] | None]] = {}

    for _, config_name, _ in STAGE1_MODELS:
        checkpoint = find_latest_checkpoint(STAGE1_OUTPUT / config_name)
        results[config_name] = {}
        for key, dataset_name in DATASETS.items():
            results[config_name][key] = get_metrics(checkpoint, dataset_name)

    # Find best values per column
    best = {}
    for col in ["ort_wer", "ort_cer", "tr2_wer", "tr2_cer"]:
        best[col] = None
    for config_name in [m[1] for m in STAGE1_MODELS]:
        for key in DATASETS:
            metrics = results[config_name].get(key)
            if metrics is None:
                continue
            wer, cer = metrics
            for suffix, val in [("wer", wer), ("cer", cer)]:
                k = f"{key}_{suffix}"
                if best[k] is None or val < best[k]:
                    best[k] = val

    print(r"\begin{table}[t]")
    print(r"  \centering")
    print(
        r"  \caption{Distillation stage~1: WER and CER (\%) on BAS-RVG1 test sets. "
        r"Teacher: Omnilingual ASR 300M. Best per column in \textbf{bold}.}"
    )
    print(r"  \label{tab:distil_stage1}")
    print(r"  \begin{tabular}{lrcccc}")
    print(r"    \toprule")
    print(r"    Student & Params & \multicolumn{2}{c}{ORT} & \multicolumn{2}{c}{TR2} \\")
    print(r"    \cmidrule(lr){3-4} \cmidrule(lr){5-6}")
    print(r"     & & WER & CER & WER & CER \\")
    print(r"    \midrule")
    print(r"    Teacher (300M) & 300M & -- & -- & -- & -- \\")
    print(r"    \midrule")

    for display_name, config_name, params in STAGE1_MODELS:
        ort = results[config_name].get("ort")
        tr2 = results[config_name].get("tr2")
        vals = {
            "ort_wer": ort[0] if ort else None,
            "ort_cer": ort[1] if ort else None,
            "tr2_wer": tr2[0] if tr2 else None,
            "tr2_cer": tr2[1] if tr2 else None,
        }
        cols = [fmt(vals[k], vals[k] is not None and vals[k] == best[k]) for k in ["ort_wer", "ort_cer", "tr2_wer", "tr2_cer"]]
        print(f"    {display_name} & {params} & {' & '.join(cols)} \\\\")

    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")


def generate_stage2():
    """Generate tab:distil_stage2 table."""
    # Collect all WER/CER values for best-detection
    all_wer = []
    all_cer = []
    rows = []

    for student_name, configs in STAGE2_CONFIGS:
        for stream_name, stage2_config, stage1_config in configs:
            if stage1_config is not None:
                # Stage 1 baseline
                checkpoint = find_latest_checkpoint(STAGE1_OUTPUT / stage1_config)
            else:
                # Stage 2 streaming
                checkpoint = find_latest_checkpoint(STAGE2_OUTPUT / stage2_config)

            metrics = get_metrics(checkpoint, "rvg1_de")
            wer = metrics[0] if metrics else None
            cer = metrics[1] if metrics else None
            if wer is not None:
                all_wer.append(wer)
            if cer is not None:
                all_cer.append(cer)
            rows.append((student_name, stream_name, wer, cer))

    best_wer = min(all_wer) if all_wer else None
    best_cer = min(all_cer) if all_cer else None

    print(r"\begin{table}[t]")
    print(r"  \centering")
    print(
        r"  \caption{Distillation stage~2: streaming WER and CER (\%) on BAS-RVG1 ORT test set. "
        r"Best per column in \textbf{bold}.}"
    )
    print(r"  \label{tab:distil_stage2}")
    print(r"  \begin{tabular}{llcc}")
    print(r"    \toprule")
    print(r"    Student & Streaming config & WER & CER \\")
    print(r"    \midrule")

    prev_student = None
    for student_name, stream_name, wer, cer in rows:
        if prev_student is not None and student_name != prev_student:
            print(r"    \midrule")

        name_col = student_name if student_name != prev_student else ""
        wer_str = fmt(wer, wer is not None and wer == best_wer)
        cer_str = fmt(cer, cer is not None and cer == best_cer)
        print(f"    {name_col} & {stream_name} & {wer_str} & {cer_str} \\\\")
        prev_student = student_name

    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", action="store_true", help="Generate stage 1 table only")
    parser.add_argument("--stage2", action="store_true", help="Generate stage 2 table only")
    args = parser.parse_args()

    both = not args.stage1 and not args.stage2

    if both or args.stage1:
        generate_stage1()
        if both:
            print()
    if both or args.stage2:
        generate_stage2()


if __name__ == "__main__":
    main()
