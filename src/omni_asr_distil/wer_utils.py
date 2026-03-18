"""Minimal WER/CER computation for distillation validation.

Performs greedy CTC decoding (argmax + unique_consecutive) and computes
WER/CER via editdistance. No LLM beam search support needed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import final

import editdistance
import torch
from torch import Tensor
from torcheval.metrics import Metric
from typing_extensions import Self, override

from fairseq2.data.tokenizers import TokenDecoder, Tokenizer
from fairseq2.datasets import Seq2SeqBatch
from fairseq2.metrics import MetricBag
from fairseq2.metrics.text import WerMetric
from fairseq2.nn import BatchLayout
from fairseq2.nn.utils.padding import pad_seqs
from fairseq2.utils.tensor import to_tensor


@final
class CerMetric(Metric[Tensor]):
    """Character Error Rate metric."""

    def __init__(self, *, device: torch.device | None = None) -> None:
        super().__init__(device=device)
        self.char_err: Tensor
        self.char_len: Tensor
        self._add_state("char_err", torch.zeros((), device=device, dtype=torch.int64))
        self._add_state("char_len", torch.zeros((), device=device, dtype=torch.int64))

    @override
    @torch.inference_mode()
    def update(self, refs: Sequence[str], hyps: Sequence[str]) -> Self:
        for ref, hyp in zip(refs, hyps):
            self.char_err += editdistance.eval(list(hyp), list(ref))
            self.char_len += len(ref)
        return self

    @override
    @torch.inference_mode()
    def compute(self) -> Tensor:
        if self.char_len:
            return self.char_err * 100.0 / self.char_len
        return to_tensor(-1.0, dtype=torch.float32)

    @override
    @torch.inference_mode()
    def merge_state(self, metrics: Iterable[CerMetric]) -> Self:
        for metric in metrics:
            self.char_err += metric.char_err.to(self.device)
            self.char_len += metric.char_len.to(self.device)
        return self


def greedy_ctc_decode(
    logits: Tensor,
    logit_layout: BatchLayout,
    blank_label: int = 0,
    pad_idx: int = 1,
) -> tuple[Tensor, BatchLayout]:
    """Greedy CTC decoding: argmax + unique_consecutive + remove blanks."""
    hyp_seqs = []
    for sample_logits, seq_len in zip(logits, logit_layout.seq_lens):
        hyp = sample_logits[:seq_len].argmax(-1).unique_consecutive()
        hyp = hyp[hyp != blank_label]
        if hyp.numel() == 0:
            hyp = torch.tensor([pad_idx], device=hyp.device)
        hyp_seqs.append(hyp)
    return pad_seqs(hyp_seqs, pad_value=pad_idx)


def compute_wer_cer(
    batch: Seq2SeqBatch,
    logits: Tensor,
    logit_layout: BatchLayout,
    text_decoder: TokenDecoder,
    metric_bag: MetricBag,
    pad_idx: int = 1,
) -> None:
    """Decode student logits via greedy CTC and update WER/CER metrics."""
    ref_seqs, _ = batch.as_target_input()
    hyp_seqs, hyp_layout = greedy_ctc_decode(logits, logit_layout, pad_idx=pad_idx)

    refs = [text_decoder(s) for s in ref_seqs]
    hyps = [text_decoder(s) for s in hyp_seqs]

    _, ref_layout = batch.as_target_input()
    metric_bag.get("wer", WerMetric).update(
        refs,
        ref_seqs.cpu(),
        ref_layout,
        hyps,
        hyp_seqs.cpu(),
        hyp_layout,
    )
    metric_bag.get("cer", CerMetric).update(refs, hyps)
