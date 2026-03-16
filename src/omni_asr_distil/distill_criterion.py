"""Distillation criterion: orchestrates teacher + student forward passes.

Follows the pattern of ``Wav2Vec2AsrCriterion`` but computes the combined
distillation loss (CTC ground-truth + KL logit + cosine hidden).
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import final

import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn.functional import ctc_loss, log_softmax

from fairseq2.datasets import Seq2SeqBatch
from fairseq2.metrics import Mean, MetricBag
from fairseq2.models.wav2vec2.asr.model import Wav2Vec2AsrModel

from .hidden_extractor import HiddenStateExtractor
from .losses import DistillationLoss


@final
class DistillCriterion:
    """Distillation training criterion.

    Performs:
    1. Teacher forward pass (no grad) → logits + hidden states
    2. Student forward pass → logits + hidden states + CTC loss
    3. Combined distillation loss computation
    """

    def __init__(
        self,
        teacher: Wav2Vec2AsrModel,
        student: Wav2Vec2AsrModel,
        distill_loss: DistillationLoss,
        teacher_extractor: HiddenStateExtractor,
        student_extractor: HiddenStateExtractor,
        layer_mapping: dict[int, int],
        projection_layers: dict[int, Linear],
    ) -> None:
        self._teacher = teacher
        self._student = student
        self._distill_loss = distill_loss
        self._teacher_extractor = teacher_extractor
        self._student_extractor = student_extractor
        self._layer_mapping = layer_mapping
        self._projection_layers = projection_layers

    def prepare_metric_bag(self, metric_bag: MetricBag) -> None:
        """Register custom distillation metrics."""
        metric_bag.add("ctc_loss", Mean())
        metric_bag.add("kd_logit_loss", Mean())
        metric_bag.add("hid_cosine_loss", Mean())
        metric_bag.add("distill_loss", Mean())

    def __call__(
        self,
        batch: Seq2SeqBatch,
        metric_bag: MetricBag,
    ) -> tuple[Tensor, int]:
        """Compute distillation loss for a batch.

        Returns:
            Tuple of (total_loss, batch_size).
        """
        source_seqs, source_seqs_layout = batch.as_source_input()
        target_seqs, target_seqs_layout = batch.as_target_input()

        # --- Teacher forward (no gradient) ---
        with torch.no_grad():
            teacher_logits, teacher_logits_layout = self._teacher(
                source_seqs, source_seqs_layout
            )
            teacher_hiddens = self._teacher_extractor.get_hiddens()

        # --- Student forward ---
        # Get logits (no CTC loss yet — we compute it manually for more control)
        student_logits, student_logits_layout = self._student(
            source_seqs, source_seqs_layout
        )
        student_hiddens = self._student_extractor.get_hiddens()

        # --- CTC loss on ground-truth labels ---
        log_probs = log_softmax(student_logits, dim=-1, dtype=torch.float32)
        log_probs_t = log_probs.transpose(0, 1)  # (S, N, V)

        ctc_gt_loss = ctc_loss(
            log_probs=log_probs_t,
            input_lengths=student_logits_layout.seq_lens_pt,
            targets=target_seqs,
            target_lengths=target_seqs_layout.seq_lens_pt,
            reduction="sum",
            zero_infinity=True,
        )
        # Normalize by batch size
        ctc_gt_loss = ctc_gt_loss / batch.batch_size

        # --- Combined distillation loss ---
        # Use student's sequence lengths for masking (teacher may have different output lengths
        # due to different encoder, but with shared frontend they should be the same)
        seq_lens = student_logits_layout.seq_lens_pt

        total_loss, loss_dict = self._distill_loss(
            ctc_loss=ctc_gt_loss,
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_hiddens=student_hiddens,
            teacher_hiddens=teacher_hiddens,
            layer_mapping=self._layer_mapping,
            projection_layers=self._projection_layers,
            seq_lens=seq_lens,
        )

        # --- Update metrics ---
        n = batch.batch_size
        metric_bag.get("ctc_loss", Mean).update(loss_dict["ctc_gt"], weight=n)
        metric_bag.get("kd_logit_loss", Mean).update(loss_dict["kd_logit"], weight=n)
        metric_bag.get("hid_cosine_loss", Mean).update(loss_dict["hid_cosine"], weight=n)
        metric_bag.get("distill_loss", Mean).update(loss_dict["total"], weight=n)

        return total_loss, batch.batch_size

    def process_metric_values(self, values: MutableMapping[str, object]) -> None:
        """Post-process metric values for logging (no-op for now)."""
        pass
