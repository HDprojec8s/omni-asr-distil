"""Distillation TrainUnit: wraps DistillCriterion for the fairseq2 trainer.

Follows the pattern of ``Wav2Vec2AsrTrainUnit`` but keeps the CNN encoder
always frozen and manages the DynamicChunkBias training/eval mode.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import final

from fairseq2.datasets import Seq2SeqBatch
from fairseq2.logging import log
from fairseq2.metrics import MetricBag
from fairseq2.recipe.model import RecipeModel
from fairseq2.recipe.trainer import TrainUnit
from torch import Tensor
from typing_extensions import override

from .distill_criterion import DistillCriterion
from .streaming import DynamicChunkBias


@final
class DistillTrainUnit(TrainUnit[Seq2SeqBatch]):
    """Training unit for knowledge distillation.

    The CNN encoder frontend is always frozen (shared from teacher).
    For streaming (Stage 2), manages DynamicChunkBias chunk sampling.
    """

    _criterion: DistillCriterion
    _model: RecipeModel
    _dynamic_bias: DynamicChunkBias | None

    def __init__(
        self,
        criterion: DistillCriterion,
        model: RecipeModel,
        *,
        dynamic_bias: DynamicChunkBias | None = None,
    ) -> None:
        self._criterion = criterion
        self._model = model
        self._dynamic_bias = dynamic_bias

    @override
    def set_step_nr(self, step_nr: int) -> None:
        """Called before each training step. Samples chunk size for DCT."""
        if self._dynamic_bias is not None:
            chunk_size = self._dynamic_bias.sample_chunk_size()
            if step_nr == 1:
                log.info(
                    f"Dynamic Chunk Training enabled with chunk sizes "
                    f"{self._dynamic_bias._chunk_sizes}"
                )

    @override
    def prepare_metric_bag(self, metric_bag: MetricBag) -> None:
        self._criterion.prepare_metric_bag(metric_bag)

    @override
    def process_batch(
        self, batch: Seq2SeqBatch, metric_bag: MetricBag
    ) -> tuple[Tensor, int]:
        return self._criterion(batch, metric_bag)

    @override
    def process_metric_values(self, values: MutableMapping[str, object]) -> None:
        self._criterion.process_metric_values(values)

    @property
    @override
    def model(self) -> RecipeModel:
        return self._model
