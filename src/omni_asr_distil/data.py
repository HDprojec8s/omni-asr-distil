"""Data loading utilities for distillation.

Reuses the MIXTURE_PARQUET dataset pipeline from omnilingual-asr.
This module registers the dataset family and provides a helper to create
data readers from the recipe config.
"""

from __future__ import annotations

from fairseq2.composition import register_dataset_family
from fairseq2.runtime.dependency import DependencyContainer

from omnilingual_asr.datasets.impl.mixture_parquet_asr_dataset import (
    MIXTURE_PARQUET_ASR_DATASET,
    MixtureParquetAsrDataset,
    MixtureParquetAsrDatasetConfig,
    open_mixture_parquet_asr_dataset,
)


def register_distill_datasets(container: DependencyContainer) -> None:
    """Register dataset families used by the distillation recipe."""
    register_dataset_family(
        container,
        MIXTURE_PARQUET_ASR_DATASET,
        MixtureParquetAsrDataset,
        MixtureParquetAsrDatasetConfig,
        opener=open_mixture_parquet_asr_dataset,
    )
