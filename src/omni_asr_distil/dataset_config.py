"""Dataset configuration and selector inlined from omnilingual-asr workflows.

The ``omnilingual_asr.workflows`` package is not part of the installed
``omnilingual_asr`` distribution, so we inline the relevant enums, dataclass,
and selector here.  All underlying dataset/storage/task classes come from the
installed ``omnilingual_asr.datasets`` package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import NoReturn, Tuple, Union

from fairseq2.recipe.base import RecipeContext
from fairseq2.recipe.config import DatasetSection

from omnilingual_asr.datasets.impl.manifest_asr_dataset import ManifestAsrDataset
from omnilingual_asr.datasets.impl.mixture_parquet_asr_dataset import (
    MixtureParquetAsrDataset,
)
from omnilingual_asr.datasets.storage.manifest_storage import ManifestStorageConfig
from omnilingual_asr.datasets.storage.mixture_parquet_storage import (
    MixtureParquetStorageConfig,
)
from omnilingual_asr.datasets.tasks.asr_task import AsrTaskConfig


class StorageMode(Enum):
    """Storage backends for Wav2Vec2AsrRecipe"""

    MANIFEST = "MANIFEST"
    MIXTURE_PARQUET = "MIXTURE_PARQUET"


class TaskMode(Enum):
    """Task backends for Wav2Vec2AsrRecipe"""

    ASR = "ASR"


@dataclass(kw_only=True)
class Wav2Vec2AsrDatasetSection(DatasetSection):
    """Recipe-specific dataset section that supports mixing storage + task interfaces."""

    train_split: str | None = "train"
    """The name of the training data split. Only ``None`` during evaluation."""

    valid_split: str | None = "dev"
    """The name of the validation data split(s).
    Format multiple splits interspersed by ``,`` and without spaces
    (``'valid,dev_clean,test_clean'``).
    """

    storage_mode: StorageMode = StorageMode.MANIFEST
    """Storage format for the dataset (e.g., MANIFEST, PARQUET)."""

    task_mode: TaskMode = TaskMode.ASR
    """Task type for training (e.g., ASR, SSL)."""

    manifest_storage_config: ManifestStorageConfig = field(
        default_factory=ManifestStorageConfig
    )
    """Configuration for manifest-based dataset storage. Used when storage_mode is MANIFEST."""

    mixture_parquet_storage_config: MixtureParquetStorageConfig = field(
        default_factory=MixtureParquetStorageConfig
    )
    """Configuration for parquet-based dataset storage. Used when storage_mode is MIXTURE_PARQUET."""

    asr_task_config: AsrTaskConfig = field(default_factory=AsrTaskConfig)
    """Configuration for ASR task parameters. Used when task_mode is ASR."""


class Wav2Vec2AsrDatasetSelector:
    """Type-safe dataset selection based on storage/task modes."""

    @classmethod
    def get_dataset_and_configs(
        cls, config, context: RecipeContext
    ) -> Union[
        Tuple[ManifestAsrDataset, ManifestStorageConfig, AsrTaskConfig],
        Tuple[MixtureParquetAsrDataset, MixtureParquetStorageConfig, AsrTaskConfig],
    ]:
        combination = (config.dataset.storage_mode, config.dataset.task_mode)

        if combination == (StorageMode.MANIFEST, TaskMode.ASR):
            return cls._get_manifest_asr(config, context)
        elif combination == (StorageMode.MIXTURE_PARQUET, TaskMode.ASR):
            return cls._get_mixture_parquet_asr(config, context)
        else:
            cls._raise_unsupported_combination(combination)

    @classmethod
    def _get_manifest_asr(
        cls, config, context: RecipeContext
    ) -> Tuple[ManifestAsrDataset, ManifestStorageConfig, AsrTaskConfig]:
        dataset = context.default_dataset.as_(ManifestAsrDataset)
        return (
            dataset,
            config.dataset.manifest_storage_config,
            config.dataset.asr_task_config,
        )

    @classmethod
    def _get_mixture_parquet_asr(
        cls, config, context: RecipeContext
    ) -> Tuple[MixtureParquetAsrDataset, MixtureParquetStorageConfig, AsrTaskConfig]:
        dataset = context.default_dataset.as_(MixtureParquetAsrDataset)
        return (
            dataset,
            config.dataset.mixture_parquet_storage_config,
            config.dataset.asr_task_config,
        )

    @classmethod
    def _raise_unsupported_combination(cls, combination) -> NoReturn:
        supported = [
            (StorageMode.MANIFEST, TaskMode.ASR),
            (StorageMode.MIXTURE_PARQUET, TaskMode.ASR),
        ]
        raise ValueError(
            f"Unsupported combination {combination}. Supported: {supported}"
        )
