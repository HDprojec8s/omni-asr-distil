"""Distillation recipe: two-stage knowledge distillation for streaming ASR.

Follows the pattern of ``Wav2Vec2AsrRecipe`` but orchestrates a teacher
and student model with combined distillation objectives.

Stage 1: Teacher (frozen) → Student (size reduction, full context)
Stage 2: Student_N (frozen) → Student_S (streaming conversion, DCT)
"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import cast, final

import torch
from torch.nn import Linear, ModuleDict

from fairseq2.datasets import Seq2SeqBatch, SyncMode
from fairseq2.logging import log
from fairseq2.metrics import MetricBag, format_as_float
from fairseq2.metrics.recorders import MetricDescriptor
from fairseq2.models.wav2vec2.asr.model import Wav2Vec2AsrModel
from fairseq2.nn.utils.module import freeze_parameters, share_parameters
from fairseq2.recipe.base import RecipeContext, TrainRecipe
from fairseq2.recipe.config import (
    ADAMW_OPTIMIZER,
    TRI_STAGE_LR,
    AdamWConfig,
    CommonSection,
    CompileOptions,
    GangSection,
    GradAccumulationConfig,
    LRSchedulerSection,
    MixedPrecisionConfig,
    ModelSection,
    OptimizerSection,
    ReferenceModelSection,
    RegimeSection,
    TokenizerSection,
    TrainerSection,
    TriStageLRConfig,
)
from fairseq2.recipe.evaluator import EvalUnit
from fairseq2.recipe.model import RecipeModel
from fairseq2.recipe.trainer import Trainer
from typing_extensions import override

from .dataset_config import Wav2Vec2AsrDatasetSelector, Wav2Vec2AsrDatasetSection

from .data import register_distill_datasets
from .distill_criterion import DistillCriterion
from .distill_train_unit import DistillTrainUnit
from .hidden_extractor import HiddenStateExtractor
from .losses import DistillationLoss
from .streaming import DynamicChunkBias
from .student_config import register_student_configs
from .student_factory import (
    apply_streaming_bias,
    compute_layer_mapping,
    enable_causal_conv,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class DistillTrainerSection(TrainerSection):
    """Trainer section — no encoder freezing (CNN is always frozen)."""

    pass


@dataclass(kw_only=True)
class DistillStreamingSection:
    """Streaming configuration for Stage 2 (DCT)."""

    chunk_sizes: list[int] = field(default_factory=lambda: [16, 24, 32, 48, 64])
    eval_chunk_size: int = 48
    full_context_prob: float = 0.1
    causal_conv: bool = True


@dataclass(kw_only=True)
class DistillRecipeConfig:
    """Two-stage distillation recipe configuration."""

    # Student model (created from arch, e.g., "distill_s_medium")
    model: ModelSection = field(
        default_factory=lambda: ModelSection(
            family="wav2vec2_asr",
            compile=False,
            compile_options=CompileOptions(fullgraph=False, dynamic=False),
        )
    )

    # Teacher model (loaded from checkpoint, e.g., "omniASR_CTC_300M_v2")
    teacher: ReferenceModelSection = field(
        default_factory=lambda: ReferenceModelSection(
            name="",
            family="wav2vec2_asr",
            compile=False,
            compile_options=CompileOptions(fullgraph=False, dynamic=False),
        )
    )

    # Dataset
    dataset: Wav2Vec2AsrDatasetSection = field(
        default_factory=lambda: Wav2Vec2AsrDatasetSection()
    )

    tokenizer: TokenizerSection = field(
        default_factory=lambda: TokenizerSection(name="omniASR_tokenizer_written_v2")
    )

    gang: GangSection = field(default_factory=lambda: GangSection())

    trainer: DistillTrainerSection = field(
        default_factory=lambda: DistillTrainerSection(
            mixed_precision=MixedPrecisionConfig(dtype=torch.bfloat16),
            grad_accumulation=GradAccumulationConfig(num_batches=4),
        )
    )

    optimizer: OptimizerSection = field(
        default_factory=lambda: OptimizerSection(
            name=ADAMW_OPTIMIZER,
            config=AdamWConfig(
                lr=3e-4,
                betas=(0.9, 0.98),
                eps=1e-08,
                weight_decay=0.01,
            ),
        )
    )

    lr_scheduler: LRSchedulerSection = field(
        default_factory=lambda: LRSchedulerSection(
            name=TRI_STAGE_LR,
            config=TriStageLRConfig(
                stage_ratio=(0.1, 0.4, 0.5),
                start_lr_scale=0.01,
                final_lr_scale=0.05,
            ),
        )
    )

    regime: RegimeSection = field(
        default_factory=lambda: RegimeSection(
            num_steps=50_000,
            score_metric="distill_loss",
            validate_after_n_steps=5_000,
            validate_every_n_steps=2_000,
            publish_metrics_every_n_steps=100,
            checkpoint_every_n_steps=5_000,
        )
    )

    common: CommonSection = field(default_factory=lambda: CommonSection())

    # --- Distillation-specific ---
    lambda_ctc: float = 0.3
    lambda_kd: float = 0.3
    lambda_hid: float = 0.4
    temperature: float = 3.0
    layer_mapping_strategy: str = "double_plus_one"

    # Stage 2 streaming (None for Stage 1)
    streaming: DistillStreamingSection | None = None


# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------


@final
class DistillRecipe(TrainRecipe):
    """Two-stage knowledge distillation recipe for streaming ASR."""

    # Stored after prepare_model for use in create_trainer
    _teacher: Wav2Vec2AsrModel | None = None
    _teacher_extractor: HiddenStateExtractor | None = None
    _student_extractor: HiddenStateExtractor | None = None
    _layer_mapping: dict[int, int] | None = None
    _projection_layers: ModuleDict | None = None
    _dynamic_bias: DynamicChunkBias | None = None

    @override
    def register(self, container) -> None:
        register_distill_datasets(container)
        register_student_configs(container)

        # Register distillation metric descriptors so the trainer
        # can use them as score_metric for checkpointing.
        for name, display in [
            ("distill_loss",    "Distill Loss"),
            ("kd_logit_loss",   "KD Logit Loss"),
            ("hid_cosine_loss", "Hidden Cosine Loss"),
        ]:
            container.collection.register_instance(
                MetricDescriptor,
                MetricDescriptor(name, display, 100, format_as_float),
            )

    @override
    def prepare_model(self, context: RecipeContext, model: RecipeModel) -> RecipeModel:
        config = context.config.as_(DistillRecipeConfig)
        student = cast(Wav2Vec2AsrModel, model.base_module)

        # --- Load teacher ---
        teacher_model = context.bootstrap_model("teacher")
        teacher = cast(Wav2Vec2AsrModel, teacher_model.module)

        # Share CNN feature extractor from teacher to student
        share_parameters(
            teacher.encoder_frontend.feature_extractor,
            student.encoder_frontend.feature_extractor,
        )

        # Freeze teacher entirely
        freeze_parameters(teacher)
        teacher.eval()

        # Freeze CNN feature extractor on student
        freeze_parameters(student.encoder_frontend.feature_extractor)

        log.info(
            f"Teacher loaded: {config.teacher.name} "
            f"({sum(p.numel() for p in teacher.parameters()) / 1e6:.1f}M params, frozen)"
        )
        log.info(
            f"Student: {config.model.arch} "
            f"({sum(p.numel() for p in student.parameters()) / 1e6:.1f}M params)"
        )

        # --- Streaming modifications (Stage 2) ---
        if config.streaming is not None:
            streaming_cfg = config.streaming
            log.info(
                f"Applying streaming: chunks={streaming_cfg.chunk_sizes}, "
                f"causal_conv={streaming_cfg.causal_conv}"
            )

            self._dynamic_bias = DynamicChunkBias(
                chunk_sizes=streaming_cfg.chunk_sizes,
                eval_chunk_size=streaming_cfg.eval_chunk_size,
                full_context_prob=streaming_cfg.full_context_prob,
            )
            apply_streaming_bias(student.encoder, self._dynamic_bias)

            if streaming_cfg.causal_conv:
                enable_causal_conv(student.encoder)

        # --- Layer mapping & projection layers ---
        student_layers = len(student.encoder.layers)
        teacher_layers = len(teacher.encoder.layers)

        self._layer_mapping = compute_layer_mapping(
            student_layers, teacher_layers, config.layer_mapping_strategy
        )
        log.info(f"Layer mapping ({config.layer_mapping_strategy}): {self._layer_mapping}")

        # Create projection layers for dimension alignment (student_dim → teacher_dim)
        s_dim = student.model_dim
        t_dim = teacher.model_dim

        projection_dict = {}
        if s_dim != t_dim:
            for s_idx in self._layer_mapping:
                projection_dict[str(s_idx)] = Linear(s_dim, t_dim, bias=False)

        self._projection_layers = ModuleDict(projection_dict)

        # Move projection layers to the right device
        device = next(student.parameters()).device
        self._projection_layers = self._projection_layers.to(device)

        # --- Register hidden state extractors ---
        teacher_layer_indices = list(set(self._layer_mapping.values()))
        student_layer_indices = list(self._layer_mapping.keys())

        self._teacher_extractor = HiddenStateExtractor(teacher.encoder, teacher_layer_indices)
        self._student_extractor = HiddenStateExtractor(student.encoder, student_layer_indices)

        self._teacher = teacher

        return model

    @override
    def create_trainer(self, context: RecipeContext) -> Trainer:
        config = context.config.as_(DistillRecipeConfig)

        # Access context.model first — this lazily resolves RecipeModel,
        # which triggers prepare_model() via the dependency system.
        student = cast(Wav2Vec2AsrModel, context.model.base_module)

        assert self._teacher is not None, "prepare_model must be called before create_trainer"
        assert self._teacher_extractor is not None
        assert self._student_extractor is not None
        assert self._layer_mapping is not None
        assert self._projection_layers is not None

        # Build projection layers dict with int keys for the criterion
        proj_dict: dict[int, Linear] = {}
        for key, proj in self._projection_layers.items():
            proj_dict[int(key)] = proj

        # --- Dataset ---
        dataset, storage_config, task_config = (
            Wav2Vec2AsrDatasetSelector.get_dataset_and_configs(config, context)
        )

        # --- Distillation criterion ---
        distill_loss = DistillationLoss(
            lambda_ctc=config.lambda_ctc,
            lambda_kd=config.lambda_kd,
            lambda_hid=config.lambda_hid,
            temperature=config.temperature,
        )

        criterion = DistillCriterion(
            teacher=self._teacher,
            student=student,
            distill_loss=distill_loss,
            teacher_extractor=self._teacher_extractor,
            student_extractor=self._student_extractor,
            layer_mapping=self._layer_mapping,
            projection_layers=proj_dict,
        )

        # --- Train unit ---
        unit = DistillTrainUnit(
            criterion,
            context.model,
            dynamic_bias=self._dynamic_bias,
        )

        # --- Data reader ---
        if config.dataset.train_split is None:
            raise ValueError("dataset.train_split must be defined for training.")

        task_config.seed = config.common.seed

        data_reader = dataset.create_reader(
            split=config.dataset.train_split,
            tokenizer=context.default_tokenizer,
            gangs=context.gangs,
            dtype=config.trainer.mixed_precision.dtype,
            num_accumulate=config.trainer.grad_accumulation.num_batches,
            storage_config=storage_config,
            task_config=task_config,
        )

        # --- Validation (optional) ---
        valid_units = []
        valid_data_readers = []

        if config.dataset.valid_split is not None:
            valid_splits = config.dataset.valid_split.split(",")

            valid_criterion = DistillCriterion(
                teacher=self._teacher,
                student=student,
                distill_loss=distill_loss,
                teacher_extractor=self._teacher_extractor,
                student_extractor=self._student_extractor,
                layer_mapping=self._layer_mapping,
                projection_layers=proj_dict,
            )

            for split in valid_splits:
                valid_unit = DistillEvalUnit(valid_criterion, context.model)
                valid_units.append(valid_unit)

                task_config.batch_shuffle_window = 1
                task_config.seed = config.common.seed + 1
                storage_config.sync_mode = SyncMode.UNTIL_LAST

                valid_data_reader = dataset.create_reader(
                    split=split,
                    tokenizer=context.default_tokenizer,
                    gangs=context.gangs,
                    dtype=config.trainer.mixed_precision.dtype,
                    num_accumulate=1,
                    storage_config=storage_config,
                    task_config=task_config,
                )
                valid_data_readers.append(valid_data_reader)

        return context.create_trainer(unit, data_reader, valid_units, valid_data_readers)

    @override
    def has_static_autograd_graph(self, context: RecipeContext) -> bool:
        return True  # CNN always frozen, graph is static

    @property
    @override
    def config_kls(self) -> type[object]:
        return DistillRecipeConfig


# ---------------------------------------------------------------------------
# Eval unit
# ---------------------------------------------------------------------------


@final
class DistillEvalUnit(EvalUnit[Seq2SeqBatch]):
    """Evaluation unit for distillation — computes combined loss on validation."""

    _criterion: DistillCriterion
    _model: RecipeModel

    def __init__(self, criterion: DistillCriterion, model: RecipeModel) -> None:
        self._criterion = criterion
        self._model = model

    @override
    def prepare_metric_bag(self, metric_bag: MetricBag) -> None:
        self._criterion.prepare_metric_bag(metric_bag)

    @override
    def process_batch(self, batch: Seq2SeqBatch, metric_bag: MetricBag) -> None:
        self._criterion(batch, metric_bag)

    @override
    def process_metric_values(self, values: MutableMapping[str, object]) -> None:
        self._criterion.process_metric_values(values)

    @property
    @override
    def model(self) -> RecipeModel:
        return self._model
