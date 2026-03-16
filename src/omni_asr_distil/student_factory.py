"""Student model factory with optional streaming support.

Creates Wav2Vec2AsrModel students that share the CNN feature extractor with
the teacher. For Stage 2, applies streaming modifications:
- DynamicChunkBias on self-attention layers
- Causal depthwise convolution in Conformer blocks
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn
from torch.nn import Conv1d

from fairseq2.models.conformer import ConformerBlock, ConformerConvolution
from fairseq2.models.transformer import TransformerEncoder
from fairseq2.models.transformer.attention_bias import AttentionBias
from fairseq2.models.wav2vec2 import Wav2Vec2EncoderFactory
from fairseq2.models.wav2vec2.asr.config import Wav2Vec2AsrConfig
from fairseq2.models.wav2vec2.asr.model import Wav2Vec2AsrModel
from fairseq2.nn import Linear

from .streaming import DynamicChunkBias


@dataclass
class StreamingConfig:
    """Configuration for streaming modifications (Stage 2)."""

    chunk_sizes: list[int] = field(default_factory=lambda: [16, 24, 32, 48, 64])
    eval_chunk_size: int = 48
    full_context_prob: float = 0.1
    causal_conv: bool = True


class StreamingEncoderFactory(Wav2Vec2EncoderFactory):
    """Encoder factory that supports causal convolution and custom attention bias.

    Subclasses ``Wav2Vec2EncoderFactory`` to override Conformer convolution
    creation with ``causal_depthwise_conv=True`` for streaming.
    """

    def __init__(
        self,
        config: Wav2Vec2AsrConfig,
        *,
        causal_conv: bool = False,
    ) -> None:
        super().__init__(config.encoder_config)
        self._causal_conv = causal_conv

    def create_conformer_conv(self) -> ConformerConvolution:
        config = self._config
        return ConformerConvolution(
            config.model_dim,
            config.depthwise_conv_kernel_size,
            causal_depthwise_conv=self._causal_conv,
        )


def _init_final_projection(proj: Linear) -> None:
    nn.init.xavier_uniform_(proj.weight)
    if proj.bias is not None:
        nn.init.zeros_(proj.bias)


def create_student_model(
    config: Wav2Vec2AsrConfig,
    teacher: Wav2Vec2AsrModel,
    *,
    streaming: StreamingConfig | None = None,
) -> Wav2Vec2AsrModel:
    """Create a student model, sharing the CNN frontend with the teacher.

    Args:
        config: Student architecture configuration.
        teacher: Teacher model whose frontend will be shared.
        streaming: If provided, applies streaming modifications (Stage 2).

    Returns:
        Student Wav2Vec2AsrModel with shared encoder_frontend.
    """
    causal_conv = streaming is not None and streaming.causal_conv

    # Create encoder via factory (possibly with causal conv)
    if causal_conv:
        factory = StreamingEncoderFactory(config, causal_conv=True)
    else:
        factory = Wav2Vec2EncoderFactory(config.encoder_config)

    encoder = factory.create_encoder()

    # Create final projection (vocab prediction head)
    final_proj = Linear(
        config.encoder_config.model_dim,
        config.target_vocab_size,
        bias=True,
        init_fn=_init_final_projection,
    )

    # Build student with teacher's frontend (shared)
    student = Wav2Vec2AsrModel(
        config.encoder_config.model_dim,
        teacher.encoder_frontend,
        encoder,
        final_proj,
        masker=None,
        final_dropout_p=config.final_dropout_p,
    )

    # Apply streaming attention bias if needed
    if streaming is not None:
        dynamic_bias = DynamicChunkBias(
            chunk_sizes=streaming.chunk_sizes,
            eval_chunk_size=streaming.eval_chunk_size,
            full_context_prob=streaming.full_context_prob,
        )
        apply_streaming_bias(student.encoder, dynamic_bias)

    return student


def apply_streaming_bias(
    encoder: TransformerEncoder, bias: AttentionBias
) -> None:
    """Replace attention bias on all encoder layers with the given bias.

    Walks the encoder's layers and sets ``layer.self_attn.sdpa.bias``.
    """
    for layer in encoder.layers:
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "sdpa"):
            layer.self_attn.sdpa.bias = bias


def enable_causal_conv(encoder: TransformerEncoder) -> None:
    """Enable causal depthwise convolution on all Conformer layers in-place.

    Modifies existing ConformerConvolution modules to use left-only padding.
    Weights are preserved (only the padding changes).
    """
    for layer in encoder.layers:
        if not isinstance(layer, ConformerBlock):
            continue

        conv_module = layer.conv
        if not isinstance(conv_module, ConformerConvolution):
            continue

        if conv_module.causal_depthwise_conv:
            continue  # already causal

        conv_module.causal_depthwise_conv = True

        # Re-create depthwise conv with padding=0 (causal padding applied in forward)
        old_conv = conv_module.depthwise_conv
        new_conv = Conv1d(
            old_conv.in_channels,
            old_conv.out_channels,
            old_conv.kernel_size[0],
            padding=0,
            groups=old_conv.groups,
            bias=False,
        )
        new_conv.weight = old_conv.weight  # share the parameter tensor
        conv_module.depthwise_conv = new_conv


def compute_layer_mapping(
    student_layers: int,
    teacher_layers: int,
    strategy: str = "double_plus_one",
) -> dict[int, int]:
    """Compute student→teacher layer index mapping for hidden distillation.

    Args:
        student_layers: Number of student encoder layers.
        teacher_layers: Number of teacher encoder layers.
        strategy: Mapping strategy. ``"double_plus_one"`` maps student layer i
            to teacher layer 2i+1. ``"uniform"`` distributes evenly.

    Returns:
        Dict mapping student layer indices to teacher layer indices.
    """
    if strategy == "double_plus_one":
        # g(i) = 2i + 1, clamped to teacher range; final layer always mapped
        mapping = {}
        for i in range(student_layers):
            t_idx = min(2 * i + 1, teacher_layers - 1)
            mapping[i] = t_idx
        # Ensure final student layer maps to final teacher layer
        mapping[student_layers - 1] = teacher_layers - 1
        return mapping

    elif strategy == "uniform":
        # Evenly spaced mapping
        mapping = {}
        for i in range(student_layers):
            t_idx = round(i * (teacher_layers - 1) / max(student_layers - 1, 1))
            mapping[i] = t_idx
        return mapping

    else:
        raise ValueError(f"Unknown layer mapping strategy: {strategy}")
