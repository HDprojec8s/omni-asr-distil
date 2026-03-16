"""Student architecture configurations for distillation.

Registers Conformer-based student configs with the wav2vec2_asr family
via fairseq2's ConfigRegistrar. All students use:
- Conformer blocks (use_conformer=True)
- Rotary position encoding (pos_encoder_type="rotary")
- Same CNN feature extractor layout as the teacher (large_lv60k)
- No masking
- POST norm order (Conformer requirement)
"""

from __future__ import annotations

from fairseq2.models.transformer import TransformerNormOrder
from fairseq2.models.wav2vec2.asr.config import Wav2Vec2AsrConfig
from fairseq2.models.wav2vec2.config import Wav2Vec2EncoderConfig
from fairseq2.runtime.config_registry import ConfigRegistrar
from fairseq2.runtime.dependency import DependencyContainer


def _base_student(
    model_dim: int,
    ffn_inner_dim: int,
    num_heads: int,
    num_layers: int,
    depthwise_conv_kernel_size: int = 31,
    vocab_size: int = 10288,
) -> Wav2Vec2AsrConfig:
    """Create a base student config with Conformer encoder and RoPE."""
    encoder_config = Wav2Vec2EncoderConfig(
        model_dim=model_dim,
        max_seq_len=4096,
        feature_dim=512,
        use_fbank=False,
        use_conformer=True,
        pos_encoder_type="rotary",
        num_encoder_layers=num_layers,
        num_encoder_attn_heads=num_heads,
        ffn_inner_dim=ffn_inner_dim,
        depthwise_conv_kernel_size=depthwise_conv_kernel_size,
        dropout_p=0.1,
        attn_dropout_p=0.1,
        ffn_inner_dropout_p=0.0,
        layer_drop_p=0.0,
        norm_order=TransformerNormOrder.POST,
        # Same CNN feature extractor as teacher (large_lv60k layout)
        feature_extractor_layer_descs=[(512, 10, 5)]
        + [(512, 3, 2)] * 4
        + [(512, 2, 2)] * 2,
        feature_extractor_bias=True,
        feature_extractor_layer_norm_convs=True,
        layer_norm_features=False,
        feature_grad_scale=1.0,
    )

    return Wav2Vec2AsrConfig(
        encoder_config=encoder_config,
        target_vocab_size=vocab_size,
        use_masking=False,
        max_temporal_mask_prob=0.0,
        max_spatial_mask_prob=0.0,
    )


def register_student_configs(container: DependencyContainer) -> None:
    """Register student architectures in the wav2vec2_asr config registry."""
    arch = ConfigRegistrar(container, Wav2Vec2AsrConfig)

    @arch("distill_s_large")
    def s_large() -> Wav2Vec2AsrConfig:
        """~45M params: 512 dim, 2048 FFN, 8 heads, 12 layers."""
        return _base_student(512, 2048, 8, 12)

    @arch("distill_s_medium")
    def s_medium() -> Wav2Vec2AsrConfig:
        """~28M params: 384 dim, 1536 FFN, 8 heads, 12 layers."""
        return _base_student(384, 1536, 8, 12)

    @arch("distill_s_small")
    def s_small() -> Wav2Vec2AsrConfig:
        """~15M params: 256 dim, 1024 FFN, 8 heads, 10 layers."""
        return _base_student(256, 1024, 8, 10)
