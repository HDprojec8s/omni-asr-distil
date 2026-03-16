"""Dynamic Chunk Training attention bias for streaming ASR.

Implements ``DynamicChunkBias``, an ``AttentionBias`` subclass that randomly
samples chunk sizes during training and uses a fixed chunk size during eval.
This enables a single model to support multiple latency targets at inference.
"""

from __future__ import annotations

import random
from typing import final

import torch
from torch import Tensor
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.models.transformer.attention_bias import AttentionBias


@final
class DynamicChunkBias(AttentionBias):
    """Attention bias for Dynamic Chunk Training (DCT).

    During training, randomly samples a chunk size per forward pass from a
    predefined set (or uses full context with some probability).
    During eval, uses a fixed chunk size for deterministic streaming behaviour.

    The mask is causal within each chunk: position i can attend to position j
    iff j is in the same chunk as i AND j <= i.

    Follows the same 0/1 float convention as ``ChunkedAttentionBias``.
    """

    def __init__(
        self,
        chunk_sizes: list[int] | None = None,
        eval_chunk_size: int = 48,
        full_context_prob: float = 0.1,
    ) -> None:
        """
        :param chunk_sizes: Candidate chunk sizes sampled during training.
        :param eval_chunk_size: Fixed chunk size used during evaluation.
        :param full_context_prob: Probability of using full context (no chunking)
            during training, following the ∞-chunk augmentation strategy.
        """
        self._chunk_sizes = chunk_sizes or [16, 24, 32, 48, 64]
        self._eval_chunk_size = eval_chunk_size
        self._full_context_prob = full_context_prob
        self._training = True
        self._current_chunk_size: int | None = None

    def set_training(self, mode: bool = True) -> None:
        """Toggle training / eval mode."""
        self._training = mode

    def sample_chunk_size(self) -> int | None:
        """Sample a chunk size for the current forward pass.

        Must be called before each forward pass during training.
        Returns ``None`` for full context (no chunking).
        """
        if not self._training:
            self._current_chunk_size = self._eval_chunk_size
            return self._current_chunk_size

        if random.random() < self._full_context_prob:
            self._current_chunk_size = None
            return None

        self._current_chunk_size = random.choice(self._chunk_sizes)
        return self._current_chunk_size

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DynamicChunkBias):
            return (
                self._chunk_sizes == other._chunk_sizes
                and self._eval_chunk_size == other._eval_chunk_size
                and self._training == other._training
                and self._current_chunk_size == other._current_chunk_size
            )
        return NotImplemented

    def __hash__(self) -> int:
        if self._training:
            return hash(("dct_train", self._current_chunk_size))
        return hash(("dct_eval", self._eval_chunk_size))

    @override
    def create_bias_tensor(
        self, q_len: int, k_len: int, device: Device, dtype: DataType
    ) -> Tensor:
        if q_len != k_len:
            raise ValueError(f"`q_len` and `k_len` must be equal: {q_len} != {k_len}")

        chunk_size = self._current_chunk_size if self._training else self._eval_chunk_size

        if chunk_size is None:
            # Full context: no masking (all 1s following ChunkedAttentionBias convention)
            return torch.ones((q_len, k_len), device=device, dtype=dtype)

        # Reuse the exact same logic as ChunkedAttentionBias
        pos = torch.arange(q_len, device=device)

        block_pos = pos.unsqueeze(0) // chunk_size - pos.unsqueeze(1) // chunk_size
        token_pos = pos.unsqueeze(0) - pos.unsqueeze(1)

        mask: Tensor = (block_pos == 0) & (token_pos <= 0)
        mask = mask.to(dtype)

        return mask

    def __repr__(self) -> str:
        if self._training:
            return (
                f"DynamicChunkBias(training, chunks={self._chunk_sizes}, "
                f"full_ctx_prob={self._full_context_prob})"
            )
        return f"DynamicChunkBias(eval, chunk_size={self._eval_chunk_size})"
