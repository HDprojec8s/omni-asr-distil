"""Hook-based hidden state extraction from TransformerEncoder layers."""

from __future__ import annotations

from torch import Tensor
from torch.utils.hooks import RemovableHandle

from fairseq2.models.transformer import TransformerEncoder
from fairseq2.nn import BatchLayout


class HiddenStateExtractor:
    """Collects hidden states from specified encoder layers using the
    ``TransformerEncoder.register_layer_hook()`` API.

    Usage::

        extractor = HiddenStateExtractor(encoder, layer_indices=[1, 3, 5])
        output = encoder(seqs, seqs_layout)  # triggers hooks
        hiddens = extractor.get_hiddens()    # {1: tensor, 3: tensor, 5: tensor}
    """

    def __init__(
        self,
        encoder: TransformerEncoder,
        layer_indices: list[int],
    ) -> None:
        """
        :param encoder: The TransformerEncoder to hook into.
        :param layer_indices: Which layer indices to capture (0-based).
        """
        self._layer_indices = set(layer_indices)
        self._hiddens: dict[int, Tensor] = {}
        self._handle: RemovableHandle = encoder.register_layer_hook(self._hook)

    def _hook(
        self,
        layer_idx: int,
        layer_output: Tensor,
        layer_output_layout: BatchLayout,
        num_layers: int,
    ) -> bool:
        """TransformerEncoderLayerHook protocol implementation."""
        if layer_idx in self._layer_indices:
            self._hiddens[layer_idx] = layer_output
        return True  # always continue execution

    def get_hiddens(self) -> dict[int, Tensor]:
        """Return collected hidden states and clear the buffer.

        Call this after each encoder forward pass.
        """
        hiddens = self._hiddens
        self._hiddens = {}
        return hiddens

    def remove(self) -> None:
        """Remove the hook from the encoder."""
        self._handle.remove()
