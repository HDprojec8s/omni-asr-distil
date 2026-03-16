"""Distillation losses: KL-Divergence on logits, Cosine Similarity on hiddens."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Module


def kd_logit_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float,
    seq_lens: Tensor,
) -> Tensor:
    """KL(teacher || student) on temperature-scaled CTC logits.

    Args:
        student_logits: (N, S, V) student CTC logits.
        teacher_logits: (N, S, V) teacher CTC logits.
        temperature: Temperature for softening distributions.
        seq_lens: (N,) valid sequence lengths for masking padding.

    Returns:
        Scalar KL divergence loss, scaled by T^2.
    """
    batch_size, max_len, _ = student_logits.shape

    # (N, S) mask for valid (non-padding) positions
    mask = torch.arange(max_len, device=seq_lens.device).unsqueeze(0) < seq_lens.unsqueeze(1)

    # Temperature-scaled log-softmax / softmax
    s_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    t_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # KL(teacher || student) per position per token
    kl = F.kl_div(s_log_probs, t_probs, reduction="none").sum(dim=-1)  # (N, S)

    # Average over valid positions only
    kl = (kl * mask).sum() / mask.sum().clamp(min=1)

    # Scale by T^2 (Hinton et al., 2015)
    return kl * (temperature**2)


def hidden_cosine_loss(
    student_hiddens: dict[int, Tensor],
    teacher_hiddens: dict[int, Tensor],
    layer_mapping: dict[int, int],
    projection_layers: dict[int, Linear],
    seq_lens: Tensor,
) -> Tensor:
    """Cosine similarity loss on hidden representations with linear projection.

    Args:
        student_hiddens: {layer_idx: (N, S, D_s)} collected via hooks.
        teacher_hiddens: {layer_idx: (N, S, D_t)} collected via hooks.
        layer_mapping: {student_layer: teacher_layer} index mapping.
        projection_layers: {student_layer: Linear(D_s, D_t)} dimension alignment.
        seq_lens: (N,) valid sequence lengths.

    Returns:
        Scalar 1 - cos_sim, averaged over mapped layer pairs and valid positions.
    """
    device = seq_lens.device
    total_loss = torch.tensor(0.0, device=device)
    num_pairs = 0

    for s_idx, t_idx in layer_mapping.items():
        if s_idx not in student_hiddens or t_idx not in teacher_hiddens:
            continue

        s_hidden = student_hiddens[s_idx]  # (N, S, D_s)
        t_hidden = teacher_hiddens[t_idx]  # (N, S, D_t)

        # Project student to teacher dimension
        if s_idx in projection_layers:
            s_hidden = projection_layers[s_idx](s_hidden)  # (N, S, D_t)

        # Use the shorter sequence length (should be equal, but be safe)
        min_len = min(s_hidden.shape[1], t_hidden.shape[1])
        s_hidden = s_hidden[:, :min_len]
        t_hidden = t_hidden[:, :min_len]

        # Mask for valid positions
        mask = torch.arange(min_len, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)
        mask = mask[:, :min_len]

        # 1 - cos_sim at each position
        cos_sim = F.cosine_similarity(s_hidden, t_hidden, dim=-1)  # (N, S)
        layer_loss = ((1.0 - cos_sim) * mask).sum() / mask.sum().clamp(min=1)

        total_loss = total_loss + layer_loss
        num_pairs += 1

    if num_pairs == 0:
        return total_loss

    return total_loss / num_pairs


class DistillationLoss(Module):
    """Combined distillation loss: L_ctc_gt + L_kd_logit + L_hid_cosine.

    L_total = lambda_ctc * L_ctc + lambda_kd * L_kd + lambda_hid * L_hid
    """

    def __init__(
        self,
        lambda_ctc: float = 0.3,
        lambda_kd: float = 0.3,
        lambda_hid: float = 0.4,
        temperature: float = 3.0,
    ) -> None:
        super().__init__()
        self.lambda_ctc = lambda_ctc
        self.lambda_kd = lambda_kd
        self.lambda_hid = lambda_hid
        self.temperature = temperature

    def forward(
        self,
        ctc_loss: Tensor,
        student_logits: Tensor,
        teacher_logits: Tensor,
        student_hiddens: dict[int, Tensor],
        teacher_hiddens: dict[int, Tensor],
        layer_mapping: dict[int, int],
        projection_layers: dict[int, Linear],
        seq_lens: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute combined distillation loss.

        Returns:
            total_loss: Weighted sum of all three loss components.
            loss_dict: Individual loss values for logging.
        """
        l_kd = kd_logit_loss(student_logits, teacher_logits, self.temperature, seq_lens)

        l_hid = hidden_cosine_loss(
            student_hiddens, teacher_hiddens, layer_mapping, projection_layers, seq_lens
        )

        total = self.lambda_ctc * ctc_loss + self.lambda_kd * l_kd + self.lambda_hid * l_hid

        loss_dict = {
            "ctc_gt": ctc_loss.item(),
            "kd_logit": l_kd.item(),
            "hid_cosine": l_hid.item(),
            "total": total.item(),
        }

        return total, loss_dict
