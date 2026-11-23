"""ResNet-18 student and the distillation losses."""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from discipulus.teacher import prompt_logits


def make_student(out_dim: int) -> nn.Module:
    """ImageNet-initialized ResNet-18 whose fc is a fresh linear head: 14 logits (supervised) or 512-d embedding."""
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, out_dim)
    return m


def embedding_loss(s_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
    """Arm A: cosine distance to the teacher image embedding."""
    return 1 - F.cosine_similarity(s_emb.float(), t_emb.float(), dim=-1).mean()


def score_loss(
    s_emb: torch.Tensor, t_emb: torch.Tensor, bank: torch.Tensor, seen: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Arm B: per-finding KL(teacher || student) over the (pos, neg) prompt pair.

    Both sides go through the frozen text bank (training template only, seen findings only). Logits use
    CheXzero's learned logit scale; AUROC is unchanged by this monotone rescaling, but it keeps the
    targets from collapsing to 0.5.
    """
    b = bank[:1, seen]
    t = (scale * prompt_logits(t_emb, b)).log_softmax(-1)
    s = (scale * prompt_logits(s_emb, b)).log_softmax(-1)
    return F.kl_div(s, t, log_target=True, reduction="none").sum(-1).mean()
