import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE = -1  # píxeles "sin datos" en las etiquetas de Sen1Floods11


def bce_loss(logits, target):
    valid = target != IGNORE
    return F.binary_cross_entropy_with_logits(logits[valid], target[valid].float())


def dice_loss(logits, target, eps=1.0):
    valid = (target != IGNORE).float()
    prob = torch.sigmoid(logits) * valid
    tgt = target.clamp(min=0).float() * valid
    inter = (prob * tgt).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + tgt.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (union + eps)).mean()


class SegLoss(nn.Module):
    """kind: 'bce', 'dice' o 'bce_dice'."""

    def __init__(self, kind="bce_dice"):
        super().__init__()
        if kind not in ("bce", "dice", "bce_dice"):
            raise ValueError(kind)
        self.kind = kind

    def forward(self, logits, target):
        if self.kind == "bce":
            return bce_loss(logits, target)
        if self.kind == "dice":
            return dice_loss(logits, target)
        return 0.5 * bce_loss(logits, target) + 0.5 * dice_loss(logits, target)
