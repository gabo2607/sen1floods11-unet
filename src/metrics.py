import torch

from .losses import IGNORE


class WaterMetrics:
    """Acumula TP/FP/FN/TN de la clase 'agua' ignorando los píxeles sin datos."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.tp = self.fp = self.fn = self.tn = 0
        self.chip_ious = []

    @torch.no_grad()
    def update(self, logits, target):
        pred = torch.sigmoid(logits) > self.threshold
        valid = target != IGNORE
        water = target == 1
        tp = (pred & water & valid).sum(dim=(1, 2, 3))
        fp = (pred & ~water & valid).sum(dim=(1, 2, 3))
        fn = (~pred & water & valid).sum(dim=(1, 2, 3))
        tn = (~pred & ~water & valid).sum(dim=(1, 2, 3))
        self.tp += tp.sum().item()
        self.fp += fp.sum().item()
        self.fn += fn.sum().item()
        self.tn += tn.sum().item()
        # IoU por chip (métrica de los autores); se omiten chips sin agua real ni predicha
        denom = tp + fp + fn
        self.chip_ious += (tp[denom > 0] / denom[denom > 0]).tolist()

    def compute(self):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        div = lambda a, b: a / b if b else 0.0
        precision, recall = div(tp, tp + fp), div(tp, tp + fn)
        return {
            "iou": div(tp, tp + fp + fn),
            "iou_mean_chip": div(sum(self.chip_ious), len(self.chip_ious)),
            "f1": div(2 * precision * recall, precision + recall),
            "precision": precision,
            "recall": recall,
            "omission": div(fn, tp + fn),
            "commission": div(fp, tp + fp),
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        }
