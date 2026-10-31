"""Evalúa un modelo entrenado sobre los splits dorados de Sen1Floods11.

    python -m src.evaluate --name gold_bce_dice --splits valid test bolivia

Guarda en runs/<name>/:
  eval.json           métricas globales por split (IoU, F1, precisión, recall, matriz de confusión)
  perchip_<split>.csv TP/FP/FN/TN e IoU de cada chip (para análisis por evento y de errores)
"""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .losses import IGNORE
from .metrics import WaterMetrics
from .models.unet import UNet
from .train import hand_dataset


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    a = ckpt["args"]
    model = UNet(2, a["base_channels"], a["depth"], use_skip=not a["no_skip"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_split(model, ds, device, batch_size=4, workers=0):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    metrics, rows, i = WaterMetrics(), [], 0
    for img, label in loader:
        img, label = img.to(device), label.to(device)
        logits = model(img)
        metrics.update(logits, label)
        pred = torch.sigmoid(logits) > 0.5
        valid, water = label != IGNORE, label == 1
        for b in range(img.size(0)):
            c = {k: int(m[b].sum()) for k, m in {
                "tp": pred & water & valid, "fp": pred & ~water & valid,
                "fn": ~pred & water & valid, "tn": ~pred & ~water & valid}.items()}
            name = ds.pairs[i][0]
            d = c["tp"] + c["fp"] + c["fn"]
            rows.append({"chip": name, "event": name.split("_")[0], **c,
                         "iou": c["tp"] / d if d else float("nan")})
            i += 1
    return metrics.compute(), rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--root", default="data/sen1floods11")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--splits", nargs="+", default=["valid", "test", "bolivia", "lima"])
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.runs_dir) / args.name
    model = load_model(out / "best.pt", device)

    results = json.loads((out / "eval.json").read_text()) if (out / "eval.json").exists() else {}
    for split in args.splits:
        r, rows = evaluate_split(model, hand_dataset(Path(args.root), split), device, workers=args.workers)
        results[split] = r
        with open(out / f"perchip_{split}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"[{args.name}] {split:8s} IoU {r['iou']:.4f}  IoU/chip {r['iou_mean_chip']:.4f}  "
              f"F1 {r['f1']:.4f}  P {r['precision']:.4f}  R {r['recall']:.4f}")
    (out / "eval.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
