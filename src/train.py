"""Entrena la U-Net sobre Sen1Floods11.

    python -m src.train --name gold --data gold --loss bce_dice --epochs 40
    python -m src.train --name mixed --data mixed --weak 1000
    python -m src.train --name pre --data weak --weak 1000 --epochs 20
    python -m src.train --name ft --data gold --init runs/pre/best.pt --lr 3e-4 --epochs 20

--data:  gold  = solo los 252 chips dorados de entrenamiento
         weak  = solo chips de etiqueta débil (Otsu sobre S1)
         mixed = dorados + débiles mezclados
Todas las variantes usan el MISMO número de pasos por época (por defecto, el de
'gold'), de modo que la comparación es a igual cómputo. La validación siempre se
hace sobre los chips dorados de validación.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler

from .data.dataset import FloodDataset
from .losses import SegLoss
from .metrics import WaterMetrics
from .models.unet import UNet

HAND = "v1.1/data/flood_events/HandLabeled"
WEAK = "v1.1/data/flood_events/WeaklyLabeled"
SPLITS = "v1.1/splits/flood_handlabeled"
WEAK_CSV = "v1.1/splits/weak_selected.csv"


def hand_dataset(root, split, train=False, crop=256):
    return FloodDataset(
        root / SPLITS / f"flood_{split}_data.csv",
        root / HAND / "S1Hand",
        root / HAND / "LabelHand",
        train=train,
        crop=crop,
    )


def weak_dataset(root, n, crop=256):
    ds = FloodDataset(root / WEAK_CSV, root / WEAK / "S1Weak", root / WEAK / "S1OtsuLabelWeak",
                      train=True, crop=crop)
    if n > len(ds.pairs):
        raise SystemExit(f"Se pidieron {n} chips débiles pero solo hay {len(ds.pairs)} descargados. "
                         f"Ejecuta: python -m src.data.download --weak-limit {n}")
    ds.pairs = ds.pairs[:n]
    return ds


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    metrics = WaterMetrics()
    for img, label in loader:
        img, label = img.to(device), label.to(device)
        metrics.update(model(img), label)
    return metrics.compute()


def seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="base")
    p.add_argument("--root", default="data/sen1floods11")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--data", default="gold", choices=["gold", "weak", "mixed"])
    p.add_argument("--weak", type=int, default=1000, help="chips débiles usados en weak/mixed")
    p.add_argument("--init", default=None, help="checkpoint para inicializar (afinado)")
    p.add_argument("--loss", default="bce_dice", choices=["bce", "dice", "bce_dice"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--steps", type=int, default=None, help="pasos por época (def.: los de 'gold')")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--no-skip", action="store_true")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.root)
    out = Path(args.runs_dir) / args.name
    out.mkdir(parents=True, exist_ok=True)

    gold = hand_dataset(root, "train", train=True, crop=args.crop)
    if args.data == "gold":
        train_ds = gold
    elif args.data == "weak":
        train_ds = weak_dataset(root, args.weak, args.crop)
    else:
        train_ds = ConcatDataset([gold, weak_dataset(root, args.weak, args.crop)])
    steps = args.steps or len(gold) // args.batch_size
    sampler = RandomSampler(train_ds, num_samples=steps * args.batch_size,
                            generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.workers, drop_last=True, worker_init_fn=seed_worker,
                              persistent_workers=args.workers > 0)
    valid_loader = DataLoader(hand_dataset(root, "valid"), batch_size=4, num_workers=args.workers)
    print(f"[{args.name}] datos={args.data} chips={len(train_ds)} pasos/época={steps} "
          f"pérdida={args.loss} dispositivo={device}")

    model = UNet(2, args.base_channels, args.depth, use_skip=not args.no_skip).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device, weights_only=True)["model"])
        print(f"Inicializado desde {args.init}")
    criterion = SegLoss(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    best_iou, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for img, label in train_loader:
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device, enabled=device == "cuda"):
                loss = criterion(model(img).float(), label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        scheduler.step()

        val = evaluate(model, valid_loader, device)
        history.append({"epoch": epoch, "train_loss": total / steps, **val})
        print(f"época {epoch:3d}  loss {total / steps:.4f}  IoU val {val['iou']:.4f}  "
              f"F1 val {val['f1']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if val["iou"] > best_iou:
            best_iou = val["iou"]
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, out / "best.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))

    (out / "DONE").write_text(f"{best_iou:.6f}\n")
    print(f"Mejor IoU de validación: {best_iou:.4f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
