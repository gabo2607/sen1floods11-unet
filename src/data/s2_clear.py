"""Bono: selecciona los chips dorados cuya imagen Sentinel-2 está (casi) libre de nubes.

    python -m src.data.s2_clear

Un píxel se considera nube si es brillante en el azul (B2 > 0.20 de reflectancia, el agua y la
vegetación están muy por debajo) o si la banda de cirros lo detecta (B10 > 0.01). Un chip es
"despejado" si tiene menos de MAX_CLOUD de nubes y menos de 10 % de píxeles sin dato óptico.
Escribe v1.1/splits/flood_handlabeled/flood_<split>_s2clear_data.csv para train/valid/test/bolivia.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio

SPLITS = "v1.1/splits/flood_handlabeled"
S2 = "v1.1/data/flood_events/HandLabeled/S2Hand"
MAX_CLOUD = 0.10


def cloud_fraction(path):
    with rasterio.open(path) as src:
        b2, b10 = src.read([2, 11]).astype(np.float32) / 10000.0
    empty = b2 <= 0
    cloud = ((b2 > 0.20) | (b10 > 0.01)) & ~empty
    return float(cloud.mean()), float(empty.mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/sen1floods11")
    p.add_argument("--max-cloud", type=float, default=MAX_CLOUD)
    a = p.parse_args()
    root = Path(a.root)
    for split in ("train", "valid", "test", "bolivia"):
        with open(root / SPLITS / f"flood_{split}_data.csv", newline="") as f:
            pairs = [row[:2] for row in csv.reader(f) if row]
        keep = []
        for s1, lab in pairs:
            c, e = cloud_fraction(root / S2 / s1.replace("_S1Hand", "_S2Hand"))
            if c < a.max_cloud and e < 0.10:
                keep.append((s1, lab))
        with open(root / SPLITS / f"flood_{split}_s2clear_data.csv", "w", newline="") as f:
            csv.writer(f).writerows(keep)
        print(f"{split:8s}: {len(keep):3d} de {len(pairs):3d} chips sin nubes (< {a.max_cloud:.0%})")


if __name__ == "__main__":
    main()
