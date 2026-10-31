import csv
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from ..losses import IGNORE

DB_MIN, DB_MAX = -50.0, 1.0  # rango de recorte de VV/VH en dB


def read_pairs(csv_path):
    with open(csv_path, newline="") as f:
        return [tuple(row[:2]) for row in csv.reader(f) if row]


def load_s1(path):
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)  # (2, H, W): VV, VH
    nodata = np.isnan(img).any(axis=0)
    img = np.nan_to_num(img, nan=DB_MIN)
    img = (np.clip(img, DB_MIN, DB_MAX) - DB_MIN) / (DB_MAX - DB_MIN)
    return img, nodata


class FloodDataset(Dataset):
    """Pares (imagen S1, máscara) listados en un CSV de splits de Sen1Floods11.

    train=True aplica recorte aleatorio de crop x crop y volteos;
    train=False devuelve el chip completo de 512 x 512.
    """

    def __init__(self, csv_path, img_dir, label_dir, train=False, crop=256):
        self.pairs = read_pairs(csv_path)
        self.img_dir, self.label_dir = Path(img_dir), Path(label_dir)
        self.train, self.crop = train, crop

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img_name, label_name = self.pairs[i]
        img, nodata = load_s1(self.img_dir / img_name)
        with rasterio.open(self.label_dir / label_name) as src:
            label = src.read(1).astype(np.int64)
        label[(label != 0) & (label != 1)] = IGNORE  # -1 = sin datos en las etiquetas doradas
        label[nodata] = IGNORE

        if self.train:
            h, w = label.shape
            y = np.random.randint(0, h - self.crop + 1)
            x = np.random.randint(0, w - self.crop + 1)
            img = img[:, y : y + self.crop, x : x + self.crop]
            label = label[y : y + self.crop, x : x + self.crop]
            if np.random.rand() < 0.5:
                img, label = img[:, :, ::-1], label[:, ::-1]
            if np.random.rand() < 0.5:
                img, label = img[:, ::-1, :], label[::-1, :]

        img = torch.from_numpy(np.ascontiguousarray(img))
        label = torch.from_numpy(np.ascontiguousarray(label)).unsqueeze(0)
        return img, label
