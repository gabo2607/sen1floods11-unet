"""Figuras para el informe a partir de runs/summary.json (lo genera run_all.py).

    python -m src.figures --runs-dir runs

Salida en runs/figures/:
  curves.png              IoU de validación por época, todos los experimentos
  confusion_<split>.png   matriz de confusión del mejor modelo (test y Bolivia)
  events.png              IoU por evento (test) y Bolivia, mejor modelo
  errors_<split>.png      los 6 chips con más píxeles mal clasificados
  chips/<split>/*.png     paneles individuales de los 20 peores chips (para elegir ejemplos)
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from .data.dataset import DB_MAX, DB_MIN
from .evaluate import load_model
from .losses import IGNORE
from .train import hand_dataset

# Paleta categórica validada (claro, superficie #fcfcfb) — orden fijo, nunca ciclado
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE, INK, INK2, GRID, NODATA = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#bdbcb6"
BLUE_SEQ = ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"]
# Mapa de errores: fondo, acierto agua (gris), falso positivo (naranja), falso negativo (azul), sin datos
ERR_CMAP = ListedColormap([SURFACE, "#8a8984", SERIES[1], SERIES[0], "#d9d8d3"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
})


def save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ->", path)


def run_curve(runs, parts):
    ys, pre_len = [], None
    for part in parts:
        h = json.loads((runs / part / "history.json").read_text())
        pre_len = pre_len if pre_len is not None else len(h)
        ys += [r["iou"] for r in h]
    return np.array(ys), (pre_len if len(parts) > 1 else None)


def plot_curves(summary, runs, out):
    """Media entre semillas (línea) y rango mín–máx (banda). (a) datos, (b) pérdida."""
    exps = summary["experiments"]
    groups = [("(a) Estrategia de datos · BCE+Dice", exps[:3]),
              ("(b) Función de pérdida", [e for e in exps if e["strategy"] == exps[3]["strategy"]
                                          and e["loss"] in ("bce_dice", "bce", "dice")])]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3), sharey=True)
    for ax, (title, group) in zip(axes, groups):
        for i, exp in enumerate(group):
            curves = [run_curve(runs, r["parts"]) for r in exp["runs"]]
            ys = np.stack([c[0] for c in curves])
            x = np.arange(1, ys.shape[1] + 1)
            c = SERIES[i]
            label = exp["label"].split(" · ")[0 if ax is axes[0] else 1]
            ax.fill_between(x, ys.min(0), ys.max(0), color=c, alpha=0.15, lw=0)
            ax.plot(x, ys.mean(0), color=c, lw=2, label=label)
            ax.annotate(f"{ys.mean(0)[-1]:.2f}", (x[-1], ys.mean(0)[-1]), xytext=(4, 0),
                        textcoords="offset points", color=INK2, fontsize=8, va="center")
            if curves[0][1]:
                ax.axvline(curves[0][1] + 0.5, color=c, lw=1, ls=":")
        ax.set_title(title)
        ax.set_xlabel("Época")
        ax.set_ylim(0, 0.75)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel(f"IoU agua (validación)\nmedia de {len(exps[0]['runs'])} semillas")
    save(fig, out / "curves.png")


def plot_confusion(ev, split, out):
    c = ev[split]["confusion"]
    m = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]], dtype=float)
    pct = m / m.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.imshow(pct, cmap=LinearSegmentedColormap.from_list("seq", BLUE_SEQ), vmin=0, vmax=1)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{pct[i, j]:.1%}\n({int(m[i, j]):,} px)", ha="center", va="center",
                    fontsize=8, color=SURFACE if pct[i, j] > 0.55 else INK)
    ax.set_xticks([0, 1], ["no agua", "agua"])
    ax.set_yticks([0, 1], ["no agua", "agua"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta dorada")
    ax.set_title(f"Matriz de confusión — {split}")
    save(fig, out / f"confusion_{split}.png")


def read_perchip(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("tp", "fp", "fn", "tn"):
            r[k] = int(r[k])
    return rows


def plot_events(best_dir, out):
    stats = {}
    special = {"bolivia": "Bolivia (retenido)", "lima": "Lima (EMSN038)"}
    for split in ("test", "bolivia", "lima"):
        for r in read_perchip(best_dir / f"perchip_{split}.csv"):
            key = special.get(split, r["event"])
            s = stats.setdefault(key, [0, 0, 0, 0])
            s[0] += r["tp"]; s[1] += r["fp"]; s[2] += r["fn"]; s[3] += 1
    items = sorted(((k, s[0] / max(s[0] + s[1] + s[2], 1), s[3]) for k, s in stats.items()),
                   key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(5.6, 0.32 * len(items) + 0.8))
    names, ious, ns = zip(*items)
    colors = [SERIES[1] if ("Bolivia" in n or "Lima" in n) else SERIES[0] for n in names]
    ax.barh(range(len(items)), ious, color=colors, height=0.7)
    for i, (v, n) in enumerate(zip(ious, ns)):
        ax.text(v + 0.01, i, f"{v:.2f}  (n={n})", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(items)), names)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("IoU agua (global por evento)")
    ax.grid(axis="y", visible=False)
    ax.set_title("IoU por evento — test (azul) y regiones nunca vistas (naranja)")
    save(fig, out / "events.png")


def error_map(label, pred):
    e = np.zeros(label.shape, dtype=np.int64)
    e[(pred == 1) & (label == 1)] = 1
    e[(pred == 1) & (label == 0)] = 2
    e[(pred == 0) & (label == 1)] = 3
    e[label == IGNORE] = 4
    return e


def to_db(x):
    return x * (DB_MAX - DB_MIN) + DB_MIN


@torch.no_grad()
def chip_panels(model, ds, rows, device, k):
    worst = sorted(range(len(rows)), key=lambda i: rows[i]["fp"] + rows[i]["fn"], reverse=True)[:k]
    out = []
    for i in worst:
        img, label = ds[i]
        prob = torch.sigmoid(model(img[None].to(device)))[0, 0].cpu().numpy()
        out.append((rows[i], img.numpy(), label[0].numpy(), (prob > 0.5).astype(np.int64)))
    return out


def draw_row(axes, row, img, label, pred):
    vv, vh = to_db(img[0]), to_db(img[1])
    lab = np.where(label == IGNORE, np.nan, label.astype(float))
    d = row["tp"] + row["fp"] + row["fn"]
    iou = row["tp"] / d if d else float("nan")
    panels = [
        (vv, dict(cmap="gray", vmin=-25, vmax=0), f"VV (dB)\n{row['chip'].replace('_S1Hand.tif', '')}"),
        (vh, dict(cmap="gray", vmin=-30, vmax=-5), "VH (dB)"),
        (lab, dict(cmap=ListedColormap([SURFACE, SERIES[0]]), vmin=0, vmax=1), "Etiqueta dorada"),
        (pred, dict(cmap=ListedColormap([SURFACE, SERIES[0]]), vmin=0, vmax=1), f"Predicción (IoU {iou:.2f})"),
        (error_map(label, pred), dict(cmap=ERR_CMAP, vmin=0, vmax=4, interpolation="nearest"),
         f"Errores  FP {row['fp']:,} · FN {row['fn']:,}"),
    ]
    for ax, (a, kw, t) in zip(axes, panels):
        ax.imshow(a, **kw)
        ax.set_title(t, fontsize=7, fontweight="normal")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)


def error_legend(fig):
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#8a8984", label="agua acertada (TP)"),
                        Patch(color=SERIES[1], label="falso positivo (FP)"),
                        Patch(color=SERIES[0], label="falso negativo (FN)"),
                        Patch(color="#d9d8d3", label="sin datos")],
               loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.01))


def plot_errors(model, best_dir, root, split, out, device, n_panel=6, n_chips=20):
    ds = hand_dataset(root, split)
    rows = read_perchip(best_dir / f"perchip_{split}.csv")
    chips = chip_panels(model, ds, rows, device, n_chips)

    fig, axes = plt.subplots(n_panel, 5, figsize=(9, 1.9 * n_panel))
    for r, ch in enumerate(chips[:n_panel]):
        draw_row(axes[r], *ch)
    error_legend(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out / f"errors_{split}.png")

    import shutil
    shutil.rmtree(out / "chips" / split, ignore_errors=True)
    (out / "chips" / split).mkdir(parents=True, exist_ok=True)
    for rank, ch in enumerate(chips, 1):
        fig, axes = plt.subplots(1, 5, figsize=(10, 2.4))
        draw_row(axes, *ch)
        error_legend(fig)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(out / "chips" / split / f"{rank:02d}_{ch[0]['chip'].replace('_S1Hand.tif', '')}.png",
                    dpi=130, bbox_inches="tight")
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/sen1floods11")
    p.add_argument("--runs-dir", default="runs")
    args = p.parse_args()
    runs, root = Path(args.runs_dir), Path(args.root)
    out = runs / "figures"
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((runs / "summary.json").read_text(encoding="utf-8"))
    best_dir = runs / summary["best"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Figuras (mejor modelo: {summary['best']})")
    plot_curves(summary, runs, out)
    ev = json.loads((best_dir / "eval.json").read_text())
    for split in ("test", "bolivia", "lima"):
        plot_confusion(ev, split, out)
    plot_events(best_dir, out)
    model = load_model(best_dir / "best.pt", device)
    for split in ("test", "bolivia", "lima"):
        plot_errors(model, best_dir, root, split, out, device)


if __name__ == "__main__":
    main()
