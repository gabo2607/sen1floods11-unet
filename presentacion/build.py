"""Llena la presentación con los resultados y la exporta a PDF.

    python presentacion/build.py --runs ../sen1floods11-unet/runs

1. Lee eval.json de todos los entrenamientos (ronda 1: semilla 0; ronda 2: sufijos _s1, _s2).
2. Calcula media ± desviación por experimento y escribe presentacion/datos.js.
3. Copia las figuras de runs/figures a presentacion/img.
4. Exporta presentacion/presentacion.pdf con Chrome sin interfaz.
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXPS = {"gold": "gold_bce_dice", "mixed": "mixed_bce_dice", "preft": "preft_bce_dice",
        "bce": "gold_bce", "dice": "gold_dice"}
METRICS = {"val": ("valid", "iou"), "test": ("test", "iou"), "test_chip": ("test", "iou_mean_chip"),
           "test_f1": ("test", "f1"), "test_p": ("test", "precision"), "test_r": ("test", "recall"),
           "bol": ("bolivia", "iou"), "bol_chip": ("bolivia", "iou_mean_chip"), "bol_p": ("bolivia", "precision"),
           "bol_r": ("bolivia", "recall"), "lima": ("lima", "iou"), "lima_p": ("lima", "precision"),
           "lima_r": ("lima", "recall")}
FIGS = ["curves", "confusion_test", "confusion_bolivia", "confusion_lima", "events", "backscatter",
        "errors_test", "errors_bolivia", "errors_lima"]
ERROR_CHIPS = {"err_veg": "Pakistan_849790", "err_suelo": "Pakistan_70625", "err_urbano": "Spain_7387658"}
CHROME = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
          r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]


def render_error_chips(runs, best_run, root):
    """Paneles VV/VH/etiqueta/predicción/errores de chips elegidos, con el mejor modelo."""
    import sys
    sys.path.insert(0, str(HERE.parent))
    import matplotlib.pyplot as plt
    import torch
    from src.evaluate import evaluate_split, load_model
    from src.figures import draw_row
    from src.train import hand_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(runs / best_run / "best.pt", device)
    ds = hand_dataset(root, "test")
    names = [a for a, _ in ds.pairs]
    _, rows = evaluate_split(model, ds, device)
    for out, key in ERROR_CHIPS.items():
        i = names.index(f"{key}_S1Hand.tif")
        img, label = ds[i]
        with torch.no_grad():
            prob = torch.sigmoid(model(img[None].to(device)))[0, 0].cpu().numpy()
        fig, axes = plt.subplots(1, 5, figsize=(10, 2.25))
        draw_row(axes, rows[i], img.numpy(), label[0].numpy(), (prob > 0.5).astype(np.int64))
        fig.tight_layout()
        fig.savefig(HERE / "img" / f"{out}.png", dpi=150, bbox_inches="tight", facecolor="#ffffff")
        plt.close(fig)
    print("Chips de error dibujados:", ", ".join(ERROR_CHIPS.values()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="../sen1floods11-unet/runs")
    p.add_argument("--root", default=str(HERE.parent / "data" / "sen1floods11"))
    p.add_argument("--no-pdf", action="store_true")
    a = p.parse_args()
    runs = Path(a.runs)

    data, seeds = {}, 0
    for key, base in EXPS.items():
        names = [n for n in [base, f"{base}_s1", f"{base}_s2"] if (runs / n / "eval.json").exists()]
        evs = [json.loads((runs / n / "eval.json").read_text()) for n in names]
        seeds = max(seeds, len(evs))
        d = {"n": len(evs), "runs": names}
        for k, (split, m) in METRICS.items():
            v = np.array([e[split][m] for e in evs if split in e])
            if v.size:
                d[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else None}
        data[key] = d

    best_key = max(data, key=lambda k: data[k]["val"]["mean"])
    best_run = max(data[best_key]["runs"],
                   key=lambda n: json.loads((runs / n / "eval.json").read_text())["valid"]["iou"])
    ev = json.loads((runs / best_run / "eval.json").read_text())
    single = {}
    for split in ev:
        c = ev[split]["confusion"]
        single[split] = {**{k: ev[split][k] for k in ("iou", "iou_mean_chip", "f1", "precision", "recall")},
                         "fn_rate": c["fn"] / max(c["tp"] + c["fn"], 1),
                         "fp_rate": c["fp"] / max(c["fp"] + c["tn"], 1)}
    out = {"seeds": seeds, "best": best_key, "best_run": best_run, "exp": data, "single": single}
    (HERE / "datos.js").write_text("window.R = " + json.dumps(out, indent=1) + ";\n", encoding="utf-8")
    print(f"datos.js: {seeds} semilla(s); mejor experimento {best_key} ({best_run})")

    for f in FIGS:
        src = runs / "figures" / f"{f}.png"
        if src.exists():
            shutil.copy(src, HERE / "img" / f"{f}.png")
        else:
            print(f"  (falta {src.name}, se mantiene la anterior si existe)")

    if (runs / best_run / "best.pt").exists():
        render_error_chips(runs, best_run, Path(a.root))
    else:  # sin pesos locales: usar los paneles que dibujó src/figures.py con el mismo modelo
        from PIL import Image
        for out, key in ERROR_CHIPS.items():
            cands = sorted((runs / "figures" / "chips" / "test").glob(f"*_{key}.png"),
                           key=lambda f: f.stat().st_mtime)
            if not cands:
                print(f"  (no hay panel para {key})")
                continue
            im = Image.open(cands[-1])
            im.crop((0, 0, im.width, int(im.height * 0.86))).save(HERE / "img" / f"{out}.png")
        print("Paneles de error tomados de runs/figures/chips/test")

    if not a.no_pdf:
        chrome = next((c for c in CHROME if Path(c).exists()), None)
        if chrome is None:
            print("No se encontró Chrome/Edge; abre presentacion.html e imprime a PDF.")
            return
        pdf = HERE / "presentacion.pdf"
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        "--virtual-time-budget=8000", f"--print-to-pdf={pdf}",
                        (HERE / "presentacion.html").as_uri()], check=True, capture_output=True)
        print(f"PDF: {pdf}")


if __name__ == "__main__":
    main()
