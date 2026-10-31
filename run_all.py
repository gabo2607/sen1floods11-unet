"""Pipeline completo de la Pregunta 2 con un único comando:

    python run_all.py

1. Descarga los chips dorados de Sen1Floods11 y N chips débiles (sin Bolivia ni val/test).
2. Ablación (a), estrategia de datos con pérdida BCE+Dice:
     gold   solo dorados
     mixed  dorados + débiles mezclados
     pre-ft preentrenar con débiles (E/2 épocas) y afinar con dorados (E/2 épocas)
3. Ablación (b), con la mejor estrategia de (a) según IoU de validación: BCE sola y Dice sola.
4. Evalúa todo en valid / test / Bolivia y escribe runs/summary.json, results.md y ablation.tex.
5. Genera las figuras del informe (src/figures.py).

Es reanudable: un entrenamiento con archivo DONE no se repite.
Prueba rápida de humo: python run_all.py --quick
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

LABELS = {"gold": "Solo dorados", "mixed": "Dorados + débiles", "preft": "Pre. débiles → afin. dorados"}
LOSSES = {"bce_dice": "BCE+Dice", "bce": "BCE", "dice": "Dice"}


def run(*cmd):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    subprocess.run([sys.executable, *map(str, cmd)], check=True)


def parts_for(strategy, loss, a):
    """Lista de (nombre, argumentos de train) de un experimento."""
    name = f"{strategy}_{loss}"
    common = ["--root", a.root, "--runs-dir", a.runs_dir, "--loss", loss, "--weak", a.weak,
              "--workers", a.workers] + (["--steps", a.steps] if a.steps else [])
    if strategy == "preft":
        half = max(a.epochs // 2, 1)
        pre = f"{name}__pre"
        return [(pre, ["--data", "weak", "--epochs", half, *common]),
                (name, ["--data", "gold", "--epochs", half, "--lr", 3e-4,
                        "--init", Path(a.runs_dir) / pre / "best.pt", *common])]
    return [(name, ["--data", strategy, "--epochs", a.epochs, *common])]


def train_experiment(strategy, loss, a):
    parts = parts_for(strategy, loss, a)
    for name, args in parts:
        if (Path(a.runs_dir) / name / "DONE").exists():
            print(f"[{name}] ya entrenado, se omite")
            continue
        run("-m", "src.train", "--name", name, *args)
    name = parts[-1][0]
    return {"name": name, "strategy": strategy, "loss": loss, "parts": [p for p, _ in parts],
            "label": f"{LABELS[strategy]} · {LOSSES[loss]}",
            "val_iou": float((Path(a.runs_dir) / name / "DONE").read_text())}


def write_tables(exps, a):
    runs = Path(a.runs_dir)
    rows = []
    for e in exps:
        ev = json.loads((runs / e["name"] / "eval.json").read_text())
        e["eval"] = {s: {k: v for k, v in ev[s].items()} for s in ev}
        t, b = ev["test"], ev["bolivia"]
        rows.append((e["label"], ev["valid"]["iou"], t["iou"], t["iou_mean_chip"], t["f1"],
                     t["precision"], t["recall"], b["iou"], b["iou_mean_chip"], b["f1"]))

    head = ["Experimento", "IoU val", "IoU test", "IoU/chip test", "F1 test", "P test", "R test",
            "IoU Bolivia", "IoU/chip Bolivia", "F1 Bolivia"]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join([r[0]] + [f"{v:.3f}" for v in r[1:]]) + " |" for r in rows]
    (runs / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    tex = [r"\begin{tabular}{lccccc|ccc}", r"\toprule",
           r"Experimento & IoU$_{val}$ & IoU & IoU$_{chip}$ & F1 & P / R & IoU & IoU$_{chip}$ & F1 \\",
           r" & & \multicolumn{4}{c|}{Test (en distribución)} & \multicolumn{3}{c}{Bolivia (fuera)} \\",
           r"\midrule"]
    for r in rows:
        tex.append(f"{r[0].replace('→', '$\\to$')} & {r[1]:.3f} & {r[2]:.3f} & {r[3]:.3f} & {r[4]:.3f} & "
                   f"{r[5]:.2f} / {r[6]:.2f} & {r[7]:.3f} & {r[8]:.3f} & {r[9]:.3f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (runs / "ablation.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/sen1floods11")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--weak", type=int, default=1000)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--quick", action="store_true", help="prueba de humo: 2 épocas, 3 pasos, 16 débiles")
    a = p.parse_args()
    if a.quick:
        a.epochs, a.steps, a.weak = 2, 3, 16
        a.runs_dir = a.runs_dir + "_quick"
    Path(a.runs_dir).mkdir(parents=True, exist_ok=True)

    if not a.skip_download:
        run("-m", "src.data.download", "--root", a.root, "--weak-limit", a.weak)

    # (a) estrategia de datos
    exps = [train_experiment(s, "bce_dice", a) for s in ("gold", "mixed", "preft")]
    best_strategy = max(exps, key=lambda e: e["val_iou"])["strategy"]
    print(f"\nMejor estrategia de datos (IoU val): {best_strategy}")
    # (b) función de pérdida
    exps += [train_experiment(best_strategy, loss, a) for loss in ("bce", "dice")]

    for e in exps:
        if not (Path(a.runs_dir) / e["name"] / "eval.json").exists():
            run("-m", "src.evaluate", "--name", e["name"], "--root", a.root, "--runs-dir", a.runs_dir)
    write_tables(exps, a)

    best = max(exps, key=lambda e: e["val_iou"])["name"]  # selección SOLO por validación
    summary = {"best": best, "best_strategy": best_strategy, "epochs": a.epochs, "weak": a.weak,
               "experiments": exps}
    (Path(a.runs_dir) / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                    encoding="utf-8")
    print(f"\nMejor modelo (por IoU de validación): {best}")

    run("-m", "src.figures", "--root", a.root, "--runs-dir", a.runs_dir)


if __name__ == "__main__":
    main()
