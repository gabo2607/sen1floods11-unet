"""Pipeline completo de la Pregunta 2 con un único comando:

    python run_all.py

1. Descarga los chips dorados de Sen1Floods11 y N chips débiles (sin Bolivia ni val/test).
2. Construye la región de prueba de Lima (S1 17-mar-2017 + EMSN038, src/data/lima.py).
3. Ablación (a), estrategia de datos con pérdida BCE+Dice:
     gold   solo dorados
     mixed  dorados + débiles mezclados
     preft  preentrenar con débiles (E/2 épocas) y afinar con dorados (E/2 épocas)
4. Ablación (b): con la estrategia ganadora de (a) en la ronda 1 (semilla 0), BCE sola y Dice sola.
5. Ronda 1 = semilla 0; ronda 2 = repetición de los 5 experimentos con las semillas 1 y 2, para
   medir la variabilidad entre entrenamientos. Se reporta media ± desviación estándar.
6. Evalúa todo en valid / test / Bolivia / Lima y escribe runs/summary.json, results.md y ablation.tex.
7. Genera las figuras del informe (src/figures.py) con el mejor entrenamiento según validación.

Es reanudable: un entrenamiento con archivo DONE no se repite.
Prueba rápida de humo: python run_all.py --quick
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

LABELS = {"gold": "Solo dorados", "mixed": "Dorados + débiles", "preft": "Pre. débiles → afin. dorados"}
LOSSES = {"bce_dice": "BCE+Dice", "bce": "BCE", "dice": "Dice"}
SPLITS = ["valid", "test", "bolivia", "lima"]


def run(*cmd):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    subprocess.run([sys.executable, *map(str, cmd)], check=True)


def run_name(strategy, loss, seed):
    return f"{strategy}_{loss}" + (f"_s{seed}" if seed else "")


def parts_for(strategy, loss, seed, a):
    """Lista de (nombre, argumentos de train) de un entrenamiento."""
    name = run_name(strategy, loss, seed)
    common = ["--root", a.root, "--runs-dir", a.runs_dir, "--loss", loss, "--weak", a.weak,
              "--workers", a.workers, "--seed", seed] + (["--steps", a.steps] if a.steps else [])
    if strategy == "preft":
        half = max(a.epochs // 2, 1)
        pre = f"{name}__pre"
        return [(pre, ["--data", "weak", "--epochs", half, *common]),
                (name, ["--data", "gold", "--epochs", half, "--lr", 3e-4,
                        "--init", Path(a.runs_dir) / pre / "best.pt", *common])]
    return [(name, ["--data", strategy, "--epochs", a.epochs, *common])]


def train_run(strategy, loss, seed, a):
    parts = parts_for(strategy, loss, seed, a)
    for name, args in parts:
        if (Path(a.runs_dir) / name / "DONE").exists():
            print(f"[{name}] ya entrenado, se omite")
            continue
        run("-m", "src.train", "--name", name, *args)
    name = parts[-1][0]
    return {"name": name, "seed": seed, "parts": [p for p, _ in parts],
            "val_iou": float((Path(a.runs_dir) / name / "DONE").read_text())}


def evaluate_run(r, a):
    ev_path = Path(a.runs_dir) / r["name"] / "eval.json"
    done = json.loads(ev_path.read_text()) if ev_path.exists() else {}
    missing = [s for s in SPLITS if s not in done]
    if missing:
        run("-m", "src.evaluate", "--name", r["name"], "--root", a.root, "--runs-dir", a.runs_dir,
            "--splits", *missing)
    r["eval"] = json.loads(ev_path.read_text())


METRICS = [("valid", "iou"), ("test", "iou"), ("test", "iou_mean_chip"), ("test", "f1"),
           ("test", "precision"), ("test", "recall"), ("bolivia", "iou"), ("bolivia", "iou_mean_chip"),
           ("bolivia", "f1"), ("lima", "iou"), ("lima", "f1"), ("lima", "precision"), ("lima", "recall")]


def aggregate(exp):
    exp["stats"] = {}
    for split, m in METRICS:
        v = np.array([r["eval"][split][m] for r in exp["runs"]])
        exp["stats"][f"{split}_{m}"] = {"mean": float(v.mean()),
                                        "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                                        "values": v.tolist()}


def write_tables(exps, a):
    runs = Path(a.runs_dir)
    n = len(exps[0]["runs"])
    head = ["Experimento"] + [f"{m} {s}" for s, m in METRICS]
    md = [f"Media ± desviación estándar sobre {n} semillas.\n",
          "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for e in exps:
        md.append("| " + " | ".join([e["label"]] + [
            f"{e['stats'][f'{s}_{m}']['mean']:.3f} ± {e['stats'][f'{s}_{m}']['std']:.3f}"
            for s, m in METRICS]) + " |")
    md += ["", "Por entrenamiento (IoU val / test / Bolivia / Lima):", ""]
    for e in exps:
        for r in e["runs"]:
            ev = r["eval"]
            md.append(f"- {r['name']}: {ev['valid']['iou']:.3f} / {ev['test']['iou']:.3f} / "
                      f"{ev['bolivia']['iou']:.3f} / {ev['lima']['iou']:.3f}")
    (runs / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    def pm(e, key, d=3):
        s = e["stats"][key]
        return f"{s['mean']:.{d}f}\\,$\\pm$\\,{s['std']:.{d}f}"

    def mean(e, key):
        return f"{e['stats'][key]['mean']:.2f}"

    eol = " \\\\"
    tex = [r"\begin{tabular}{lcccc|c|cc}", r"\toprule",
           r" & & \multicolumn{3}{c|}{Test (en distribución)} & Bolivia & \multicolumn{2}{c}{Lima}" + eol,
           r"Experimento & IoU$_{val}$ & IoU & IoU$_{chip}$ & P / R & IoU & IoU & P / R" + eol,
           r"\midrule"]
    for i, e in enumerate(exps):
        if i == 3:
            tex.append(r"\midrule")
        name = e["label"].replace("→", r"$\to$")
        tex.append(f"{name} & {pm(e, 'valid_iou')} & {pm(e, 'test_iou')} & {pm(e, 'test_iou_mean_chip')} & "
                   f"{mean(e, 'test_precision')} / {mean(e, 'test_recall')} & {pm(e, 'bolivia_iou')} & "
                   f"{pm(e, 'lima_iou')} & {mean(e, 'lima_precision')} / {mean(e, 'lima_recall')}" + eol)
    tex += [r"\bottomrule", r"\end{tabular}"]
    (runs / "ablation.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/sen1floods11")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--weak", type=int, default=1000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--quick", action="store_true", help="prueba de humo: 2 épocas, 3 pasos, 16 débiles")
    a = p.parse_args()
    if a.quick:
        a.epochs, a.steps, a.weak, a.seeds = 2, 3, 16, [0, 1]
        a.runs_dir = a.runs_dir + "_quick"
    Path(a.runs_dir).mkdir(parents=True, exist_ok=True)

    if not a.skip_download:
        run("-m", "src.data.download", "--root", a.root, "--weak-limit", a.weak)
    if not (Path(a.root) / "lima" / "flood_lima_data.csv").exists():
        run("-m", "src.data.lima", "--root", a.root)

    def experiment(strategy, loss):
        return {"strategy": strategy, "loss": loss, "label": f"{LABELS[strategy]} · {LOSSES[loss]}",
                "runs": [train_run(strategy, loss, s, a) for s in a.seeds]}

    # Ronda 1 (semilla 0) decide la estrategia para la ablación de pérdida
    first = {s: train_run(s, "bce_dice", a.seeds[0], a)["val_iou"] for s in ("gold", "mixed", "preft")}
    best_strategy = max(first, key=first.get)
    print(f"\nRonda 1: mejor estrategia de datos (IoU val, semilla {a.seeds[0]}): {best_strategy}")

    exps = [experiment(s, "bce_dice") for s in ("gold", "mixed", "preft")]
    exps += [experiment(best_strategy, loss) for loss in ("bce", "dice")]

    for e in exps:
        for r in e["runs"]:
            evaluate_run(r, a)
        aggregate(e)
    write_tables(exps, a)

    # Mejor experimento por IoU val medio; dentro de él, el entrenamiento con mejor IoU val
    best_exp = max(exps, key=lambda e: e["stats"]["valid_iou"]["mean"])
    best = max(best_exp["runs"], key=lambda r: r["val_iou"])["name"]
    summary = {"best": best, "best_experiment": best_exp["label"], "best_strategy_round1": best_strategy,
               "seeds": a.seeds, "epochs": a.epochs, "weak": a.weak, "experiments": exps}
    (Path(a.runs_dir) / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                    encoding="utf-8")
    print(f"\nMejor experimento (IoU val medio): {best_exp['label']}  ->  entrenamiento {best}")

    run("-m", "src.figures", "--root", a.root, "--runs-dir", a.runs_dir)


if __name__ == "__main__":
    main()
