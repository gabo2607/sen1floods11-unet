"""Verificación rápida (2-3 minutos): evalúa el modelo final publicado y compara con el informe.

    python reproducir.py

1. Descarga los chips dorados de Sen1Floods11 (si faltan) y usa los chips de Lima del repositorio.
2. Descarga el modelo final (U-Net, pérdida BCE, semilla 1) desde el Release v1.0 de GitHub.
3. Lo evalúa en validación, test, Bolivia y Lima, e imprime cada IoU junto al valor del informe.

Para reentrenar todo desde cero (≈2.5 h en una GPU T4): python run_all.py
"""
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE / "data" / "sen1floods11"
RUNS = HERE / "runs_verificacion"
NAME = "modelo_final"
URL = "https://github.com/gabo2607/sen1floods11-unet/releases/download/v1.0/modelo_final_gold_bce_s1.pt"
INFORME = {"valid": 0.631, "test": 0.655, "bolivia": 0.701, "lima": 0.002}  # Tabla y Sección 6-7 del informe


def run(*args, tries=3):
    for i in range(tries):  # reintenta ante cortes de red al descargar
        if subprocess.run([sys.executable, *args], cwd=HERE).returncode == 0:
            return
        print(f"Falló ({i + 1}/{tries}), reintentando...", flush=True)
    sys.exit(f"No se pudo ejecutar: {' '.join(args)}")


def main():
    run("-m", "src.data.download", "--root", str(ROOT))
    lima = ROOT / "lima"
    if not (lima / "flood_lima_data.csv").exists():
        shutil.copytree(HERE / "lima" / "chips", lima, dirs_exist_ok=True)

    ckpt = RUNS / NAME / "best.pt"
    if not ckpt.exists():
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        print(f"Descargando el modelo final desde {URL}", flush=True)
        for i in range(3):
            try:
                urllib.request.urlretrieve(URL, ckpt)
                break
            except OSError as e:
                print(f"Descarga fallida ({e}), reintentando...")
        else:
            sys.exit("No se pudo descargar el modelo; bájalo a mano desde el Release v1.0.")

    run("-m", "src.evaluate", "--name", NAME, "--root", str(ROOT), "--runs-dir", str(RUNS))

    ev = json.loads((RUNS / NAME / "eval.json").read_text())
    print("\nComparación con el informe (IoU de la clase agua):")
    print(f"{'conjunto':10s} {'obtenido':>9s} {'informe':>9s}")
    ok = True
    for split, ref in INFORME.items():
        got = ev[split]["iou"]
        match = abs(got - ref) < 0.002
        ok &= match
        print(f"{split:10s} {got:9.3f} {ref:9.3f}  {'OK' if match else 'DIFIERE'}")
    print("\nResultado:", "los números coinciden con el informe." if ok else "hay diferencias; revisar versión de datos.")


if __name__ == "__main__":
    main()
