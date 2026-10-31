"""Descarga Sen1Floods11 (v1.1) desde el bucket público de Google Cloud.

    python -m src.data.download                    # chips dorados (S1Hand + LabelHand) y splits
    python -m src.data.download --weak-limit 1000  # + 1000 chips de etiqueta débil (S1 + Otsu)

Los chips débiles se eligen al azar (semilla fija) EXCLUYENDO:
  * todo el evento Bolivia (región de generalización reservada), y
  * cualquier chip cuya ubicación coincida con un chip dorado de validación o test.
La lista elegida se guarda en v1.1/splits/weak_selected.csv.
"""
import argparse
import csv
import json
import random
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://storage.googleapis.com/sen1floods11/"
LIST = "https://storage.googleapis.com/storage/v1/b/sen1floods11/o"
SPLITS = "v1.1/splits/flood_handlabeled/"
HAND = "v1.1/data/flood_events/HandLabeled/"
WEAK = "v1.1/data/flood_events/WeaklyLabeled/"
WEAK_CSV = "v1.1/splits/weak_selected.csv"
HELDOUT_EVENT = "Bolivia"


def list_objects(prefix):
    names, token = [], None
    while True:
        query = {"prefix": prefix, "fields": "items(name),nextPageToken"}
        if token:
            query["pageToken"] = token
        with urllib.request.urlopen(LIST + "?" + urllib.parse.urlencode(query)) as r:
            page = json.load(r)
        names += [item["name"] for item in page.get("items", [])]
        token = page.get("nextPageToken")
        if not token:
            return names


def fetch(name, root):
    dest = root / name
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(BASE + urllib.parse.quote(name), tmp)
    tmp.rename(dest)


def fetch_all(names, root, workers):
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda n: fetch(n, root), names))


def chip_key(filename):
    """'Ghana_313799_S1Hand.tif' -> 'Ghana_313799'"""
    return "_".join(Path(filename).name.split("_")[:2])


def reserved_keys(root):
    keys = set()
    for split in ("valid", "test", "bolivia"):
        with open(root / SPLITS / f"flood_{split}_data.csv", newline="") as f:
            keys |= {chip_key(row[0]) for row in csv.reader(f) if row}
    return keys


def select_weak(root, limit, seed):
    reserved = reserved_keys(root)
    keys = sorted(
        chip_key(n) for n in list_objects(WEAK + "S1Weak/")
        if n.endswith(".tif")
        and not chip_key(n).startswith(HELDOUT_EVENT)
        and chip_key(n) not in reserved
    )
    random.Random(seed).shuffle(keys)
    if limit >= 0:
        keys = keys[:limit]
    print(f"Chips débiles elegidos: {len(keys)} (sin {HELDOUT_EVENT} ni val/test)")
    return keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/sen1floods11")
    parser.add_argument("--weak-limit", type=int, default=0,
                        help="chips débiles a descargar (0 = ninguno, -1 = todos los permitidos)")
    parser.add_argument("--s2", action="store_true", help="descarga también Sentinel-2 de los chips dorados (bono)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    root = Path(args.root)

    prefixes = [SPLITS, HAND + "S1Hand/", HAND + "LabelHand/"] + ([HAND + "S2Hand/"] if args.s2 else [])
    for prefix in prefixes:
        names = [n for n in list_objects(prefix) if not n.endswith("/")]
        print(f"{prefix}: {len(names)} archivos")
        fetch_all(names, root, args.workers)

    if args.weak_limit:
        keys = select_weak(root, args.weak_limit, args.seed)
        pairs = [(f"{k}_S1Weak.tif", f"{k}_S1OtsuLabelWeak.tif") for k in keys]
        names = [WEAK + "S1Weak/" + a for a, _ in pairs] + [WEAK + "S1OtsuLabelWeak/" + b for _, b in pairs]
        fetch_all(names, root, args.workers)
        with open(root / WEAK_CSV, "w", newline="") as f:
            csv.writer(f).writerows(pairs)
        print(f"Lista guardada en {root / WEAK_CSV}")


if __name__ == "__main__":
    main()
