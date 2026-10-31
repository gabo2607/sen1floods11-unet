"""Construye la región de prueba de Lima (marzo de 2017) en el formato de Sen1Floods11.

    python -m src.data.lima

Fuentes independientes:
  * Imagen: Sentinel-1B IW GRDH dual-pol (VV+VH) del 17-mar-2017 (órbita descendente), producto
    RTC de Microsoft Planetary Computer (acceso anónimo). Es la escena dual-pol más cercana a la
    delineación óptica (18 y 21-mar) y una de las que lista el propio informe de EMSN038.
  * Etiquetas: activación Copernicus EMS EMSN038, capa "Potential Maximum Flood Extent"
    (delineación sobre WorldView-2), exportada de su geodatabase a lima/emsn038_layers.geojson.

Preprocesamiento para igualar a Sen1Floods11 (σ⁰ en dB, EPSG:4326, 8.983e-5° ≈ 10 m, chips de 512×512):
  * El RTC entrega γ⁰ (lineal, aplanado por terreno). Se aproxima σ⁰ ≈ γ⁰·cos θ, con θ interpolado
    linealmente a lo ancho del barrido IW (29.1° en rango cercano → 46.0° en rango lejano). En una
    pasada descendente, mirando a la derecha, el rango cercano es el borde este de la escena.
  * Etiqueta: 1 = inundación EMSN038 ∪ cauces (Rímac, Chillón) ∪ lagunas (Sen1Floods11 también marca
    el agua permanente); -1 = mar, fuera del AOI de EMSN038 o sin datos SAR; 0 = resto.
  * Se conservan los chips que contienen al menos MIN_FLOOD píxeles del polígono de inundación.
"""
import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-1-rtc/items/"
ITEM = "S1B_IW_GRDH_1SDV_20170317T103823_20170317T103848_004747_0084AE_rtc"
TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-rtc"
RES = 8.983152841195215e-05  # resolución de los chips de Sen1Floods11
CHIP = 512
NEAR_DEG, FAR_DEG = 29.1, 46.0  # ángulos de incidencia del barrido IW
MIN_FLOOD = 500  # píxeles de inundación EMSN038 para conservar un chip (~5 ha)
LAYERS = Path(__file__).resolve().parents[2] / "lima" / "emsn038_layers.geojson"


def get_json(url, tries=6):
    for i in range(tries):  # Planetary Computer responde 429 si hay muchas peticiones seguidas
        try:
            with urllib.request.urlopen(url) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 or i == tries - 1:
                raise
            time.sleep(10 * 2 ** i)


def grid_for(layers):
    """Grilla EPSG:4326 alineada a la resolución de Sen1Floods11 que cubre la inundación + 1 km."""
    flood = [f["geometry"] for f in layers["features"] if f["properties"]["cls"] == "flood"]
    xs, ys = [], []
    for g in flood:
        for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
            for ring in poly:
                xs += [p[0] for p in ring]
                ys += [p[1] for p in ring]
    pad = 0.01
    west, north = min(xs) - pad, max(ys) + pad
    width = int(np.ceil((max(xs) + pad - west) / RES / CHIP)) * CHIP
    height = int(np.ceil((north - (min(ys) - pad)) / RES / CHIP)) * CHIP
    return from_origin(west, north, RES, RES), width, height


def read_band(href, transform, width, height):
    with rasterio.open(href) as src, WarpedVRT(src, crs="EPSG:4326", transform=transform, width=width,
                                                height=height, resampling=Resampling.bilinear,
                                                src_nodata=src.nodata, nodata=np.nan) as vrt:
        return vrt.read(1).astype(np.float32)


def incidence_cos(item, transform, width):
    """cos θ por columna: θ lineal entre el borde este (cercano) y oeste (lejano) de la escena."""
    west, _, east, _ = item["bbox"]
    lon = transform.c + (np.arange(width) + 0.5) * transform.a
    frac = np.clip((east - lon) / (east - west), 0, 1)
    return np.cos(np.deg2rad(NEAR_DEG + frac * (FAR_DEG - NEAR_DEG)))[None, :]


def burn(layers, classes, transform, width, height):
    geoms = [f["geometry"] for f in layers["features"] if f["properties"]["cls"] in classes]
    if not geoms:
        return np.zeros((height, width), bool)
    return rasterize(geoms, (height, width), transform=transform, fill=0, default_value=1,
                     dtype="uint8").astype(bool)


def write_tif(path, arr, transform, nodata=None, descriptions=None):
    arr = arr if arr.ndim == 3 else arr[None]
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[2], height=arr.shape[1],
                       count=arr.shape[0], dtype=arr.dtype, crs="EPSG:4326", transform=transform,
                       nodata=nodata, compress="deflate") as dst:
        dst.write(arr)
        if descriptions:
            dst.descriptions = descriptions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/sen1floods11")
    p.add_argument("--min-flood", type=int, default=MIN_FLOOD)
    args = p.parse_args()
    out = Path(args.root) / "lima"
    (out / "S1Hand").mkdir(parents=True, exist_ok=True)
    (out / "LabelHand").mkdir(parents=True, exist_ok=True)

    layers = json.loads(LAYERS.read_text(encoding="utf-8"))
    transform, width, height = grid_for(layers)
    print(f"Grilla Lima: {width}x{height} px ({width // CHIP}x{height // CHIP} chips)")

    item = get_json(STAC + ITEM)
    token = get_json(TOKEN)["token"]
    print(f"Escena: {ITEM}  ({item['properties']['sat:orbit_state']})")
    cos_t = incidence_cos(item, transform, width)
    bands = []
    for b in ("vv", "vh"):
        gamma0 = read_band(item["assets"][b]["href"] + "?" + token, transform, width, height)
        sigma0 = gamma0 * cos_t
        with np.errstate(divide="ignore", invalid="ignore"):
            db = np.where(sigma0 > 0, 10 * np.log10(sigma0), np.nan).astype(np.float32)
        bands.append(db)
        print(f"  {b.upper()}: mediana {np.nanmedian(db):.1f} dB, sin datos {np.isnan(db).mean():.1%}")
    img = np.stack(bands)

    flood = burn(layers, {"flood"}, transform, width, height)
    water = flood | burn(layers, {"river", "lake"}, transform, width, height)
    aoi = burn(layers, {"aoi"}, transform, width, height)
    sea = burn(layers, {"sea"}, transform, width, height)
    label = water.astype(np.int16)
    label[sea | ~aoi | np.isnan(img).any(0)] = -1

    write_tif(out / "lima_full_S1.tif", img, transform, nodata=np.nan, descriptions=("VV", "VH"))
    write_tif(out / "lima_full_label.tif", label, transform, nodata=-1)

    rows = []
    for r in range(height // CHIP):
        for c in range(width // CHIP):
            win = np.s_[r * CHIP:(r + 1) * CHIP, c * CHIP:(c + 1) * CHIP]
            n_flood = int((flood[win] & (label[win] == 1)).sum())
            if n_flood < args.min_flood:
                continue
            key = f"Lima_{r:02d}{c:02d}"
            t = rasterio.windows.transform(rasterio.windows.Window(c * CHIP, r * CHIP, CHIP, CHIP), transform)
            write_tif(out / "S1Hand" / f"{key}_S1Hand.tif", img[:, win[0], win[1]], t, nodata=np.nan,
                      descriptions=("VV", "VH"))
            write_tif(out / "LabelHand" / f"{key}_LabelHand.tif", label[win], t, nodata=-1)
            rows.append((f"{key}_S1Hand.tif", f"{key}_LabelHand.tif"))
            lv = label[win]
            print(f"  {key}: agua {(lv == 1).mean():.1%}  inundación {n_flood:,} px  ignorado {(lv == -1).mean():.0%}")
    with open(out / "flood_lima_data.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"{len(rows)} chips de Lima -> {out}")


if __name__ == "__main__":
    main()
