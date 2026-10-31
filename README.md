# Mapeo de inundaciones con Sentinel-1 y U-Net (Sen1Floods11)

Pregunta 2 del examen parcial de *Redes Neuronales y Aprendizaje Profundo* (UNI FIIS, 2026-II).
U-Net implementada desde cero en PyTorch que segmenta, píxel a píxel, el agua de inundación a partir
de las bandas SAR **VV y VH** de Sentinel-1.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gabo2607/sen1floods11-unet/blob/main/notebooks/colab.ipynb)

## Cómo replicar los resultados

**Verificación rápida (2–3 minutos, sirve con CPU):** descarga el modelo final publicado en el
[Release v1.0](https://github.com/gabo2607/sen1floods11-unet/releases/tag/v1.0), lo evalúa y compara
cada IoU con el del informe.

```bash
pip install -r requirements.txt
python reproducir.py
```

```
conjunto    obtenido   informe
valid          0.631     0.631  OK
test           0.655     0.655  OK
bolivia        0.701     0.701  OK
lima           0.002     0.002  OK
```

**Replicación completa (≈2.5 h en una GPU T4):** reentrena desde cero los 15 modelos de la ablación y
los 6 del bono, y regenera tablas y figuras. Es el comando único descrito a continuación (o el notebook
de Colab).

## Comando único

```bash
pip install -r requirements.txt
python run_all.py
```

`run_all.py` hace todo de extremo a extremo:

1. **Descarga** Sen1Floods11 v1.1 desde el bucket público de Google Cloud: los 446 chips dorados con
   sus particiones oficiales y 1000 chips de etiqueta débil (Otsu sobre S1) elegidos al azar.
   Los débiles **excluyen todo Bolivia y cualquier chip de validación o test** para que no se filtre
   información a la evaluación.
2. **Ablación (a): cómo combinar etiquetas doradas y débiles** (pérdida BCE+Dice):
   `gold` (solo dorados), `mixed` (dorados + débiles mezclados) y `preft` (preentrenar con débiles y
   afinar con dorados).
3. **Ablación (b): función de pérdida** — con la mejor estrategia de (a) se entrena con BCE sola y
   Dice sola.
4. **Evaluación** de los 5 modelos en validación, test (en distribución), Bolivia y **Lima** (fuera de
   distribución): IoU global, IoU promedio por chip (métrica de los autores), F1, precisión, recall,
   omisión, comisión y matriz de confusión.
5. **Figuras** para el informe: curvas, matrices de confusión, IoU por evento y paneles de errores.

**Dos rondas de entrenamiento.** En la ronda 1 cada experimento se entrena una vez (semilla 0); con
ella se elige la estrategia de datos para la ablación de pérdida. Como el IoU de validación oscila
bastante entre épocas, en la ronda 2 se repiten los 5 experimentos con las semillas 1 y 2, y todas las
tablas reportan **media ± desviación estándar sobre 3 semillas**.

Todos los experimentos usan el **mismo número de pasos de optimización** (40 épocas × 31 pasos ×
lotes de 8 recortes de 256×256), de modo que se comparan a igual cómputo. El mejor modelo se elige
**solo por IoU de validación**; test y Bolivia no se usan para tomar decisiones.

En Colab (GPU T4) cada entrenamiento tarda unos 6 minutos (≈1.5 h en total); usa el notebook `notebooks/colab.ipynb`, que
guarda los resultados en Google Drive y se puede reanudar. Prueba de humo en un par de minutos:
`python run_all.py --quick`.

## Bono: fusión SAR + Sentinel-2

Al final, `run_all.py` descarga las 13 bandas de Sentinel-2 de los chips dorados y conserva los chips
**sin nubes** (`src/data/s2_clear.py`: píxel nuboso si B2 > 0.20 o B10 > 0.01; chip despejado si tiene
menos de 10 % de nubes). Quedan 142 de entrenamiento, 50 de validación, 56 de test y 6 de Bolivia.
Sobre **esos mismos chips** entrena, con 3 semillas, una U-Net **solo SAR** (VV, VH) y otra con **SAR +
óptico** (VV, VH + B2, B3, B4, B8, B11, B12, 8 canales), y escribe la comparación en `runs/bono.md` y
`runs/bono.tex`. Lima no entra en el bono porque no se preparó Sentinel-2 para esa escena.

Resultado (test, media de 3 semillas): solo SAR IoU 0.662 ± 0.002 → SAR + Sentinel-2 IoU 0.870 ± 0.001.

## Estructura

```
run_all.py              comando único (descarga → ablación → evaluación → figuras)
src/models/unet.py      U-Net propia (profundidad y conexiones de salto configurables)
src/losses.py           BCE, Dice y BCE+Dice, ignorando píxeles sin datos (-1)
src/metrics.py          IoU, F1, precisión, recall, omisión, comisión, matriz de confusión
src/data/download.py    descarga y selección de chips débiles sin fuga hacia val/test/Bolivia
src/data/dataset.py     lectura GeoTIFF, dB → [0, 1], recorte aleatorio y volteos
src/train.py            entrenamiento (--data gold|weak|mixed, --loss, --init para afinar)
src/evaluate.py         métricas globales y por chip
src/figures.py          figuras del informe
notebooks/colab.ipynb   ejecución en Colab
```

Salidas en `runs/`: un directorio por experimento (`best.pt`, `history.json`, `eval.json`,
`perchip_*.csv`), más `results.md`, `ablation.tex`, `summary.json` y `figures/`.

## Generalización geográfica: Lima (marzo de 2017) y Bolivia

**Región de prueba propia para el Perú** (`src/data/lima.py`), construida con dos fuentes independientes:

- **Imagen SAR**: Sentinel-1B IW GRDH dual-pol (VV+VH) del **17 de marzo de 2017**, órbita
  descendente (`S1B_IW_GRDH_1SDV_20170317T103823_…`), una de las escenas que lista el propio informe
  de EMSN038. Se obtiene sin cuenta del catálogo RTC de Microsoft Planetary Computer.
- **Etiquetas**: activación **Copernicus EMS EMSN038**, capa *Potential Maximum Flood Extent*
  (delineada sobre WorldView-2 del 18 y 21 de marzo). Del geodatabase oficial
  (`EMSN038_Geodatabase.zip`, 2.2 GB) se exportaron a `lima/emsn038_layers.geojson` solo las capas
  necesarias: inundación, cauces, lagunas, mar y AOI.

Preprocesamiento para igualar el formato de Sen1Floods11: reproyección a EPSG:4326 con 8.98e-5°
(≈10 m), conversión γ⁰ (RTC) → σ⁰ ≈ γ⁰·cos θ con θ interpolado a lo ancho del barrido IW, paso a dB
y corte en chips de 512×512. Etiqueta: 1 = inundación ∪ cauces ∪ lagunas; −1 = mar, fuera del AOI o
sin datos; 0 = resto. Se conservan los **19 chips** con al menos 500 píxeles de inundación.

**Limitaciones de esta referencia** (también las reconoce el informe oficial de EMSN038): la
delineación es óptica y del 21 de marzo, con el agua ya en retroceso; el agua cargada de sedimentos es
difícil de separar del suelo desnudo, y en el 25 % del AOI no se pudo delinear la inundación. Por eso
la tomamos como supervisión débil, no como *ground truth*.

**Bolivia** (los 15 chips dorados de la partición oficial, excluidos también de los chips débiles) se
reporta como segunda región nunca vista, etiquetada con el mismo protocolo que el entrenamiento. Así
se puede separar la parte de la brecha que se debe al cambio de región de la que se debe a la
diferencia de etiqueta.

## Requisitos

Python ≥ 3.10, PyTorch ≥ 2.2, `numpy`, `rasterio`, `matplotlib` (ver `requirements.txt`).
Una GPU con 4 GB alcanza (desarrollado en una GTX 1650 Ti; pensado para correr en Colab con T4).

## Referencia del dataset

Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020). *Sen1Floods11: a georeferenced
dataset to train and test deep learning flood algorithms for Sentinel-1*. CVPR Workshops.
