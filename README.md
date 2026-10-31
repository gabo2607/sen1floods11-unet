# Mapeo de inundaciones con Sentinel-1 y U-Net (Sen1Floods11)

Pregunta 2 del examen parcial de *Redes Neuronales y Aprendizaje Profundo* (UNI FIIS, 2026-II).
U-Net implementada desde cero en PyTorch que segmenta, píxel a píxel, el agua de inundación a partir
de las bandas SAR **VV y VH** de Sentinel-1.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gabo2607/sen1floods11-unet/blob/main/notebooks/colab.ipynb)

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
4. **Evaluación** de los 5 modelos en validación, test (en distribución) y Bolivia (fuera de
   distribución): IoU global, IoU promedio por chip (métrica de los autores), F1, precisión, recall,
   omisión, comisión y matriz de confusión.
5. **Figuras** para el informe: curvas, matrices de confusión, IoU por evento y paneles de errores.

Todos los experimentos usan el **mismo número de pasos de optimización** (40 épocas × 31 pasos ×
lotes de 8 recortes de 256×256), de modo que se comparan a igual cómputo. El mejor modelo se elige
**solo por IoU de validación**; test y Bolivia no se usan para tomar decisiones.

En Colab (GPU T4) se estima alrededor de una hora; usa el notebook `notebooks/colab.ipynb`, que
guarda los resultados en Google Drive y se puede reanudar. Prueba de humo en un par de minutos:
`python run_all.py --quick`.

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

## Generalización geográfica: Bolivia en lugar de Lima

El enunciado propone construir una región de prueba en Lima con Sentinel-1 de marzo de 2017 y la
delineación EMSN038 de Copernicus EMS, y permite usar Bolivia como respaldo justificado. Usamos
**Bolivia** (los 15 chips dorados de la partición oficial, evento nunca visto en entrenamiento, ni
siquiera entre los débiles) porque:

- **Plazo**: construir la región de Lima exige ubicar y preprocesar una escena Sentinel-1 GRD
  (calibración a σ⁰ en dB, corrección de terreno, remuestreo a 10 m y recorte en chips de 512×512)
  y rasterizar los polígonos de EMSN038 sobre esa grilla; no fue posible completarlo y verificarlo
  dentro del plazo del proyecto.
- **Calidad de la etiqueta**: EMSN038 se delineó con imagen óptica WorldView-2 de muy alta
  resolución del 18 y 21 de marzo de 2017; cualquier desfase con la pasada SAR (en ríos encañonados
  como el Rímac y el Chillón el agua retrocede rápido) y la diferencia de criterio frente a las
  etiquetas doradas harían que la brecha medida mezcle error del modelo con error de la etiqueta.
- **Tipo de cobertura**: las zonas afectadas son urbanas y periurbanas, con lodo y huaicos más que
  lámina de agua abierta, donde el doble rebote y la sombra de radar dominan la señal.

Bolivia, en cambio, tiene etiquetas doradas hechas con el mismo protocolo que el entrenamiento, por
lo que la brecha frente a test refleja solo el cambio de región. Estos puntos se discuten como
hipótesis y limitaciones en el informe.

## Requisitos

Python ≥ 3.10, PyTorch ≥ 2.2, `numpy`, `rasterio`, `matplotlib` (ver `requirements.txt`).
Una GPU con 4 GB alcanza (desarrollado en una GTX 1650 Ti; pensado para correr en Colab con T4).

## Referencia del dataset

Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020). *Sen1Floods11: a georeferenced
dataset to train and test deep learning flood algorithms for Sentinel-1*. CVPR Workshops.
