# Guion — Pregunta 2 (5 minutos)

Ritmo: unas 130 palabras por minuto. Los tiempos son acumulados.

---

## Diapositiva 1 · Portada — 0:00 a 0:15

> Buenas noches. Les presento la Pregunta 2: mapeo de inundaciones con radar
> Sentinel-1 usando una U-Net que implementamos desde cero.

---

## Diapositiva 2 · El problema y los datos — 0:15 a 1:05

> Cuando hay una inundación, las nubes no dejan ver a los satélites ópticos. El
> radar de Sentinel-1 sí atraviesa las nubes, de día y de noche.
>
> Usamos Sen1Floods11: 11 inundaciones en el mundo. Tiene **446 chips con
> etiquetas doradas**, es decir, dibujadas a mano por expertos, y más de **4 mil
> con etiquetas débiles**, generadas automáticamente con un umbral, que son
> baratas pero tienen errores.
>
> Como el dataset no tiene ningún evento peruano, **construimos nuestra propia
> región de prueba en Lima**: una imagen Sentinel-1 del 17 de marzo de 2017 y la
> delineación oficial de Copernicus EMSN038 de los desbordes del Rímac. Son 19
> chips, a la derecha se ve Huachipa.
>
> Un detalle importante: los chips débiles **excluyen Bolivia y las zonas de
> validación y test**, para que no haya fuga de información.

---

## Diapositiva 3 · Método — 1:05 a 1:50

> La red es una U-Net escrita desde cero: un codificador que comprime la imagen
> de 32 a 512 canales, un decodificador que la reconstruye, y conexiones de salto
> que recuperan el detalle. Tiene 7.8 millones de parámetros y la entrada son las
> dos polarizaciones del radar, VV y VH.
>
> Hicimos dos ablaciones: **qué datos usar** —solo dorados, mezclados, o
> preentrenar con débiles y afinar con dorados— y **qué pérdida** —BCE, Dice o la
> combinación.
>
> Para que sea justo, todos entrenan con el mismo número de pasos y todo se
> decide **solo con validación**. Y como el resultado oscilaba bastante entre
> épocas, hicimos **una segunda ronda**: repetimos todo con tres semillas y
> reportamos media y desviación.

---

## Diapositiva 4 · ¿Ayudan las etiquetas débiles? — 1:50 a 2:40

> Este es el resultado central. **Las etiquetas débiles no ayudaron.**
>
> Mezclarlas fue lo peor: cuatro de cada cinco ejemplos que veía la red venían
> del umbral automático, y la red aprendió sus errores. Preentrenar y afinar
> recupera casi todo, pero no supera a entrenar solo con las doradas.
> Calidad le ganó a cantidad.
>
> Mezclar le cuesta unos 0.07 de IoU, y eso se repitió en las tres semillas: es
> real, no azar.
>
> En cambio, la pérdida casi no importa: BCE sola y BCE más Dice empatan. Y aquí
> está el valor de la segunda ronda: con una sola semilla parecía que BCE más Dice
> era mejor; al repetir, la diferencia desapareció.

---

## Diapositiva 5 · Test: aciertos y errores — 2:40 a 3:35

> En el test dorado, el mejor modelo logra un **IoU de 0.655 y un F1 de
> 0.79**. La matriz de confusión muestra que el error dominante es la
> **omisión**: se pierde el 21 % del agua, mientras que menos del 3 % de la
> tierra se marca como agua.
>
> Encontramos tres errores típicos del radar:
> uno, **vegetación inundada**: los tallos rebotan la señal y el agua no se ve
> oscura, así que la red no la detecta;
> dos, **suelo húmedo o liso**, que se ve tan oscuro como el agua y genera falsos
> positivos;
> y tres, **doble rebote urbano**: entre edificios, el agua devuelve un eco muy
> brillante, al revés de lo normal.

---

## Diapositiva 6 · Generalización — 3:35 a 4:35

> ¿Funciona en regiones nuevas? En **Bolivia**, que la red nunca vio, el IoU se
> mantiene alrededor de 0.64 a 0.70: no hay una brecha real.
>
> En **Lima** todos los modelos colapsan, con un IoU de casi cero. Pero no nos quedamos ahí: medimos el
> brillo del radar. En el test y en Bolivia el agua es unos **8 decibeles más
> oscura** que la tierra. En Lima, el agua de la referencia y la tierra **tienen
> el mismo brillo**: las curvas se superponen. Ningún modelo que use solo radar
> podría separarlas.
>
> Nuestras hipótesis: eran huaicos, agua turbulenta con lodo, que no refleja como
> un espejo; hay doble rebote urbano en Huachipa; y la etiqueta de Copernicus
> marca la extensión máxima y es del 21 de marzo, cuando el agua ya bajaba.

---

## Diapositiva 7 · Conclusiones — 4:35 a 5:00

> En resumen: la U-Net funciona bien con radar; las etiquetas débiles no
> ayudaron; y generaliza a otra inundación de llanura, pero no al caso peruano.
> Para usarla en el Perú hace falta afinar con ejemplos locales y sumar datos de
> elevación u ópticos.
>
> Todo el código corre con un solo comando y está en GitHub. Gracias.

---

## Preguntas probables del profesor

**¿Por qué no usaron más etiquetas débiles?**
Usamos 1,000 de 4,384 por tiempo de cómputo. Con las 3 semillas, la diferencia
entre doradas y mezcladas fue consistente, pero con todas las débiles o
ponderándolas por confianza el resultado podría cambiar. Está en limitaciones.

**¿Por qué el IoU por chip es menor que el global?**
El global suma todos los píxeles. El por chip promedia chip por chip, y en chips
con muy poca agua unos pocos píxeles errados bajan mucho el valor.

**¿Cómo saben que el problema en Lima es del dato y no del modelo?**
Por la figura de distribuciones: en Lima el agua etiquetada y la tierra tienen el
mismo brillo de radar (−8.7 frente a −9.0 dB en VV). En los datos de
entrenamiento difieren unos 8 dB.

**¿Por qué ganó BCE si empata con BCE+Dice?**
Por protocolo: el modelo final se elige solo por IoU de validación, y BCE tuvo
la media más alta por 0.002. Pero la diferencia está dentro del ruido; BCE+Dice
es igual de válida y más estable entre semillas (±0.005 frente a ±0.020).

**¿Por qué Bolivia rinde igual o mejor que el test?**
Es una inundación de llanura con agua abierta y calma, el mismo tipo de
fenómeno que domina el entrenamiento, y sus etiquetas siguen el mismo protocolo.

**¿Hicieron el bono de Sentinel-2?**
No. Lo dejamos como trabajo futuro: el óptico ayudaría justo donde falla el
radar, siempre que no haya nubes.

**¿Qué es una semilla?**
El número que fija el azar del entrenamiento (inicialización de pesos y orden de
los datos). Tres semillas = tres entrenamientos independientes del mismo
experimento.
