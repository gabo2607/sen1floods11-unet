# Guía de exposición — Pregunta 2 (primera persona, 8 minutos, 10 diapositivas)

Cada diapositiva responde a un entregable del enunciado; el número aparece arriba a la derecha.
Entre corchetes van las indicaciones: qué señalar o dónde hacer una pausa. Los tiempos son acumulados.

---

## 1 · Portada (0:00 – 0:15)

> Buenas noches, profesor, compañeros. Soy Gabriel Nizama, del grupo 4, y les voy a presentar la
> Pregunta 2: mapeo de inundaciones con el radar Sentinel-1, usando una U-Net que implementamos desde
> cero.

---

## 2 · El problema y los datos (0:15 – 1:00)

> Cuando hay una inundación casi siempre hay nubes, y los satélites ópticos no ven el suelo. El radar
> de Sentinel-1 atraviesa las nubes, de día y de noche. El agua calma refleja la señal lejos del sensor
> y por eso se ve **oscura**. [Señalo el chip] Esa es la firma que aprende la red.
>
> [Señalo las cifras] Trabajé con Sen1Floods11: once inundaciones en el mundo, con 446 chips
> **dorados**, dibujados a mano por expertos, y más de cuatro mil **débiles**, generados por un umbral
> automático: muchos y gratis, pero con errores.
>
> [Señalo la tabla] Además de las particiones oficiales, evaluamos en Bolivia, que la red nunca ve, y en
> 19 chips de Lima que construimos nosotros. Y un cuidado importante: de las débiles excluí Bolivia y las
> zonas de validación y test, para no tener fuga de información.

---

## 3 · Entregable 1: U-Net propia (1:00 – 1:45)

> El primer entregable pide una U-Net implementada a mano, con VV y VH como entrada.
>
> [Señalo el diagrama] La parte que baja, el codificador, comprime la imagen de 32 a 512 canales para
> entender el contexto. La parte que sube, el decodificador, la reconstruye para clasificar cada píxel.
> Y estas conexiones de salto le devuelven el detalle fino que se pierde al comprimir.
>
> Tiene 7.8 millones de parámetros, no usa ninguna librería de segmentación, y cada entrenamiento de
> 40 épocas toma unos seis minutos en una GPU de Colab.

---

## 4 · Entregable 2: diseño de la ablación (1:45 – 2:30)

> El segundo entregable pide dos comparaciones.
>
> [Señalo la tabla] Primero, **qué datos usar**: solo doradas, doradas mezcladas con débiles, o primero
> débiles y luego afinar con doradas. Segundo, **qué pérdida**: BCE, que mide el error píxel a píxel;
> Dice, que mide cuánto se superpone el agua predicha con la real; o la combinación.
>
> Para que sea justo, todos entrenan la misma cantidad y todo se decide solo con validación.
>
> [Señalo "Dos rondas"] Y algo importante: en la primera ronda vi que el resultado oscilaba mucho entre
> épocas, así que hicimos una segunda ronda: repetimos los cinco experimentos con tres semillas
> distintas y reportamos el promedio y la desviación.

---

## 5 · Entregable 2: resultados (2:30 – 3:35)

> Este es el resultado central. [Pausa] **Las etiquetas débiles no ayudaron.**
>
> [Señalo la fila naranja] Mezclarlas baja el IoU unos 0.065, y pasó igual en las tres semillas: no es
> azar. La razón es que cuatro de cada cinco ejemplos que veía la red venían del umbral automático, y
> aprendió sus errores. Preentrenar y afinar recupera casi todo, pero no supera a las doradas solas.
> **Calidad le ganó a cantidad.**
>
> [Señalo las filas de BCE y Dice] En la pérdida, BCE y BCE más Dice empatan. Y aquí se ve el valor de la
> segunda ronda: con una sola semilla parecía que BCE más Dice ganaba; al repetir, la diferencia
> desapareció. Sin esa ronda habríamos sacado una conclusión falsa.

---

## 6 · Entregable 3: evaluación en distribución (3:35 – 4:25)

> El tercer entregable pide las métricas en el test y la matriz de confusión comentada.
>
> [Señalo las cifras] El modelo final logra un IoU de 0.655, F1 de 0.79, precisión de 0.80 y recall de
> 0.79.
>
> [Señalo la matriz] La matriz muestra que su error típico es **quedarse corto**: no ve el 21 % del
> agua, pero casi nunca inventa agua, menos del 3 % de la tierra. El IoU por chip es menor que el global
> porque en chips con poca agua unos pocos errores pesan mucho. Y depende del tipo de inundación: 0.87
> en Nigeria, con agua abierta, y 0.20 en Pakistán, con agua somera entre cultivos.

---

## 7 · Entregable 4: construcción de Lima (4:25 – 5:10)

> El cuarto entregable pide construir nuestra propia región de prueba en el Perú.
>
> [Señalo las tarjetas] Usé la imagen de radar de Sentinel-1 del **17 de marzo de 2017**, la más cercana
> al evento, y la delineación oficial de Copernicus **EMSN038**. La llevé al mismo formato que
> Sen1Floods11 —decibeles, 10 metros, chips de 512— y rasterizé los polígonos sobre la misma grilla.
>
> [Señalo la imagen] Aquí se ve Huachipa y el río Rímac, con la inundación de Copernicus en amarillo.
> Resultado: 19 chips. Y evalué también Bolivia como región de respaldo.

---

## 8 · Entregable 4: brecha e hipótesis (5:10 – 6:20)

> [Señalo las cifras] En Bolivia, que la red nunca vio, el resultado se mantiene: la brecha es pequeña.
> En Lima, en cambio, todos los modelos colapsan, con un IoU de casi cero.
>
> No me quedé con ese número. [Señalo las curvas] Medí el brillo del radar: lo azul es el agua, lo
> naranja la tierra. En el test y en Bolivia están separadas, el agua es unos 8 decibeles más oscura. En
> Lima, las dos curvas están **una encima de la otra**: lo que Copernicus marcó como agua tiene el mismo
> brillo que la tierra.
>
> [Señalo las hipótesis] ¿Por qué? Uno: fueron huaicos, agua turbulenta con lodo, que no refleja como
> espejo. Dos: doble rebote urbano en Huachipa. Tres: la etiqueta marca la extensión **máxima** del 21 de
> marzo, con el agua ya bajando, y el radar es del 17. Cuatro: son franjas muy angostas para 10 metros.
>
> **Conclusión: en Lima no falla la red, falla la correspondencia entre lo que ve el radar y la
> etiqueta.**

---

## 9 · Entregable 5: errores del radar (6:20 – 7:15)

> El quinto entregable pide al menos tres tipos de error propios del radar. [Señalo la leyenda] Azul es
> agua que no vio; naranja, agua que inventó.
>
> Uno: **vegetación inundada**. En estos arrozales de Pakistán las plantas sobresalen del agua, rebotan
> la señal, y el agua no se ve oscura: la red no la detecta.
>
> Dos: **suelo húmedo o liso**. Un campo mojado se ve tan oscuro como el agua, y la red lo marca como
> agua.
>
> Tres: **doble rebote urbano**. En una ciudad inundada la señal rebota en el agua y luego en la pared, y
> vuelve muy brillante: al revés de lo normal.

---

## 10 · Bono y conclusiones (7:15 – 8:00)

> El bono pedía sumar bandas ópticas de Sentinel-2. [Señalo las cifras] En los mismos chips sin nubes, el
> IoU sube de **0.66 a 0.87**, porque en infrarrojo el agua absorbe casi todo. Pero el óptico solo existe
> sin nubes, justo lo que falta en una tormenta: por eso el radar sigue siendo la base.
>
> [Señalo las conclusiones] En resumen: la U-Net funciona bien con radar; las etiquetas débiles no
> ayudaron; repetir con tres semillas separó lo real del azar; y el modelo generaliza a Bolivia pero no a
> Lima, que necesita ejemplos locales.
>
> Todo está en GitHub, con el modelo entrenado y un comando que reproduce estos números en tres
> minutos. Muchas gracias.

---

## Recordatorios

- Las diapositivas **5 y 8** son las más importantes: no las apures.
- Si vas tarde: en la 3 no expliques el entrenamiento y en la 9 nombra los tres errores sin detalle.
- Si el profesor pregunta por un punto del enunciado, ve directo a la diapositiva de ese entregable
  (el número está arriba a la derecha).

## Preguntas probables

**¿Por qué no usaron más etiquetas débiles o una mezcla 50/50?** Por tiempo usamos 1,000 de 4,384 y
mezcla uniforme. La diferencia fue consistente en 3 semillas, pero con más datos débiles o ponderándolos
por confianza podría cambiar. Está en las limitaciones del informe.

**¿Por qué ganó BCE si empata con BCE+Dice?** Por protocolo: se elige solo por validación y BCE tuvo la
media más alta por 0.002. La diferencia está dentro del ruido; BCE+Dice es igual de válida y más estable.

**¿Cómo saben que en Lima falla el dato y no el modelo?** Por la distribución de brillo: en Lima el agua
etiquetada y la tierra tienen el mismo brillo (−8.7 frente a −9.0 dB). Además el modelo casi no marca
falsos positivos allí: no está confundido, simplemente no hay firma de agua.

**¿En Lima agregaron ríos y lagunas a la etiqueta?** Sí, porque Sen1Floods11 también marca el agua
permanente; era para comparar con el mismo criterio. Aun así el IoU es casi cero.

**¿Por qué Bolivia rinde igual o mejor que el test?** Es una inundación de llanura con agua abierta, el
mismo fenómeno que domina el entrenamiento, y sus etiquetas siguen el mismo protocolo.

**¿Qué es una semilla?** El número que fija el azar del entrenamiento: pesos iniciales y orden de los
datos. Tres semillas son tres entrenamientos independientes del mismo experimento.

**¿Cómo se puede verificar?** En el GitHub, `python reproducir.py` descarga el modelo y reproduce los
IoU del informe en unos 3 minutos; `python run_all.py` reentrena todo.
