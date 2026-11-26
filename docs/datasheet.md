# Datasheet: NutriVisión MX

Formato basado en *Datasheets for Datasets* (Gebru et al., 2021).

## 1. Selección y motivación (B1)

| Campo | Valor |
|---|---|
| Fuente | **Food-101**: Bossard, Guillaumin & Van Gool, ECCV 2014. <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/> |
| Origen de las imágenes | Fotos subidas por usuarios a *foodspotting.com* (sitio ya cerrado), tomadas con celular en restaurantes y casas |
| Licencia / términos de uso | Publicado por ETH Zürich para **investigación**. Las imágenes pertenecen a sus autores y se usan bajo *fair use* académico. **No redistribuimos imágenes**: el script `data/download.py` las descarga de la fuente original |
| Subconjunto usado | 8 clases de platillos mexicanos + clase `otro` (muestra determinista de ~1.6 % de cada uno de los 93 platillos restantes) |
| Número de imágenes | **9,474** en total: 8 × 1,000 platillos mexicanos + 1,474 en `otro` |
| Clases | Burrito (`breakfast_burrito`), Ceviche, Quesadilla (`chicken_quesadilla`), Churros, Guacamole, Huevos rancheros, Nachos, Tacos, Otro platillo |

**¿Por qué este conjunto representa el problema y al usuario?**

- Son **fotos reales de celular** tomadas por comensales: con luz de restaurante, encuadres imperfectos, platos con varios alimentos y desenfoque. Es justo lo que verá la app en manos de un usuario, y no fotos de estudio.
- Es el único dataset grande y público que incluye **varios platillos mexicanos con la misma calidad de etiquetado**. Así podemos medir con rigor sin etiquetar miles de imágenes nosotros.
- La clase `otro` (93 platillos internacionales) representa lo que **no** queremos clasificar como comida mexicana. Sin ella, el modelo forzaría cualquier plato a una de las 8 clases.

**Limitación de representatividad:** foodspotting.com era usado mayormente en EE. UU. y Europa, así que muchas fotos son de **versiones "Tex-Mex"** (por ejemplo, `breakfast_burrito` y `nachos` con queso amarillo). Medimos la caída de desempeño con otro conjunto independiente, ISIA Food-500 (sección 6); fotos tomadas en México quedan como trabajo futuro.

## 2. Proceso de etiquetado (B2)

- Food-101 se etiquetó por **la categoría del platillo que el usuario escribió** al subir la foto en foodspotting.com. Los autores eligieron los 101 platillos más populares.
- **Prueba (250/clase):** revisada manualmente por los autores del dataset.
- **Entrenamiento (750/clase):** los autores **no la limpiaron a propósito**. Advierten que contiene ruido: colores muy saturados y algunas etiquetas incorrectas.
- **Nuestra clase `otro`:** se forma automáticamente a partir de la etiqueta original. El nombre del archivo conserva el platillo de origen (`pizza__12345.jpg`).

## 3. Distribución de clases

| Clase | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| Burrito | 643 | 106 | 250 | 999 |
| Ceviche | 641 | 107 | 250 | 998 |
| Quesadilla | 640 | 108 | 250 | 998 |
| Churros | 640 | 108 | 250 | 998 |
| Guacamole | 643 | 107 | 250 | 1000 |
| Huevos rancheros | 643 | 106 | 250 | 999 |
| Nachos | 643 | 107 | 250 | 1000 |
| Tacos | 643 | 107 | 250 | 1000 |
| Otro platillo | 943 | 157 | 374 | 1474 |
| **Total** | **6,079** | **1,013** | **2,374** | **9,466** |

Hay **desbalance moderado**: `otro` tiene ~1.47 veces más imágenes que cada platillo. Lo atendemos con pesos de clase en la pérdida, aumento de datos y F1 macro como métrica (ver `notebooks/01_exploracion_datos.ipynb`). La figura está en `docs/figuras/distribucion_clases.png`.

## 4. Representación por subgrupo

Food-101 **no contiene datos personales ni metadatos** de autor, lugar o cámara. Los subgrupos relevantes para el usuario son las **condiciones de captura**, que medimos a partir de cada imagen (`src/subgroups.py`):

| Subgrupo | Cómo se mide | Niveles |
|---|---|---|
| Iluminación | Luminancia media | oscura / media / clara (terciles) |
| Temperatura de color | Promedio de R − B (luz cálida de restaurante) | fría / neutra / cálida |
| Nitidez | Varianza del Laplaciano | borrosa / media / nítida |
| Origen del conjunto | Fuente de la imagen | Food-101 (prueba) / ISIA Food-500 (fuera de distribución) |

El desempeño por subgrupo está en `eval/auditoria_subgrupos.ipynb`.

**Sesgo de origen geográfico:** las fotos son mayormente de restaurantes de EE. UU. y Europa. Es una representación sesgada de cómo se ven estos platillos en México: tortilla de maíz contra harina, salsas, presentación en plato desechable o de puesto callejero.

## 5. Particiones sin fuga de información (B3)

Ver `src/make_splits.py`.

1. **Test = partición oficial de Food-101.** No se toca durante el desarrollo; se evalúa **una sola vez** al final. Los hiperparámetros y el epoch se eligen **solo con validación**.
2. **Casi-duplicados:** calculamos un hash perceptual (pHash de 64 bits) para cada imagen. Dos imágenes a distancia de Hamming ≤ 4 se consideran la misma foto (re-subida, re-compresión o recorte leve).
3. **6 imágenes de train eran casi-duplicadas de test** y se eliminaron. Están listadas en `data/splits/removidas_por_fuga.csv`.
4. **Val se separa de train por clúster de duplicados** (`StratifiedGroupKFold`): las copias de una misma foto nunca quedan repartidas entre train y val. Se encontraron 4 clústeres.
5. **Limitación:** Food-101 no publica el ID del usuario que subió cada foto, así que no podemos separar por autor. Un mismo usuario podría tener fotos parecidas (no idénticas) en train y test.

## 6. Prueba fuera de distribución (B6)

La rúbrica permite usar imágenes "capturadas por el equipo (o de otro conjunto)". Como el proyecto es individual y no había comida disponible para fotografiar, se usó **otro conjunto: ISIA Food-500** (Min et al., 2020), que se recolectó de forma independiente a Food-101 (otras búsquedas web, otros fotógrafos, otro año).

- **80 imágenes** de la partición de prueba de Food-500 (réplica pública `advancedcv/Food500Cap` en Hugging Face), descargadas con `python -m src.download_ood`:
  - 10 por cada clase mexicana que existe en Food-500: tacos, nachos, huevos rancheros, quesadilla, burrito y ceviche (60 en total).
  - 20 de "otro platillo": **una imagen de cada una de 20 categorías** no mexicanas elegidas al azar (semilla 42).
  - Churros y guacamole **no existen** en Food-500, así que no se evalúan aquí.
- **Exclusiones:** se descartaron de "otro" categorías que visualmente son un platillo mexicano (por ejemplo `Avocado_sauce` = guacamole, `Mission_burrito`, `Tostada`), para no castigar al modelo por un acierto.
- **Sin fuga:** se compararon por pHash (Hamming ≤ 4) contra todo Food-101; **0 duplicados**.
- Las etiquetas y el origen de cada imagen quedan en `data/ood/labels.csv` (columnas `food500_cat` y `food500_row`).
- **Resultado:** F1 macro 0.767 y exactitud 0.750, contra 0.837 y 0.828 en la prueba de Food-101 con las mismas 7 clases (detalle en `eval/robustez.ipynb` y `docs/robustez_y_despliegue.md`).
- **Limitación:** siguen siendo fotos de internet. No sustituyen fotos tomadas en puestos y casas de México; ese es el siguiente paso recomendado.
- El script `src/prepare_ood.py` queda listo para incorporar fotos propias (elimina EXIF y anota origen y luz).

## 7. Revisión manual de calidad de etiquetas (B5)

- **Muestra:** 40 imágenes de train elegidas al azar (semilla 42).
- **Resultados:** se revisaron una por una. Ver `docs/revision_etiquetas.csv` y `notebooks/03_revision_etiquetas.ipynb`.

| Veredicto | n | % |
|---|---:|---:|
| Correcta | 33 | 82.5 % |
| Dudosa (el platillo puede estar, pero casi no se ve o se parece a otro) | 5 | 12.5 % |
| **Incorrecta** | **2** | **5.0 %** |

- **Errores encontrados:**
  - #18 `breakfast_burrito`: solo se ven papas y salsas.
  - #27 `churros`: es un postre en vaso, sin churros visibles.
- **Corrección:**
  1. Las 2 imágenes incorrectas se **excluyen de train** automáticamente en `make_splits.py`.
  2. Las dudosas se conservan.
  3. Como esto implica que train tiene un ~5 % de ruido en sus etiquetas, se usa **label smoothing de 0.1** en la pérdida.

## 8. Mantenimiento

El conjunto es estático; versión 1.0, noviembre de 2026. Para reproducirlo: `python data/download.py && python -m src.make_splits`.
