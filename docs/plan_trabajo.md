# Plan de trabajo (proyecto individual, uso interno)

> Rúbrica A3: los commits deben tener contenido técnico. Como el proyecto es individual, todos salen de tu cuenta de GitHub. Haz **varios commits por partes**, no uno solo con todo: el historial muestra cómo se construyó el proyecto.

## Commits sugeridos (en este orden)

| # | Mensaje de commit | Archivos |
|---|---|---|
| 1 | `Estructura del repo, licencia y dependencias` | `README.md`, `LICENSE`, `.gitignore`, `requirements.txt` |
| 2 | `Descarga de Food-101 y particiones sin fuga (pHash)` | `src/config.py`, `src/utils.py`, `data/`, `src/make_splits.py` |
| 3 | `EDA y revisión manual de etiquetas` | `notebooks/01_*`, `notebooks/03_*`, `docs/revision_etiquetas.csv`, `docs/datasheet.md` |
| 4 | `Dataset, modelos y entrenamiento` | `src/data.py`, `src/models.py`, `src/train.py`, `notebooks/02_*` |
| 5 | `Evaluación, Grad-CAM y comparación de arquitecturas` | `src/evaluate.py`, `src/gradcam.py`, `src/compare.py`, `docs/interpretabilidad.md` |
| 6 | `Prueba OOD con ISIA Food-500, robustez y subgrupos` | `src/download_ood.py`, `src/prepare_ood.py`, `src/robustness.py`, `src/subgroups.py`, `eval/` |
| 7 | `Exportación ONNX y cuantización INT8` | `src/export.py`, `docs/robustez_y_despliegue.md` |
| 8 | `App en tiempo real` | `app/` (sin el mp4) |
| 9 | `Resultados, model card, costos y documentos` | `runs/` (sin pesos), `docs/`, `MODEL_CARD.md`, `NutriVision_Colab.ipynb` |

## Calendario

| Fecha | Qué |
|---|---|
| **Jue 26 nov** | Repo público + `git tag v1.0-final` + `git push --tags` + enlace en Moodle. Pesos (`best.pt`, `model.onnx`, `model_int8.onnx`) en Releases |
| Antes de la defensa | Grabar el video de respaldo (1–2 min) y poner el enlace en el README |
| Lun 30 nov – Jue 3 dic | Defensa: 12 min + 8 min de preguntas |

## Preguntas de código probables (E1–E3): practícalas

1. `src/make_splits.py`: ¿por qué pHash y por qué separar por clúster de duplicados?
   - Con `HAMMING_THR = 0` no se detectan re-compresiones, así que hay más fuga.
   - Con 20 se agrupan por error fotos distintas.
2. `src/data.py::class_weights`: ¿qué pasa sin pesos? El modelo favorece a "otro" y baja el recall de los platillos.
3. `src/train.py`: ¿por qué un learning rate 10 veces mayor en la cabeza? ¿Qué hace el early stopping? ¿Por qué no se usa el test para elegir el epoch?
4. `src/gradcam.py::__call__`: explicar línea por línea (pesos = gradiente promedio por canal, suma ponderada, ReLU, normalización).
5. `app/demo.py`: ¿qué hace el suavizado EMA? Con `--smooth 0` la etiqueta parpadea; con `--smooth 0.95` tarda en reaccionar.
6. `src/export.py`: ¿qué es la cuantización INT8 estática y por qué necesita datos de calibración? ¿Por qué con MinMax el F1 caía a 0.06 y con percentil sube a 0.81? Los valores atípicos de *hardswish* y SE estiran la escala.
7. ¿Por qué MobileNetV3 y no EfficientNet-B0 si tienen el mismo F1? Por la latencia: 7.2 contra 12.6 ms en CPU.
8. ¿Por qué Food-500 sirve como prueba fuera de distribución? Es otra recolección, sin duplicados con Food-101 (pHash). ¿Por qué no hay churros ni guacamole en esa prueba?
