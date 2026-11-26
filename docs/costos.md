# Costos del proyecto

Las cifras en dinero son **estimaciones** con precios públicos aproximados de noviembre de 2026. Los tiempos son los registrados en `runs/*/train_info.json` y `runs/*/export/latencia.md`.

## 1. Cómputo de entrenamiento

| Concepto | Hardware | Tiempo | Costo real | Costo equivalente en la nube* |
|---|---|---:|---:|---:|
| Entrenamiento MobileNetV3-Large (15 epochs) | Colab, GPU T4 | 15.3 min | $0 (Colab gratuito) | ~0.10 USD |
| Entrenamiento EfficientNet-B0 (12 epochs) | Colab, GPU T4 | 12.9 min | $0 | ~0.08 USD |
| Descarga, particiones, evaluación, Grad-CAM, exportación, notebooks | Colab, GPU T4 y CPU | ~30 min | $0 | ~0.20 USD |
| Experimentos previos y de cuantización | MacBook M4 (propia) | ~3 h | Solo electricidad, < 0.5 kWh | — |
| **Total** | | **~1 h de GPU** | **$0** | **< 0.50 USD** |

\* Con una GPU T4 rentada a ~0.35–0.40 USD por hora.

El costo de cómputo es bajo porque se hace **fine-tuning** de un modelo preentrenado: no se entrena desde cero, y bastan ~6,000 imágenes y 15 minutos por arquitectura.

## 2. Datos

| Concepto | Costo |
|---|---|
| Food-101 (uso académico) | $0. Se descarga como *stream*: lee ~5 GB por red y guarda ~300 MB en disco |
| ISIA Food-500, 80 imágenes por API pública | $0 |
| Etiquetado | $0. Se usan las etiquetas del dataset |
| Revisión manual de 40 etiquetas | ~1 h de trabajo |
| Etiquetar un conjunto propio en México (trabajo futuro) | ~1,000 fotos × 10 s ≈ 3 h por persona, más las fotos |

## 3. Despliegue e inferencia

| Opción | Costo por predicción | Latencia | Privacidad |
|---|---|---|---|
| **Elegida: ONNX en el dispositivo (CPU)** | **$0**. Unos 7 ms de CPU, del orden de 10⁻⁵ Wh | 7 ms (M4) | La foto no sale del dispositivo |
| Servidor propio con GPU | ~0.35 USD/h fijos aunque no haya usuarios, más la red | 50–200 ms con la red | La foto viaja al servidor |
| API comercial de visión o un LLM multimodal | ~0.001–0.01 USD por imagen | 0.5–3 s | Datos de salud en un tercero |

- **Ejemplo:** con 1,000 usuarios que registran 3 comidas diarias, son ~90,000 imágenes al mes.
  - En el dispositivo cuesta $0.
  - Con una API costaría ~90–900 USD al mes.
- **Tamaño de la app:**
  - FP32: 16.8 MB, aceptable para una app móvil.
  - INT8: 8.4 MB, a cambio de unos 5 puntos de F1.

## 4. Costo de los errores

| Error | Consecuencia | Costo para el usuario | Mitigación |
|---|---|---|---|
| Platillo equivocado con confianza alta (1.3 % del test) | Registro calórico erróneo, por ejemplo nachos (650 kcal) anotados como guacamole (230 kcal) | Hasta ~400 kcal de error en un registro | Top-3 y confirmación del usuario |
| Abstención ("No estoy seguro", ~18 % de las fotos con umbral 0.55) | El usuario acerca la cámara o registra a mano | Unos segundos de fricción | Umbral ajustable |
| Platillo mexicano predicho como "otro" | No se sugiere el registro | Fricción | Registro manual |

En esta aplicación es preferible **abstenerse** que equivocarse con seguridad. Por eso el umbral por defecto (0.55) sacrifica cobertura (82 %) para subir la exactitud de las respuestas a 92.5 %.
