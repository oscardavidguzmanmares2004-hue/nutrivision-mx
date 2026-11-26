# Model card: NutriVisión MX

Formato basado en Mitchell et al. (2019), *Model Cards for Model Reporting*.

## 1. Detalles del modelo

| | |
|---|---|
| **Nombre** | NutriVisión MX v1.0 |
| **Tarea** | Clasificación de imágenes en 9 clases: 8 platillos mexicanos + "otro platillo" |
| **Arquitectura** | MobileNetV3-Large (4.2 M de parámetros), preentrenada en ImageNet, con fine-tuning completo y una cabeza nueva de 9 salidas |
| **Alternativa evaluada** | EfficientNet-B0 con el mismo protocolo (ver `docs/comparacion_arquitecturas.md`) |
| **Entrenamiento** | AdamW (lr 3e-4; ×10 en la cabeza), decaimiento coseno, batch 64, pérdida ponderada por clase con label smoothing de 0.1, early stopping sobre F1 macro de validación (paciencia 4), semilla 42 |
| **Hardware de entrenamiento** | Google Colab, GPU NVIDIA T4; 15.3 min (15 epochs, mejor en el 13) |
| **Formato de despliegue** | ONNX FP32 (16.8 MB) en ONNX Runtime CPU; variante INT8 (8.4 MB) opcional |
| **Fecha y versión** | Noviembre de 2026, v1.0 |
| **Licencia** | Código MIT. Pesos solo para uso académico, igual que Food-101 |
| **Autor** | Oscar David Guzmán Mares (219872688) |

## 2. Uso previsto

- **Uso principal:** ayudar a una persona a **registrar lo que come**. Apunta la cámara a su plato y el sistema sugiere el platillo y unas calorías aproximadas por porción, todo en su propio dispositivo.
- **Usuarios:** personas que llevan control de su dieta; apps de bienestar como componente de sugerencia.
- **Fuera de alcance (no usar para):**
  - Decisiones médicas: dosis de insulina, tratamiento de diabetes o trastornos de la conducta alimentaria.
  - Calcular calorías exactas. El valor es un promedio por porción típica y **no mide la cantidad servida**.
  - Platillos fuera de las 8 clases. El modelo los manda a "otro" o se equivoca.
  - Vigilar lo que comen otras personas.

## 3. Factores relevantes

- **Condiciones de captura:** iluminación, temperatura de color y nitidez (auditadas, sección 6).
- **Origen y estilo del platillo:** Food-101 está dominado por versiones Tex-Mex de restaurantes de EE. UU. Los platillos caseros o de puesto en México se ven distintos.
- **Medio:** foto de comida real o foto mostrada en la pantalla de un celular (moiré y reflejos).
- **Contexto en el plato:** acompañamientos como pico de gallo, lechuga o totopos influyen en la predicción (ver `docs/interpretabilidad.md`).

## 4. Métricas

- **Principal: F1 macro.** Pesa igual a todas las clases, aunque "otro" tenga 1.5 veces más imágenes.
- **Secundarias:** exactitud, exactitud top-3, matriz de confusión, y cobertura y exactitud con abstención por umbral.
- **Selección del modelo:** solo con validación. **Test se evaluó una sola vez**, al final.

## 5. Datos

- **Entrenamiento y validación:** Food-101, subconjunto de 9 clases.
  - Train: 6,079 imágenes. Val: 1,013.
  - Sin casi-duplicados entre particiones (pHash).
  - Se excluyeron 2 imágenes con etiqueta incorrecta. Detalle en `docs/datasheet.md`.
- **Prueba:** partición oficial de Food-101, 2,374 imágenes.
- **Fuera de distribución:** 80 imágenes de ISIA Food-500, otro conjunto independiente.
- **Preprocesamiento:** recorte central cuadrado, 224×224, normalización ImageNet.
- **Aumento en train:** recorte aleatorio (escala 0.5–1), volteo, rotación de 15°, `ColorJitter`, desenfoque y borrado aleatorio.

## 6. Análisis cuantitativo

### 6.1 Prueba (Food-101, n = 2,374)

| Métrica | MobileNetV3-L (elegido) | EfficientNet-B0 |
|---|---:|---:|
| F1 macro | **0.859** | 0.859 |
| Exactitud | 0.856 | 0.855 |
| Top-3 | 0.965 | 0.966 |
| Latencia ONNX CPU (M4) | **7.2 ms** | 12.6 ms |

F1 por clase (MobileNetV3-L):

| Burrito | Ceviche | Quesadilla | Churros | Guacamole | Huevos rancheros | Nachos | Tacos | Otro |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.834 | 0.869 | 0.861 | 0.953 | 0.946 | 0.809 | 0.860 | 0.802 | 0.792 |

### 6.2 Desempeño por subgrupo

| Atributo | F1 macro por tercil | Brecha |
|---|---|---:|
| Iluminación (oscura / media / clara) | 0.850 / 0.856 / 0.862 | 1.2 pts |
| Temperatura (fría / neutra / cálida) | 0.843 / 0.851 / 0.869 | 2.6 pts |
| Nitidez (borrosa / media / nítida) | 0.857 / 0.861 / 0.850 | 1.1 pts |

### 6.3 Fuera de distribución y robustez

- **ISIA Food-500, 7 clases:** F1 macro **0.767** contra 0.837 en Food-101 con las mismas clases (−7 pts). Quesadilla y "otro" son las clases más frágiles.
- **Perturbaciones:**
  - Robusto a brillo: −2 pts entre 0.4× y 1.4×.
  - Frágil a ruido σ ≥ 20 y a desenfoque σ ≥ 3 (más de −10 pts).
  - Foto en pantalla de celular: −0.7 pts en nivel leve y −24 pts en el nivel más fuerte.
- Detalle en `docs/robustez_y_despliegue.md`.

### 6.4 Umbral de confianza

| Umbral | Cobertura | Exactitud en las que responde |
|---:|---:|---:|
| 0.40 | 93.7 % | 88.3 % |
| **0.55 (default)** | 82.4 % | 92.5 % |
| 0.70 | 69.9 % | 96.0 % |

### 6.5 Cuantización

| Variante | Tamaño | F1 macro test | Latencia (M4, CPU) |
|---|---:|---:|---:|
| ONNX FP32 | 16.8 MB | 0.859 | 7.2 ms |
| ONNX INT8 (percentil, primera y última capa en FP32) | 8.4 MB | 0.808 | 4.4 ms |

## 7. Consideraciones éticas

- **Salud:** un error en las calorías puede afectar a personas con diabetes o con trastornos de la conducta alimentaria. Por eso la app muestra siempre "~X kcal (aprox.)" y se abstiene con confianza baja.
- **Sesgo cultural:** el modelo aprende la versión Tex-Mex de los platillos. Puede fallar más con comida casera mexicana, que justamente es la de su usuario objetivo.
- **Privacidad:** la inferencia es local, así que la foto no sale del dispositivo.
- Detalle en `docs/informe_etico.pdf`.

## 8. Limitaciones conocidas

1. **Solo 8 platillos.** No reconoce enchiladas, pozole, chilaquiles, tamales ni muchos más.
2. **No estima la porción.** Las calorías son un promedio por platillo, no de la foto.
3. **Atajos de contexto:** los acompañamientos (pico de gallo, totopos, taza de chocolate) influyen en la predicción.
4. **31 errores con confianza ≥ 80 %** en la prueba (1.3 %). El umbral no los atrapa.
5. **Datos de internet de EE. UU. y Europa.** La prueba fuera de distribución también es de internet; falta validar con fotos tomadas en México.
6. **Food-101 tiene ruido de etiquetas** (~5 % en la muestra revisada), así que las métricas de prueba tienen esa incertidumbre.

## 9. Recomendaciones

- Mostrar el top-3 y pedir **confirmación del usuario** antes de guardar un registro.
- Re-entrenar con fotos de México y volver a auditar por subgrupos.
- Calibrar las probabilidades (*temperature scaling*) antes de usar el umbral como garantía.
- Para equipos lentos, usar INT8 sabiendo que cuesta unos 5 puntos de F1.
