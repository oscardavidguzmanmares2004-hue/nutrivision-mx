# Caso de sector: salud y nutrición

## 1. Problema

México tiene una de las prevalencias de sobrepeso y obesidad más altas del mundo: según la ENSANUT, alrededor de **3 de cada 4 adultos** viven con sobrepeso u obesidad. Llevar un registro de lo que se come es una de las intervenciones con más evidencia para bajar de peso, pero la gente lo abandona porque **anotar cada comida a mano es tedioso**.

Las apps de conteo de calorías que reconocen fotos se entrenaron sobre todo con comida estadounidense y europea. Por eso es probable que **platillos mexicanos comunes** (tacos, quesadillas, huevos rancheros, ceviche) se reconozcan mal o se confundan con equivalentes extranjeros, y que el usuario termine capturando los datos a mano. Este proyecto construye y mide un reconocedor específico para esos platillos.

## 2. Usuario

| Usuario | Necesidad | Cómo lo usa |
|---|---|---|
| **Persona que lleva control de su dieta** (principal) | Registrar comidas rápido, sin buscar en una base de datos | Apunta la cámara al plato, confirma el platillo y la porción aproximada |
| Nutriólogo/a | Revisar lo que comen sus pacientes entre consultas | Recibe el diario fotográfico ya clasificado |
| App de bienestar (integrador) | Un módulo de reconocimiento que funcione con comida local | Integra el modelo ONNX (17 MB, u 8 MB en INT8) en su app |

## 3. Qué hace el sistema

1. La cámara captura cuadros continuamente.
2. Un clasificador (MobileNetV3, con fine-tuning) identifica uno de **8 platillos mexicanos** o los marca como **"otro platillo"**.
3. Si la confianza supera el umbral, muestra el platillo y una **estimación de calorías por porción típica**. Si no la supera, pide acercar el plato en lugar de adivinar.

## 4. Qué NO hace (alcance)

- No mide la porción real ni los ingredientes: las calorías son **promedios de referencia**.
- No es un dispositivo médico ni sustituye la orientación de un profesional de la salud.
- Reconoce 8 platillos; cualquier otro cae en "otro platillo".

## 5. Costo de los errores

| Error | Consecuencia | Gravedad |
|---|---|---|
| Platillo mexicano → otro platillo mexicano (p. ej., quesadilla → burrito) | Calorías mal estimadas (±100–200 kcal) | Media |
| Comida no mexicana → platillo mexicano | Registro falso en el diario | Media |
| Platillo → "no estoy seguro" | El usuario lo captura a mano (solo molestia) | Baja |

Por eso preferimos **abstenernos con confianza baja** (umbral ajustable) antes que dar una cifra equivocada con aparente seguridad, y reportamos **F1 macro** en lugar de solo exactitud.

## 6. Valor esperado

- Registrar una comida pasa de ~1 minuto (buscar y capturar) a ~3 segundos.
- El modelo es lo bastante ligero (17 MB y 7 ms por imagen en el CPU de una laptop) para correr **en el teléfono, sin enviar fotos a un servidor**, lo que también protege la privacidad.
- **Resultado obtenido:**
  - F1 macro de **0.859** en la prueba de Food-101 y 0.767 en imágenes de otro conjunto (ISIA Food-500).
  - Con el umbral por defecto, la app responde en 82 % de las fotos y acierta 92.5 % de las veces que responde.
  - Detalle en el README y en `MODEL_CARD.md`.
