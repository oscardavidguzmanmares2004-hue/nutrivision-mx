# Informe ético: NutriVisión MX

## 1. Usos previstos y no previstos

| Previsto ✅ | No previsto ❌ |
|---|---|
| Ayuda para **registrar comidas** en un diario de alimentación personal | Diagnóstico, tratamiento o plan nutricional clínico |
| **Estimación orientativa** de calorías de 8 platillos comunes | Cálculo exacto de calorías, macronutrientes o porciones |
| Prototipo académico y demostración de visión por computadora | Decisiones de salud de personas con diabetes, trastornos de la conducta alimentaria (TCA) u otras condiciones que requieren conteo preciso |
| Base para integrarse en apps de bienestar, siempre con confirmación del usuario | Vigilancia de lo que comen otras personas (empleados, estudiantes, pacientes) sin su consentimiento |

## 2. Riesgos identificados

1. **Daño por estimaciones erróneas.** Una quesadilla puede tener entre 300 y 900 kcal según su tamaño y relleno. Mostrar un solo número puede dar una falsa precisión.
2. **Trastornos de la conducta alimentaria.** Las apps de conteo de calorías pueden reforzar conductas obsesivas en personas vulnerables.
3. **Sesgo cultural y geográfico.** El dataset se formó con fotos de restaurantes de EE. UU. y Europa (versiones Tex-Mex). El modelo puede funcionar peor con la comida como se sirve en México: puestos callejeros, tortilla de maíz, plato desechable. Esto afectaría justo al usuario objetivo.
4. **Sesgo por condiciones de captura.** Las fotos con poca luz o luz cálida (taquerías, cenas) pueden clasificarse peor. Esas condiciones se asocian con ciertos contextos socioeconómicos y horarios. Ver la auditoría de subgrupos.
5. **Privacidad.** Las fotos de comida pueden revelar ubicación (EXIF/GPS), hábitos y horarios. Una app que sube fotos a un servidor acumula datos sensibles de salud.
6. **Sobreconfianza.** Un clasificador siempre elige una clase aunque la foto no sea comida.

## 3. Sesgos medidos

Ver `eval/auditoria_subgrupos.ipynb` (prueba de Food-101) y `eval/robustez.ipynb` (prueba fuera de distribución con ISIA Food-500 y perturbaciones). Los resultados numéricos están resumidos en `MODEL_CARD.md`, sección "Desempeño por subgrupo".

## 4. Mitigaciones implementadas

| # | Mitigación | Riesgo que atiende | Dónde |
|---|---|---|---|
| 1 | **Umbral de confianza ajustable con abstención** ("No estoy seguro, acerca el platillo") en lugar de forzar una predicción | 1, 4, 6 | `app/demo.py` |
| 2 | Clase **"otro platillo"** para no etiquetar como mexicana cualquier comida | 6 | `src/config.py`, entrenamiento |
| 3 | **Aumento de datos de brillo, color y desenfoque** en entrenamiento | 4 | `src/data.py` |
| 4 | Calorías mostradas como **"~X kcal por porción (aprox.)"**, nunca como valor exacto | 1 | `app/demo.py` |
| 5 | **Inferencia local** (ONNX de 17 MB FP32 u 8 MB INT8 en CPU): las fotos no salen del dispositivo | 5 | `src/export.py`, `app/demo.py` |
| 6 | **Eliminación de EXIF** (GPS, modelo de teléfono) para cualquier foto propia que se agregue | 5 | `src/prepare_ood.py` |
| 7 | Evaluación **fuera de distribución con otro conjunto (ISIA Food-500)** para medir la caída de desempeño en vez de suponerla | 3 | `eval/robustez.ipynb` |

## 5. Recomendaciones para un despliegue real

- Pedir **confirmación del usuario** antes de guardar el registro (humano en el ciclo).
- Agregar un aviso: "no apto para personas en tratamiento de TCA o diabetes sin supervisión".
- Re-entrenar con **fotos tomadas en México** (puestos, mercados, comida casera) y volver a auditar por subgrupos.
- No usar el sistema para monitorear a terceros.

## 6. Datos personales y consentimiento

- Food-101 no contiene personas identificables como sujeto de la foto.
- La prueba fuera de distribución usa imágenes públicas de ISIA Food-500 (uso académico); son fotos de platillos, sin personas como sujeto.
- No se recolectaron fotos propias ni datos de usuarios. Si en el futuro se agregan, `src/prepare_ood.py` elimina el EXIF y se recortan manos o rostros antes de publicarlas.
