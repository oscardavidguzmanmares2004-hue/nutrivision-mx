# Guion de la defensa (12 minutos, presentación individual)

> Base para `docs/defensa.pdf`.

| Min | Diapositiva / acción | Mensaje clave |
|---|---|---|
| 0:00–1:00 | **Problema y usuario** | Registrar comidas es tedioso y las apps no reconocen la comida mexicana. Usuario: persona que lleva su dieta. Todo corre en su dispositivo |
| 1:00–2:30 | **Datos**: distribución, ejemplos, revisión de etiquetas | Food-101, 9 clases, 9,466 imágenes. 6 duplicados de test eliminados con pHash. 5 % de etiquetas incorrectas en la muestra revisada. Desbalance de "otro": pesos de clase |
| 2:30–4:00 | **Arquitectura**: diagrama y tabla comparativa | MobileNetV3-L contra EfficientNet-B0, mismo protocolo: F1 0.859 contra 0.859, pero 7.2 contra 12.6 ms en CPU. A igual F1, gana la latencia |
| 4:00–5:00 | **Curvas de entrenamiento** | Val loss baja sin repuntar; early stopping en el epoch 13; label smoothing y aumento de datos |
| 5:00–7:00 | **Métricas, matriz de confusión, errores y Grad-CAM** | F1 macro 0.859, top-3 0.965. 5 tipos de error. Atajo de acompañamientos: pico de gallo → tacos, chocolate → churros |
| 7:00–8:30 | **Fuera de distribución, robustez, subgrupos y cuantización** | Food-500: 0.767 contra 0.837 (−7 pts). Brechas ≤ 2.6 pts. INT8 con MinMax se rompía (F1 0.06); con percentil, 0.81. Se queda FP32 |
| 8:30–11:00 | **DEMO EN VIVO** | 1) `python app/demo.py --source app/demo.mp4` (opción segura). 2) Webcam con una foto en el celular. 3) Mover el umbral y mostrar "No estoy seguro". 4) Tapar la cámara: "Sin imagen". Señalar los FPS en pantalla |
| 11:00–12:00 | **Limitaciones, ética y costos** | Calorías aproximadas, no es un dispositivo médico, sesgo Tex-Mex. Inferencia local a $0 por predicción. Abstención por umbral |

## Checklist antes de entrar

- [ ] Laptop cargada, con `app/demo.mp4` en la carpeta `app/` y la demo probada.
- [ ] Ventana de la demo abierta una vez antes (la primera carga es más lenta).
- [ ] Video de respaldo descargado localmente, sin depender de internet.
- [ ] 3–4 fotos de platillos en el celular, con brillo al 70 % y sin reflejos.
- [ ] Repasar `docs/plan_trabajo.md` → "Preguntas de código probables".
