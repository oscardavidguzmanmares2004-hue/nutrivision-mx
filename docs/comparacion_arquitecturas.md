# Comparación de arquitecturas

| arquitectura | parámetros (M) | F1 macro val | F1 macro test | exactitud test | top-3 test | epochs (mejor) | min. entrenamiento | latencia PyTorch CPU (ms) | latencia ONNX CPU (ms) |
|---|---|---|---|---|---|---|---|---|---|
| efficientnet_b0 | 4.02 | 0.8393 | 0.8587 | 0.8551 | 0.9663 | 12 (8) | 12.9 | 138.68 | 12.64 |
| mobilenet_v3_large | 4.21 | 0.8353 | 0.8585 | 0.8555 | 0.9646 | 15 (13) | 15.3 | 72.94 | 7.22 |
