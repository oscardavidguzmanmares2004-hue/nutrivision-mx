Hardware: arm | macOS-26.6.2-arm64-arm-64bit | 4 hilos

| variante | tamaño_MB | f1_macro_test | lat_media_ms | lat_p50_ms | lat_p95_ms | fps_modelo |
|---|---|---|---|---|---|---|
| PyTorch FP32 (eager, CPU) | 16.1 | 0.8715 | 138.68 | 132.02 | 175.36 | 7.21 |
| ONNX Runtime FP32 (CPU) | 16.1 | 0.8715 | 12.64 | 12.33 | 14.45 | 79.09 |
| ONNX Runtime INT8 (CPU) | 4.9 | 0.7399 | 8.3 | 8.31 | 9.27 | 120.47 |
