Hardware: arm | macOS-26.6.2-arm64-arm-64bit | 4 hilos

| variante | tamaño_MB | f1_macro_test | lat_media_ms | lat_p50_ms | lat_p95_ms | fps_modelo |
|---|---|---|---|---|---|---|
| PyTorch FP32 (eager, CPU) | 16.9 | 0.8585 | 72.94 | 70.73 | 93.5 | 13.71 |
| ONNX Runtime FP32 (CPU) | 16.8 | 0.8585 | 7.22 | 6.95 | 8.42 | 138.57 |
| ONNX Runtime INT8 (CPU) | 8.4 | 0.8076 | 4.36 | 4.21 | 4.71 | 229.17 |
