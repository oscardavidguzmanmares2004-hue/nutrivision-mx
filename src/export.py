"""Exporta el modelo a ONNX (FP32 e INT8) y mide latencia antes/después.

Uso:
    python -m src.export --arch mobilenet_v3_large

Genera:
    runs/<arch>/model.onnx         ONNX FP32 (el que usa la demo por defecto)
    runs/<arch>/model_int8.onnx    ONNX cuantizado INT8 (cuantización estática QDQ,
                                   calibrada con 200 imágenes de VALIDACIÓN)
    runs/<arch>/export/latencia.csv y latencia.md   tabla antes/después
"""
import argparse
import platform
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import onnx
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                      QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process
from sklearn.metrics import f1_score

from src.config import IMG_SIZE, RUNS_DIR, SEED
from src.data import CLASS_TO_IDX, FoodDataset, get_transforms, split_csv
from src.evaluate import load_model


def export_onnx(model, path):
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    kw = dict(input_names=["input"], output_names=["logits"], opset_version=17)
    try:
        torch.onnx.export(model, x, path, dynamo=False, **kw)
    except Exception as e:  # exportador clásico no disponible -> dynamo
        print(f"Exportador clásico falló ({e.__class__.__name__}); usando dynamo=True")
        torch.onnx.export(model, x, path, dynamo=True, **kw)


class ValReader(CalibrationDataReader):
    def __init__(self, n=200):
        ds = FoodDataset(split_csv("val"), get_transforms(train=False))
        idx = np.random.default_rng(SEED).choice(len(ds), size=min(n, len(ds)), replace=False)
        self.items = iter([{"input": ds[i][0].unsqueeze(0).numpy()} for i in idx])

    def get_next(self):
        return next(self.items, None)


def bench(fn, x, warmup=20, runs=200):
    for _ in range(warmup):
        fn(x)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t) * 1000)
    t = np.array(times)
    return {"lat_media_ms": t.mean(), "lat_p50_ms": np.median(t), "lat_p95_ms": np.percentile(t, 95),
            "fps_modelo": 1000 / t.mean()}


def onnx_f1(path, ds):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    preds = [int(sess.run(None, {"input": ds[i][0].unsqueeze(0).numpy()})[0].argmax())
             for i in range(len(ds))]
    return f1_score(ds.targets, preds, average="macro")


@torch.no_grad()
def torch_f1(model, ds):
    preds = [int(model(ds[i][0].unsqueeze(0)).argmax()) for i in range(len(ds))]
    return f1_score(ds.targets, preds, average="macro")


def to_markdown(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def main(args):
    torch.set_num_threads(args.threads)
    run = RUNS_DIR / args.arch
    out = run / "export"
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(args.arch, "cpu")

    fp32, int8 = run / "model.onnx", run / "model_int8.onnx"
    export_onnx(model, fp32)
    pre = out / "model_pre.onnx"
    quant_pre_process(str(fp32), str(pre))
    # Con calibración MinMax las activaciones atípicas (hardswish / bloques SE) arruinan la escala
    # y el F1 cae a ~0.1. Percentil 99.999 recorta esos extremos; la primera convolución y el
    # clasificador (las capas más sensibles) se dejan en FP32. Ver docs/robustez_y_despliegue.md.
    graph = onnx.load(str(pre)).graph
    convs = [n.name for n in graph.node if n.op_type == "Conv"]
    head = [n.name for n in graph.node if n.op_type in ("Gemm", "MatMul")]
    quantize_static(str(pre), str(int8), ValReader(args.calib_n), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    per_channel=True, calibrate_method=CalibrationMethod.Percentile,
                    nodes_to_exclude=convs[:1] + head)
    pre.unlink()

    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    xn = x.numpy()
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    rows = []

    def add(name, fn, size_mb, f1):
        r = bench(fn, x if name.startswith("PyTorch") else xn)
        rows.append({"variante": name, "tamaño_MB": round(size_mb, 1), "f1_macro_test": round(f1, 4),
                     **{k: round(v, 2) for k, v in r.items()}})
        print(rows[-1])

    test_ds = FoodDataset(split_csv("test"), get_transforms(train=False))
    if args.f1_subset:
        test_ds.df = test_ds.df.sample(args.f1_subset, random_state=SEED).reset_index(drop=True)
        test_ds.targets = test_ds.df["label"].map(CLASS_TO_IDX).to_numpy()

    pt_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    with torch.no_grad():
        add("PyTorch FP32 (eager, CPU)", lambda t: model(t), pt_size, torch_f1(model, test_ds))
    for name, path in (("ONNX Runtime FP32 (CPU)", fp32), ("ONNX Runtime INT8 (CPU)", int8)):
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        add(name, lambda a, s=sess: s.run(None, {"input": a}), path.stat().st_size / 1e6, onnx_f1(path, test_ds))

    df = pd.DataFrame(rows)
    df.to_csv(out / "latencia.csv", index=False)
    hw = f"{platform.processor() or platform.machine()} | {platform.platform()} | {args.threads} hilos"
    (out / "latencia.md").write_text(f"Hardware: {hw}\n\n" + to_markdown(df) + "\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mobilenet_v3_large")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--f1-subset", type=int, default=0,
                    help="evaluar F1 en una muestra del test (0 = test completo)")
    ap.add_argument("--calib-n", type=int, default=200,
                    help="imágenes de validación para calibrar INT8 (bajar si falta RAM)")
    main(ap.parse_args())
