"""Tabla comparativa de arquitecturas (justificación de la elección).

Uso:
    python -m src.compare
Lee runs/*/train_info.json, eval_test/metrics.json y export/latencia.csv y
escribe docs/comparacion_arquitecturas.md.
"""
import pandas as pd

from src.config import ROOT, RUNS_DIR
from src.export import to_markdown
from src.utils import load_json


def main():
    rows = []
    for run in sorted(RUNS_DIR.iterdir()):
        info_p, met_p = run / "train_info.json", run / "eval_test" / "metrics.json"
        if not (info_p.exists() and met_p.exists()):
            continue
        info, met = load_json(info_p), load_json(met_p)
        row = {
            "arquitectura": run.name,
            "parámetros (M)": round(info["params"] / 1e6, 2),
            "F1 macro val": round(info["best_val_f1_macro"], 4),
            "F1 macro test": round(met["f1_macro"], 4),
            "exactitud test": round(met["accuracy"], 4),
            "top-3 test": round(met["top3_accuracy"], 4),
            "epochs (mejor)": f'{info["epochs_run"]} ({info["best_epoch"]})',
            "min. entrenamiento": round(info["train_minutes"], 1),
        }
        lat = run / "export" / "latencia.csv"
        if lat.exists():
            l = pd.read_csv(lat).set_index("variante")
            row["latencia PyTorch CPU (ms)"] = l.loc["PyTorch FP32 (eager, CPU)", "lat_media_ms"]
            row["latencia ONNX CPU (ms)"] = l.loc["ONNX Runtime FP32 (CPU)", "lat_media_ms"]
        rows.append(row)
    df = pd.DataFrame(rows)
    out = ROOT / "docs" / "comparacion_arquitecturas.md"
    out.write_text("# Comparación de arquitecturas\n\n" + to_markdown(df) + "\n")
    print(df.to_string(index=False))
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()
