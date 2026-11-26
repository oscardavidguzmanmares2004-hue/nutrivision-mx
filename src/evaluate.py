"""Evaluación cuantitativa de un modelo entrenado.

Uso:
    python -m src.evaluate --arch mobilenet_v3_large --split test
    python -m src.evaluate --arch mobilenet_v3_large --split ood   # fotos del equipo

Métricas: F1 macro (principal), F1/precisión/recall por clase, exactitud,
exactitud top-3 y matriz de confusión. Se guardan en runs/<arch>/eval_<split>/.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report,  # noqa: E402
                             confusion_matrix, f1_score)
from torch.utils.data import DataLoader  # noqa: E402

from src.config import CLASSES, DISPLAY, OOD_DIR, ROOT, RUNS_DIR  # noqa: E402
from src.data import FoodDataset, get_transforms, split_csv  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import get_device, save_json  # noqa: E402


def load_model(arch: str, device="cpu"):
    ckpt = torch.load(RUNS_DIR / arch / "best.pt", map_location="cpu")
    model = build_model(ckpt["arch"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def csv_for(split: str) -> Path:
    return OOD_DIR / "labels.csv" if split == "ood" else split_csv(split)


@torch.no_grad()
def predict(model, csv_path, device, batch=64, transform=None):
    ds = FoodDataset(csv_path, transform or get_transforms(train=False))
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2)
    probs = []
    for x, _ in dl:
        probs.append(torch.softmax(model(x.to(device)), 1).cpu())
    return ds.df.copy(), ds.targets, torch.cat(probs).numpy()


def compute_metrics(y, probs) -> dict:
    pred = probs.argmax(1)
    top3 = np.argsort(-probs, 1)[:, :3]
    present = sorted(set(y))
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro", labels=present)),
        "top3_accuracy": float(np.mean([t in row for t, row in zip(y, top3)])),
        "per_class": classification_report(
            y, pred, labels=list(range(len(CLASSES))), target_names=CLASSES,
            output_dict=True, zero_division=0),
    }


def plot_confusion(y, pred, path, title):
    cm = confusion_matrix(y, pred, labels=range(len(CLASSES)), normalize="true")
    names = [DISPLAY[c] for c in CLASSES]
    plt.figure(figsize=(9, 7.5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=names,
                yticklabels=names, vmin=0, vmax=1, cbar_kws={"label": "Proporción (recall)"})
    plt.xlabel("Predicción"); plt.ylabel("Etiqueta real"); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def plot_errors(df, path, n=15):
    """Galería de los errores con mayor confianza (los más 'graves')."""
    err = df[df.label != df.pred].sort_values("conf", ascending=False).head(n)
    if err.empty:
        return
    cols = 5
    rows = int(np.ceil(len(err) / cols))
    plt.figure(figsize=(cols * 3, rows * 3.3))
    for i, (_, r) in enumerate(err.iterrows()):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(Image.open(ROOT / r.path).convert("RGB"))
        plt.title(f"Real: {DISPLAY[r.label]}\nPred: {DISPLAY[r.pred]} ({r.conf:.0%})", fontsize=9)
        plt.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def main(args):
    device = get_device(args.device)
    model = load_model(args.arch, device)
    df, y, probs = predict(model, csv_for(args.split), device)
    pred = probs.argmax(1)
    df["pred"] = [CLASSES[i] for i in pred]
    df["conf"] = probs.max(1)
    for i, c in enumerate(CLASSES):
        df[f"p_{c}"] = probs[:, i]

    out = RUNS_DIR / args.arch / f"eval_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    m = compute_metrics(y, probs)
    save_json(m, out / "metrics.json")
    df.to_csv(out / "predictions.csv", index=False)
    plot_confusion(y, pred, out / "confusion.png", f"{args.arch} — {args.split}")
    plot_errors(df, out / "errores.png")
    print(f"{args.arch} [{args.split}] n={m['n']}  F1 macro={m['f1_macro']:.4f}  "
          f"acc={m['accuracy']:.4f}  top3={m['top3_accuracy']:.4f}")
    print(pd.DataFrame(m["per_class"]).T.loc[CLASSES, ["precision", "recall", "f1-score", "support"]].round(3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mobilenet_v3_large")
    ap.add_argument("--split", default="test", choices=["val", "test", "ood"])
    ap.add_argument("--device", default="auto")
    main(ap.parse_args())
