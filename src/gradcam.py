"""Grad-CAM: ¿en qué parte de la imagen se fija el modelo?

Grad-CAM (Selvaraju et al., 2017) pondera cada canal de la última capa
convolucional por el gradiente promedio de la clase objetivo respecto a ese
canal. El resultado es un mapa de calor de las zonas que más empujan la
predicción hacia esa clase.

Uso:
    python -m src.gradcam --arch mobilenet_v3_large
Genera en docs/figuras/:
    gradcam_aciertos.png   aciertos con alta confianza (uno por clase)
    gradcam_errores.png    errores con alta confianza
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from src.config import CLASSES, DISPLAY, FIG_DIR, IMG_SIZE, ROOT, RUNS_DIR  # noqa: E402
from src.data import get_transforms  # noqa: E402
from src.evaluate import load_model  # noqa: E402
from src.models import gradcam_layer  # noqa: E402


class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.acts = self.grads = None
        layer.register_forward_hook(self._save_acts)

    def _save_acts(self, _module, _inp, out):
        self.acts = out.detach()
        out.register_hook(self._save_grads)  # gradiente de la salida de la capa

    def _save_grads(self, grad):
        self.grads = grad.detach()

    def __call__(self, x, class_idx=None):
        """x: tensor (1,3,H,W). Devuelve (mapa HxW en [0,1], clase, probs)."""
        self.model.zero_grad()
        logits = self.model(x)
        probs = torch.softmax(logits, 1)[0]
        if class_idx is None:
            class_idx = int(probs.argmax())
        logits[0, class_idx].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)      # importancia de cada canal
        cam = F.relu((weights * self.acts).sum(1, keepdim=True))  # combinación lineal + ReLU
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.numpy(), class_idx, probs.detach().numpy()


def display_image(path):
    """La misma imagen que ve el modelo (resize + center crop), sin normalizar."""
    img = Image.open(path).convert("RGB")
    s = int(IMG_SIZE * 1.14)
    img = img.resize((s, int(s * img.height / img.width)) if img.width < img.height
                     else (int(s * img.width / img.height), s))
    left, top = (img.width - IMG_SIZE) // 2, (img.height - IMG_SIZE) // 2
    return np.asarray(img.crop((left, top, left + IMG_SIZE, top + IMG_SIZE)))


def grid(cam_fn, rows_df, path, title):
    tf = get_transforms(train=False)
    n = len(rows_df)
    fig, axes = plt.subplots(2, n, figsize=(n * 2.6, 6.2))
    axes = np.atleast_2d(axes).reshape(2, n)
    for j, (_, r) in enumerate(rows_df.iterrows()):
        x = tf(Image.open(ROOT / r.path).convert("RGB")).unsqueeze(0)
        cam, idx, probs = cam_fn(x)
        img = display_image(ROOT / r.path)
        axes[0, j].imshow(img)
        axes[0, j].set_title(f"Real: {DISPLAY[r.label]}", fontsize=9)
        axes[1, j].imshow(img)
        axes[1, j].imshow(cam, cmap="jet", alpha=0.45)
        axes[1, j].set_title(f"Pred: {DISPLAY[CLASSES[idx]]} ({probs[idx]:.0%})", fontsize=9)
        for a in axes[:, j]:
            a.axis("off")
    fig.suptitle(title)
    plt.tight_layout(h_pad=2.0); plt.savefig(path, dpi=120); plt.close()


def main(args):
    model = load_model(args.arch, "cpu")
    cam = GradCAM(model, gradcam_layer(model, args.arch))
    preds = pd.read_csv(RUNS_DIR / args.arch / f"eval_{args.split}" / "predictions.csv")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ok = preds[preds.label == preds.pred].sort_values("conf", ascending=False)
    ok = ok.groupby("label", sort=False).head(1).set_index("label").reindex(CLASSES).dropna(subset=["path"]).reset_index()
    grid(cam, ok, FIG_DIR / "gradcam_aciertos.png", f"Grad-CAM — aciertos ({args.arch})")

    err = preds[preds.label != preds.pred].sort_values("conf", ascending=False).head(args.n_errors)
    if len(err):
        grid(cam, err, FIG_DIR / "gradcam_errores.png", f"Grad-CAM — errores con alta confianza ({args.arch})")
    print(f"Figuras guardadas en {FIG_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mobilenet_v3_large")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-errors", type=int, default=8)
    main(ap.parse_args())
