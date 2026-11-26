"""Crea un video de entrada para la demo (plan seguro para la defensa).

Arma un video a 30 FPS con fotos de PRUEBA (nunca vistas en entrenamiento)
y, si existen, fotos propias del equipo (data/ood/). Incluye movimiento suave
(zoom/paneo) y transiciones, de modo que el modelo procesa un flujo continuo
de cuadros distintos, igual que con una cámara.

Uso:
    python app/make_demo_video.py --out app/demo.mp4 --per-class 3
    python app/demo.py --source app/demo.mp4
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CLASSES, OOD_DIR, ROOT, SEED, SPLITS_DIR  # noqa: E402

W, H, FPS = 960, 720, 30


def fit(img, zoom=1.0, dx=0.0):
    """Encaja la imagen en WxH con un zoom/paneo dado (efecto 'cámara en mano')."""
    h, w = img.shape[:2]
    scale = max(W / w, H / h) * zoom
    img = cv2.resize(img, (int(w * scale) + 1, int(h * scale) + 1))
    h, w = img.shape[:2]
    x0 = int((w - W) / 2 + dx * (w - W) / 2)
    y0 = (h - H) // 2
    return img[y0:y0 + H, max(0, x0):max(0, x0) + W]


def main(args):
    test = pd.read_csv(SPLITS_DIR / "test.csv")
    picks = test.groupby("label").sample(args.per_class, random_state=SEED)
    paths = list(picks.sample(frac=1, random_state=SEED)["path"])
    ood = OOD_DIR / "labels.csv"
    if ood.exists():
        paths += list(pd.read_csv(ood).sample(frac=1, random_state=SEED)["path"].head(args.ood))

    out = ROOT / args.out
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    n_hold, n_fade = int(args.secs * FPS), int(0.4 * FPS)
    prev = None
    for i, p in enumerate(paths):
        img = cv2.imread(str(ROOT / p))
        if img is None:
            continue
        direction = 1 if i % 2 else -1
        for t in range(n_hold):
            a = t / n_hold
            frame = fit(img, zoom=1.0 + 0.12 * a, dx=direction * 0.3 * a)
            if prev is not None and t < n_fade:
                frame = cv2.addWeighted(prev, 1 - t / n_fade, frame, t / n_fade, 0)
            vw.write(frame)
        prev = frame
        # 0.5 s de "mesa vacía" cada 4 imágenes para mostrar el caso sin platillo
        if i % 4 == 3:
            blank = np.full((H, W, 3), 25, np.uint8)
            for _ in range(int(0.5 * FPS)):
                vw.write(blank)
            prev = blank
    vw.release()
    print(f"Video: {out}  ({len(paths)} imágenes, {len(paths) * args.secs:.0f} s aprox.; clases: {len(CLASSES)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="app/demo.mp4")
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--ood", type=int, default=10)
    ap.add_argument("--secs", type=float, default=2.5)
    main(ap.parse_args())
