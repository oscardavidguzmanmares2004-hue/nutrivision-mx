"""Perturbaciones para medir robustez (usadas en eval/robustez.ipynb).

Cada perturbación simula una condición real de la demo:
  brillo      -> salón oscuro / pantalla al máximo
  desenfoque  -> cámara fuera de foco o movimiento
  ruido       -> webcam barata con poca luz
  jpeg        -> compresión de video / redes sociales
  pantalla    -> foto mostrada en celular frente a la webcam
                 (patrón moiré + cambio de gamma + reflejo)
"""
import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

PERTURBATIONS = {
    "brillo": [0.4, 0.6, 1.0, 1.4, 1.8],
    "desenfoque": [0, 1, 2, 3, 5],
    "ruido": [0, 10, 20, 35, 50],
    "jpeg": [95, 50, 25, 10, 5],
    "pantalla": [0, 1, 2, 3],
}


def apply(img: Image.Image, kind: str, level) -> Image.Image:
    if kind == "brillo":
        return ImageEnhance.Brightness(img).enhance(level)
    if kind == "desenfoque":
        return img.filter(ImageFilter.GaussianBlur(level)) if level else img
    if kind == "ruido":
        a = np.asarray(img).astype(np.float32)
        a += np.random.default_rng(0).normal(0, level, a.shape)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if kind == "jpeg":
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=int(level))
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    if kind == "pantalla":
        return simulate_screen(img, level)
    raise ValueError(kind)


def simulate_screen(img: Image.Image, severity: int) -> Image.Image:
    """Aproximación de fotografiar una pantalla: moiré, gamma y reflejo."""
    if severity == 0:
        return img
    a = np.asarray(img).astype(np.float32) / 255
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    freq = 0.35 + 0.1 * severity
    moire = 1 - 0.06 * severity * (0.5 + 0.5 * np.sin(freq * (xx * 0.8 + yy * 0.6)))
    a = a * moire[..., None]
    a = a ** (1 + 0.15 * severity)                       # pantalla más contrastada
    a[..., 2] *= 1 + 0.05 * severity                      # tinte azul de pantalla
    glare = np.exp(-(((xx - w * 0.7) ** 2 + (yy - h * 0.3) ** 2) / (2 * (w * 0.18) ** 2)))
    a = a + 0.25 * severity / 3 * glare[..., None]        # reflejo de luz
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
