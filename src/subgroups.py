"""Atributos de imagen para la auditoría por subgrupos.

Food-101 no trae metadatos (no hay datos de personas, lugar ni cámara), así
que los subgrupos se definen a partir de propiedades medibles de la foto que
sí cambian en el uso real:

  iluminacion   oscura / media / clara        (luminancia media, terciles)
  temperatura   fría / neutra / cálida        (R - B medio: luz amarilla de restaurante)
  nitidez       borrosa / media / nítida      (varianza del Laplaciano, terciles)
  origen        (solo OOD) directo / pantalla / impreso — columna en data/ood/labels.csv
"""
import cv2
import numpy as np
import pandas as pd

from src.config import ROOT


def image_attributes(paths) -> pd.DataFrame:
    rows = []
    for p in paths:
        bgr = cv2.imread(str(ROOT / p))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        b, _, r = [c.astype(np.float32).mean() for c in cv2.split(bgr)]
        rows.append({
            "path": p,
            "luminancia": float(gray.mean()),
            "calidez": float(r - b),
            "nitidez": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        })
    return pd.DataFrame(rows)


def add_subgroups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["iluminacion"] = pd.qcut(df["luminancia"], 3, labels=["oscura", "media", "clara"])
    df["temperatura"] = pd.qcut(df["calidez"], 3, labels=["fría", "neutra", "cálida"])
    df["nitidez_grupo"] = pd.qcut(df["nitidez"], 3, labels=["borrosa", "media", "nítida"])
    return df
