"""Conjunto FUERA DE DISTRIBUCIÓN tomado de otro dataset: ISIA Food-500 (B6).

Fuente: Min et al., "ISIA Food-500: A Dataset for Large-Scale Food Recognition
via Stacked Global-Local Attention Network", ACM Multimedia 2020. Se usa la
partición de prueba publicada en Hugging Face como `advancedcv/Food500Cap`.

Por qué es fuera de distribución respecto a Food-101:
  * Otra fuente: imágenes recolectadas de buscadores web (Google, Bing, Baidu),
    no de la red social foodspotting.com.
  * Otro etiquetado y otra época de recolección.

Se descargan, vía la API pública del visor de Hugging Face (sin bajar los 3 GB
del dataset completo):
  * las 10 imágenes de prueba de cada platillo mexicano que existe en Food-500
    (tacos, nachos, huevos rancheros, quesadilla, burrito, ceviche);
  * OTHER_N imágenes de platillos NO mexicanos elegidos al azar -> clase "otro".

Antes de guardar, cada imagen se compara (pHash) contra las imágenes de
Food-101 del proyecto, para descartar fotos repetidas entre ambos datasets.

Uso:
    python -m src.download_ood
"""
import io
import json
import random
import time
import urllib.parse
import urllib.request

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from src.config import OOD_DIR, RESIZE_SAVE, ROOT, SEED, SPLITS_DIR

API = "https://datasets-server.huggingface.co"
DATASET = "advancedcv/Food500Cap"
SPLIT = "test"
CLASS_MAP = {  # categoría de Food-500 -> clase del proyecto
    "Tacos": "tacos",
    "Nachos": "nachos",
    "Huevos_rancheros": "huevos_rancheros",
    "Quesadilla": "chicken_quesadilla",
    "Burrito": "breakfast_burrito",
    "Ceviche": "ceviche",
}
# Categorías de Food-500 con raíz mexicana/latina que NO deben caer en "otro"
EXCLUDE_FROM_OTHER = {"Mission_burrito", "Tostada", "Champurrado", "Enchilada", "Tamale",
                      "Chilaquiles", "Pozole", "Mole", "Fajita", "Chimichanga", "Taquito",
                      "Elote", "Arepa", "Empanada", "Pupusa",
                      "Avocado_sauce"}  # visualmente es guacamole
OTHER_N = 20
HAMMING_THR = 4


def _get(url, tries=5):
    """GET con reintentos (la API pública a veces tarda o devuelve error temporal)."""
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            print(f"[reintento {i + 1}] {e.__class__.__name__}")
            time.sleep(3 * (i + 1))


def api(path, **params):
    return json.loads(_get(f"{API}/{path}?" + urllib.parse.urlencode(params)))


def fetch_image(src):
    img = Image.open(io.BytesIO(_get(src))).convert("RGB")
    img.thumbnail((RESIZE_SAVE, RESIZE_SAVE), Image.Resampling.LANCZOS)
    return img


def main():
    rng = random.Random(SEED)
    common = dict(dataset=DATASET, config="default", split=SPLIT)

    # hashes de Food-101 (train+val+test) para descartar repetidas
    f101 = pd.concat([pd.read_csv(SPLITS_DIR / f"{s}.csv") for s in ("train", "val", "test")])
    f101_hashes = np.array([int(str(imagehash.phash(Image.open(ROOT / p))), 16) for p in f101.path], dtype=np.uint64)

    rows = []
    for cat, label in CLASS_MAP.items():
        res = api("filter", where=f"\"cat\"='{cat}'", length=100, **common)
        for r in res["rows"]:
            rows.append((label, cat, r["row_idx"], r["row"]["image"]["src"]))

    total = api("size", dataset=DATASET)["size"]["splits"]
    n_test = next(s["num_rows"] for s in total if s["split"] == SPLIT)
    excluded = set(CLASS_MAP) | EXCLUDE_FROM_OTHER
    seen_cats = set()
    while sum(1 for r in rows if r[0] == "otro") < OTHER_N:
        off = rng.randrange(n_test)
        r = api("rows", offset=off, length=1, **common)["rows"][0]
        cat = r["row"]["cat"]
        if cat in excluded or cat in seen_cats:
            continue
        seen_cats.add(cat)  # una imagen por categoría: máxima variedad
        rows.append(("otro", cat, r["row_idx"], r["row"]["image"]["src"]))

    out_rows, dropped = [], 0
    for label, cat, idx, src in rows:
        img = fetch_image(src)
        h = np.uint64(int(str(imagehash.phash(img)), 16))
        if np.bitwise_count(f101_hashes ^ h).min() <= HAMMING_THR:
            dropped += 1
            continue
        path = OOD_DIR / label / f"food500_{cat}_{idx}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path, "JPEG", quality=92)
        out_rows.append({"path": str(path.relative_to(ROOT)), "label": label, "origen": "food500_web",
                         "luz": "desconocida", "food500_cat": cat, "food500_row": idx})

    df = pd.DataFrame(out_rows)
    df.to_csv(OOD_DIR / "labels.csv", index=False)
    print(df.label.value_counts().to_string())
    print(f"Total: {len(df)} | descartadas por duplicado con Food-101: {dropped}")


if __name__ == "__main__":
    main()
