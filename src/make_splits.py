"""Construye las particiones train / val / test SIN fuga de información.

Decisiones (ver README, sección "Particiones"):
1. TEST = partición oficial de prueba de Food-101 (250 img/clase). Nunca se usa
   para elegir hiperparámetros: solo se evalúa una vez al final.
2. Detección de casi-duplicados con hash perceptual (pHash, 64 bits). Dos
   imágenes con distancia de Hamming <= HAMMING_THR se consideran la misma foto
   (re-subidas, recortes, re-compresiones).
3. Cualquier imagen de train que sea casi-duplicado de una de test se ELIMINA
   de train (fuga train→test).
4. Las imágenes marcadas como "incorrecta" en la revisión manual de etiquetas
   (docs/revision_etiquetas.csv, rúbrica B5) se excluyen de train.
5. Las imágenes de train se agrupan en "clústeres de duplicados" (union-find) y
   la partición train/val se hace POR CLÚSTER, de modo que dos copias de la
   misma foto nunca queden una en train y otra en val.

Uso:
    python -m src.make_splits
"""
import argparse
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from src.config import CLASSES, MEX_CLASSES, OTHER_CLASS, RAW_DIR, ROOT, SEED, SPLITS_DIR
from src.utils import save_json

HAMMING_THR = 4


def read_official_split() -> dict:
    """Devuelve {'clase/id': 'train'|'test'} según meta/*.txt de Food-101."""
    split = {}
    for name in ("train", "test"):
        for line in (RAW_DIR / "meta" / f"{name}.txt").read_text().split():
            split[line.strip()] = name
    return split


def collect_images(official: dict) -> pd.DataFrame:
    rows = []
    for cls in CLASSES:
        for p in sorted((RAW_DIR / cls).glob("*.jpg")):
            if cls == OTHER_CLASS:
                orig_cls, fname = p.stem.split("__")
                image_id = f"{orig_cls}/{fname}"
            else:
                orig_cls, image_id = cls, f"{cls}/{p.stem}"
            rows.append({
                "path": str(p.relative_to(RAW_DIR.parent.parent)),
                "label": cls,
                "orig_class": orig_cls,
                "image_id": image_id,
                "official_split": official[image_id],
            })
    return pd.DataFrame(rows)


def phash_all(df: pd.DataFrame, root: Path) -> np.ndarray:
    hashes = []
    for rel in tqdm(df["path"], desc="pHash"):
        h = imagehash.phash(Image.open(root / rel))
        hashes.append(int(str(h), 16))
    return np.array(hashes, dtype=np.uint64)


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.bitwise_count(a[:, None] ^ b[None, :])


def duplicate_clusters(hashes: np.ndarray, thr: int) -> np.ndarray:
    """Union-find sobre pares con distancia <= thr. Devuelve id de clúster."""
    n = len(hashes)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for start in range(0, n, 1024):
        block = hamming_matrix(hashes[start:start + 1024], hashes)
        ii, jj = np.nonzero(block <= thr)
        for i, j in zip(ii + start, jj):
            if i < j:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    return np.array([find(i) for i in range(n)])


def main(val_frac: float = 0.15) -> None:
    root = RAW_DIR.parent.parent
    df = collect_images(read_official_split())
    print(df.groupby(["label", "official_split"]).size().unstack())
    df["phash"] = phash_all(df, root)

    train = df[df.official_split == "train"].reset_index(drop=True)
    test = df[df.official_split == "test"].reset_index(drop=True)

    # 1) fuga train -> test
    h_tr = train["phash"].to_numpy(np.uint64)
    h_te = test["phash"].to_numpy(np.uint64)
    min_dist = hamming_matrix(h_tr, h_te).min(axis=1)
    leak = min_dist <= HAMMING_THR
    print(f"Imágenes de train casi-duplicadas de test (eliminadas): {leak.sum()}")
    leaked = train[leak][["path", "label"]]
    train = train[~leak].reset_index(drop=True)

    # 2) corrección de etiquetas tras la revisión manual
    review = ROOT / "docs" / "revision_etiquetas.csv"
    wrong = set()
    if review.exists():
        rv = pd.read_csv(review)
        wrong = set(rv.loc[rv.veredicto == "incorrecta", "path"])
        train = train[~train["path"].isin(wrong)].reset_index(drop=True)
    print(f"Imágenes excluidas por etiqueta incorrecta (revisión manual): {len(wrong)}")

    # 3) clústeres de duplicados dentro de train y split agrupado
    train["dup_group"] = duplicate_clusters(train["phash"].to_numpy(np.uint64), HAMMING_THR)
    n_groups_dup = (train.groupby("dup_group").size() > 1).sum()
    print(f"Clústeres de duplicados dentro de train: {n_groups_dup}")

    n_splits = round(1 / val_frac)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    tr_idx, va_idx = next(sgkf.split(train, train["label"], groups=train["dup_group"]))
    val = train.iloc[va_idx].copy()
    train = train.iloc[tr_idx].copy()
    assert set(train.dup_group).isdisjoint(set(val.dup_group))

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["path", "label", "orig_class", "image_id"]
    for name, part in (("train", train), ("val", val), ("test", test)):
        part[cols].sort_values("path").to_csv(SPLITS_DIR / f"{name}.csv", index=False)
    leaked.to_csv(SPLITS_DIR / "removidas_por_fuga.csv", index=False)

    report = {
        "hamming_threshold": HAMMING_THR,
        "removed_train_dups_of_test": int(leak.sum()),
        "dup_clusters_in_train": int(n_groups_dup),
        "removed_wrong_label": sorted(wrong),
        "counts": {
            name: part["label"].value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict()
            for name, part in (("train", train), ("val", val), ("test", test))
        },
        "mex_classes": MEX_CLASSES,
        "seed": SEED,
    }
    save_json(report, SPLITS_DIR / "reporte_particiones.json")
    print(pd.DataFrame(report["counts"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.15)
    main(ap.parse_args().val_frac)
