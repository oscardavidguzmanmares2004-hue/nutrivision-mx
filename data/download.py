"""Descarga Food-101 y conserva solo las clases del proyecto.

El archivo original pesa ~5 GB. Para no necesitar ese espacio en disco, el
.tar.gz se lee como *stream* (sin guardarlo) y solo se escriben:
  * las 8 clases de platillos mexicanos (1,000 imágenes cada una),
  * una muestra determinista de ~1.6 % de cada una de las otras 93 clases
    (clase "otro"),
  * los archivos meta/train.txt y meta/test.txt con la partición oficial.

Cada imagen se redimensiona a lado mayor RESIZE_SAVE px (~300 MB en total).

Uso:
    python data/download.py
"""
import hashlib
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (FOOD101_URL, MEX_CLASSES, OTHER_CLASS,  # noqa: E402
                        OTHER_SAMPLE_PER_MILLE, RAW_DIR, RESIZE_SAVE)


def keep_other(image_id: str) -> bool:
    """Muestreo determinista (no depende de semillas ni del orden del tar)."""
    h = int(hashlib.md5(image_id.encode()).hexdigest(), 16)
    return h % 1000 < OTHER_SAMPLE_PER_MILLE


def save_resized(data: bytes, out_path: Path) -> None:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((RESIZE_SAVE, RESIZE_SAVE), Image.Resampling.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "JPEG", quality=92)


def main() -> None:
    done_flag = RAW_DIR / ".complete"
    if done_flag.exists():
        print(f"Ya descargado en {RAW_DIR}. Borra esa carpeta para repetir.")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "meta").mkdir(exist_ok=True)

    print(f"Leyendo {FOOD101_URL} como stream (~5 GB, no se guarda en disco)...")
    req = urllib.request.Request(FOOD101_URL, headers={"User-Agent": "python-urllib"})
    kept = {c: 0 for c in MEX_CLASSES + [OTHER_CLASS]}
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0)) or None
        bar = tqdm(total=total, unit="B", unit_scale=True)

        class Counting(io.RawIOBase):
            def readable(self):
                return True

            def readinto(self, b):
                n = resp.readinto(b)
                bar.update(n)
                return n

        stream = io.BufferedReader(Counting(), buffer_size=1 << 20)
        with tarfile.open(fileobj=stream, mode="r|gz") as tar:
            for m in tar:
                if not m.isfile():
                    continue
                parts = m.name.split("/")  # food-101/images/<clase>/<id>.jpg
                if len(parts) == 3 and parts[1] == "meta" and parts[2] in ("train.txt", "test.txt", "classes.txt"):
                    (RAW_DIR / "meta" / parts[2]).write_bytes(tar.extractfile(m).read())
                    continue
                if len(parts) != 4 or parts[1] != "images":
                    continue
                cls, fname = parts[2], parts[3]
                image_id = f"{cls}/{fname[:-4]}"
                if cls in MEX_CLASSES:
                    target = cls
                elif keep_other(image_id):
                    target = OTHER_CLASS
                else:
                    continue
                # En "otro" conservamos la clase original en el nombre del archivo
                out_name = fname if target != OTHER_CLASS else f"{cls}__{fname}"
                save_resized(tar.extractfile(m).read(), RAW_DIR / target / out_name)
                kept[target] += 1
        bar.close()

    missing = [f for f in ("train.txt", "test.txt") if not (RAW_DIR / "meta" / f).exists()]
    if missing:
        raise RuntimeError(f"No se encontraron los archivos meta {missing}")
    done_flag.write_text("ok\n")
    print("Imágenes guardadas por clase:", kept)


if __name__ == "__main__":
    main()
