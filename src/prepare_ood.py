"""Prepara el conjunto FUERA DE DISTRIBUCIÓN (fotos tomadas por el equipo).

Organización esperada (fotos originales del celular):

    data/ood_originales/
        tacos/directo_natural_001.jpg
        tacos/pantalla_artificial_002.jpg
        nachos/impreso_natural_003.jpg
        otro/directo_artificial_004.jpg     <- comida NO mexicana (pizza, sushi...)
        ...

Nombre del archivo = <origen>_<luz>_<num>.jpg
    origen: directo (comida real) | pantalla (foto en celular frente a la webcam) | impreso
    luz:    natural | artificial

El script redimensiona (igual que Food-101), quita metadatos EXIF (ubicación
GPS, modelo del teléfono) y escribe data/ood/labels.csv.

Uso:
    python -m src.prepare_ood
"""
import pandas as pd
from PIL import Image, ImageOps

from src.config import CLASSES, DATA_DIR, OOD_DIR, RESIZE_SAVE, ROOT

SRC = DATA_DIR / "ood_originales"


def main():
    rows = []
    for cls_dir in sorted(p for p in SRC.iterdir() if p.is_dir()):
        if cls_dir.name not in CLASSES:
            print(f"[aviso] carpeta ignorada (no es una clase): {cls_dir.name}")
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".heic", ".webp"):
                continue
            parts = p.stem.lower().split("_")
            origen = parts[0] if parts[0] in ("directo", "pantalla", "impreso") else "desconocido"
            luz = parts[1] if len(parts) > 1 and parts[1] in ("natural", "artificial") else "desconocida"
            img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")  # respeta rotación, descarta EXIF
            img.thumbnail((RESIZE_SAVE, RESIZE_SAVE), Image.Resampling.LANCZOS)
            out = OOD_DIR / cls_dir.name / f"{p.stem}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            img.save(out, "JPEG", quality=92)
            rows.append({"path": str(out.relative_to(ROOT)), "label": cls_dir.name,
                         "origen": origen, "luz": luz})
    df = pd.DataFrame(rows)
    df.to_csv(OOD_DIR / "labels.csv", index=False)
    print(df.groupby(["label", "origen"]).size().unstack(fill_value=0))
    print(f"\nTotal: {len(df)} imágenes -> {OOD_DIR / 'labels.csv'}")


if __name__ == "__main__":
    main()
