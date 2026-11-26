"""Captura fotos fuera de distribución con la webcam (rúbrica B6).

Muestra la cámara en vivo. Apunta a un platillo (comida real, foto en el
celular / otra pantalla o foto impresa) y presiona la tecla de su clase para
guardar el cuadro en data/ood_originales/<clase>/<origen>_<luz>_<n>.jpg

Teclas:
    1 Burrito   2 Ceviche   3 Quesadilla   4 Churros   5 Guacamole
    6 Huevos rancheros   7 Nachos   8 Tacos   9 Otro platillo (NO mexicano)
    o  cambiar origen (pantalla -> directo -> impreso)
    l  cambiar luz    (artificial -> natural)
    q  salir

Uso:
    python app/capture_ood.py              # webcam 0
    python app/capture_ood.py --source 1   # otra cámara

Después:
    python -m src.prepare_ood
    python -m src.evaluate --split ood
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CLASSES, DATA_DIR, DISPLAY  # noqa: E402

OUT = DATA_DIR / "ood_originales"
ORIGENES = ["pantalla", "directo", "impreso"]
LUCES = ["artificial", "natural"]


def count_saved():
    return {c: len(list((OUT / c).glob("*.jpg"))) if (OUT / c).exists() else 0 for c in CLASSES}


def main(args):
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la cámara {args.source}")
    origen, luz = 0, 0
    counts = count_saved()
    flash = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        vis = frame.copy()
        h, w = vis.shape[:2]
        s = min(h, w)
        x0, y0 = (w - s) // 2, (h - s) // 2
        cv2.rectangle(vis, (x0, y0), (x0 + s, y0 + s), (255, 255, 255) if not flash else (80, 200, 80), 3)
        flash = max(0, flash - 1)
        lines = [f"origen [o]: {ORIGENES[origen]}   luz [l]: {LUCES[luz]}   total: {sum(counts.values())}"]
        lines += [f"{i + 1} {DISPLAY[c]}: {counts[c]}" for i, c in enumerate(CLASSES)]
        for i, t in enumerate(lines):
            cv2.putText(vis, t, (12, 28 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, t, (12, 28 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Captura OOD - NutriVision MX", vis)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("o"):
            origen = (origen + 1) % len(ORIGENES)
        if k == ord("l"):
            luz = (luz + 1) % len(LUCES)
        if ord("1") <= k <= ord("9"):
            cls = CLASSES[k - ord("1")]
            counts[cls] += 1
            path = OUT / cls / f"{ORIGENES[origen]}_{LUCES[luz]}_{counts[cls]:03d}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), frame[y0:y0 + s, x0:x0 + s])  # mismo recorte que ve el modelo
            print(f"Guardada: {path.relative_to(DATA_DIR.parent)}")
            flash = 6
    cap.release()
    cv2.destroyAllWindows()
    print("Capturas por clase:", count_saved())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    main(ap.parse_args())
