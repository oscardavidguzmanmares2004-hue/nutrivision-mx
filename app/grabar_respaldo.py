"""Graba el video de respaldo de la demo, sin cámara ni ventana.

Pasa cada cuadro de app/demo.mp4 por el mismo pipeline de app/demo.py (recorte, ONNX,
suavizado EMA, umbral, dibujo) y guarda el resultado con las predicciones en pantalla.
Para mostrar la abstención, a mitad del video el umbral sube a 0.90 durante unos segundos.

Uso:
    python app/grabar_respaldo.py                       # -> app/demo_respaldo.mp4
    python app/grabar_respaldo.py --int8 --out otro.mp4
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import demo  # noqa: E402
from src.config import RUNS_DIR  # noqa: E402


def main(args):
    model = RUNS_DIR / args.arch / ("model_int8.onnx" if args.int8 else "model.onnx")
    clf = demo.Classifier(model)
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"No existe {args.source}. Genéralo con: python app/make_demo_video.py")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(3)), int(cap.get(4))
    raw = Path(args.out).with_suffix(".raw.mp4")
    BAR = 34  # franja inferior para la nota (no tapa la interfaz de la app)
    out = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h + BAR))

    alto = (int(n * 0.62), int(n * 0.74))  # tramo con umbral alto (0.90)
    ema, fps_ema, lats, dudas = None, 0.0, [], [0, 0]
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        thr = 0.90 if alto[0] <= i < alto[1] else args.threshold
        state, lat = "ok", 0.0
        if demo.is_blank(frame):
            state, ema = "blank", None
        else:
            t = time.perf_counter()
            p = clf(frame)
            lat = (time.perf_counter() - t) * 1000
            lats.append(lat)
            ema = p if ema is None else args.smooth * ema + (1 - args.smooth) * p
        if ema is not None and ema.max() < thr:
            dudas[int(alto[0] <= i < alto[1])] += 1
        demo.draw(frame, ema, thr, fps_ema, lat, state)
        inst = 1.0 / max(time.perf_counter() - t0, 1e-6)  # FPS reales del pipeline
        fps_ema = inst if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst
        nota = "Grabacion automatica de app/demo.py | MobileNetV3-Large ONNX " + ("INT8" if args.int8 else "FP32")
        if alto[0] <= i < alto[1]:
            nota += " | umbral subido a 90%: con menos confianza responde \"No estoy seguro\""
        frame = cv2.copyMakeBorder(frame, 0, BAR, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        cv2.putText(frame, nota, (12, h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        out.write(frame)
    out.release()

    final = Path(args.out)
    if shutil.which("ffmpeg"):  # H.264 para que se reproduzca en el navegador y en GitHub
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "26", "-movflags", "+faststart", str(final)], check=True)
        raw.unlink()
    else:
        raw.rename(final)
    print(f"Guardado {final} | {i + 1} cuadros | latencia media del modelo {sum(lats) / len(lats):.1f} ms | "
          f"cuadros \"No estoy seguro\": {dudas[0]} con umbral {args.threshold}, {dudas[1]} con umbral 0.90")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="app/demo.mp4")
    ap.add_argument("--out", default="app/demo_respaldo.mp4")
    ap.add_argument("--arch", default="mobilenet_v3_large")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--smooth", type=float, default=0.7)
    main(ap.parse_args())
