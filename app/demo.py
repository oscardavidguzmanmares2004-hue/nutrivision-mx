"""NutriVisión MX — demo en tiempo real.

Clasifica platillos mexicanos cuadro por cuadro desde la webcam o un video y
muestra: platillo, confianza, top-3, calorías aproximadas, FPS y latencia.

Uso:
    python app/demo.py                          # webcam 0
    python app/demo.py --source demo.mp4        # video en flujo (se repite en bucle)
    python app/demo.py --source 1               # otra cámara (p. ej. celular)
    python app/demo.py --int8                   # usar el modelo cuantizado INT8

Teclas:
    q / ESC  salir            + / -   subir / bajar umbral de confianza
    espacio  pausar           s       guardar captura en docs/figuras/
"""
import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (CLASSES, DISPLAY, FIG_DIR, IMAGENET_MEAN,  # noqa: E402
                        IMAGENET_STD, IMG_SIZE, KCAL, OTHER_CLASS, RUNS_DIR)

WIN = "NutriVision MX"
GREEN, YELLOW, GRAY, WHITE, BLACK = (80, 200, 80), (0, 200, 255), (160, 160, 160), (255, 255, 255), (0, 0, 0)
MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
STD = np.array(IMAGENET_STD, dtype=np.float32)


# ------------------------------------------------------------------ modelo
class Classifier:
    def __init__(self, onnx_path: Path):
        if not onnx_path.exists():
            raise SystemExit(f"No existe {onnx_path}. Corre primero: python -m src.export")
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    @staticmethod
    def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
        """Recorte cuadrado central -> 224x224 -> RGB normalizado (igual que en entrenamiento)."""
        h, w = frame_bgr.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        crop = frame_bgr[y0:y0 + s, x0:x0 + s]
        img = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        return img.transpose(2, 0, 1)[None]  # (1,3,H,W)

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        logits = self.sess.run(None, {"input": self.preprocess(frame_bgr)})[0][0]
        e = np.exp(logits - logits.max())
        return e / e.sum()


# ------------------------------------------------------------------ fuente de video
def open_source(src: str):
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la fuente de video: {src}")
    return cap


def is_blank(frame: np.ndarray) -> bool:
    """Cuadro negro / tapado / sin contenido: no tiene sentido clasificarlo."""
    small = cv2.resize(frame, (64, 64))
    return small.std() < 8


# ------------------------------------------------------------------ dibujo
def put(img, text, org, scale=0.6, color=WHITE, thick=1, bg=True):
    """Texto legible sobre cualquier fondo (caja oscura semitransparente detrás)."""
    if not text:
        return
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    if bg:
        x1, y1 = max(x - 4, 0), max(y - th - 5, 0)
        x2, y2 = min(x + tw + 4, img.shape[1]), min(y + base + 3, img.shape[0])
        roi = img[y1:y2, x1:x2]
        img[y1:y2, x1:x2] = (roi * 0.35).astype(img.dtype)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw(frame, probs, thr, fps, lat_ms, state):
    h, w = frame.shape[:2]
    s = min(h, w)
    x0, y0 = (w - s) // 2, (h - s) // 2

    order = np.argsort(-probs) if probs is not None else []
    top = order[0] if len(order) else None
    conf = float(probs[top]) if top is not None else 0.0
    cls = CLASSES[top] if top is not None else None

    # decidir qué mostrar
    if state == "blank":
        color, title, sub = GRAY, "Sin imagen", "La camara parece tapada u oscura"
    elif conf < thr:
        color, title, sub = YELLOW, "No estoy seguro", f"Confianza {conf:.0%} < umbral {thr:.0%}: acerca el platillo"
    elif cls == OTHER_CLASS:
        color, title, sub = GRAY, "Otro platillo (no mexicano)", f"Confianza {conf:.0%}"
    else:
        kcal = KCAL.get(cls)
        color, title = GREEN, f"{DISPLAY[cls]}  {conf:.0%}"
        sub = f"~{kcal} kcal por porcion (aprox.)" if kcal else ""

    # marco de la zona que ve el modelo
    cv2.rectangle(frame, (x0 + 2, y0 + 2), (x0 + s - 3, y0 + s - 3), color, 3)

    # panel superior
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 92), BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    put(frame, title, (14, 38), 1.0, color, 2)
    put(frame, sub, (14, 72), 0.6, WHITE)

    # barra de confianza
    bx, by, bw = w - 230, 22, 200
    cv2.rectangle(frame, (bx, by), (bx + bw, by + 16), GRAY, 1)
    cv2.rectangle(frame, (bx, by), (bx + int(bw * conf), by + 16), color, -1)
    tx = bx + int(bw * thr)
    cv2.line(frame, (tx, by - 4), (tx, by + 20), WHITE, 2)
    put(frame, f"umbral {thr:.0%}", (bx, by + 40), 0.5)

    # top-3
    if len(order) and state != "blank":
        for i, k in enumerate(order[:3]):
            put(frame, f"{i + 1}. {DISPLAY[CLASSES[k]]}: {probs[k]:.0%}", (14, h - 92 + i * 24), 0.55)

    # rendimiento
    perf_color = GREEN if fps >= 10 else (0, 0, 255)
    put(frame, f"FPS {fps:5.1f} | modelo {lat_ms:4.1f} ms", (w - 300, h - 16), 0.6, perf_color, 2)
    put(frame, "q salir  +/- umbral  espacio pausa", (14, h - 16), 0.45, GRAY)
    return frame


# ------------------------------------------------------------------ bucle principal
def main(args):
    onnx_path = Path(args.model) if args.model else RUNS_DIR / args.arch / ("model_int8.onnx" if args.int8 else "model.onnx")
    clf = Classifier(onnx_path)
    cap = open_source(args.source)
    is_file = not args.source.isdigit()

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    thr = args.threshold
    cv2.createTrackbar("umbral %", WIN, int(thr * 100), 99, lambda v: None)

    ema_probs, fps_ema, paused, frame = None, 0.0, False, None
    lat_hist, fps_hist = deque(maxlen=2000), deque(maxlen=2000)
    fails, t_prev = 0, time.perf_counter()
    print(f"Modelo: {onnx_path.name} | fuente: {args.source}")

    while True:
        if not paused:
            ok, new = cap.read()
            if not ok or new is None:
                if is_file:                       # video terminado -> repetir
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                fails += 1                        # webcam: reintentar sin caerse
                if fails > 50:
                    print("La cámara dejó de responder; reabriendo...")
                    cap.release(); cap = open_source(args.source); fails = 0
                continue
            fails = 0
            frame = new
            if frame.ndim == 2:                   # cámara en escala de grises
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        thr = max(cv2.getTrackbarPos("umbral %", WIN), 1) / 100
        vis = frame.copy()
        state, lat_ms = "ok", 0.0
        try:
            if is_blank(frame):
                state, ema_probs = "blank", None
            else:
                t = time.perf_counter()
                probs = clf(frame)
                lat_ms = (time.perf_counter() - t) * 1000
                lat_hist.append(lat_ms)
                # suavizado temporal: evita que la etiqueta "parpadee" entre cuadros
                ema_probs = probs if ema_probs is None else args.smooth * ema_probs + (1 - args.smooth) * probs
        except Exception as e:                    # entrada inesperada: avisar, no caerse
            state, ema_probs = "blank", None
            print(f"[aviso] cuadro ignorado: {e}")

        now = time.perf_counter()
        inst = 1.0 / max(now - t_prev, 1e-6)
        t_prev = now
        fps_ema = inst if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst
        if not paused:
            fps_hist.append(inst)

        cv2.imshow(WIN, draw(vis, ema_probs, thr, fps_ema, lat_ms, state))
        delay = max(1, int(1000 / cap.get(cv2.CAP_PROP_FPS))) if is_file and cap.get(cv2.CAP_PROP_FPS) > 0 and args.realtime else 1
        key = cv2.waitKey(delay) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("+"), ord("=")):
            cv2.setTrackbarPos("umbral %", WIN, min(99, int(thr * 100) + 5))
        if key == ord("-"):
            cv2.setTrackbarPos("umbral %", WIN, max(1, int(thr * 100) - 5))
        if key == ord(" "):
            paused = not paused
        if key == ord("s"):
            FIG_DIR.mkdir(parents=True, exist_ok=True)
            p = FIG_DIR / f"captura_demo_{int(time.time())}.png"
            cv2.imwrite(str(p), vis); print(f"Captura guardada: {p}")
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    if lat_hist:
        lat = np.array(lat_hist); fps = np.array(fps_hist)
        print(f"\nResumen: {len(lat)} cuadros | latencia modelo media {lat.mean():.1f} ms "
              f"(p95 {np.percentile(lat, 95):.1f} ms) | FPS de la app medio {fps.mean():.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="índice de cámara (0, 1, ...) o ruta a video")
    ap.add_argument("--arch", default="mobilenet_v3_large")
    ap.add_argument("--model", default=None, help="ruta a un .onnx específico")
    ap.add_argument("--int8", action="store_true", help="usar el modelo cuantizado INT8")
    ap.add_argument("--threshold", type=float, default=0.55, help="umbral de confianza inicial")
    ap.add_argument("--smooth", type=float, default=0.7, help="suavizado temporal (0 = sin suavizar)")
    ap.add_argument("--realtime", action="store_true", default=True,
                    help="reproducir el video a su velocidad original")
    main(ap.parse_args())
