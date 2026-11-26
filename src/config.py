"""Configuración central del proyecto NutriVisión MX.

Todo lo que define el experimento (clases, semilla, rutas, tamaño de imagen)
vive aquí para que el entrenamiento sea reproducible y fácil de auditar.
"""
from pathlib import Path

# ---------------------------------------------------------------- rutas
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # imágenes descargadas y redimensionadas
SPLITS_DIR = DATA_DIR / "splits"    # CSV de particiones train/val/test
OOD_DIR = DATA_DIR / "ood"          # fotos propias del equipo (fuera de distribución)
RUNS_DIR = ROOT / "runs"            # pesos, historiales y métricas por arquitectura
FIG_DIR = ROOT / "docs" / "figuras"

# ---------------------------------------------------------------- reproducibilidad
SEED = 42

# ---------------------------------------------------------------- datos
FOOD101_URL = "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"

# Clases de Food-101 que son platillos mexicanos (nombre original del dataset).
MEX_CLASSES = [
    "breakfast_burrito",
    "ceviche",
    "chicken_quesadilla",
    "churros",
    "guacamole",
    "huevos_rancheros",
    "nachos",
    "tacos",
]
OTHER_CLASS = "otro"                # muestra de los otros 93 platillos de Food-101
CLASSES = MEX_CLASSES + [OTHER_CLASS]
NUM_CLASSES = len(CLASSES)

# Fracción de imágenes que se toman de cada clase NO mexicana para formar "otro".
# 16/1000 * 93 clases * 1000 imgs ≈ 1,500 imágenes.
OTHER_SAMPLE_PER_MILLE = 16

# Nombres para mostrar en la app (español).
DISPLAY = {
    "breakfast_burrito": "Burrito",
    "ceviche": "Ceviche",
    "chicken_quesadilla": "Quesadilla",
    "churros": "Churros",
    "guacamole": "Guacamole",
    "huevos_rancheros": "Huevos rancheros",
    "nachos": "Nachos",
    "tacos": "Tacos",
    "otro": "Otro platillo",
}

# Calorías aproximadas por porción típica (kcal). Valores de referencia
# redondeados a partir de tablas nutricionales públicas (USDA FoodData Central
# y Sistema Mexicano de Alimentos Equivalentes). Son PROMEDIOS: ver MODEL_CARD.
KCAL = {
    "breakfast_burrito": 550,
    "ceviche": 200,
    "chicken_quesadilla": 500,
    "churros": 350,      # 3 piezas
    "guacamole": 230,    # 1/2 taza
    "huevos_rancheros": 450,
    "nachos": 650,
    "tacos": 450,        # 3 tacos
    "otro": None,
}

# ---------------------------------------------------------------- modelo
IMG_SIZE = 224
RESIZE_SAVE = 320                  # lado mayor al guardar (acelera el entrenamiento)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_ARCH = "mobilenet_v3_large"
