"""Dataset, transformaciones y pesos de clase."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from src.config import CLASSES, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, ROOT, SPLITS_DIR

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def get_transforms(train: bool) -> T.Compose:
    """Aumento de datos solo en entrenamiento.

    Las transformaciones imitan lo que verá la cámara en la demo: cambios de
    luz y color (restaurante, pantalla de celular), encuadres distintos,
    rotaciones leves y algo de desenfoque.
    """
    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if train:
        return T.Compose([
            T.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
            T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
            *norm,
            T.RandomErasing(p=0.2),
        ])
    return T.Compose([T.Resize(int(IMG_SIZE * 1.14)), T.CenterCrop(IMG_SIZE), *norm])


class FoodDataset(Dataset):
    def __init__(self, csv_path, transform=None, root: Path = ROOT):
        self.df = pd.read_csv(csv_path)
        self.root = Path(root)
        self.transform = transform
        self.targets = self.df["label"].map(CLASS_TO_IDX).to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img = Image.open(self.root / self.df.iloc[i]["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(self.targets[i])


def split_csv(name: str) -> Path:
    return SPLITS_DIR / f"{name}.csv"


def class_weights(targets: np.ndarray) -> torch.Tensor:
    """Pesos inversamente proporcionales a la frecuencia (normalizados a media 1).

    Atienden el desbalance de la clase "otro", que tiene más imágenes que
    cada platillo mexicano.
    """
    counts = np.bincount(targets, minlength=len(CLASSES)).astype(float)
    w = counts.sum() / (len(CLASSES) * np.maximum(counts, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)
