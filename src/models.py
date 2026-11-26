"""Arquitecturas comparadas.

* mobilenet_v3_large  (principal: ~5.4 M parámetros, pensada para CPU/móvil)
* efficientnet_b0     (alternativa: ~5.3 M parámetros, más FLOPs)
* resnet50            (alternativa pesada: ~25.6 M parámetros)

Todas parten de pesos preentrenados en ImageNet y se les reemplaza la última
capa por una de NUM_CLASSES salidas (fine-tuning completo).
"""
import torch.nn as nn
from torchvision import models

from src.config import NUM_CLASSES

ARCHS = ("mobilenet_v3_large", "mobilenet_v3_small", "efficientnet_b0", "resnet50")


def build_model(arch: str, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    if arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights="IMAGENET1K_V2" if pretrained else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif arch == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == "resnet50":
        m = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"Arquitectura desconocida: {arch}. Opciones: {ARCHS}")
    return m


def head_parameters(model: nn.Module, arch: str):
    """Parámetros de la cabeza nueva (se entrenan con learning rate mayor)."""
    if arch.startswith("mobilenet") or arch.startswith("efficientnet"):
        return list(model.classifier.parameters())
    return list(model.fc.parameters())


def gradcam_layer(model: nn.Module, arch: str) -> nn.Module:
    """Última capa convolucional, usada por Grad-CAM."""
    if arch.startswith("mobilenet") or arch.startswith("efficientnet"):
        return model.features[-1]
    return model.layer4[-1]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
