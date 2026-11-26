"""Entrenamiento (fine-tuning) con early stopping sobre F1 macro de validación.

Uso:
    python -m src.train --arch mobilenet_v3_large
    python -m src.train --arch efficientnet_b0

Salidas en runs/<arch>/:
    best.pt          pesos del mejor epoch (según F1 macro en VALIDACIÓN)
    history.csv      pérdida y métricas por epoch (curvas de entrenamiento)
    train_info.json  hiperparámetros, tiempo, parámetros, epoch elegido
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.config import CLASSES, RUNS_DIR, SEED
from src.data import FoodDataset, class_weights, get_transforms, split_csv
from src.models import build_model, count_params, head_parameters
from src.utils import get_device, save_json, seed_worker, set_seed


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, preds, targets = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds.append(logits.argmax(1).cpu())
            targets.append(y.cpu())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return {
        "loss": total_loss / len(loader.dataset),
        "acc": float((preds == targets).mean()),
        "f1_macro": float(f1_score(targets, preds, average="macro")),
    }


def main(args):
    set_seed(SEED)
    device = get_device(args.device)
    out = RUNS_DIR / args.arch
    out.mkdir(parents=True, exist_ok=True)
    print(f"Dispositivo: {device} | arquitectura: {args.arch}")

    train_ds = FoodDataset(split_csv("train"), get_transforms(train=True))
    val_ds = FoodDataset(split_csv("val"), get_transforms(train=False))
    g = torch.Generator().manual_seed(SEED)
    common = dict(num_workers=args.workers, worker_init_fn=seed_worker,
                  persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, generator=g, **common)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, **common)

    model = build_model(args.arch).to(device)
    # Pesos de clase contra el desbalance + label smoothing contra etiquetas ruidosas
    weights = class_weights(train_ds.targets).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    eval_criterion = nn.CrossEntropyLoss()

    # Learning rate diferenciado: la cabeza nueva aprende 10x más rápido
    head = head_parameters(model, args.arch)
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": args.lr},
        {"params": head, "lr": args.lr * 10},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history, best_f1, best_epoch, bad_epochs = [], -1.0, 0, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        te = time.time()
        tr = run_epoch(model, train_dl, criterion, device, optimizer)
        va = run_epoch(model, val_dl, eval_criterion, device)
        scheduler.step()
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
               **{f"val_{k}": v for k, v in va.items()},
               "lr": optimizer.param_groups[0]["lr"], "secs": time.time() - te}
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(f"[{epoch:02d}] train loss {tr['loss']:.3f} f1 {tr['f1_macro']:.3f} | "
              f"val loss {va['loss']:.3f} f1 {va['f1_macro']:.3f} | {row['secs']:.0f}s")

        if va["f1_macro"] > best_f1:
            best_f1, best_epoch, bad_epochs = va["f1_macro"], epoch, 0
            torch.save({"arch": args.arch, "classes": CLASSES,
                        "state_dict": model.state_dict()}, out / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping: {args.patience} epochs sin mejorar (mejor epoch {best_epoch}).")
                break

    info = {
        "arch": args.arch, "seed": SEED, "device": str(device),
        "epochs_run": len(history), "best_epoch": best_epoch, "best_val_f1_macro": best_f1,
        "train_minutes": (time.time() - t0) / 60, "params": count_params(model),
        "hyperparams": vars(args), "n_train": len(train_ds), "n_val": len(val_ds),
        "class_weights": dict(zip(CLASSES, np.round(weights.cpu().numpy(), 3).tolist())),
    }
    save_json(info, out / "train_info.json")
    print(f"Listo. Mejor F1 macro val = {best_f1:.4f} (epoch {best_epoch}). Pesos en {out/'best.pt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mobilenet_v3_large")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    main(ap.parse_args())
