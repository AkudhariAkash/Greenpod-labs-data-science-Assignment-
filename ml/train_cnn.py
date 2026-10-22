from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import discover_records, load_splits, make_splits, save_splits, seed_everything, validate_image


def train(data_dir: str = "data/raw", epochs: int = 10, batch_size: int = 32, learning_rate: float = 3e-4, image_size: int = 224, seed: int = 42) -> dict:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import datasets, models, transforms
    except ImportError as exc:
        raise RuntimeError("CNN training requires torch and torchvision") from exc
    records = discover_records(data_dir)
    for r in records: r.update(validate_image(r["path"]))
    splits = load_splits() or make_splits(records, seed=seed); save_splits(splits); seed_everything(seed)
    classes = sorted({r["label"] for r in records if r.get("readable", True)})
    class_to_idx = {name: i for i, name in enumerate(classes)}
    root = Path("data/processed/cnn"); root.mkdir(parents=True, exist_ok=True)
    # ImageFolder is used through a generated, local manifest-free directory tree.
    for split, items in splits.items():
        for item in items:
            destination = root / split / item["label"] / Path(item["path"]).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists(): destination.write_bytes(Path(item["path"]).read_bytes())
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    train_tf = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10), transforms.ColorJitter(brightness=.15, contrast=.15), transforms.ToTensor(), transforms.Normalize(weights.transforms().mean, weights.transforms().std)])
    eval_tf = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize(weights.transforms().mean, weights.transforms().std)])
    train_ds = datasets.ImageFolder(root / "train", transform=train_tf); val_ds = datasets.ImageFolder(root / "validation", transform=eval_tf)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v3_small(weights=weights); model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes)); model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate); loss_fn = nn.CrossEntropyLoss(); best = float("inf"); patience_left = 3; history = []
    out = Path("artifacts/cnn"); out.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        model.train(); train_loss = 0.0
        for xb, yb in DataLoader(train_ds, batch_size=batch_size, shuffle=True):
            xb, yb = xb.to(device), yb.to(device); optimizer.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); optimizer.step(); train_loss += loss.item() * len(yb)
        model.eval(); val_loss = 0.0; correct = 0; total = 0
        with torch.no_grad():
            for xb, yb in DataLoader(val_ds, batch_size=batch_size):
                xb, yb = xb.to(device), yb.to(device); logits = model(xb); val_loss += loss_fn(logits, yb).item() * len(yb); correct += (logits.argmax(1) == yb).sum().item(); total += len(yb)
        row = {"epoch": epoch + 1, "train_loss": train_loss / max(len(train_ds), 1), "validation_loss": val_loss / max(len(val_ds), 1), "validation_accuracy": correct / max(total, 1)}; history.append(row); print(row)
        if row["validation_loss"] < best: best = row["validation_loss"]; patience_left = 3; torch.save(model.state_dict(), out / "best_checkpoint.pt")
        else: patience_left -= 1
        if patience_left <= 0: break
    config = {"classes": classes, "class_to_idx": class_to_idx, "image_size": image_size, "mean": list(weights.transforms().mean), "std": list(weights.transforms().std), "device": str(device), "history": history, "status": "trained"}
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8"); return config


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--data-dir", default="data/raw"); p.add_argument("--epochs", type=int, default=10); p.add_argument("--batch-size", type=int, default=32); p.add_argument("--learning-rate", type=float, default=3e-4); p.add_argument("--image-size", type=int, default=224); a = p.parse_args(); print(train(a.data_dir, a.epochs, a.batch_size, a.learning_rate, a.image_size))
