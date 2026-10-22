from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def discover_records(data_dir: str | Path, mapping_csv: str | Path | None = None) -> list[dict]:
    root = Path(data_dir)
    if mapping_csv:
        import pandas as pd
        frame = pd.read_csv(mapping_csv)
        required = {"path", "label"}
        if not required.issubset(frame.columns):
            raise ValueError("Mapping CSV must contain path and label columns")
        rows = []
        for row in frame.itertuples(index=False):
            path = Path(str(row.path))
            if not path.is_absolute():
                path = root / path
            rows.append({"path": str(path), "label": str(row.label)})
        return rows
    rows = []
    if not root.exists():
        return rows
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS):
            rows.append({"path": str(path), "label": class_dir.name})
    return rows


def image_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_image(path: str | Path) -> dict:
    result = {"path": str(path), "readable": False, "reason": None}
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        result["reason"] = "missing_or_empty"
        return result
    try:
        with Image.open(p) as image:
            image.verify()
        with Image.open(p) as image:
            result.update({"readable": True, "width": image.width, "height": image.height, "mode": image.mode, "format": image.format})
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        result["reason"] = f"unreadable:{type(exc).__name__}"
    return result


def make_splits(records: list[dict], seed: int = 42, test_size: float = 0.15, validation_size: float = 0.15) -> dict[str, list[dict]]:
    valid = [r for r in records if r.get("readable", True)]
    if len(valid) < 6 or len({r["label"] for r in valid}) < 2:
        raise ValueError("At least 6 readable images and 2 classes are required for a train/validation/test split")
    labels = [r["label"] for r in valid]
    try:
        train_val, test = train_test_split(valid, test_size=test_size, random_state=seed, stratify=labels)
        relative_val = validation_size / (1 - test_size)
        train, val = train_test_split(train_val, test_size=relative_val, random_state=seed, stratify=[r["label"] for r in train_val])
    except ValueError as exc:
        raise ValueError(f"Stratified split failed; a class may be too small: {exc}") from exc
    return {"train": train, "validation": val, "test": test}


def save_splits(splits: dict, path: str | Path = "artifacts/splits.json") -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(splits, indent=2), encoding="utf-8")


def load_splits(path: str | Path = "artifacts/splits.json") -> dict | None:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
