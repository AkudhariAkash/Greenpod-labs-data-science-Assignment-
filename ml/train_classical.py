from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .dataset import discover_records, load_splits, make_splits, save_splits, validate_image
from .features import extract_features


def train(data_dir: str = "data/raw", mapping_csv: str | None = None, seed: int = 42) -> dict:
    records = discover_records(data_dir, mapping_csv)
    for r in records:
        r.update(validate_image(r["path"]))
    splits = load_splits() or make_splits(records, seed=seed)
    save_splits(splits)
    x = np.stack([extract_features(r["path"]) for r in splits["train"]])
    y = [r["label"] for r in splits["train"]]
    model = Pipeline([("scaler", StandardScaler()), ("classifier", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed))])
    model.fit(x, y)
    out = Path("artifacts/classical"); out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "classical_pipeline.joblib")
    (out / "classes.json").write_text(json.dumps(sorted(model.classes_.tolist()), indent=2), encoding="utf-8")
    return {"status": "trained", "classes": sorted(model.classes_.tolist()), "train_images": len(y)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--data-dir", default="data/raw"); p.add_argument("--mapping-csv"); p.add_argument("--seed", type=int, default=42); a = p.parse_args(); print(train(a.data_dir, a.mapping_csv, a.seed))
