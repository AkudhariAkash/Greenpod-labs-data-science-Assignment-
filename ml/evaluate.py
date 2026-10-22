from __future__ import annotations

import argparse, json, time
from pathlib import Path
import joblib, numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from .dataset import load_splits
from .features import extract_features

def metrics(y_true, y_pred, labels, latency_ms, model_name):
    return {"model": model_name, "accuracy": accuracy_score(y_true, y_pred), "macro_precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), "macro_recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0), "balanced_accuracy": balanced_accuracy_score(y_true, y_pred), "classification_report": classification_report(y_true, y_pred, labels=labels, target_names=labels, output_dict=True, zero_division=0), "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(), "labels": labels, "single_image_latency_ms": latency_ms}

def evaluate_classical(splits):
    model = joblib.load("artifacts/classical/classical_pipeline.joblib"); y_true=[]; y_pred=[]; elapsed=[]
    for r in splits["test"]:
        start=time.perf_counter(); pred=model.predict([extract_features(r["path"])])[0]; elapsed.append((time.perf_counter()-start)*1000); y_true.append(r["label"]); y_pred.append(pred)
    return metrics(y_true,y_pred,sorted(model.classes_.tolist()),float(np.mean(elapsed)),"classical")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--splits",default="artifacts/splits.json"); a=p.parse_args(); splits=load_splits(a.splits)
    if not splits: raise RuntimeError("No split manifest found; train a model first")
    out=Path("artifacts/metrics"); out.mkdir(parents=True,exist_ok=True); results=[]
    if Path("artifacts/classical/classical_pipeline.joblib").exists(): results.append(evaluate_classical(splits))
    # CNN evaluation is intentionally optional; its artifact is consumed by the API when present.
    for result in results: (out/f"{result['model']}.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    (out/"comparison.json").write_text(json.dumps(results,indent=2),encoding="utf-8"); print(json.dumps(results,indent=2))
if __name__=="__main__": main()
