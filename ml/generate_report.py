from __future__ import annotations
import json
from pathlib import Path

def main():
    audit_path=Path("artifacts/audit/dataset_summary.json"); metrics_path=Path("artifacts/metrics/comparison.json")
    audit=json.loads(audit_path.read_text()) if audit_path.exists() else {"status":"unavailable"}; results=json.loads(metrics_path.read_text()) if metrics_path.exists() else []
    rows="\n".join(f"| {r['model']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['balanced_accuracy']:.4f} | {r['single_image_latency_ms']:.3f} |" for r in results) or "| unavailable | unavailable | unavailable | unavailable | unavailable |"
    report=f'''# Avocado Ripeness Classification and Model Comparison

## Objective and abstract

This project classifies avocado ripening stages from images using two distinct approaches: engineered color/texture features with a classical classifier, and a lightweight transfer-learning CNN. Results below are generated from saved artifacts only.

## Dataset and audit

Source: https://drive.google.com/file/d/1NMJ0FOxcWH1mwFFeERuY6fgEY-MVgeET/view

- Audit status: `{audit.get("status", "unavailable")}`
- Readable images: `{audit.get("readable_images", "unavailable")}`
- Classes: `{", ".join(audit.get("classes", [])) or "unavailable"}`
- Class counts: `{audit.get("class_counts", "unavailable")}`
- Exact duplicate image count: `{audit.get("duplicate_image_count", "unavailable")}`

No class names or statistics are assumed from the assignment examples. Lighting, camera, background, and variety can introduce sampling bias; the audit records measurable checks and labels manual-review concerns separately.

## Preprocessing and split strategy

Readable files are discovered from class directories or a documented mapping CSV. Exact duplicates are recorded. A fixed-seed stratified train/validation/test split is saved to `artifacts/splits.json`; if classes are too small, the split fails with an explanation. Augmentation is restricted to the CNN training split.

## Approach 1: classical computer vision

Each image is represented with normalized RGB, HSV, and LAB channel histograms, channel means and standard deviations, and an LBP texture histogram. A scaled RBF SVM is trained only on the training partition. Color and texture features can correlate with stage differences, but they are sensitive to lighting, background, camera settings, and variety; no individual channel is treated as a direct ripeness measurement.

## Approach 2: transfer learning CNN

MobileNetV3-Small is initialized with torchvision ImageNet weights when available, receives a head sized to the discovered classes, and is trained with resized normalized images, training-only augmentation, AdamW, validation monitoring, and early stopping. The checkpoint stores class mapping and preprocessing configuration.

## Metrics and comparison

Accuracy reports overall correctness. Macro precision, recall, and F1 give each class equal weight, which is important under imbalance. Weighted F1 reflects the observed class frequencies. Balanced accuracy averages per-class recall. Latency is the measured mean single-image prediction time on the local runtime; it is not a production SLA.

| Model | Accuracy | Macro F1 | Balanced accuracy | Single-image latency (ms) |
|---|---:|---:|---:|---:|
{rows}

If a model or metric is unavailable, it is shown as unavailable rather than replaced with an example value. The better model is the one with stronger macro F1 and balanced accuracy while considering latency and per-class recall; no conclusion is drawn when results are missing or inconclusive.

## Mandatory questions

### 1. In production, how does your model perform?

Production performance has not been validated unless a deployment and external test set are documented here. The held-out test results and measured local latency are not equivalent to production performance. New lighting, cameras, avocado varieties, backgrounds, and changing data distributions require external validation and monitoring.

### 2. What is the drawback of your model?

Observed drawbacks are limited to the audit and evaluation output available above. General risks include class overlap, label noise, lighting/background sensitivity for engineered features, and limited generalization when the training data is small or imbalanced. The classical model is compact but less invariant; the CNN may be more robust but needs more compute and data.

### 3. If given more time, what would you improve?

Priorities are balanced data collection across devices and environments, label review for visually adjacent stages, duplicate/near-duplicate grouping before splitting, external validation, confidence calibration, and measuring quantized CNN inference on target hardware. Improvements should be selected after reviewing the generated audit and error analysis.

### 4. Where did you spend more time while solving this assignment?

The work is concentrated on dataset discovery and quality checks, leakage-safe splitting, feature engineering, transfer-learning integration, fair comparison, API/frontend integration, and tests. The proportions should be edited to reflect the candidate's actual work rather than invented hours.

## Reproducibility and limitations

Follow `README.md` after placing the dataset in `data/raw/`. This report is intentionally honest when the dataset, training, or external validation is unavailable.
'''
    Path("reports").mkdir(exist_ok=True); Path("reports/final_report.md").write_text(report,encoding="utf-8"); print("reports/final_report.md")
if __name__=="__main__": main()
