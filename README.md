# Avocado Ripeness Classification and Model Comparison Platform

This repository implements the Greenpod Labs Data Science Assignment 2 as a reproducible project. It contains:

- a dataset audit with corruption, duplicate, dimension, quality, and class-balance checks;
- a classical OpenCV/scikit-learn pipeline using HSV, LAB, RGB statistics, and LBP texture features;
- a lightweight transfer-learning classifier using MobileNetV3-Small when torchvision/pretrained weights are available;
- fair evaluation on one held-out test split;
- a FastAPI service that remains usable when artifacts are missing;
- a React/Vite frontend that reads dataset and metric results from the API;
- tests for feature extraction, preprocessing, and API behavior.

No dataset, model metrics, predictions, or production claims are included until the supplied dataset has actually been downloaded and processed.

## Dataset setup

Download the dataset from the assignment source:

<https://drive.google.com/file/d/1NMJ0FOxcWH1mwFFeERuY6fgEY-MVgeET/view>

Extract the archive under `data/raw/`. The preferred layout is:

```text
data/raw/<class-name>/*.jpg
data/raw/<class-name>/*.png
```

Class names are discovered from the actual directory names. A flat image directory can also be mapped with a CSV containing `path` and `label` columns; pass it with `--mapping-csv` to the audit/training scripts. Do not rename classes based on examples in the assignment.

## Windows PowerShell quick start

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python -m ml.audit_dataset --data-dir data/raw
python -m ml.train_classical --data-dir data/raw
python -m ml.train_cnn --data-dir data/raw --epochs 10
python -m ml.evaluate --data-dir data/raw
python -m ml.generate_report

uvicorn backend.app.main:app --reload
```

Open the API docs at `http://127.0.0.1:8000/docs`. In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Linux/macOS use `python3 -m venv .venv`, `source .venv/bin/activate`, and the equivalent `npm` commands.

## Outputs

- `artifacts/audit/dataset_summary.json` and audit images;
- `artifacts/classical/classical_pipeline.joblib`;
- `artifacts/cnn/best_checkpoint.pt` and `config.json`;
- `artifacts/metrics/*.json` and comparison charts;
- `reports/final_report.md`.

Scripts fail clearly when the dataset is absent or too small for a valid split. They do not fabricate results. The API reports missing artifacts instead of starting training automatically.

## API and testing

```powershell
python -m pytest -q
```

Useful endpoints are `/api/health`, `/api/classes`, `/api/dataset/summary`, `/api/models`, `/api/metrics`, `/api/report`, and `POST /api/predict`.

## Reproducibility

The default seed is 42. The split manifest is saved in `artifacts/splits.json`; training scripts consume the same manifest when it exists. Test data is not used for model selection. Training augmentation is only applied to the training dataset.

## Assignment questions

The generated report answers all four mandatory questions using observed audit/evaluation output where available and explicitly labels unmeasured production behavior as a limitation.
