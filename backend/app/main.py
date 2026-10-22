from __future__ import annotations

import io, json, os, time
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
MAX_BYTES = 10 * 1024 * 1024
ALLOWED = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}

app = FastAPI(title="Avocado Ripeness Classification API", version="1.0.0")
configured_origins = os.getenv("FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,https://greenpod-avocado-frontend.onrender.com")
app.add_middleware(CORSMiddleware, allow_origins=[origin.strip() for origin in configured_origins.split(",") if origin.strip()], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


def read_json(path: Path, default: Any) -> Any:
    try: return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except (OSError, json.JSONDecodeError): return default


def model_status() -> dict:
    classical = (ARTIFACTS / "classical/classical_pipeline.joblib").exists()
    cnn = (ARTIFACTS / "cnn/best_checkpoint.pt").exists() and (ARTIFACTS / "cnn/config.json").exists()
    return {"classical": {"available": classical, "status": "trained" if classical else "unavailable"}, "cnn": {"available": cnn, "status": "trained" if cnn else "unavailable"}}


@app.get("/api/health")
def health(): return {"status": "ok", "models": model_status()}


@app.get("/api/classes")
def classes():
    cnn = read_json(ARTIFACTS / "cnn/config.json", {})
    if cnn.get("classes"): return {"classes": cnn["classes"]}
    return {"classes": read_json(ARTIFACTS / "classical/classes.json", [])}


@app.get("/api/dataset/summary")
def dataset_summary(): return read_json(ARTIFACTS / "audit/dataset_summary.json", {"status": "unavailable", "message": "Run the dataset audit after placing images in data/raw/."})


@app.get("/api/models")
def models(): return model_status()


@app.get("/api/metrics")
def metrics(): return {"results": read_json(ARTIFACTS / "metrics/comparison.json", []), "status": "available" if (ARTIFACTS / "metrics/comparison.json").exists() else "unavailable"}


@app.get("/api/models/{model_name}/metrics")
def model_metrics(model_name: str):
    if model_name not in {"classical", "cnn"}: raise HTTPException(404, "Unknown model")
    result = read_json(ARTIFACTS / f"metrics/{model_name}.json", None)
    if result is None: raise HTTPException(404, "Metrics are unavailable for this model")
    return result


@app.get("/api/report", response_class=PlainTextResponse)
def report():
    path = ROOT / "reports/final_report.md"
    if not path.exists(): raise HTTPException(404, "Report is unavailable; run ml.generate_report first")
    return path.read_text(encoding="utf-8")


def validate_upload(data: bytes, content_type: str | None) -> Image.Image:
    if not data: raise HTTPException(400, "Uploaded file is empty")
    if len(data) > MAX_BYTES: raise HTTPException(413, "Uploaded file exceeds 10 MB")
    if content_type and content_type not in ALLOWED: raise HTTPException(415, "Unsupported image content type")
    try: return Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError): raise HTTPException(400, "Uploaded content is not a valid image")


def predict_classical(image: Image.Image):
    from ml.features import extract_features
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        image.save(f, format="PNG"); path=f.name
    try:
        model=joblib.load(ARTIFACTS / "classical/classical_pipeline.joblib"); start=time.perf_counter(); probabilities=model.predict_proba([extract_features(path)])[0]; duration=(time.perf_counter()-start)*1000; classes=model.classes_.tolist()
        return classes[int(probabilities.argmax())], float(probabilities.max()), dict(zip(classes, map(float, probabilities))), duration
    finally: Path(path).unlink(missing_ok=True)


_CNN_MODEL = None
_CNN_CONFIG = None


def predict_cnn(image: Image.Image):
    global _CNN_MODEL, _CNN_CONFIG
    try:
        import torch
        from torchvision import models, transforms
        from torch import nn
    except ImportError as exc:
        raise HTTPException(503, "CNN inference dependencies are unavailable") from exc
    if _CNN_MODEL is None:
        config = read_json(ARTIFACTS / "cnn/config.json", {})
        classes = config.get("classes", [])
        if not classes:
            raise HTTPException(503, "CNN configuration is unavailable")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
        try:
            state = torch.load(ARTIFACTS / "cnn/best_checkpoint.pt", map_location=device, weights_only=True)
            model.load_state_dict(state)
        except (OSError, RuntimeError, TypeError) as exc:
            raise HTTPException(503, "CNN checkpoint could not be loaded") from exc
        model.to(device).eval()
        _CNN_MODEL, _CNN_CONFIG = model, {"classes": classes, "device": device, "image_size": config.get("image_size", 224), "mean": config.get("mean", [0.485, 0.456, 0.406]), "std": config.get("std", [0.229, 0.224, 0.225])}
    import torch
    transform = transforms.Compose([transforms.Resize((_CNN_CONFIG["image_size"], _CNN_CONFIG["image_size"])), transforms.ToTensor(), transforms.Normalize(_CNN_CONFIG["mean"], _CNN_CONFIG["std"])])
    tensor = transform(image).unsqueeze(0).to(_CNN_CONFIG["device"])
    start = time.perf_counter()
    with torch.inference_mode():
        probabilities = torch.softmax(_CNN_MODEL(tensor), dim=1)[0].detach().cpu().numpy()
    duration = (time.perf_counter() - start) * 1000
    classes = _CNN_CONFIG["classes"]
    return classes[int(probabilities.argmax())], float(probabilities.max()), dict(zip(classes, map(float, probabilities))), duration


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), model: str = "classical"):
    if model not in {"classical", "cnn"}: raise HTTPException(400, "Unknown model; choose classical or cnn")
    if not model_status()[model]["available"]: raise HTTPException(503, f"{model} model artifact is unavailable")
    image=validate_upload(await file.read(), file.content_type)
    label, confidence, scores, duration = predict_classical(image) if model == "classical" else predict_cnn(image)
    return {"model": model, "predicted_label": label, "confidence": confidence, "scores": scores, "inference_duration_ms": duration}


@app.post("/api/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...), model: str = "classical"):
    if len(files) > 32: raise HTTPException(400, "Maximum batch size is 32")
    results=[]
    for file in files:
        result=await predict(file, model); results.append({"filename": file.filename, **result})
    return {"results": results}
