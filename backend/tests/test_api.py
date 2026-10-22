from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_dataset_summary_is_honest_when_missing():
    response = client.get("/api/dataset/summary")
    assert response.status_code == 200
    assert "status" in response.json()

def test_empty_upload_rejected():
    response = client.post("/api/predict", files={"file": ("empty.png", b"", "image/png")})
    assert response.status_code in {400, 503}

def test_report_missing_or_available():
    assert client.get("/api/report").status_code in {200, 404}
