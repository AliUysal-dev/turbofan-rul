import pytest
from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_health_endpoint():
    """Ö-07: /health uç noktasının HTTP 200 ve JSON şeması döndürdüğünü doğrular."""
    with TestClient(app) as test_client:
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data


def test_predict_validation_under_20_cycles():
    """Ö-03: 20 döngünün altındaki isteklerin Pydantic tarafından HTTP 422 ile reddedildiğini doğrular."""
    short_payload = {
        "unit_number": 1,
        "history": [
            {
                "time_in_cycles": 1,
                "sensors": {f"sensor_{s}": 10.0 for s in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]}
            }
        ]
    }
    response = client.post("/predict", json=short_payload)
    assert response.status_code == 422


def test_predict_validation_missing_sensor():
    """Ö-04: Eksik sensörlü isteklerin HTTP 422 ile reddedildiğini ve eksik sensörün raporlandığını doğrular."""
    # 20 döngü var ama sensor_21 eksik
    faulty_history = []
    for i in range(1, 21):
        sensors_dict = {f"sensor_{s}": 10.0 for s in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20]}  # sensor_21 yok
        faulty_history.append({"time_in_cycles": i, "sensors": sensors_dict})

    payload = {
        "unit_number": 1,
        "history": faulty_history
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "Eksik sensör" in response.text