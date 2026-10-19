
import os

os.environ["MODEL_URI"] = "deployment_model"

import pytest
from monitor_app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


def test_home(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json["status"] == "running"


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json["status"] == "healthy"


def test_prediction(client):
    response = client.post(
        "/predict",
        json={
            "features": [5.1, 3.5, 1.4, 0.2]
        }
    )

    assert response.status_code == 200
    assert response.json["status"] == "success"
    assert response.json["prediction"] in [0, 1, 2]


def test_invalid_prediction(client):
    response = client.post(
        "/predict",
        json={
            "features": [5.1, 3.5]
        }
    )

    assert response.status_code == 400
    assert response.json["status"] == "error"


def test_metrics(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert b"prediction_requests_total" in response.data
