
from flask import Flask, request, jsonify, Response
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST
)

import mlflow
import numpy as np
import time
import os

# ------------------------------------------
# 1. Initialize Flask Application
# ------------------------------------------

app = Flask(__name__)

# ------------------------------------------
# 2. Configure MLflow and Load Model
# ------------------------------------------

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
)

# Local: loads the registered MLflow model
# Docker: loads the exported deployment_model
MODEL_URI = os.getenv(
    "MODEL_URI",
    "models:/Iris_RandomForest_Model/1"
)

print(f"Loading ML model from: {MODEL_URI}")

model = mlflow.pyfunc.load_model(MODEL_URI)

print("ML model loaded successfully!")

# ------------------------------------------
# 3. Initialize Prometheus Metrics
# ------------------------------------------

REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
    ["status"]
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time taken to process predictions"
)

# ------------------------------------------
# 4. Home Endpoint
# ------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "application": "Iris ML Monitoring API",
        "status": "running",
        "model": "Iris_RandomForest_Model",
        "version": "1",
        "monitoring": "Prometheus"
    })


# ------------------------------------------
# 5. Prediction Endpoint
# ------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    start_time = time.time()

    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict) or "features" not in data:
            raise ValueError(
                "Request must contain a 'features' field"
            )

        features = np.asarray(
            data["features"],
            dtype=float
        )

        if features.shape != (4,):
            raise ValueError(
                "Exactly 4 numeric features are required"
            )

        if not np.isfinite(features).all():
            raise ValueError(
                "All feature values must be finite numbers"
            )

        prediction = model.predict(
            features.reshape(1, -1)
        )

        REQUEST_COUNT.labels(
            status="success"
        ).inc()

        return jsonify({
            "prediction": int(prediction[0]),
            "status": "success"
        })

    except (ValueError, TypeError, KeyError) as e:

        REQUEST_COUNT.labels(
            status="error"
        ).inc()

        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400

    except Exception:

        app.logger.exception("Prediction failed")

        REQUEST_COUNT.labels(
            status="error"
        ).inc()

        return jsonify({
            "error": "Prediction failed",
            "status": "error"
        }), 500

    finally:

        PREDICTION_LATENCY.observe(
            time.time() - start_time
        )


# ------------------------------------------
# 6. Prometheus Metrics Endpoint
# ------------------------------------------

@app.route("/metrics", methods=["GET"])
def metrics():

    return Response(
        generate_latest(),
        mimetype=CONTENT_TYPE_LATEST
    )


# ------------------------------------------
# 7. Health Check Endpoint
# ------------------------------------------

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })


# ------------------------------------------
# 8. Run Flask Application
# ------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5003
    )
 