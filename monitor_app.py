
from flask import Flask, request, jsonify, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import mlflow
import numpy as np
import time

app = Flask(__name__)

mlflow.set_tracking_uri("sqlite:///mlflow.db")

model = mlflow.pyfunc.load_model(
    "models:/Iris_RandomForest_Model/1"
)

REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
    ["status"]
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time taken to process predictions"
)

@app.route("/")
def home():
    return jsonify(
        application="Iris ML Monitoring API",
        status="running"
    )

@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()

    try:
        data = request.get_json()
        features = np.array(data["features"], dtype=float)

        if features.shape != (4,):
            raise ValueError("Exactly 4 features required")

        prediction = model.predict(
            features.reshape(1, -1)
        )

        REQUEST_COUNT.labels(status="success").inc()

        return jsonify({
            "prediction": int(prediction[0]),
            "status": "success"
        })

    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()

        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400

    finally:
        PREDICTION_LATENCY.observe(
            time.time() - start_time
        )

@app.route("/metrics")
def metrics():
    return Response(
        generate_latest(),
        mimetype=CONTENT_TYPE_LATEST
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003)
