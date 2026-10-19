
import mlflow
import shutil
from pathlib import Path

mlflow.set_tracking_uri("sqlite:///mlflow.db")

model_uri = "models:/Iris_RandomForest_Model/1"
destination = Path("deployment_model")

if destination.exists():
    shutil.rmtree(destination)

mlflow.artifacts.download_artifacts(
    artifact_uri=model_uri,
    dst_path=str(destination)
)

print("Model downloaded successfully!")
print("Deployment folder:", destination.resolve())
