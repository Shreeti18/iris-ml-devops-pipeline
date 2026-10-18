
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Configure MLflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Iris_Classification_Experiment")

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train models with different parameters
for n_estimators in [10, 50, 100]:

    with mlflow.start_run(run_name=f"RF_{n_estimators}_Trees"):

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, average="weighted",
            zero_division=0
        )
        recall = recall_score(
            y_test, predictions, average="weighted",
            zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, average="weighted",
            zero_division=0
        )

        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)

        # Log evaluation metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        
        # Log trained model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="iris_model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        print(f"\nModel: Random Forest ({n_estimators} trees)")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("-" * 40)

print("\nAll experiments completed successfully!")
