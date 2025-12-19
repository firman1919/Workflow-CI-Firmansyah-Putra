import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
import sys
import os

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # =====================
    # Read arguments from MLProject
    # =====================
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 35
    dataset_path = sys.argv[3] if len(sys.argv) > 3 else "train_pca.csv"

    # Make sure dataset path works in CI
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(os.path.dirname(__file__), dataset_path)

    # =====================
    # Load dataset
    # =====================
    data = pd.read_csv(dataset_path)

    X = data.drop("Credit_Score", axis=1)
    y = data["Credit_Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    input_example = X_train.head(5)

    # =====================
    # MLflow Training
    # =====================
    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )
