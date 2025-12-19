import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import os
import sys


def main():
    # ===============================
    # Ambil parameter dari MLflow Project
    # ===============================
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "coffeshop_preprocessing.csv"

    # Pastikan path aman di CI
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(BASE_DIR, dataset_path)

    # ===============================
    # Load dataset
    # ===============================
    df = pd.read_csv(dataset_path)

    # Ambil fitur numerik
    X = df[['avg_price', 'total_order']]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)

    # Aktifkan autolog
    mlflow.sklearn.autolog()

    # Training
    with mlflow.start_run():
        kmeans.fit(X_scaled)


if __name__ == "__main__":
    main()
