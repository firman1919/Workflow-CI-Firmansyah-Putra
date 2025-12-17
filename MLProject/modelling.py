import argparse
import pandas as pd
import os
import mlflow
from sklearn.cluster import KMeans

def main(input_path):
    df = pd.read_csv(input_path)
    print("Dataset loaded. Rows:", len(df))

    # Contoh preprocessing sederhana (anggap sudah scaled)
    X = df[['avg_price', 'total_order']]

    # Training KMeans (misal 3 cluster)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    df['cluster'] = kmeans.labels_

    # Logging ke MLflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment("CoffeeShop_Menu_Clustering")
    with mlflow.start_run():
        mlflow.log_param("n_clusters", 3)
        mlflow.log_metric("inertia", kmeans.inertia_)
        # Simpan model
        os.makedirs("artifacts", exist_ok=True)
        model_path = os.path.join("artifacts", "kmeans_model.pkl")
        import joblib
        joblib.dump(kmeans, model_path)
        mlflow.log_artifact(model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    args = parser.parse_args()
    main(args.input_path)
