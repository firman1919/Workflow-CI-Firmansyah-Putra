import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn


def main():
    # Set MLflow tracking (LOCAL)
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment("CoffeeShop_Menu_Clustering")

    # Load dataset hasil preprocessing
    df = pd.read_csv("coffeshop_preprocessing.csv")

    # Ambil fitur numerik
    X = df[['avg_price', 'total_order']]

    # Scaling (aman meskipun sudah di-scale sebelumnya)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model KMeans (BASIC, tanpa tuning)
    kmeans = KMeans(n_clusters=3, random_state=42)

    # Aktifkan autolog
    mlflow.sklearn.autolog()

    # Training
    with mlflow.start_run():
        kmeans.fit(X_scaled)


if __name__ == "__main__":
    main()