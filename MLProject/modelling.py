import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True)
args = parser.parse_args()

df = pd.read_csv(args.input_path)

X = df.select_dtypes(include="number")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with mlflow.start_run():
    model = KMeans(n_clusters=3, random_state=42)
    model.fit(X_scaled)

    mlflow.sklearn.log_model(model, "model")
