"""Weekly training entry point.

Replace `build_training_frame` and the estimator with the approved target/features.
The current application has no trained prediction model, so this is a safe scaffold,
not a claim that the existing rule-based rainfall page is ML-backed.
"""
from __future__ import annotations
import argparse, json, os, tempfile
from datetime import datetime, timezone
from pathlib import Path
import boto3, joblib, mlflow, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def download_file(bucket: str, key: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    boto3.client("s3").download_file(bucket, key, str(destination))


def build_training_frame(path: Path, target: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available columns: {list(df.columns)[:30]}")
    y = df[target]
    X = df.drop(columns=[target]).select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)
    if X.empty:
        raise ValueError("No numeric model features were found.")
    return X, y


def production_accuracy(bucket: str, metadata_key: str) -> float:
    try:
        obj = boto3.client("s3").get_object(Bucket=bucket, Key=metadata_key)
        return float(json.loads(obj["Body"].read())["accuracy"])
    except Exception:
        return float("-inf")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-bucket", required=True)
    p.add_argument("--data-key", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--model-bucket", required=True)
    p.add_argument("--model-key", default="nyc-resilience/models/production/model.pkl")
    p.add_argument("--metadata-key", default="nyc-resilience/models/production/metadata.json")
    args = p.parse_args()
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "nyc-resilience-weekly-training"))
    with tempfile.TemporaryDirectory() as td, mlflow.start_run(run_name="weekly-training"):
        data_path = Path(td) / Path(args.data_key).name
        download_file(args.data_bucket, args.data_key, data_path)
        X, y = build_training_frame(data_path, args.target)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(Xtr, ytr)
        accuracy = float(accuracy_score(yte, model.predict(Xte)))
        baseline = production_accuracy(args.model_bucket, args.metadata_key)
        mlflow.log_metrics({"accuracy": accuracy, "production_accuracy": baseline})
        mlflow.log_params({"target": args.target, "feature_count": X.shape[1], "rows": len(X)})
        candidate = Path(td) / "model.pkl"; joblib.dump(model, candidate)
        mlflow.log_artifact(str(candidate), artifact_path="candidate")
        if accuracy <= baseline:
            print(json.dumps({"promoted": False, "accuracy": accuracy, "production_accuracy": baseline}))
            raise SystemExit(2)
        metadata = {"accuracy": accuracy, "trained_at": datetime.now(timezone.utc).isoformat(), "features": list(X.columns), "mlflow_run_id": mlflow.active_run().info.run_id}
        meta_path = Path(td) / "metadata.json"; meta_path.write_text(json.dumps(metadata, indent=2))
        s3 = boto3.client("s3")
        s3.upload_file(str(candidate), args.model_bucket, args.model_key)
        s3.upload_file(str(meta_path), args.model_bucket, args.metadata_key)
        print(json.dumps({"promoted": True, **metadata}))

if __name__ == "__main__": main()
