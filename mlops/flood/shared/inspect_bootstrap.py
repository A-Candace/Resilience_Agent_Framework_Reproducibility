from __future__ import annotations

import os

import boto3
import pandas as pd


BUCKET = os.getenv(
    "DATA_S3_BUCKET",
    "nyc-resilience-data",
)

MERGED_KEY = (
    "mlops/flood/bootstrap/"
    "merged_mrms_floodnet.parquet"
)

HOURLY_KEY = (
    "mlops/flood/bootstrap/"
    "all_sensor_hourly_model_1mile.parquet"
)


def _s3_uri(key: str) -> str:
    return f"s3://{BUCKET}/{key}"


def inspect_merged() -> None:
    uri = _s3_uri(MERGED_KEY)

    print("\n=== merged_mrms_floodnet.parquet ===")
    print(f"Source: {uri}")

    df = pd.read_parquet(
        uri,
        columns=[
            "deployment_id",
            "time",
            "depth_proc_mm",
        ],
    )

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True,
        errors="coerce",
    )

    print(f"Rows: {len(df):,}")
    print(
        f"Sensors: "
        f"{df['deployment_id'].nunique():,}"
    )
    print(
        f"Start: {df['time'].min()}"
    )
    print(
        f"End:   {df['time'].max()}"
    )

    print("\nMissing values:")
    print(
        df[
            [
                "deployment_id",
                "time",
                "depth_proc_mm",
            ]
        ]
        .isna()
        .sum()
    )


def inspect_hourly() -> None:
    uri = _s3_uri(HOURLY_KEY)

    print(
        "\n=== "
        "all_sensor_hourly_model_1mile.parquet "
        "==="
    )
    print(f"Source: {uri}")

    df = pd.read_parquet(uri)

    print(f"Rows: {len(df):,}")

    if "deployment_id" in df.columns:
        print(
            f"Sensors: "
            f"{df['deployment_id'].nunique():,}"
        )

    if "hour" in df.columns:
        df["hour"] = pd.to_datetime(
            df["hour"],
            utc=True,
            errors="coerce",
        )

        print(
            f"Start: {df['hour'].min()}"
        )
        print(
            f"End:   {df['hour'].max()}"
        )

    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")

    print("\nMissing values:")
    print(
        df.isna().sum().sort_values(
            ascending=False
        )
    )


def main() -> None:
    inspect_merged()
    inspect_hourly()


if __name__ == "__main__":
    main()