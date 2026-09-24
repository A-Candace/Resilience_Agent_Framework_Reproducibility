"""Download required NYC Resilience datasets from S3."""

from __future__ import annotations

import os
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        raise RuntimeError(
            f"Required environment variable is missing: {name}"
        )

    return value


def download_object(
    s3_client,
    bucket: str,
    key: str,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Downloading s3://{bucket}/{key} "
        f"-> {destination}"
    )

    s3_client.download_file(
        bucket,
        key,
        str(destination),
    )


def download_prefix_flat(
    s3_client,
    bucket: str,
    prefix: str,
    destination: Path,
) -> int:
    """
    Download all objects under an S3 prefix into one local directory.

    This is appropriate for shapefile components because they must remain
    together and share the same basename.
    """

    normalized_prefix = (
        prefix.strip("/") + "/"
    )

    paginator = s3_client.get_paginator(
        "list_objects_v2"
    )

    downloaded = 0

    for page in paginator.paginate(
        Bucket=bucket,
        Prefix=normalized_prefix,
    ):
        for item in page.get(
            "Contents",
            [],
        ):
            key = item["Key"]

            if key.endswith("/"):
                continue

            filename = Path(key).name

            target = (
                destination / filename
            )

            download_object(
                s3_client=s3_client,
                bucket=bucket,
                key=key,
                destination=target,
            )

            downloaded += 1

    return downloaded


def validate_runtime_files(
    runtime_dir: Path,
) -> None:
    required = [
        runtime_dir
        / "Risk_Attributes_Table_v4.xlsx",
        runtime_dir
        / "nyc_flood_risk_grid_4km.geojson",
        runtime_dir
        / "nyc_census_tracts.shp",
        runtime_dir
        / "nyc_census_tracts.shx",
        runtime_dir
        / "nyc_census_tracts.dbf",
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        formatted = "\n".join(
            f"  - {path}"
            for path in missing
        )

        raise RuntimeError(
            "S3 synchronization completed, but required files "
            "are missing:\n"
            f"{formatted}"
        )


def validate_forecast_files(
    forecast_root: Path,
    spatial_root: Path,
) -> None:
    required = []

    for mode in (
        "today",
        "tomorrow",
    ):
        mode_dir = (
            forecast_root / mode
        )

        required.extend(
            [
                mode_dir
                / "grid_flood_forecast.parquet",
                mode_dir
                / "grid_flood_forecast_summary.json",
                mode_dir
                / "grid_precipitation_forecast_summary.json",
            ]
        )

    required.append(
        spatial_root
        / "grid_imputation_reference.geojson"
    )

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        formatted = "\n".join(
            f"  - {path}"
            for path in missing
        )

        raise RuntimeError(
            "S3 synchronization completed, but required "
            "operational forecast files are missing:\n"
            f"{formatted}"
        )


def main() -> None:
    bucket = required_env(
        "DATA_S3_BUCKET"
    )

    region = os.getenv(
        "AWS_REGION",
        "us-east-1",
    ).strip()

    runtime_dir = Path(
        os.getenv(
            "DATA_RUNTIME_DIR",
            "/app/data/runtime",
        )
    )

    forecast_root = Path(
        os.getenv(
            "GRID_FLOOD_FORECAST_ROOT",
            "/app/artifacts/flood/forecasting",
        )
    )

    spatial_root = Path(
        os.getenv(
            "GRID_SPATIAL_ROOT",
            "/app/artifacts/flood/spatial",
        )
    )

    census_prefix = os.getenv(
        "S3_CENSUS_TRACTS_PREFIX",
        "data/raw/geospatial/census-tracts",
    )

    risk_xlsx_key = os.getenv(
        "S3_RISK_XLSX_KEY",
        "data/raw/tabular/tract-attributes/"
        "Risk_Attributes_Table_v4.xlsx",
    )

    flood_grid_key = os.getenv(
        "S3_FLOOD_GRID_KEY",
        "data/raw/geospatial/flood-forecast/"
        "nyc_flood_risk_grid_4km.geojson",
    )

    forecast_prefix = os.getenv(
        "S3_FORECAST_PREFIX",
        "artifacts/flood/forecasting",
    ).strip("/")

    grid_imputation_geojson_key = os.getenv(
        "S3_GRID_IMPUTATION_GEOJSON_KEY",
        "artifacts/flood/spatial/"
        "grid_imputation_reference.geojson",
    )

    runtime_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    spatial_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    s3 = boto3.client(
        "s3",
        region_name=region,
    )

    try:
        census_count = (
            download_prefix_flat(
                s3_client=s3,
                bucket=bucket,
                prefix=census_prefix,
                destination=runtime_dir,
            )
        )

        download_object(
            s3_client=s3,
            bucket=bucket,
            key=risk_xlsx_key,
            destination=(
                runtime_dir
                / "Risk_Attributes_Table_v4.xlsx"
            ),
        )

        download_object(
            s3_client=s3,
            bucket=bucket,
            key=flood_grid_key,
            destination=(
                runtime_dir
                / "nyc_flood_risk_grid_4km.geojson"
            ),
        )

        for mode in (
            "today",
            "tomorrow",
        ):
            for filename in (
                "grid_flood_forecast.parquet",
                "grid_flood_forecast_summary.json",
                "grid_precipitation_forecast_summary.json",
            ):
                key = (
                    f"{forecast_prefix}/"
                    f"{mode}/"
                    f"{filename}"
                )

                destination = (
                    forecast_root
                    / mode
                    / filename
                )

                download_object(
                    s3_client=s3,
                    bucket=bucket,
                    key=key,
                    destination=destination,
                )

        download_object(
            s3_client=s3,
            bucket=bucket,
            key=grid_imputation_geojson_key,
            destination=(
                spatial_root
                / "grid_imputation_reference.geojson"
            ),
        )

        validate_runtime_files(
            runtime_dir
        )

        validate_forecast_files(
            forecast_root,
            spatial_root,
        )

        print(
            "S3 asset synchronization completed successfully. "
            f"Downloaded {census_count} census-tract files, "
            "hydrated today/tomorrow operational forecasts, "
            "and hydrated the 1-km forecasting GeoJSON."
        )

    except (
        BotoCoreError,
        ClientError,
    ) as exc:
        raise RuntimeError(
            f"Could not synchronize S3 assets: {exc}"
        ) from exc


if __name__ == "__main__":
    main()