"""
Progress manifest utilities for the canonical flood history build.

The manifest records the status of each monthly partition so that
long historical builds are auditable, resumable, and easier to debug.

Statuses:
- completed
- skipped
- failed
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3

from mlops.flood.shared.config import (
    FLOOD_ARTIFACT_DIR,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_S3_BUCKET = os.getenv(
    "DATA_S3_BUCKET",
    "nyc-resilience-data",
)

MANIFEST_S3_KEY = os.getenv(
    "FLOOD_HISTORY_MANIFEST_KEY",
    "mlops/flood/manifests/canonical_history_manifest.json",
)

LOCAL_MANIFEST_PATH = (
    FLOOD_ARTIFACT_DIR
    / "canonical_history_manifest.json"
)


# ============================================================
# HELPERS
# ============================================================

def utc_now_iso() -> str:
    """
    Return the current UTC timestamp in ISO-8601 format.
    """

    return datetime.now(
        timezone.utc
    ).isoformat()


def month_key(
    year: int,
    month: int,
) -> str:
    """
    Return a stable YYYY-MM identifier for a monthly partition.
    """

    return (
        f"{year:04d}-"
        f"{month:02d}"
    )


# ============================================================
# EMPTY MANIFEST
# ============================================================

def empty_manifest() -> dict[str, Any]:
    """
    Create a new manifest structure.
    """

    now = utc_now_iso()

    return {
        "schema_version": 1,
        "created_at": now,
        "updated_at": now,
        "partitions": {},
    }


# ============================================================
# LOCAL LOAD / SAVE
# ============================================================

def load_local_manifest() -> dict[str, Any]:
    """
    Load the local manifest if one exists.

    Otherwise return a new empty manifest.
    """

    if not LOCAL_MANIFEST_PATH.exists():
        return empty_manifest()

    with LOCAL_MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_local_manifest(
    manifest: dict[str, Any],
) -> Path:
    """
    Persist the manifest locally.
    """

    LOCAL_MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest["updated_at"] = utc_now_iso()

    with LOCAL_MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            sort_keys=True,
        )

    return LOCAL_MANIFEST_PATH


# ============================================================
# S3 LOAD / SAVE
# ============================================================

def load_manifest_from_s3() -> dict[str, Any]:
    """
    Load the persistent manifest from S3.

    If it does not exist yet, return an empty manifest.
    """

    client = boto3.client(
        "s3"
    )

    try:
        response = client.get_object(
            Bucket=DATA_S3_BUCKET,
            Key=MANIFEST_S3_KEY,
        )

    except client.exceptions.NoSuchKey:
        return empty_manifest()

    except client.exceptions.ClientError as exc:
        code = (
            exc.response
            .get("Error", {})
            .get("Code")
        )

        if code in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            return empty_manifest()

        raise

    body = (
        response["Body"]
        .read()
        .decode("utf-8")
    )

    return json.loads(
        body
    )


def upload_manifest_to_s3(
    manifest: dict[str, Any],
) -> None:
    """
    Upload the manifest JSON to S3.
    """

    manifest["updated_at"] = utc_now_iso()

    payload = json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")

    boto3.client(
        "s3"
    ).put_object(
        Bucket=DATA_S3_BUCKET,
        Key=MANIFEST_S3_KEY,
        Body=payload,
        ContentType="application/json",
    )


# ============================================================
# MANIFEST LOAD
# ============================================================

def load_manifest() -> dict[str, Any]:
    """
    Load the authoritative manifest.

    S3 is preferred because it survives across machines and
    future automated jobs.
    """

    manifest = load_manifest_from_s3()

    save_local_manifest(
        manifest
    )

    return manifest


# ============================================================
# PARTITION STATUS
# ============================================================

def get_partition_status(
    manifest: dict[str, Any],
    partition: str,
) -> str | None:
    """
    Return the current status for one monthly partition.
    """

    record = (
        manifest
        .get("partitions", {})
        .get(partition)
    )

    if not record:
        return None

    return record.get(
        "status"
    )


# ============================================================
# UPDATE PARTITION
# ============================================================

def update_partition(
    manifest: dict[str, Any],
    *,
    partition: str,
    status: str,
    start_time: str,
    end_time: str,
    s3_key: str | None = None,
    rows: int | None = None,
    sensors: int | None = None,
    observed_response_rows: int | None = None,
    positive_duration_rows: int | None = None,
    missing_response_rows: int | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """
    Add or update one monthly partition record.
    """

    allowed_statuses = {
        "completed",
        "skipped",
        "failed",
    }

    if status not in allowed_statuses:
        raise ValueError(
            "Unsupported manifest status: "
            f"{status}"
        )

    partitions = manifest.setdefault(
        "partitions",
        {},
    )

    record = {
        "status": status,
        "start_time": start_time,
        "end_time": end_time,
        "updated_at": utc_now_iso(),
    }

    if s3_key is not None:
        record["s3_key"] = s3_key

    if rows is not None:
        record["rows"] = int(
            rows
        )

    if sensors is not None:
        record["sensors"] = int(
            sensors
        )

    if observed_response_rows is not None:
        record[
            "observed_response_rows"
        ] = int(
            observed_response_rows
        )

    if positive_duration_rows is not None:
        record[
            "positive_duration_rows"
        ] = int(
            positive_duration_rows
        )

    if missing_response_rows is not None:
        record[
            "missing_response_rows"
        ] = int(
            missing_response_rows
        )

    if error is not None:
        record["error"] = str(
            error
        )

    partitions[
        partition
    ] = record

    manifest["updated_at"] = utc_now_iso()

    return manifest


# ============================================================
# PERSIST MANIFEST
# ============================================================

def persist_manifest(
    manifest: dict[str, Any],
) -> None:
    """
    Save the manifest both locally and to S3.
    """

    save_local_manifest(
        manifest
    )

    upload_manifest_to_s3(
        manifest
    )