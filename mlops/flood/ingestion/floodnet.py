"""
FloodNet ingestion utilities for the NYC flood MLOps pipeline.

This module retrieves:

1. The current FloodNet deployment inventory.
2. Water-depth observations for one or more deployments.

The implementation is based on the research notebook
Floodnet_Sensor_Analysis, but is structured for reusable,
production-oriented ingestion.

FloodNet API
------------
Base:
    https://api.floodnet.nyc/api/rest

Deployments:
    /deployments/flood

Depth:
    /deployments/flood/{deployment_id}/depth
"""

from __future__ import annotations

import os
import time
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# CONFIGURATION
# ============================================================

FLOODNET_BASE_URL = os.getenv(
    "FLOODNET_BASE_URL",
    "https://api.floodnet.nyc/api/rest",
)

DEPTH_CHUNK_DAYS = int(
    os.getenv(
        "FLOODNET_DEPTH_CHUNK_DAYS",
        "6",
    )
)

REQUEST_TIMEOUT_SECONDS = int(
    os.getenv(
        "FLOODNET_REQUEST_TIMEOUT_SECONDS",
        "60",
    )
)

MAX_RETRIES = int(
    os.getenv(
        "FLOODNET_MAX_RETRIES",
        "5",
    )
)

RETRY_BACKOFF_SECONDS = float(
    os.getenv(
        "FLOODNET_RETRY_BACKOFF_SECONDS",
        "1.0",
    )
)


# ============================================================
# STANDARD COLUMN NAMES
# ============================================================

SENSOR_ID = "deployment_id"

SENSOR_LAT = "sensor_lat"

SENSOR_LON = "sensor_lon"

TIME_COLUMN = "time"

DEPTH_PROCESSED_COLUMN = "depth_proc_mm"

DEPTH_RAW_COLUMN = "depth_raw_mm"


# ============================================================
# HTTP SESSION
# ============================================================

def create_session() -> requests.Session:
    """
    Create a requests session with retry behavior.

    Retries transient connection and server-side failures.
    """

    retry = Retry(
        total=MAX_RETRIES,
        connect=MAX_RETRIES,
        read=MAX_RETRIES,
        status=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF_SECONDS,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504,
        ],
        allowed_methods=[
            "GET",
        ],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry
    )

    session = requests.Session()

    session.mount(
        "https://",
        adapter,
    )

    session.mount(
        "http://",
        adapter,
    )

    session.headers.update(
        {
            "User-Agent": (
                "nyc-resilience-agent/"
                "floodnet-ingestion"
            )
        }
    )

    return session


# ============================================================
# API REQUEST
# ============================================================

def get_json(
    session: requests.Session,
    url: str,
    *,
    params: dict | None = None,
) -> dict:
    """
    Perform a FloodNet GET request and return decoded JSON.
    """

    response = session.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    response.raise_for_status()

    data = response.json()

    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(
            f"FloodNet API error: {data['error']}"
        )

    return data


# ============================================================
# COORDINATE EXTRACTION
# ============================================================

def extract_coordinates(
    location,
) -> tuple[float, float]:
    """
    Convert FloodNet GeoJSON location to latitude/longitude.

    FloodNet GeoJSON coordinates are:

        [longitude, latitude]

    Returns
    -------
    tuple
        (latitude, longitude)
    """

    if not isinstance(
        location,
        dict,
    ):
        return (
            np.nan,
            np.nan,
        )

    coordinates = location.get(
        "coordinates"
    )

    if (
        not isinstance(
            coordinates,
            (list, tuple),
        )
        or len(coordinates) < 2
    ):
        return (
            np.nan,
            np.nan,
        )

    try:
        longitude = float(
            coordinates[0]
        )

        latitude = float(
            coordinates[1]
        )

    except (
        TypeError,
        ValueError,
    ):
        return (
            np.nan,
            np.nan,
        )

    return (
        latitude,
        longitude,
    )


# ============================================================
# DEPLOYMENT INVENTORY
# ============================================================

def get_deployments(
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """
    Retrieve all FloodNet flood deployments.

    Returns a normalized deployment inventory containing
    sensor latitude and longitude for downstream MRMS mapping.
    """

    owns_session = (
        session is None
    )

    if session is None:
        session = create_session()

    try:
        url = (
            f"{FLOODNET_BASE_URL}"
            "/deployments/flood"
        )

        payload = get_json(
            session,
            url,
        )

        deployments = payload.get(
            "deployments",
            [],
        )

        if not deployments:
            raise RuntimeError(
                "FloodNet returned no flood deployments."
            )

        df = pd.DataFrame(
            deployments
        )

    finally:
        if owns_session:
            session.close()

    if SENSOR_ID not in df.columns:
        raise RuntimeError(
            "FloodNet deployment response is missing "
            "'deployment_id'."
        )

    if "location" not in df.columns:
        raise RuntimeError(
            "FloodNet deployment response is missing "
            "'location'."
        )

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    coordinates = (
        df["location"]
        .apply(
            extract_coordinates
        )
    )

    df[SENSOR_LAT] = (
        coordinates.apply(
            lambda value: value[0]
        )
    )

    df[SENSOR_LON] = (
        coordinates.apply(
            lambda value: value[1]
        )
    )

    # --------------------------------------------------------
    # Deployment dates
    # --------------------------------------------------------

    for column in [
        "date_deployed",
        "date_down",
    ]:

        if column in df.columns:

            df[column] = pd.to_datetime(
                df[column],
                utc=True,
                errors="coerce",
            )

    # --------------------------------------------------------
    # Normalize identifiers
    # --------------------------------------------------------

    df[SENSOR_ID] = (
        df[SENSOR_ID]
        .astype(str)
    )

    # We cannot map sensors without coordinates.
    df = df.dropna(
        subset=[
            SENSOR_ID,
            SENSOR_LAT,
            SENSOR_LON,
        ]
    )

    df = (
        df.drop_duplicates(
            subset=[
                SENSOR_ID,
            ],
            keep="last",
        )
        .sort_values(
            SENSOR_ID
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ============================================================
# SENSOR INVENTORY FOR MRMS MAPPING
# ============================================================

def get_sensor_inventory(
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """
    Return only the FloodNet fields needed by the MRMS mapper.
    """

    deployments = get_deployments(
        session=session
    )

    columns = [
        SENSOR_ID,
        SENSOR_LAT,
        SENSOR_LON,
    ]

    for optional in [
        "name",
        "date_deployed",
        "date_down",
        "deploy_type",
    ]:

        if optional in deployments.columns:
            columns.append(
                optional
            )

    return deployments[
        columns
    ].copy()


# ============================================================
# TIME WINDOW
# ============================================================

def resolve_time_window(
    deployment: pd.Series,
    *,
    global_start=None,
    global_end=None,
) -> tuple[
    pd.Timestamp | None,
    pd.Timestamp | None,
]:
    """
    Determine the depth-observation request window.

    If global_start/global_end are supplied, they are used for
    backfill or incremental ingestion.

    The resulting request window is clipped to the sensor's actual
    deployment period when deployment dates are available.
    """

    deployed = pd.to_datetime(
        deployment.get(
            "date_deployed",
            pd.NaT,
        ),
        utc=True,
        errors="coerce",
    )

    down = pd.to_datetime(
        deployment.get(
            "date_down",
            pd.NaT,
        ),
        utc=True,
        errors="coerce",
    )

    if global_start is not None:

        start = pd.to_datetime(
            global_start,
            utc=True,
            errors="coerce",
        )

    else:

        start = deployed

    if global_end is not None:

        end = pd.to_datetime(
            global_end,
            utc=True,
            errors="coerce",
        )

    else:

        end = down

        if pd.isna(end):
            end = pd.Timestamp.now(
                tz="UTC"
            )

    # Do not request periods before the sensor existed.
    if (
        not pd.isna(deployed)
        and not pd.isna(start)
    ):
        start = max(
            start,
            deployed,
        )

    # Do not request periods after a deployment ended.
    if (
        not pd.isna(down)
        and not pd.isna(end)
    ):
        end = min(
            end,
            down,
        )

    if (
        pd.isna(start)
        or pd.isna(end)
        or start >= end
    ):
        return (
            None,
            None,
        )

    return (
        start,
        end,
    )


# ============================================================
# NORMALIZE DEPTH RESPONSE
# ============================================================

def normalize_depth_data(
    df: pd.DataFrame,
    *,
    deployment_id: str,
    sensor_name: str | None = None,
) -> pd.DataFrame:
    """
    Normalize a FloodNet depth API response.
    """

    if df.empty:
        return df

    if TIME_COLUMN not in df.columns:
        raise ValueError(
            "FloodNet depth response is missing 'time'."
        )

    result = df.copy()

    result[TIME_COLUMN] = pd.to_datetime(
        result[TIME_COLUMN],
        utc=True,
        errors="coerce",
    )

    for column in [
        DEPTH_PROCESSED_COLUMN,
        DEPTH_RAW_COLUMN,
    ]:

        if column in result.columns:

            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    result[SENSOR_ID] = str(
        deployment_id
    )

    if sensor_name is not None:
        result["name"] = (
            sensor_name
        )

    result = result.dropna(
        subset=[
            TIME_COLUMN,
        ]
    )

    # Preserve notebook behavior:
    # one observation per deployment/timestamp.
    result = (
        result.sort_values(
            TIME_COLUMN
        )
        .drop_duplicates(
            subset=[
                SENSOR_ID,
                TIME_COLUMN,
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# FETCH ONE SENSOR
# ============================================================

def fetch_depth_for_sensor(
    deployment: pd.Series,
    *,
    global_start=None,
    global_end=None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """
    Retrieve depth observations for one FloodNet deployment.

    Requests are divided into six-day windows by default to stay
    below the FloodNet API response limit.
    """

    deployment_id = str(
        deployment[SENSOR_ID]
    )

    sensor_name = deployment.get(
        "name",
        deployment_id,
    )

    (
        start_time,
        end_time,
    ) = resolve_time_window(
        deployment,
        global_start=global_start,
        global_end=global_end,
    )

    if (
        start_time is None
        or end_time is None
    ):

        return pd.DataFrame()

    owns_session = (
        session is None
    )

    if session is None:
        session = create_session()

    url = (
        f"{FLOODNET_BASE_URL}"
        f"/deployments/flood/"
        f"{deployment_id}/depth"
    )

    chunk_duration = pd.Timedelta(
        days=DEPTH_CHUNK_DAYS
    )

    parts: list[pd.DataFrame] = []

    current = start_time

    try:

        while current < end_time:

            chunk_end = min(
                current + chunk_duration,
                end_time,
            )

            params = {
                "start_time": (
                    current.isoformat()
                ),
                "end_time": (
                    chunk_end.isoformat()
                ),
            }

            payload = get_json(
                session,
                url,
                params=params,
            )

            depth_data = payload.get(
                "depth_data",
                [],
            )

            if depth_data:

                part = pd.DataFrame(
                    depth_data
                )

                if not part.empty:
                    parts.append(
                        part
                    )

            current = chunk_end

    finally:

        if owns_session:
            session.close()

    if not parts:
        return pd.DataFrame()

    result = pd.concat(
        parts,
        ignore_index=True,
    )

    return normalize_depth_data(
        result,
        deployment_id=deployment_id,
        sensor_name=sensor_name,
    )


# ============================================================
# FETCH MULTIPLE SENSORS
# ============================================================

def fetch_depth_for_all_sensors(
    deployments: pd.DataFrame,
    *,
    global_start=None,
    global_end=None,
    deployment_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Retrieve depth observations for multiple FloodNet sensors.

    Parameters
    ----------
    deployments:
        Current FloodNet deployment inventory.

    global_start:
        Optional UTC-compatible start timestamp.

    global_end:
        Optional UTC-compatible end timestamp.

    deployment_ids:
        Optional subset of deployment IDs. If omitted, every
        deployment is evaluated.
    """

    df = deployments.copy()

    if deployment_ids is not None:

        requested = {
            str(value)
            for value
            in deployment_ids
        }

        df = df[
            df[SENSOR_ID]
            .astype(str)
            .isin(requested)
        ].copy()

    session = create_session()

    parts: list[pd.DataFrame] = []

    try:

        total = len(df)

        for index, (_, deployment) in enumerate(
            df.iterrows(),
            start=1,
        ):

            deployment_id = str(
                deployment[
                    SENSOR_ID
                ]
            )

            print(
                f"[{index}/{total}] "
                f"FloodNet {deployment_id}"
            )

            try:

                depth = (
                    fetch_depth_for_sensor(
                        deployment,
                        global_start=(
                            global_start
                        ),
                        global_end=(
                            global_end
                        ),
                        session=session,
                    )
                )

            except Exception as exc:

                print(
                    f"    ERROR: {exc}"
                )

                continue

            if not depth.empty:

                parts.append(
                    depth
                )

                print(
                    f"    rows: "
                    f"{len(depth):,}"
                )

            else:

                print(
                    "    no observations"
                )

    finally:

        session.close()

    if not parts:

        return pd.DataFrame(
            columns=[
                SENSOR_ID,
                TIME_COLUMN,
                DEPTH_PROCESSED_COLUMN,
                DEPTH_RAW_COLUMN,
                "name",
            ]
        )

    result = pd.concat(
        parts,
        ignore_index=True,
    )

    result = (
        result.sort_values(
            [
                SENSOR_ID,
                TIME_COLUMN,
            ]
        )
        .drop_duplicates(
            subset=[
                SENSOR_ID,
                TIME_COLUMN,
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# INVENTORY SUMMARY
# ============================================================

def inventory_summary(
    deployments: pd.DataFrame,
) -> dict:
    """
    Return basic information about the current FloodNet network.
    """

    active = 0

    if "date_down" in deployments.columns:

        active = int(
            deployments[
                "date_down"
            ]
            .isna()
            .sum()
        )

    return {
        "deployments": int(
            deployments[
                SENSOR_ID
            ].nunique()
        ),
        "active_deployments": (
            active
        ),
        "with_coordinates": int(
            deployments[
                [
                    SENSOR_LAT,
                    SENSOR_LON,
                ]
            ]
            .notna()
            .all(
                axis=1
            )
            .sum()
        ),
    }


# ============================================================
# SAFE LOCAL SMOKE TEST
# ============================================================

def main() -> None:
    """
    Safe smoke test.

    This retrieves the deployment inventory only.

    It deliberately does NOT download all historical depth data.
    """

    print(
        "Retrieving current FloodNet deployment inventory..."
    )

    deployments = (
        get_deployments()
    )

    summary = inventory_summary(
        deployments
    )

    print(
        "\n=== FloodNet Inventory ==="
    )

    print(
        f"Deployments: "
        f"{summary['deployments']:,}"
    )

    print(
        f"Active: "
        f"{summary['active_deployments']:,}"
    )

    print(
        "With coordinates: "
        f"{summary['with_coordinates']:,}"
    )

    print(
        "\nSample:"
    )

    display_columns = [
        SENSOR_ID,
        SENSOR_LAT,
        SENSOR_LON,
    ]

    for optional in [
        "name",
        "date_deployed",
        "date_down",
    ]:

        if optional in deployments.columns:
            display_columns.append(
                optional
            )

    print(
        deployments[
            display_columns
        ]
        .head()
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()