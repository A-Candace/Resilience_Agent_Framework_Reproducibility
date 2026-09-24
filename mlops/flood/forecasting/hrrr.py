"""
HRRR precipitation ingestion utilities for NYC flood forecasting.

Purpose
-------
This module retrieves hourly HRRR precipitation forecasts from NOAA's
public HRRR S3 archive and prepares the weather-side inputs needed by
the NYC Resilience flood forecasting pipeline.

Important modeling boundary
---------------------------
MRMS:
    Observed precipitation used for historical model training,
    testing, evaluation, and future retraining.

HRRR:
    Forecast precipitation used only for operational/day-ahead
    inference.

The flood models currently expect:

    precip_current_hour_mm
    precip_previous_6h_mm
    daily_total_precip_mm

This module initially establishes the HRRR hourly precipitation
contract. Higher-level feature construction can then derive the three
model predictors from the hourly HRRR series.

HRRR precipitation field
------------------------
The surface product contains APCP fields such as:

    APCP:surface:0-1 hour acc fcst
    APCP:surface:5-6 hour acc fcst
    APCP:surface:23-24 hour acc fcst

For operational hourly precipitation we intentionally select the
ONE-HOUR incremental APCP message, rather than the cumulative APCP
message.

The .idx sidecar files are used to identify GRIB byte ranges so that
the pipeline does not need to download the complete HRRR surface GRIB
file for every forecast lead.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import math
import os
import re
import tempfile
from typing import Iterable, Sequence

import boto3
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

HRRR_BUCKET = os.getenv(
    "HRRR_BUCKET",
    "noaa-hrrr-bdp-pds",
)

HRRR_REGION = os.getenv(
    "HRRR_REGION",
    "us-east-1",
)

HRRR_DOMAIN = os.getenv(
    "HRRR_DOMAIN",
    "conus",
)

HRRR_PRODUCT = os.getenv(
    "HRRR_PRODUCT",
    "wrfsfc",
)

# Extended HRRR cycles provide the horizon needed for day-ahead
# forecasting and complete daily precipitation totals.
EXTENDED_CYCLE_HOURS = (
    0,
    6,
    12,
    18,
)

# Current operational design uses hourly precipitation.
FORECAST_STEP_HOURS = 1

# NYC bounding box used only when a decoded HRRR field needs to be
# spatially restricted.
#
# Padding is intentionally larger than NYC so that every 1-km grid
# centroid has a nearby HRRR grid point.
NYC_MIN_LAT = 40.40
NYC_MAX_LAT = 41.10
NYC_MIN_LON = -74.40
NYC_MAX_LON = -73.50

# Expected HRRR native resolution is approximately 3 km. This value
# is documentation/provenance only; nearest-neighbor calculations use
# actual decoded HRRR coordinates.
HRRR_APPROX_RESOLUTION_KM = 3.0

DEFAULT_CACHE_DIR = Path(
    os.getenv(
        "HRRR_CACHE_DIR",
        "artifacts/flood/forecasting/hrrr_cache",
    )
)

DEFAULT_OUTPUT_DIR = Path(
    os.getenv(
        "FLOOD_FORECAST_OUTPUT_DIR",
        "artifacts/flood/forecasting",
    )
)


# ============================================================
# Data contracts
# ============================================================

@dataclass(frozen=True)
class HRRRCycle:
    """
    One HRRR model initialization.
    """

    initialization_time: datetime

    @property
    def date_string(self) -> str:
        return self.initialization_time.strftime(
            "%Y%m%d"
        )

    @property
    def cycle_hour(self) -> int:
        return self.initialization_time.hour

    @property
    def cycle_string(self) -> str:
        return self.initialization_time.strftime(
            "%H"
        )

    @property
    def cycle_id(self) -> str:
        return self.initialization_time.strftime(
            "%Y%m%dT%HZ"
        )


@dataclass(frozen=True)
class HRRRIndexRecord:
    """
    One record from a NOAA HRRR .idx file.
    """

    message_number: int
    byte_start: int
    raw_line: str
    variable: str
    level: str
    description: str

    @property
    def is_apcp(self) -> bool:
        return self.variable == "APCP"

    @property
    def is_surface(self) -> bool:
        return self.level == "surface"


@dataclass(frozen=True)
class HRRRByteRange:
    """
    Byte range containing one GRIB message.
    """

    start: int
    end: int | None

    @property
    def range_header(self) -> str:
        if self.end is None:
            return f"bytes={self.start}-"

        return (
            f"bytes={self.start}-{self.end}"
        )


# ============================================================
# AWS client
# ============================================================

def create_unsigned_s3_client():
    """
    Create an anonymous S3 client for NOAA's public HRRR bucket.

    NOAA's HRRR Open Data bucket does not require AWS credentials.
    """

    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client(
        "s3",
        region_name=HRRR_REGION,
        config=Config(
            signature_version=UNSIGNED,
        ),
    )


# ============================================================
# Time helpers
# ============================================================

def ensure_utc(
    value: datetime,
) -> datetime:
    """
    Return an aware UTC datetime.
    """

    if value.tzinfo is None:
        return value.replace(
            tzinfo=timezone.utc,
        )

    return value.astimezone(
        timezone.utc
    )


def floor_to_hour(
    value: datetime,
) -> datetime:
    """
    Remove minute/second/microsecond components.
    """

    value = ensure_utc(value)

    return value.replace(
        minute=0,
        second=0,
        microsecond=0,
    )


def candidate_extended_cycles(
    reference_time: datetime | None = None,
    lookback_cycles: int = 8,
) -> list[HRRRCycle]:
    """
    Generate recent extended-cycle candidates newest first.

    The cycle still must be checked against S3 because HRRR files may
    not yet have been published when this function is called.
    """

    if reference_time is None:
        reference_time = datetime.now(
            timezone.utc
        )

    reference_time = floor_to_hour(
        reference_time
    )

    candidates: list[HRRRCycle] = []

    # Search sufficiently far backward to obtain the requested number
    # of 00/06/12/18 UTC cycles.
    search_hours = max(
        48,
        lookback_cycles * 6 + 12,
    )

    for offset in range(
        search_hours + 1
    ):
        candidate = (
            reference_time
            - timedelta(hours=offset)
        )

        if (
            candidate.hour
            not in EXTENDED_CYCLE_HOURS
        ):
            continue

        candidates.append(
            HRRRCycle(
                initialization_time=candidate
            )
        )

        if (
            len(candidates)
            >= lookback_cycles
        ):
            break

    return candidates


# ============================================================
# HRRR object naming
# ============================================================

def hrrr_prefix(
    cycle: HRRRCycle,
) -> str:
    return (
        f"hrrr.{cycle.date_string}/"
        f"{HRRR_DOMAIN}"
    )


def hrrr_surface_filename(
    cycle: HRRRCycle,
    forecast_hour: int,
) -> str:
    """
    HRRR surface forecast filename.

    Example:
        hrrr.t00z.wrfsfcf06.grib2
    """

    if forecast_hour < 0:
        raise ValueError(
            "forecast_hour cannot be negative."
        )

    return (
        f"hrrr.t{cycle.cycle_string}z."
        f"{HRRR_PRODUCT}"
        f"f{forecast_hour:02d}.grib2"
    )


def hrrr_surface_key(
    cycle: HRRRCycle,
    forecast_hour: int,
) -> str:
    return (
        f"{hrrr_prefix(cycle)}/"
        f"{hrrr_surface_filename(cycle, forecast_hour)}"
    )


def hrrr_index_key(
    cycle: HRRRCycle,
    forecast_hour: int,
) -> str:
    return (
        hrrr_surface_key(
            cycle,
            forecast_hour,
        )
        + ".idx"
    )


# ============================================================
# S3 existence / metadata
# ============================================================

def object_exists(
    s3_client,
    key: str,
) -> bool:
    """
    Return True when the HRRR object exists.
    """

    try:
        s3_client.head_object(
            Bucket=HRRR_BUCKET,
            Key=key,
        )

    except Exception as exc:
        # Avoid importing botocore exception classes into every
        # caller. A failed HEAD means unavailable for our purposes.
        response = getattr(
            exc,
            "response",
            {},
        )

        error = response.get(
            "Error",
            {},
        )

        code = str(
            error.get(
                "Code",
                "",
            )
        )

        if code in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            return False

        # Public S3 occasionally represents missing anonymous objects
        # as 403. Test with GET only when necessary.
        if code == "403":
            return False

        raise

    return True


def get_object_size(
    s3_client,
    key: str,
) -> int:
    """
    Return object size in bytes.
    """

    response = s3_client.head_object(
        Bucket=HRRR_BUCKET,
        Key=key,
    )

    return int(
        response["ContentLength"]
    )


# ============================================================
# Cycle discovery
# ============================================================

def cycle_supports_forecast_hour(
    s3_client,
    cycle: HRRRCycle,
    forecast_hour: int,
) -> bool:
    """
    Check that both the GRIB and index exist.
    """

    grib_key = hrrr_surface_key(
        cycle,
        forecast_hour,
    )

    idx_key = hrrr_index_key(
        cycle,
        forecast_hour,
    )

    return (
        object_exists(
            s3_client,
            grib_key,
        )
        and object_exists(
            s3_client,
            idx_key,
        )
    )


def find_latest_available_extended_cycle(
    required_forecast_hour: int,
    reference_time: datetime | None = None,
    s3_client=None,
) -> HRRRCycle:
    """
    Find the newest extended HRRR cycle with the required horizon.

    We deliberately verify the required forecast-hour object rather
    than merely checking that the cycle directory exists.
    """

    if required_forecast_hour <= 0:
        raise ValueError(
            "required_forecast_hour must be positive."
        )

    if s3_client is None:
        s3_client = (
            create_unsigned_s3_client()
        )

    candidates = (
        candidate_extended_cycles(
            reference_time=reference_time,
        )
    )

    for cycle in candidates:

        if cycle_supports_forecast_hour(
            s3_client=s3_client,
            cycle=cycle,
            forecast_hour=required_forecast_hour,
        ):
            return cycle

    candidate_ids = [
        cycle.cycle_id
        for cycle in candidates
    ]

    raise RuntimeError(
        "Unable to find an available extended HRRR cycle "
        f"supporting forecast hour {required_forecast_hour}. "
        f"Checked: {candidate_ids}"
    )


# ============================================================
# Index retrieval / parsing
# ============================================================

def download_index_text(
    s3_client,
    cycle: HRRRCycle,
    forecast_hour: int,
) -> str:
    """
    Download the small HRRR .idx sidecar file.
    """

    key = hrrr_index_key(
        cycle,
        forecast_hour,
    )

    response = s3_client.get_object(
        Bucket=HRRR_BUCKET,
        Key=key,
    )

    raw = response[
        "Body"
    ].read()

    return raw.decode(
        "utf-8",
        errors="replace",
    )


def parse_index_line(
    line: str,
) -> HRRRIndexRecord:
    """
    Parse one wgrib-style HRRR index line.

    Example:

        90:60786351:d=2026082500:
        APCP:surface:5-6 hour acc fcst:
    """

    pieces = line.rstrip().split(":")

    if len(pieces) < 6:
        raise ValueError(
            "Unexpected HRRR index line: "
            + line
        )

    try:
        message_number = int(
            pieces[0]
        )

        byte_start = int(
            pieces[1]
        )

    except ValueError as exc:
        raise ValueError(
            "Unable to parse HRRR index byte metadata: "
            + line
        ) from exc

    variable = pieces[3].strip()
    level = pieces[4].strip()
    description = pieces[5].strip()

    return HRRRIndexRecord(
        message_number=message_number,
        byte_start=byte_start,
        raw_line=line.rstrip(),
        variable=variable,
        level=level,
        description=description,
    )


def parse_index(
    text: str,
) -> list[HRRRIndexRecord]:
    """
    Parse a complete HRRR .idx file.
    """

    records: list[
        HRRRIndexRecord
    ] = []

    for line in text.splitlines():

        if not line.strip():
            continue

        records.append(
            parse_index_line(
                line
            )
        )

    if not records:
        raise RuntimeError(
            "HRRR index contained no records."
        )

    return records


# ============================================================
# APCP message selection
# ============================================================

_ONE_HOUR_APCP_PATTERN = re.compile(
    r"(?P<start>\d+)-(?P<end>\d+)\s+hour\s+acc\s+fcst",
    flags=re.IGNORECASE,
)


def apcp_accumulation_interval(
    record: HRRRIndexRecord,
) -> tuple[int, int] | None:
    """
    Parse the forecast-hour accumulation interval from APCP metadata.

    Examples:
        0-1 hour acc fcst  -> (0, 1)
        5-6 hour acc fcst  -> (5, 6)
        23-24 hour acc fcst -> (23, 24)

    Cumulative records such as "0-1 day acc fcst" intentionally do
    not match this parser.
    """

    match = (
        _ONE_HOUR_APCP_PATTERN.search(
            record.description
        )
    )

    if match is None:
        return None

    return (
        int(match.group("start")),
        int(match.group("end")),
    )


def find_hourly_apcp_record(
    records: Sequence[HRRRIndexRecord],
    forecast_hour: int,
) -> HRRRIndexRecord:
    """
    Select the one-hour incremental APCP message for a forecast lead.

    For F06 this selects:
        5-6 hour acc fcst

    and explicitly does NOT select:
        0-6 hour acc fcst
    """

    if forecast_hour <= 0:
        raise ValueError(
            "Hourly APCP requires forecast_hour >= 1."
        )

    expected_start = (
        forecast_hour - 1
    )

    matches: list[
        HRRRIndexRecord
    ] = []

    for record in records:

        if not (
            record.is_apcp
            and record.is_surface
        ):
            continue

        interval = (
            apcp_accumulation_interval(
                record
            )
        )

        if interval is None:
            continue

        start, end = interval

        if (
            start == expected_start
            and end == forecast_hour
        ):
            matches.append(
                record
            )

    if len(matches) != 1:
        candidate_descriptions = [
            record.description
            for record in records
            if (
                record.is_apcp
                and record.is_surface
            )
        ]

        raise RuntimeError(
            "Expected exactly one one-hour APCP message "
            f"for forecast hour {forecast_hour}; "
            f"found {len(matches)}. "
            "APCP candidates: "
            f"{candidate_descriptions}"
        )

    return matches[0]


def byte_range_for_record(
    records: Sequence[HRRRIndexRecord],
    selected: HRRRIndexRecord,
    object_size: int,
) -> HRRRByteRange:
    """
    Determine the exact byte range for one GRIB message.

    The end byte is one byte before the next indexed message. For the
    final message, the end is the final byte in the S3 object.
    """

    ordered = sorted(
        records,
        key=lambda record: (
            record.byte_start
        ),
    )

    selected_index = None

    for index, record in enumerate(
        ordered
    ):
        if (
            record.message_number
            == selected.message_number
            and record.byte_start
            == selected.byte_start
        ):
            selected_index = index
            break

    if selected_index is None:
        raise ValueError(
            "Selected HRRR index record was not found "
            "in the supplied index."
        )

    start = selected.byte_start

    if (
        selected_index
        < len(ordered) - 1
    ):
        end = (
            ordered[
                selected_index + 1
            ].byte_start
            - 1
        )

    else:
        end = (
            object_size - 1
        )

    if end < start:
        raise RuntimeError(
            "Invalid HRRR GRIB byte range: "
            f"{start}-{end}"
        )

    return HRRRByteRange(
        start=start,
        end=end,
    )


# ============================================================
# Partial GRIB retrieval
# ============================================================

def download_hourly_apcp_grib(
    s3_client,
    cycle: HRRRCycle,
    forecast_hour: int,
    output_path: Path | None = None,
) -> tuple[Path, HRRRIndexRecord]:
    """
    Download only the one-hour APCP GRIB message.

    Returns
    -------
    path
        Local GRIB2 file containing the selected APCP message.

    record
        Index metadata describing the selected field.
    """

    index_text = (
        download_index_text(
            s3_client=s3_client,
            cycle=cycle,
            forecast_hour=forecast_hour,
        )
    )

    records = parse_index(
        index_text
    )

    selected = (
        find_hourly_apcp_record(
            records=records,
            forecast_hour=forecast_hour,
        )
    )

    grib_key = (
        hrrr_surface_key(
            cycle,
            forecast_hour,
        )
    )

    object_size = (
        get_object_size(
            s3_client,
            grib_key,
        )
    )

    byte_range = (
        byte_range_for_record(
            records=records,
            selected=selected,
            object_size=object_size,
        )
    )

    if output_path is None:

        DEFAULT_CACHE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            DEFAULT_CACHE_DIR
            / (
                f"hrrr_{cycle.cycle_id}_"
                f"f{forecast_hour:02d}_"
                "apcp_1h.grib2"
            )
        )

    else:
        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    response = s3_client.get_object(
        Bucket=HRRR_BUCKET,
        Key=grib_key,
        Range=byte_range.range_header,
    )

    payload = response[
        "Body"
    ].read()

    if not payload:
        raise RuntimeError(
            "Downloaded HRRR APCP byte range was empty."
        )

    output_path.write_bytes(
        payload
    )

    return (
        output_path,
        selected,
    )


# ============================================================
# GRIB decoding
# ============================================================

def _find_apcp_data_array(
    dataset,
):
    """
    Locate the precipitation DataArray in a cfgrib dataset.

    Depending on ecCodes/cfgrib versions, APCP is commonly exposed as
    `tp`, but we avoid assuming one exact variable name.
    """

    preferred_names = (
        "tp",
        "apcp",
        "APCP",
    )

    for name in preferred_names:
        if name in dataset.data_vars:
            return dataset[name]

    if len(
        dataset.data_vars
    ) == 1:
        name = next(
            iter(dataset.data_vars)
        )
        return dataset[name]

    raise RuntimeError(
        "Unable to identify APCP variable in decoded HRRR GRIB. "
        f"Variables: {list(dataset.data_vars)}"
    )


def decode_hourly_apcp_grib(
    grib_path: Path,
) -> pd.DataFrame:
    """
    Decode a one-message HRRR APCP GRIB file.

    Returns one row per HRRR grid point:

        hrrr_lat
        hrrr_lon
        precip_hourly_mm

    APCP GRIB units are expected to decode as kg m-2, which is
    numerically equivalent to millimeters of liquid precipitation.
    cfgrib may expose the field as accumulated precipitation (`tp`).
    """

    import xarray as xr

    grib_path = Path(
        grib_path
    )

    if not grib_path.exists():
        raise FileNotFoundError(
            grib_path
        )

    dataset = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={
            "indexpath": "",
        },
    )

    try:
        data_array = (
            _find_apcp_data_array(
                dataset
            )
        )

        if "latitude" not in dataset:
            raise RuntimeError(
                "Decoded HRRR field does not contain latitude."
            )

        if "longitude" not in dataset:
            raise RuntimeError(
                "Decoded HRRR field does not contain longitude."
            )

        latitudes = np.asarray(
            dataset["latitude"].values,
            dtype=float,
        )

        longitudes = np.asarray(
            dataset["longitude"].values,
            dtype=float,
        )

        precipitation = np.asarray(
            data_array.values,
            dtype=float,
        )

    finally:
        dataset.close()

    if not (
        latitudes.shape
        == longitudes.shape
        == precipitation.shape
    ):
        raise RuntimeError(
            "Decoded HRRR coordinate/data shapes do not match. "
            f"latitude={latitudes.shape}, "
            f"longitude={longitudes.shape}, "
            f"precipitation={precipitation.shape}"
        )

    # HRRR longitudes are commonly represented on a 0..360 system.
    # Convert to conventional -180..180 longitude.
    longitudes = (
        (longitudes + 180.0)
        % 360.0
        - 180.0
    )

    result = pd.DataFrame(
        {
            "hrrr_lat": (
                latitudes.ravel()
            ),
            "hrrr_lon": (
                longitudes.ravel()
            ),
            "precip_hourly_mm": (
                precipitation.ravel()
            ),
        }
    )

    invalid = (
        ~np.isfinite(
            result["hrrr_lat"]
        )
        | ~np.isfinite(
            result["hrrr_lon"]
        )
        | ~np.isfinite(
            result["precip_hourly_mm"]
        )
    )

    result = result.loc[
        ~invalid
    ].copy()

    # Numerical decoder noise should not produce physically negative
    # precipitation.
    result[
        "precip_hourly_mm"
    ] = (
        result[
            "precip_hourly_mm"
        ]
        .clip(lower=0.0)
    )

    return result.reset_index(
        drop=True
    )


# ============================================================
# NYC subset
# ============================================================

def subset_hrrr_to_nyc(
    hrrr_points: pd.DataFrame,
) -> pd.DataFrame:
    """
    Restrict decoded HRRR points to the padded NYC bounding box.
    """

    required = {
        "hrrr_lat",
        "hrrr_lon",
        "precip_hourly_mm",
    }

    missing = (
        required
        - set(hrrr_points.columns)
    )

    if missing:
        raise ValueError(
            "HRRR point table is missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    mask = (
        hrrr_points[
            "hrrr_lat"
        ].between(
            NYC_MIN_LAT,
            NYC_MAX_LAT,
        )
        &
        hrrr_points[
            "hrrr_lon"
        ].between(
            NYC_MIN_LON,
            NYC_MAX_LON,
        )
    )

    result = (
        hrrr_points.loc[
            mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if result.empty:
        raise RuntimeError(
            "HRRR NYC spatial subset is empty. "
            "Check decoded coordinates and bounding box."
        )

    return result


# ============================================================
# Distance / nearest-neighbor helpers
# ============================================================

def haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Great-circle distance in kilometers.
    """

    radius_km = 6371.0088

    phi1 = math.radians(
        lat1
    )

    phi2 = math.radians(
        lat2
    )

    delta_phi = math.radians(
        lat2 - lat1
    )

    delta_lambda = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(
            delta_phi / 2.0
        ) ** 2
        +
        math.cos(phi1)
        * math.cos(phi2)
        * math.sin(
            delta_lambda / 2.0
        ) ** 2
    )

    return (
        2.0
        * radius_km
        * math.asin(
            math.sqrt(a)
        )
    )


def nearest_hrrr_point(
    target_lat: float,
    target_lon: float,
    hrrr_points: pd.DataFrame,
) -> pd.Series:
    """
    Find the closest HRRR point to one target location.

    The NYC subset is small enough that a vectorized haversine search
    is simple and deterministic.
    """

    if hrrr_points.empty:
        raise ValueError(
            "Cannot search an empty HRRR point table."
        )

    lat1 = np.radians(
        float(target_lat)
    )

    lon1 = np.radians(
        float(target_lon)
    )

    lat2 = np.radians(
        hrrr_points[
            "hrrr_lat"
        ].to_numpy(
            dtype=float
        )
    )

    lon2 = np.radians(
        hrrr_points[
            "hrrr_lon"
        ].to_numpy(
            dtype=float
        )
    )

    delta_lat = (
        lat2 - lat1
    )

    delta_lon = (
        lon2 - lon1
    )

    a = (
        np.sin(
            delta_lat / 2.0
        ) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(
            delta_lon / 2.0
        ) ** 2
    )

    distances = (
        2.0
        * 6371.0088
        * np.arcsin(
            np.sqrt(a)
        )
    )

    position = int(
        np.argmin(
            distances
        )
    )

    result = (
        hrrr_points.iloc[
            position
        ]
        .copy()
    )

    result[
        "hrrr_distance_km"
    ] = float(
        distances[
            position
        ]
    )

    return result


# ============================================================
# One-hour operational extraction
# ============================================================

def retrieve_hourly_nyc_apcp(
    cycle: HRRRCycle,
    forecast_hour: int,
    s3_client=None,
    keep_grib: bool = False,
) -> pd.DataFrame:
    """
    Retrieve and decode one HRRR hourly precipitation field over NYC.

    Output columns:

        forecast_initialization
        forecast_hour
        valid_time
        hrrr_lat
        hrrr_lon
        precip_hourly_mm
        hrrr_cycle_id
        hrrr_apcp_description
    """

    if forecast_hour <= 0:
        raise ValueError(
            "forecast_hour must be >= 1."
        )

    if s3_client is None:
        s3_client = (
            create_unsigned_s3_client()
        )

    temporary_directory = None

    if keep_grib:

        DEFAULT_CACHE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        grib_path = (
            DEFAULT_CACHE_DIR
            / (
                f"hrrr_{cycle.cycle_id}_"
                f"f{forecast_hour:02d}_"
                "apcp_1h.grib2"
            )
        )

    else:

        temporary_directory = (
            tempfile.TemporaryDirectory()
        )

        grib_path = (
            Path(
                temporary_directory.name
            )
            / "apcp.grib2"
        )

    try:

        grib_path, record = (
            download_hourly_apcp_grib(
                s3_client=s3_client,
                cycle=cycle,
                forecast_hour=forecast_hour,
                output_path=grib_path,
            )
        )

        points = (
            decode_hourly_apcp_grib(
                grib_path
            )
        )

        points = (
            subset_hrrr_to_nyc(
                points
            )
        )

    finally:

        if temporary_directory is not None:
            temporary_directory.cleanup()

    initialization = (
        ensure_utc(
            cycle.initialization_time
        )
    )

    valid_time = (
        initialization
        + timedelta(
            hours=forecast_hour
        )
    )

    points.insert(
        0,
        "forecast_initialization",
        pd.Timestamp(
            initialization
        ),
    )

    points.insert(
        1,
        "forecast_hour",
        int(
            forecast_hour
        ),
    )

    points.insert(
        2,
        "valid_time",
        pd.Timestamp(
            valid_time
        ),
    )

    points[
        "hrrr_cycle_id"
    ] = cycle.cycle_id

    points[
        "hrrr_apcp_description"
    ] = record.description

    return points


# ============================================================
# Grid validation
# ============================================================

def validate_target_grid(
    grid: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate a target 1-km grid centroid table.

    Required columns:

        grid_id
        grid_lat
        grid_lon

    The higher-level spatial module can construct this table from the
    authoritative grid GeoJSON/reference artifact.
    """

    if not isinstance(
        grid,
        pd.DataFrame,
    ):
        raise TypeError(
            "Target grid must be a pandas DataFrame."
        )

    required = {
        "grid_id",
        "grid_lat",
        "grid_lon",
    }

    missing = (
        required
        - set(grid.columns)
    )

    if missing:
        raise ValueError(
            "Target grid is missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    result = grid[
        [
            "grid_id",
            "grid_lat",
            "grid_lon",
        ]
    ].copy()

    if result[
        "grid_id"
    ].isna().any():
        raise ValueError(
            "Target grid contains null grid_id values."
        )

    result[
        "grid_id"
    ] = (
        result[
            "grid_id"
        ]
        .astype(str)
    )

    if result[
        "grid_id"
    ].duplicated().any():
        raise ValueError(
            "Target grid contains duplicate grid_id values."
        )

    for column in (
        "grid_lat",
        "grid_lon",
    ):
        result[
            column
        ] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    if result[
        [
            "grid_lat",
            "grid_lon",
        ]
    ].isna().any().any():
        raise ValueError(
            "Target grid contains invalid coordinates."
        )

    return result


# ============================================================
# HRRR → 1-km target-grid mapping
# ============================================================

def map_hourly_hrrr_to_target_grid(
    grid: pd.DataFrame,
    hrrr_points: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign the nearest HRRR precipitation point to every 1-km grid.

    HRRR is approximately 3-km resolution. The NYC flood output
    remains 1-km resolution; each target grid receives precipitation
    from its geographically closest HRRR grid point.
    """

    grid = validate_target_grid(
        grid
    )

    required_hrrr = {
        "hrrr_lat",
        "hrrr_lon",
        "precip_hourly_mm",
        "forecast_initialization",
        "forecast_hour",
        "valid_time",
        "hrrr_cycle_id",
        "hrrr_apcp_description",
    }

    missing = (
        required_hrrr
        - set(hrrr_points.columns)
    )

    if missing:
        raise ValueError(
            "HRRR table is missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    if hrrr_points.empty:
        raise ValueError(
            "HRRR point table is empty."
        )

    metadata_columns = [
        "forecast_initialization",
        "forecast_hour",
        "valid_time",
        "hrrr_cycle_id",
        "hrrr_apcp_description",
    ]

    for column in metadata_columns:

        if (
            hrrr_points[
                column
            ].nunique(
                dropna=False
            )
            != 1
        ):
            raise ValueError(
                "One-hour HRRR table contains multiple values for "
                f"{column}."
            )

    metadata = {
        column: (
            hrrr_points[
                column
            ].iloc[0]
        )
        for column
        in metadata_columns
    }

    rows: list[
        dict
    ] = []

    for row in grid.itertuples(
        index=False
    ):

        nearest = (
            nearest_hrrr_point(
                target_lat=float(
                    row.grid_lat
                ),
                target_lon=float(
                    row.grid_lon
                ),
                hrrr_points=hrrr_points,
            )
        )

        rows.append(
            {
                "grid_id": (
                    row.grid_id
                ),
                "grid_lat": float(
                    row.grid_lat
                ),
                "grid_lon": float(
                    row.grid_lon
                ),
                "forecast_initialization": (
                    metadata[
                        "forecast_initialization"
                    ]
                ),
                "forecast_hour": int(
                    metadata[
                        "forecast_hour"
                    ]
                ),
                "forecast_hour_valid_time": (
                    metadata[
                        "valid_time"
                    ]
                ),
                "precip_hourly_mm": float(
                    nearest[
                        "precip_hourly_mm"
                    ]
                ),
                "hrrr_lat": float(
                    nearest[
                        "hrrr_lat"
                    ]
                ),
                "hrrr_lon": float(
                    nearest[
                        "hrrr_lon"
                    ]
                ),
                "hrrr_distance_km": float(
                    nearest[
                        "hrrr_distance_km"
                    ]
                ),
                "hrrr_cycle_id": (
                    metadata[
                        "hrrr_cycle_id"
                    ]
                ),
                "hrrr_apcp_description": (
                    metadata[
                        "hrrr_apcp_description"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Multi-hour retrieval
# ============================================================

def retrieve_grid_hourly_precipitation(
    grid: pd.DataFrame,
    cycle: HRRRCycle,
    forecast_hours: Iterable[int],
    s3_client=None,
    keep_grib: bool = False,
) -> pd.DataFrame:
    """
    Retrieve hourly HRRR precipitation for multiple forecast hours and
    map every hour to the NYC 1-km grid.

    This produces the raw hourly weather table from which the flood
    predictor variables are constructed.
    """

    grid = validate_target_grid(
        grid
    )

    hours = sorted(
        {
            int(hour)
            for hour
            in forecast_hours
        }
    )

    if not hours:
        raise ValueError(
            "No forecast hours were supplied."
        )

    if hours[0] < 1:
        raise ValueError(
            "Forecast hours must begin at F01 or later."
        )

    if s3_client is None:
        s3_client = (
            create_unsigned_s3_client()
        )

    outputs: list[
        pd.DataFrame
    ] = []

    print()
    print(
        "HRRR HOURLY PRECIPITATION RETRIEVAL"
    )
    print(
        "-----------------------------------"
    )
    print(
        "Cycle:",
        cycle.cycle_id,
    )
    print(
        "Forecast hours:",
        f"{hours[0]} -> {hours[-1]}",
    )
    print(
        "Target grids:",
        len(grid),
    )

    for position, forecast_hour in enumerate(
        hours,
        start=1,
    ):

        print(
            f"Retrieving F{forecast_hour:02d} "
            f"({position}/{len(hours)})"
        )

        hrrr_points = (
            retrieve_hourly_nyc_apcp(
                cycle=cycle,
                forecast_hour=forecast_hour,
                s3_client=s3_client,
                keep_grib=keep_grib,
            )
        )

        mapped = (
            map_hourly_hrrr_to_target_grid(
                grid=grid,
                hrrr_points=hrrr_points,
            )
        )

        outputs.append(
            mapped
        )

    result = pd.concat(
        outputs,
        ignore_index=True,
    )

    expected_rows = (
        len(grid)
        * len(hours)
    )

    if len(result) != expected_rows:
        raise RuntimeError(
            "Unexpected HRRR grid-hour row count. "
            f"Expected {expected_rows}; "
            f"received {len(result)}."
        )

    duplicate_count = int(
        result.duplicated(
            [
                "grid_id",
                "forecast_hour_valid_time",
            ]
        ).sum()
    )

    if duplicate_count:
        raise RuntimeError(
            "HRRR grid-hour table contains duplicate "
            f"grid/time rows: {duplicate_count}"
        )

    return result.sort_values(
        [
            "forecast_hour_valid_time",
            "grid_id",
        ]
    ).reset_index(
        drop=True
    )


# ============================================================
# Predictor construction
# ============================================================

def build_flood_precipitation_predictors(
    hourly_grid_precipitation: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct the three precipitation predictors used by the flood
    models.

    Definitions
    -----------
    precip_current_hour_mm:
        HRRR one-hour APCP for the forecast hour.

    precip_previous_6h_mm:
        Sum of the six PRECEDING hourly APCP values. The current hour
        is excluded.

    daily_total_precip_mm:
        Total forecast precipitation across the complete UTC calendar
        day containing the target forecast hour.

    Important
    ---------
    daily_total_precip_mm intentionally includes forecast rainfall
    occurring later in the target day. It represents forecast storm
    severity for the entire calendar day.

    Rows without a complete six-hour antecedent window are marked
    invalid and rejected. Daily totals also require a complete target
    calendar day in the supplied HRRR window.
    """

    required = {
        "grid_id",
        "forecast_hour_valid_time",
        "precip_hourly_mm",
    }

    missing = (
        required
        - set(
            hourly_grid_precipitation.columns
        )
    )

    if missing:
        raise ValueError(
            "Hourly HRRR table is missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    result = (
        hourly_grid_precipitation
        .copy()
    )

    result[
        "forecast_hour_valid_time"
    ] = pd.to_datetime(
        result[
            "forecast_hour_valid_time"
        ],
        utc=True,
    )

    result[
        "precip_hourly_mm"
    ] = pd.to_numeric(
        result[
            "precip_hourly_mm"
        ],
        errors="coerce",
    )

    if result[
        "precip_hourly_mm"
    ].isna().any():
        raise ValueError(
            "Hourly HRRR precipitation contains null/non-numeric "
            "values."
        )

    result = result.sort_values(
        [
            "grid_id",
            "forecast_hour_valid_time",
        ]
    ).reset_index(
        drop=True
    )

    result[
        "precip_current_hour_mm"
    ] = result[
        "precip_hourly_mm"
    ]

    # Previous six hours, excluding current hour.
    shifted = (
        result
        .groupby(
            "grid_id",
            sort=False,
        )[
            "precip_hourly_mm"
        ]
        .shift(1)
    )

    result[
        "precip_previous_6h_mm"
    ] = (
        shifted
        .groupby(
            result[
                "grid_id"
            ],
            sort=False,
        )
        .rolling(
            window=6,
            min_periods=6,
        )
        .sum()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    result[
        "_forecast_date"
    ] = (
        result[
            "forecast_hour_valid_time"
        ]
        .dt.floor(
            "D"
        )
    )

    # Validate that a daily total is actually based on a complete
    # calendar day rather than silently summing a partial forecast.
    day_hour_counts = (
        result
        .groupby(
            [
                "grid_id",
                "_forecast_date",
            ]
        )[
            "forecast_hour_valid_time"
        ]
        .nunique()
        .rename(
            "_day_hour_count"
        )
        .reset_index()
    )

    result = result.merge(
        day_hour_counts,
        on=[
            "grid_id",
            "_forecast_date",
        ],
        how="left",
        validate="many_to_one",
    )

    daily_totals = (
        result
        .groupby(
            [
                "grid_id",
                "_forecast_date",
            ]
        )[
            "precip_hourly_mm"
        ]
        .sum()
        .rename(
            "daily_total_precip_mm"
        )
        .reset_index()
    )

    result = result.merge(
        daily_totals,
        on=[
            "grid_id",
            "_forecast_date",
        ],
        how="left",
        validate="many_to_one",
    )

    result[
        "complete_previous_6h"
    ] = (
        result[
            "precip_previous_6h_mm"
        ].notna()
    )

    result[
        "complete_daily_total"
    ] = (
        result[
            "_day_hour_count"
        ]
        == 24
    )

    result[
        "predictor_contract_complete"
    ] = (
        result[
            "complete_previous_6h"
        ]
        &
        result[
            "complete_daily_total"
        ]
    )

    return result.drop(
        columns=[
            "_forecast_date",
        ]
    )


# ============================================================
# Audit helpers
# ============================================================

def audit_hourly_grid_precipitation(
    frame: pd.DataFrame,
) -> None:
    """
    Print an operational audit of the HRRR grid-hour artifact.
    """

    print()
    print(
        "HRRR GRID PRECIPITATION AUDIT"
    )
    print(
        "============================="
    )

    print(
        "Rows:",
        len(frame),
    )

    if "grid_id" in frame:
        print(
            "Grid cells:",
            frame[
                "grid_id"
            ].nunique(),
        )

    if (
        "forecast_hour_valid_time"
        in frame
    ):
        print(
            "Forecast hours:",
            frame[
                "forecast_hour_valid_time"
            ].nunique(),
        )

        print(
            "First valid time:",
            frame[
                "forecast_hour_valid_time"
            ].min(),
        )

        print(
            "Last valid time:",
            frame[
                "forecast_hour_valid_time"
            ].max(),
        )

    if "precip_hourly_mm" in frame:
        print()
        print(
            "Hourly precipitation (mm):"
        )
        print(
            frame[
                "precip_hourly_mm"
            ]
            .describe()
            .to_string()
        )

    if "hrrr_distance_km" in frame:
        print()
        print(
            "HRRR → 1-km grid distance (km):"
        )
        print(
            frame[
                "hrrr_distance_km"
            ]
            .describe()
            .to_string()
        )

    if (
        "predictor_contract_complete"
        in frame
    ):
        print()
        print(
            "Predictor contract complete:"
        )
        print(
            frame[
                "predictor_contract_complete"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )


# ============================================================
# Lightweight contract test
# ============================================================

def test_index_contract(
    cycle: HRRRCycle,
    forecast_hours: Sequence[int] = (
        1,
        6,
        24,
    ),
) -> None:
    """
    Verify the NOAA index contract without downloading GRIB fields.

    This is intentionally the first test to run after installing this
    module.
    """

    s3_client = (
        create_unsigned_s3_client()
    )

    print(
        "HRRR APCP INDEX CONTRACT"
    )
    print(
        "========================"
    )
    print(
        "Cycle:",
        cycle.cycle_id,
    )
    print()

    for forecast_hour in forecast_hours:

        text = (
            download_index_text(
                s3_client=s3_client,
                cycle=cycle,
                forecast_hour=forecast_hour,
            )
        )

        records = (
            parse_index(
                text
            )
        )

        selected = (
            find_hourly_apcp_record(
                records=records,
                forecast_hour=forecast_hour,
            )
        )

        print(
            f"F{forecast_hour:02d}: "
            f"{selected.description}"
        )

    print()
    print(
        "HRRR APCP INDEX CONTRACT PASSED"
    )


# ============================================================
# CLI
# ============================================================

def main() -> None:
    """
    Initial HRRR connectivity/contract test.

    This CLI deliberately does not yet launch the full 837-grid,
    multi-hour production retrieval. First we verify:

        1. public S3 access
        2. cycle availability
        3. HRRR surface naming
        4. incremental APCP selection
    """

    print(
        "NYC FLOOD HRRR INGESTION"
    )
    print(
        "========================"
    )

    print(
        "Bucket:",
        HRRR_BUCKET,
    )

    print(
        "Region:",
        HRRR_REGION,
    )

    print(
        "Domain:",
        HRRR_DOMAIN,
    )

    print(
        "Product:",
        HRRR_PRODUCT,
    )

    print()
    print(
        "Finding latest available extended cycle..."
    )

    s3_client = (
        create_unsigned_s3_client()
    )

    # F24 is sufficient for the initial contract check. Higher-level
    # day-ahead orchestration will calculate the exact horizon needed
    # for antecedent context + complete calendar-day totals.
    cycle = (
        find_latest_available_extended_cycle(
            required_forecast_hour=24,
            s3_client=s3_client,
        )
    )

    print(
        "Selected cycle:",
        cycle.cycle_id,
    )

    print()

    test_index_contract(
        cycle=cycle,
        forecast_hours=(
            1,
            6,
            24,
        ),
    )

    print()
    print(
        "HRRR INGESTION CONTRACT READY"
    )
    print(
        "No full GRIB files were downloaded."
    )


if __name__ == "__main__":
    main()