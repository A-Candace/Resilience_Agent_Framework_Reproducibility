"""
Validate one full day of NOAA MRMS hourly QPE against the
historical precipitation predictor table.

This compares:

    historical precip_current_hour_mm
vs.
    NOAA MultiSensor_QPE_01H_Pass2

using the validated time convention:

    historical feature hour H
        <- precipitation during H to H+1

    NOAA source timestamp
        = H + 1 hour

The comparison is restricted to deployment_id/hour combinations
that exist in both datasets.
"""

from __future__ import annotations

import pandas as pd

from mlops.flood.ingestion.mrms import (
    PRECIP_COLUMN,
    SENSOR_ID,
    process_hour,
)

from mlops.flood.shared.s3_bootstrap import (
    load_gcn_hourly_features,
)


# ============================================================
# CONFIGURATION
# ============================================================

VALIDATION_DATE = pd.Timestamp(
    "2026-03-12",
    tz="UTC",
)


# ============================================================
# LOAD HISTORICAL DAY
# ============================================================

def load_historical_day(
    validation_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Load historical GCN precipitation values for one UTC day.
    """

    start = validation_date.normalize()

    end = (
        start
        + pd.Timedelta(days=1)
    )

    historical = (
        load_gcn_hourly_features()
    )

    historical["hour"] = pd.to_datetime(
        historical["hour"],
        utc=True,
        errors="coerce",
    )

    historical = historical[
        (historical["hour"] >= start)
        &
        (historical["hour"] < end)
    ].copy()

    historical = historical[
        [
            SENSOR_ID,
            "hour",
            PRECIP_COLUMN,
        ]
    ].rename(
        columns={
            PRECIP_COLUMN: "historical_mm",
        }
    )

    return historical


# ============================================================
# BUILD NOAA DAY
# ============================================================

def build_noaa_day(
    validation_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build NOAA precipitation for all 24 feature hours.
    """

    start = validation_date.normalize()

    feature_hours = pd.date_range(
        start=start,
        periods=24,
        freq="h",
        tz="UTC",
    )

    parts = []

    for index, feature_hour in enumerate(
        feature_hours,
        start=1,
    ):

        print(
            f"\n[{index:02d}/24] "
            f"Feature hour: {feature_hour}"
        )

        try:

            result = process_hour(
                feature_hour
            )

        except Exception as exc:

            print(
                f"    ERROR: {exc}"
            )

            continue

        result = result[
            [
                SENSOR_ID,
                "hour",
                "mrms_source_hour",
                PRECIP_COLUMN,
            ]
        ].rename(
            columns={
                PRECIP_COLUMN: "noaa_mm",
            }
        )

        parts.append(
            result
        )

    if not parts:

        raise RuntimeError(
            "No NOAA MRMS hours were successfully processed."
        )

    return pd.concat(
        parts,
        ignore_index=True,
    )


# ============================================================
# HOURLY METRICS
# ============================================================

def calculate_hourly_metrics(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute matched-sensor validation metrics by feature hour.
    """

    rows = []

    for hour, group in comparison.groupby(
        "hour"
    ):

        if group.empty:
            continue

        difference = (
            group["noaa_mm"]
            - group["historical_mm"]
        )

        correlation = (
            group["historical_mm"]
            .corr(
                group["noaa_mm"]
            )
        )

        rows.append(
            {
                "hour": hour,
                "matched_sensors": len(
                    group
                ),
                "historical_mean_mm": (
                    group[
                        "historical_mm"
                    ].mean()
                ),
                "noaa_mean_mm": (
                    group[
                        "noaa_mm"
                    ].mean()
                ),
                "historical_max_mm": (
                    group[
                        "historical_mm"
                    ].max()
                ),
                "noaa_max_mm": (
                    group[
                        "noaa_mm"
                    ].max()
                ),
                "mae_mm": (
                    difference
                    .abs()
                    .mean()
                ),
                "bias_mm": (
                    difference
                    .mean()
                ),
                "correlation": (
                    correlation
                ),
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "hour"
    )


# ============================================================
# OVERALL METRICS
# ============================================================

def print_overall_metrics(
    comparison: pd.DataFrame,
) -> None:
    """
    Print all-day matched-observation metrics.
    """

    difference = (
        comparison["noaa_mm"]
        - comparison["historical_mm"]
    )

    correlation = (
        comparison["historical_mm"]
        .corr(
            comparison["noaa_mm"]
        )
    )

    print(
        "\n=========================================="
    )

    print(
        "FULL-DAY MATCHED SENSOR VALIDATION"
    )

    print(
        "=========================================="
    )

    print(
        "Matched sensor-hours: "
        f"{len(comparison):,}"
    )

    print(
        "Unique matched sensors: "
        f"{comparison[SENSOR_ID].nunique():,}"
    )

    print(
        "Feature hours represented: "
        f"{comparison['hour'].nunique():,}"
    )

    print(
        "\nHistorical mean:"
    )

    print(
        f"{comparison['historical_mm'].mean():.4f} mm"
    )

    print(
        "NOAA mean:"
    )

    print(
        f"{comparison['noaa_mm'].mean():.4f} mm"
    )

    print(
        "\nHistorical max:"
    )

    print(
        f"{comparison['historical_mm'].max():.4f} mm"
    )

    print(
        "NOAA max:"
    )

    print(
        f"{comparison['noaa_mm'].max():.4f} mm"
    )

    print(
        "\nCorrelation:"
    )

    print(
        f"{correlation:.4f}"
    )

    print(
        "Mean absolute error:"
    )

    print(
        f"{difference.abs().mean():.4f} mm"
    )

    print(
        "Mean bias (NOAA - historical):"
    )

    print(
        f"{difference.mean():.4f} mm"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Run full-day historical-vs-NOAA validation.
    """

    print(
        "MRMS FULL-DAY VALIDATION"
    )

    print(
        "========================"
    )

    print(
        "Validation date:"
    )

    print(
        VALIDATION_DATE.date()
    )

    # --------------------------------------------------------
    # Historical
    # --------------------------------------------------------

    print(
        "\nLoading historical predictor data..."
    )

    historical = load_historical_day(
        VALIDATION_DATE
    )

    print(
        f"Historical rows: "
        f"{len(historical):,}"
    )

    print(
        f"Historical sensors: "
        f"{historical[SENSOR_ID].nunique():,}"
    )

    print(
        f"Historical hours: "
        f"{historical['hour'].nunique():,}"
    )

    # --------------------------------------------------------
    # NOAA
    # --------------------------------------------------------

    print(
        "\nBuilding NOAA MRMS day..."
    )

    noaa = build_noaa_day(
        VALIDATION_DATE
    )

    print(
        "\nNOAA rows:"
    )

    print(
        f"{len(noaa):,}"
    )

    print(
        "NOAA sensors:"
    )

    print(
        f"{noaa[SENSOR_ID].nunique():,}"
    )

    print(
        "NOAA feature hours:"
    )

    print(
        f"{noaa['hour'].nunique():,}"
    )

    # --------------------------------------------------------
    # Join
    # --------------------------------------------------------

    comparison = historical.merge(
        noaa,
        on=[
            SENSOR_ID,
            "hour",
        ],
        how="inner",
        validate="one_to_one",
    )

    if comparison.empty:
        raise RuntimeError(
            "Historical and NOAA datasets produced no "
            "matched sensor-hour observations."
        )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    print_overall_metrics(
        comparison
    )

    # --------------------------------------------------------
    # Hourly
    # --------------------------------------------------------

    hourly = calculate_hourly_metrics(
        comparison
    )

    print(
        "\n=========================================="
    )

    print(
        "HOURLY VALIDATION"
    )

    print(
        "=========================================="
    )

    print(
        hourly.to_string(
            index=False,
            formatters={
                "historical_mean_mm": (
                    lambda value:
                    f"{value:.3f}"
                ),
                "noaa_mean_mm": (
                    lambda value:
                    f"{value:.3f}"
                ),
                "historical_max_mm": (
                    lambda value:
                    f"{value:.3f}"
                ),
                "noaa_max_mm": (
                    lambda value:
                    f"{value:.3f}"
                ),
                "mae_mm": (
                    lambda value:
                    f"{value:.3f}"
                ),
                "bias_mm": (
                    lambda value:
                    f"{value:.3f}"
                ),
                "correlation": (
                    lambda value:
                    (
                        "NaN"
                        if pd.isna(value)
                        else f"{value:.3f}"
                    )
                ),
            },
        )
    )

    # --------------------------------------------------------
    # Rain-only subset
    # --------------------------------------------------------

    rainy = comparison[
        (
            comparison["historical_mm"]
            > 0.1
        )
        |
        (
            comparison["noaa_mm"]
            > 0.1
        )
    ].copy()

    print(
        "\n=========================================="
    )

    print(
        "RAIN-ONLY MATCHED VALIDATION"
    )

    print(
        "=========================================="
    )

    if rainy.empty:

        print(
            "No rain observations found."
        )

    else:

        difference = (
            rainy["noaa_mm"]
            - rainy["historical_mm"]
        )

        print(
            "Matched rainy sensor-hours: "
            f"{len(rainy):,}"
        )

        print(
            "Correlation:"
        )

        print(
            f"{rainy['historical_mm'].corr(rainy['noaa_mm']):.4f}"
        )

        print(
            "MAE:"
        )

        print(
            f"{difference.abs().mean():.4f} mm"
        )

        print(
            "Bias:"
        )

        print(
            f"{difference.mean():.4f} mm"
        )

    print(
        "\nMRMS FULL-DAY VALIDATION COMPLETE"
    )


if __name__ == "__main__":
    main()