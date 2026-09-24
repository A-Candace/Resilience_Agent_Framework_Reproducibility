"""Census-tract geometry and attribute loading."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import streamlit as st

from resilience_app.core.shared import (
    DATA_RUNTIME_DIR,
    RISK_XLSX,
    SHAPELY_OK,
    load_boards_from_shapefile_zip_streamlit,
    load_census_tracts_from_geojson,
    load_risk_xlsx,
    merge_attrs_into_tracts,
    shp_mapping,
    shp_shape,
    shp_transform,
)


LOCAL_CENSUS_TRACTS = Path(
    os.getenv(
        "LOCAL_CENSUS_TRACTS",
        str(DATA_RUNTIME_DIR / "nyc_census_tracts.shp"),
    )
)


def ensure_boards() -> list:
    """
    Load census-tract geometry and merge the Excel attributes.

    Normal runtime behavior:
    1. scripts/s3_assets.py downloads the shapefile components and Excel
       workbook from S3 into /app/data/runtime.
    2. This module reads those local runtime files.
    3. The shapefile geometry is merged with Risk_Attributes_Table_v4.xlsx.

    Manual upload remains available as a fallback.
    """
    cached_boards = st.session_state.get("boards")

    if cached_boards:
        return cached_boards

    boards = _load_runtime_census_tracts()

    if boards:
        boards = _merge_excel_attributes(boards)

        st.session_state["boards"] = boards
        return boards

    st.warning(
        "The S3-synchronized census-tract files were not found. "
        "You may upload a GeoJSON or zipped shapefile as a temporary fallback."
    )

    boards = _load_uploaded_census_tracts()

    if boards:
        boards = _merge_excel_attributes(boards)
        st.session_state["boards"] = boards

    return boards or []


def _load_runtime_census_tracts() -> Optional[list]:
    """Load the S3-synchronized shapefile from the runtime directory."""
    shapefile_path = LOCAL_CENSUS_TRACTS

    required_files = [
        shapefile_path,
        shapefile_path.with_suffix(".shx"),
        shapefile_path.with_suffix(".dbf"),
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        st.error(
            "Required census-tract shapefile components are missing:\n\n"
            + "\n".join(f"- `{path}`" for path in missing_files)
        )
        return None

    st.info(
        f"Loading census tracts from `{shapefile_path}`..."
    )

    return _load_census_tracts_from_shapefile_direct(
        shapefile_path
    )


def _merge_excel_attributes(boards: list) -> list:
    """Merge Risk_Attributes_Table_v4.xlsx onto census tracts."""
    risk_path = Path(RISK_XLSX)

    if not risk_path.exists():
        st.error(
            "The tract-attribute workbook was not found at "
            f"`{risk_path}`. Confirm that S3 synchronization completed."
        )
        return boards

    try:
        dataframe = load_risk_xlsx(str(risk_path))

        boards, coverage = merge_attrs_into_tracts(
            boards,
            dataframe,
        )

        if coverage < 0.75:
            st.warning(
                "⚠️ Excel merge coverage: "
                f"{coverage * 100:.1f}% — check tract IDs."
            )
        else:
            st.success(
                "✅ Excel attributes merged "
                f"({coverage * 100:.1f}% coverage)."
            )

    except Exception as exc:
        st.error(
            "The tract-attribute workbook could not be merged: "
            f"{exc}"
        )

    return boards


def _load_uploaded_census_tracts() -> Optional[list]:
    """Allow a manual upload if S3 runtime assets are unavailable."""
    boards = None

    geojson_column, shapefile_column = st.columns(2)

    with geojson_column:
        uploaded_geojson = st.file_uploader(
            "Upload census tracts as GeoJSON",
            type=["geojson", "json"],
            key="geojson_upl",
        )

        if uploaded_geojson is not None:
            try:
                raw_data = uploaded_geojson.read()

                try:
                    document = json.loads(
                        raw_data.decode("utf-8")
                    )
                except UnicodeDecodeError:
                    document = json.loads(
                        raw_data.decode("latin-1")
                    )

                boards = load_census_tracts_from_geojson(
                    document
                )

            except Exception as exc:
                st.error(
                    f"Could not read uploaded GeoJSON: {exc}"
                )

    with shapefile_column:
        if boards is None:
            boards = load_boards_from_shapefile_zip_streamlit()

    return boards


def _load_census_tracts_from_shapefile_direct(
    shapefile_path: Path,
) -> Optional[list]:
    """
    Load census tracts from local shapefile components.

    The .shp, .shx, .dbf and optional .prj files must be stored in
    the same directory with the same basename.
    """
    if not SHAPELY_OK:
        st.error(
            "Shapefile support requires `shapely>=2.0`."
        )
        return None

    try:
        import shapefile
    except ImportError:
        st.error(
            "Missing dependency: `pyshp`. Install it with "
            "`pip install pyshp`."
        )
        return None

    try:
        from pyproj import CRS, Transformer
    except ImportError:
        st.error(
            "Shapefile reprojection requires `pyproj>=3.6`."
        )
        return None

    projection_path = shapefile_path.with_suffix(".prj")

    source_crs = None

    if projection_path.exists():
        try:
            projection_wkt = projection_path.read_text(
                encoding="utf-8"
            )
            source_crs = CRS.from_wkt(projection_wkt)
        except Exception as exc:
            st.warning(
                f"Could not parse `{projection_path.name}`: {exc}"
            )

    transformer = None

    if source_crs and source_crs.to_epsg() != 4326:
        try:
            transformer = Transformer.from_crs(
                source_crs,
                CRS.from_epsg(4326),
                always_xy=True,
            )
        except Exception as exc:
            st.warning(
                "Could not initialize the shapefile CRS "
                f"transformation: {exc}"
            )

    try:
        reader = shapefile.Reader(str(shapefile_path))
    except Exception as exc:
        st.error(
            f"Could not read census-tract shapefile: {exc}"
        )
        return None

    fields = [
        field[0]
        for field in reader.fields
        if field[0] != "DeletionFlag"
    ]

    def looks_like_longitude_latitude(
        x_value: float,
        y_value: float,
    ) -> bool:
        return (
            -180.0 <= x_value <= 180.0
            and -90.0 <= y_value <= 90.0
        )

    def to_geojson_geometry(geometry) -> dict:
        return json.loads(
            json.dumps(shp_mapping(geometry))
        )

    tracts = []

    for index, shape_record in enumerate(
        reader.shapeRecords()
    ):
        record = {
            fields[position]: shape_record.record[position]
            for position in range(len(fields))
        }

        unit_id = (
            record.get("TRACTCE")
            or record.get("GEOID")
            or record.get("tract_id")
            or record.get("CensusTrac")
            or f"TRACT_{index:05d}"
        )

        name = (
            record.get("NAME")
            or record.get("name")
            or str(unit_id)
        )

        raw_geometry = shape_record.shape.__geo_interface__

        try:
            geometry = shp_shape(raw_geometry)
        except Exception as exc:
            st.warning(
                f"Could not parse geometry for {unit_id}: {exc}"
            )
            continue

        if geometry.is_empty:
            continue

        active_transformer = transformer

        if active_transformer is None:
            try:
                if geometry.geom_type == "Polygon":
                    x_value, y_value = list(
                        geometry.exterior.coords
                    )[0]

                elif geometry.geom_type == "MultiPolygon":
                    first_polygon = list(geometry.geoms)[0]
                    x_value, y_value = list(
                        first_polygon.exterior.coords
                    )[0]

                else:
                    x_value, y_value = 0.0, 0.0

                if not looks_like_longitude_latitude(
                    x_value,
                    y_value,
                ):
                    active_transformer = Transformer.from_crs(
                        2263,
                        4326,
                        always_xy=True,
                    )

            except Exception:
                active_transformer = None

        if active_transformer is not None:
            try:
                geometry = shp_transform(
                    lambda x, y, z=None: (
                        active_transformer.transform(x, y)
                    ),
                    geometry,
                )
            except Exception as exc:
                st.warning(
                    f"Could not transform geometry for "
                    f"{unit_id}: {exc}"
                )
                continue

        if geometry.geom_type not in {
            "Polygon",
            "MultiPolygon",
        }:
            st.warning(
                f"Skipping {unit_id}: geometry type is "
                f"{geometry.geom_type}."
            )
            continue

        tracts.append(
            {
                "unit_id": str(unit_id),
                "name": str(name),
                "geom": geometry,
                "feature_geom": to_geojson_geometry(
                    geometry
                ),
                "attrs": dict(record),
            }
        )

    if not tracts:
        st.error(
            "No valid census-tract polygons were found in "
            "the shapefile."
        )
        return None

    st.success(
        f"✅ Loaded {len(tracts)} census tracts from shapefile."
    )

    return tracts