from io import BytesIO
import json, math, os, re
from datetime import datetime, timedelta, timezone, date
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import folium
from folium.plugins import Draw, Fullscreen
from pyproj import Geod
from streamlit_folium import st_folium

from resilience_app.core.shared import *
from resilience_app.services.data_loader import ensure_boards


_SQ_M_TO_SQ_FT = 10.7639104167
_GEOD = Geod(ellps="WGS84")

_DEFAULT_MAP_CENTER = (40.7128, -74.0060)
_DEFAULT_MAP_ZOOM = 12

# Approximate NYC bounding box:
# left, top, right, bottom = west lon, north lat, east lon, south lat
_NYC_VIEWBOX = "-74.2591,40.9176,-73.7003,40.4774"


def _polygon_area_sqft(geometry):
    """Return geodesic area (sq ft) for a GeoJSON Polygon or MultiPolygon."""
    if not geometry:
        return None

    geom_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if not coordinates:
        return None

    def ring_area_m2(ring):
        if len(ring) < 3:
            return 0.0

        lons = [pt[0] for pt in ring]
        lats = [pt[1] for pt in ring]
        area_m2, _ = _GEOD.polygon_area_perimeter(lons, lats)
        return abs(area_m2)

    def polygon_area_m2(polygon_coords):
        if not polygon_coords:
            return 0.0

        exterior = ring_area_m2(polygon_coords[0])
        holes = sum(ring_area_m2(ring) for ring in polygon_coords[1:])
        return max(exterior - holes, 0.0)

    if geom_type == "Polygon":
        area_m2 = polygon_area_m2(coordinates)
    elif geom_type == "MultiPolygon":
        area_m2 = sum(polygon_area_m2(poly) for poly in coordinates)
    else:
        return None

    return area_m2 * _SQ_M_TO_SQ_FT


def _latest_drawn_geometry(map_state):
    """Extract the most recent polygon-like geometry returned by streamlit-folium."""
    if not map_state:
        return None

    last = map_state.get("last_active_drawing")
    if isinstance(last, dict):
        geometry = last.get("geometry")
        if geometry and geometry.get("type") in {"Polygon", "MultiPolygon"}:
            return geometry

    drawings = map_state.get("all_drawings") or []
    for drawing in reversed(drawings):
        if not isinstance(drawing, dict):
            continue

        geometry = drawing.get("geometry")
        if geometry and geometry.get("type") in {"Polygon", "MultiPolygon"}:
            return geometry

    return None


@st.cache_data(ttl=86400, show_spinner=False)
def _geocode_nyc_address(address):
    """
    Resolve a user-entered NYC address using OpenStreetMap Nominatim.

    Returns:
        dict with lat, lon, display_name, or None if no NYC result is found.
    """
    query_text = address.strip()
    if not query_text:
        return None

    if "new york" not in query_text.lower():
        query_text = f"{query_text}, New York, NY"

    params = {
        "q": query_text,
        "format": "jsonv2",
        "limit": 1,
        "countrycodes": "us",
        "viewbox": _NYC_VIEWBOX,
        "bounded": 1,
        "addressdetails": 1,
    }

    url = "https://nominatim.openstreetmap.org/search?" + urlencode(params)

    request = Request(
        url,
        headers={
            "User-Agent": "NYC-Resilience-Rain-Garden-Planning-Tool/1.0"
        },
    )

    try:
        with urlopen(request, timeout=10) as response:
            results = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None

    if not results:
        return None

    result = results[0]

    try:
        lat = float(result["lat"])
        lon = float(result["lon"])
    except (KeyError, TypeError, ValueError):
        return None

    return {
        "lat": lat,
        "lon": lon,
        "display_name": result.get("display_name", query_text),
    }


def _init_rain_garden_map_state():
    """Initialize rain-garden measurement map session state."""
    if "rain_garden_map_center" not in st.session_state:
        st.session_state.rain_garden_map_center = _DEFAULT_MAP_CENTER

    if "rain_garden_map_zoom" not in st.session_state:
        st.session_state.rain_garden_map_zoom = _DEFAULT_MAP_ZOOM

    if "rain_garden_geocoded_address" not in st.session_state:
        st.session_state.rain_garden_geocoded_address = None


def _rain_garden_measurement_map(center, zoom_start, marker_label=None):
    """Render an interactive aerial map for drawing a candidate rain-garden area."""
    m = folium.Map(
        location=list(center),
        zoom_start=zoom_start,
        tiles=None,
        control_scale=True,
        max_zoom=22,
    )

    # Aerial imagery for site delineation.
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr=(
            "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, "
            "and the GIS User Community"
        ),
        name="Aerial imagery",
        overlay=False,
        control=True,
        show=True,
        max_native_zoom=19,
        max_zoom=22,
    ).add_to(m)

    # OpenStreetMap remains available as a reference layer.
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)

    if marker_label:
        folium.Marker(
            location=list(center),
            tooltip="Search result",
            popup=marker_label,
        ).add_to(m)

    Fullscreen(position="topleft").add_to(m)

    Draw(
        export=False,
        position="topleft",
        draw_options={
            "polyline": False,
            "polygon": {
                "allowIntersection": False,
                "showArea": True,
                "shapeOptions": {"weight": 3},
            },
            "rectangle": {"showArea": True},
            "circle": False,
            "marker": False,
            "circlemarker": False,
        },
        edit_options={
            "edit": True,
            "remove": True,
        },
    ).add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)

    return st_folium(
        m,
        width=None,
        height=560,
        key="rain_garden_measurement_map",
        returned_objects=["last_active_drawing", "all_drawings"],
    )


def page_rain_garden():
    st.title("🌿 Rain Garden Estimator")
    st.caption("Estimate rain garden cost and highlight candidate tracts using low slope and/or low elevation.")

    boards = ensure_boards()
    if not boards:
        st.stop()

    # ---- Cost estimator ----
    st.subheader("1) Cost estimate from measured rain-garden area")

    rg_area_sqft = st.number_input("Rain garden area (sq ft)", min_value=0.0, value=300.0, step=10.0)

    low, high = RAIN_GARDEN_COST_USD_PER_SQFT
    st.write(f"Using default unit cost range: **${low:,.0f}–${high:,.0f} per sq ft**")

    with st.expander("Optional: override unit costs", expanded=False):
        unit_low = st.number_input("Low $/sq ft (rain garden)", min_value=0.0, value=float(low), step=1.0, key="rg_low")
        unit_high = st.number_input("High $/sq ft (rain garden)", min_value=0.0, value=float(high), step=1.0, key="rg_high")
    unit_low, unit_high = float(unit_low), float(unit_high)

    est_low = rg_area_sqft * unit_low
    est_high = rg_area_sqft * unit_high

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Low estimate", f"${est_low:,.0f}")
    with c2:
        st.metric("High estimate", f"${est_high:,.0f}")

    # ---- Simple candidate-tract mapping ----
    st.divider()
    st.subheader("2) Candidate tract mapping (quick screen)")

    st.write(
        "Rain gardens often perform best where water can be directed into a planted depressed area. "
        "A simple screening approach is to look for **lower slope** and **lower elevation** tracts "
        "(not a guarantee — site conditions still matter)."
    )

    mode = st.radio(
        "Candidate rule:",
        ["Low slope", "Low elevation", "Low slope AND low elevation"],
        index=2
    )

    # Pull numeric series
    slope_series = pd.Series({b["unit_id"]: get_attr_num(b.get("attrs", {}), "Mean_slope") for b in boards})
    elev_series = pd.Series({b["unit_id"]: get_attr_num(b.get("attrs", {}), "Mean_elevation") for b in boards})

    # Choose percentile thresholds
    pct = st.slider("How strict? (percentile cut-off for 'low')", min_value=5, max_value=50, value=25, step=5)
    slope_cut = float(np.nanpercentile(slope_series.values, pct))
    elev_cut = float(np.nanpercentile(elev_series.values, pct))

    # Determine matches
    matches = set()
    for b in boards:
        uid = b["unit_id"]
        s_ok = slope_series.get(uid, 0.0) <= slope_cut
        e_ok = elev_series.get(uid, 0.0) <= elev_cut

        if mode == "Low slope" and s_ok:
            matches.add(uid)
        elif mode == "Low elevation" and e_ok:
            matches.add(uid)
        elif mode == "Low slope AND low elevation" and s_ok and e_ok:
            matches.add(uid)

    st.success(f"✅ Candidate tracts highlighted: **{len(matches)}** / {len(boards)}")

    # Map: grey all, green candidates
    fc_features = []
    for b in boards:
        uid = b["unit_id"]
        is_match = uid in matches
        fill_color = [0, 180, 0, 200] if is_match else [200, 200, 200, 90]

        fc_features.append({
            "type": "Feature",
            "properties": {
                "unit_id": uid,
                "cb": get_board_display_id(b),
                "candidate": "Yes" if is_match else "No",
                "fill_color": fill_color,
                "Mean_slope": float(slope_series.get(uid, 0.0)),
                "Mean_elevation": float(elev_series.get(uid, 0.0)),
            },
            "geometry": b["feature_geom"]
        })

    fc = {"type": "FeatureCollection", "features": fc_features}

    layer = pdk.Layer(
        "GeoJsonLayer",
        data=fc,
        pickable=True,
        stroked=True,
        filled=True,
        get_fill_color="properties.fill_color",
        get_line_color=[40, 40, 40],
        lineWidthMinPixels=1,
        opacity=1.0
    )
    bg = pdk.Layer(
        "TileLayer",
        data="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        minZoom=0, maxZoom=19, tileSize=256
    )

    lat_c, lon_c, zoom_c = boards_bbox(fc)

    deck = pdk.Deck(
        layers=[bg, layer],
        initial_view_state=pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom_c),
        tooltip={
            "html": (
                "<b>Tract:</b> {cb}<br>"
                "<b>Candidate:</b> {candidate}<br>"
                "<b>Mean slope:</b> {Mean_slope}<br>"
                "<b>Mean elevation:</b> {Mean_elevation}"
            )
        }
    )

    st.pydeck_chart(deck, use_container_width=True)

    st.markdown(
        """
        <div style="padding:10px; background:#f0f0f0; border-radius:5px; margin:10px 0;">
            <b>Legend:</b>
            <span style="color:#00b400;">● Green</span> = Candidate tracts |
            <span style="color:#c8c8c8;">● Grey</span> = Others
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---- Interactive area measurement ----
    st.divider()
    st.subheader("3) Measure a candidate rain-garden area on the map")

    st.write(
        "Use the aerial map below to measure a potential rain-garden footprint. "
        "Search for a NYC address, zoom to the site, then draw a polygon or rectangle "
        "around the proposed area."
    )

    _init_rain_garden_map_state()

    with st.form("rain_garden_address_search_form"):
        address_query = st.text_input(
            "Search for a NYC address",
            placeholder="Example: 400 W 61st St, New York, NY",
            key="rain_garden_address_query",
        )
        search_clicked = st.form_submit_button("Find address")

    if search_clicked:
        if not address_query.strip():
            st.warning("Enter an address before searching.")
        else:
            with st.spinner("Finding address..."):
                result = _geocode_nyc_address(address_query)

            if result is None:
                st.error(
                    "Address not found within New York City, or the geocoding service did not "
                    "respond. Check the address and try again."
                )
            else:
                st.session_state.rain_garden_map_center = (
                    result["lat"],
                    result["lon"],
                )
                st.session_state.rain_garden_map_zoom = 21
                st.session_state.rain_garden_geocoded_address = result["display_name"]
                st.success(f"Address found: {result['display_name']}")

    center = st.session_state.rain_garden_map_center
    zoom_start = st.session_state.rain_garden_map_zoom
    marker_label = st.session_state.rain_garden_geocoded_address

    map_state = _rain_garden_measurement_map(
        center=center,
        zoom_start=zoom_start,
        marker_label=marker_label,
    )

    geometry = _latest_drawn_geometry(map_state)
    measured_area_sqft = _polygon_area_sqft(geometry)

    if measured_area_sqft is not None and measured_area_sqft > 0:
        st.metric(
            "Measured candidate rain-garden area",
            f"{measured_area_sqft:,.0f} sq ft",
        )

        measured_low = measured_area_sqft * unit_low
        measured_high = measured_area_sqft * unit_high

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Measured-area low estimate", f"${measured_low:,.0f}")
        with m2:
            st.metric("Measured-area high estimate", f"${measured_high:,.0f}")

        st.caption(
            "Area is calculated geodesically from the polygon you drew. "
            "Aerial imagery and manually drawn boundaries may introduce measurement error."
        )
    else:
        st.info(
            "Draw a candidate rain-garden polygon or rectangle on the map to calculate its area."
        )

    st.warning(
        "**Planning Disclaimer:** This map-based measurement and cost estimate are preliminary "
        "planning tools only. Actual rain-garden feasibility depends on site-specific conditions "
        "including drainage patterns, soil infiltration, utilities, groundwater, grading, "
        "setbacks, and other engineering or regulatory requirements. Site suitability and final "
        "design should be confirmed by an appropriately qualified professional before construction."
    )

    with st.expander("Sources (click to open)", expanded=False):
        s = COST_SOURCES["rain_garden"]
        st.markdown(f"- [{s['label']}]({s['url']})")
