import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import folium
import streamlit as st
from folium.plugins import Draw, Fullscreen
from pyproj import Geod
from streamlit_folium import st_folium

from resilience_app.core.shared import *


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

    Returns a dict with lat, lon, and display_name, or None.
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
            "User-Agent": "NYC-Resilience-Green-Roof-Planning-Tool/1.0"
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


def _init_green_roof_map_state():
    """Initialize map-related Streamlit session state."""
    if "green_roof_map_center" not in st.session_state:
        st.session_state.green_roof_map_center = _DEFAULT_MAP_CENTER

    if "green_roof_map_zoom" not in st.session_state:
        st.session_state.green_roof_map_zoom = _DEFAULT_MAP_ZOOM

    if "green_roof_geocoded_address" not in st.session_state:
        st.session_state.green_roof_geocoded_address = None


def _rooftop_measurement_map(center, zoom_start, marker_label=None):
    """Render an interactive NYC map for drawing a rooftop/project polygon."""
    m = folium.Map(
        location=list(center),
        zoom_start=zoom_start,
        tiles=None,
        control_scale=True,
        max_zoom=22,
    )

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
        edit_options={"edit": True, "remove": True},
    ).add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)

    return st_folium(
        m,
        width=None,
        height=560,
        key="green_roof_measurement_map",
        returned_objects=["last_active_drawing", "all_drawings"],
    )


def page_green_roof():
    st.title("🟩 Green Roof Cost Calculator")
    st.caption(
        "Measure a candidate rooftop/project area on the map or enter an area manually, "
        "then estimate a preliminary green-roof installation cost range."
    )

    input_method = st.radio(
        "How would you like to provide the project area?",
        options=["Measure rooftop on map", "Enter area manually"],
        horizontal=True,
    )

    area_sqft = 0.0

    if input_method == "Measure rooftop on map":
        _init_green_roof_map_state()

        st.markdown(
            "**Map instructions:** Search for a NYC address, zoom to the building, then use the "
            "polygon or rectangle drawing tool to trace the candidate rooftop/project area. "
            "You can edit or redraw the shape if needed."
        )

        with st.form("green_roof_address_search_form"):
            address_query = st.text_input(
                "Search for a NYC address",
                placeholder="Example: 400 W 61st St, New York, NY",
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
                    st.session_state.green_roof_map_center = (
                        result["lat"],
                        result["lon"],
                    )
                    st.session_state.green_roof_map_zoom = 21
                    st.session_state.green_roof_geocoded_address = result["display_name"]
                    st.success(f"Address found: {result['display_name']}")

        center = st.session_state.green_roof_map_center
        zoom_start = st.session_state.green_roof_map_zoom
        marker_label = st.session_state.green_roof_geocoded_address

        map_state = _rooftop_measurement_map(
            center=center,
            zoom_start=zoom_start,
            marker_label=marker_label,
        )

        geometry = _latest_drawn_geometry(map_state)
        measured_area_sqft = _polygon_area_sqft(geometry)

        if measured_area_sqft is not None and measured_area_sqft > 0:
            area_sqft = float(measured_area_sqft)
            st.metric(
                "Measured rooftop/project area",
                f"{area_sqft:,.0f} sq ft",
            )
            st.caption(
                "Area is calculated geodesically from the polygon you drew. "
                "Aerial imagery and manually drawn boundaries may introduce measurement error."
            )
        else:
            st.info(
                "Draw a rooftop or project-area polygon on the map to calculate its area."
            )

    else:
        area_sqft = float(
            st.number_input(
                "Green roof area (sq ft)",
                min_value=0.0,
                value=1000.0,
                step=50.0,
            )
        )

    low, high = GREEN_ROOF_COST_USD_PER_SQFT
    st.write(
        f"Using default unit cost range: **${low:,.0f}–${high:,.0f} per sq ft**"
    )

    with st.expander("Optional: override unit costs", expanded=False):
        unit_low = st.number_input(
            "Low $/sq ft",
            min_value=0.0,
            value=float(low),
            step=1.0,
            key="green_roof_unit_low",
        )
        unit_high = st.number_input(
            "High $/sq ft",
            min_value=0.0,
            value=float(high),
            step=1.0,
            key="green_roof_unit_high",
        )

    unit_low = float(unit_low)
    unit_high = float(unit_high)

    if unit_high < unit_low:
        st.warning(
            "The high unit cost is below the low unit cost. Check the override values."
        )

    est_low = area_sqft * unit_low
    est_high = area_sqft * unit_high

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Low estimate", f"${est_low:,.0f}")

    with c2:
        st.metric("High estimate", f"${est_high:,.0f}")

    st.warning(
        "**Engineering Disclaimer:** The measured area and resulting cost range are preliminary "
        "planning estimates only. Rooftop load-bearing capacity, structural suitability, drainage "
        "requirements, and the area appropriate for green-roof installation are not evaluated by "
        "this tool. Actual feasibility and usable coverage should be confirmed by a licensed "
        "Professional Engineer (PE) or other appropriately qualified professional before design "
        "or installation."
    )

    with st.expander("Sources (click to open)", expanded=False):
        s = COST_SOURCES["green_roof"]
        st.markdown(f"- [{s['label']}]({s['url']})")
