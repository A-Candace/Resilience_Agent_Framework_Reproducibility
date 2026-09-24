from io import BytesIO
import json, math, os, re
from datetime import datetime, timedelta, timezone, date
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from resilience_app.core.shared import *
from resilience_app.services.data_loader import ensure_boards
def urban_features_page():
    st.title("🏙️ Urban Features (Static) - Census Tracts")
    boards = ensure_boards()
    if not boards:
        st.stop()

    fc_outlines = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": get_board_display_id(b)}, "geometry": b["feature_geom"]} for b in boards
    ]}
    lat_c, lon_c, zoom_c = boards_bbox(fc_outlines)

    # Remove label_layer from outline view
    outline = pdk.Layer(
        "GeoJsonLayer",
        data=fc_outlines,
        pickable=True, stroked=True, filled=False,
        get_line_color=[255, 0, 0], line_width_min_pixels=2
    )
    bg = pdk.Layer(
        "TileLayer",
        data="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        minZoom=0, maxZoom=19, tileSize=256
    )
    st.pydeck_chart(
        pdk.Deck(layers=[bg, outline],  # Removed label_layer here
                 initial_view_state=pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom_c),
                 tooltip={"html": "<b>Tract:</b> {name}"})
    )
    st.caption("NYC Census Tracts (outline).")

    st.subheader("Visualize a static attribute as a choropleth")
    
    # Define field mappings - now includes computed density
    field_mappings = {
        "Mean Elevation": ["Mean_elevation"],
        "Mean Slope": ["Mean_slope"],
        "Total Building Footprint (sq km)": ["Total_ftp_area"],
        "Building Footprint Density (ratio)": ["_COMPUTED_DENSITY_"],  # Special computed field
    }
    
    sel_label = st.selectbox("Attribute", list(field_mappings.keys()), index=0)
    possible_fields = field_mappings[sel_label]
    
    # Handle computed density field
    if possible_fields[0] == "_COMPUTED_DENSITY_":
        sel_attr = "_COMPUTED_DENSITY_"
        st.info(f"Using computed field: Total_ftp_area / AREA (both in sq km)")
        
        # Compute density for each tract
        vals = {}
        for b in boards:
            attrs = b.get("attrs", {})
            ftp = coerce_numeric(get_attr_ci(attrs, "Total_ftp_area"), default=0.0)  # Already in sq km
            area_sqmi = coerce_numeric(get_attr_ci(attrs, "AREA"), default=0.0)  # In square miles
            
            # Convert square miles to square kilometers (1 sq mi = 2.58999 sq km)
            area_sqkm = area_sqmi * 2.58999
            
            # Avoid division by zero
            if area_sqkm > 0:
                vals[b["unit_id"]] = ftp / area_sqkm
            else:
                vals[b["unit_id"]] = 0.0
        
        series = pd.Series(vals).sort_index()
    else:
        # Find which field actually exists in the data
        sel_attr = None
        for field in possible_fields:
            for b in boards[:5]:
                if get_attr_ci(b.get("attrs", {}), field) is not None:
                    sel_attr = field
                    break
            if sel_attr:
                break
        
        if not sel_attr:
            st.error(f"Could not find any of these fields: {possible_fields}")
            st.write("Available fields in first tract:", list(boards[0].get("attrs", {}).keys())[:20])
            st.stop()
        
        st.info(f"Using field: `{sel_attr}`")

        # Extract values using the robust numeric coercion
        vals = {}
        for b in boards:
            raw_val = get_attr_ci(b.get("attrs", {}), sel_attr)
            vals[b["unit_id"]] = coerce_numeric(raw_val, default=0.0)
        
        series = pd.Series(vals).sort_index()
    
    # Show stats
    with st.expander("📊 Data Statistics", expanded=False):
        st.write(f"**Field used:** `{sel_attr}`")
        st.write(f"**Non-zero values:** {(series > 0).sum()} / {len(series)}")
        st.dataframe(series.describe())

    bins, edges = quantile_bins_4(series)
    color_map = {0: PALETTE4[0], 1: PALETTE4[1], 2: PALETTE4[2], 3: PALETTE4[3]}
    color_series = pd.Series({uid: color_map[bins.loc[uid]] for uid in series.index})

    fc = fc_from_boards_and_values(boards, series, prop_name="value", color_series=color_series)
    
    # Remove label_layer from choropleth view too
    deck = pydeck_choropleth(fc, lat_c, lon_c, zoom_c, prop_name="value",
                             tooltip_label=sel_label, opacity=0.90, label_layer=None)  # Set to None
    c_map, c_leg = st.columns([4,1])
    with c_map:
        st.pydeck_chart(deck, use_container_width=True)
    with c_leg:
        st.markdown(legend_with_ranges_html(sel_label, edges, colors=PALETTE4), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        dl_df = series.rename(sel_attr).reset_index()
        dl_df.columns = ["unit_id", sel_attr]
        dl_df["tract_id"] = dl_df["unit_id"].map(lambda x: normalize_cb_id(x))
        st.download_button(
            "Download CSV",
            data=dl_df.to_csv(index=False).encode("utf-8"),
            file_name=f"urban_{sel_attr}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with c2:
        png_bytes = save_choropleth_png(boards, series, f"Urban Feature: {sel_label}")
        st.download_button(
            "Download PNG",
            data=png_bytes,
            file_name=f"urban_{sel_attr}.png",
            mime="image/png",
            use_container_width=True
        )

    st.divider()
    st.markdown("#### Methodology")
    st.info(METH_URBAN)
