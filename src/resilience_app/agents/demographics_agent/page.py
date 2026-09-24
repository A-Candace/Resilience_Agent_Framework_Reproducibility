from io import BytesIO
import json, math, os, re
from datetime import datetime, timedelta, timezone, date
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from resilience_app.core.shared import *
from resilience_app.services.data_loader import ensure_boards
def page_demographics():
    st.title("👥 Socio-Demographic Mapping")
    st.caption("Census tract-level population characteristics from ACS 2014-2018")
    
    boards = ensure_boards()
    if not boards:
        st.stop()
    
    # Category selection
    st.subheader("Select a demographic category")
    
    category = st.selectbox(
        "Category:",
        options=list(DEMOGRAPHIC_CATEGORIES.keys()),
        index=0,
        key="demo_category"
    )
    
    # Field selection within category
    fields_in_category = DEMOGRAPHIC_CATEGORIES[category]
    
    sel_label = st.selectbox(
        f"{category} - Select metric:",
        options=list(fields_in_category.keys()),
        index=0,
        key="demo_field"
    )
    
    sel_attr = fields_in_category[sel_label]
    
    st.info(f"📊 Using field: **{sel_attr}**")
    
    # Check if field exists
    found = False
    for b in boards[:10]:
        if get_attr_ci(b.get("attrs", {}), sel_attr) is not None:
            found = True
            break
    
    if not found:
        st.error(f"❌ Field '{sel_attr}' not found in the merged Excel data. "
                f"Please ensure Risk_Attributes_Table_v4.xlsx contains this column.")
        
        # Show available fields for debugging
        with st.expander("🔍 Available fields in data"):
            sample_attrs = boards[0].get("attrs", {})
            st.write("Sample of available field names:")
            st.write(list(sample_attrs.keys())[:80])
        st.stop()
    
    # Extract values
    vals = {}
    for b in boards:
        raw_val = get_attr_ci(b.get("attrs", {}), sel_attr)
        vals[b["unit_id"]] = coerce_numeric(raw_val, default=0.0)
    
    series = pd.Series(vals).sort_index()
    
    # Show statistics
    with st.expander("📊 Data Statistics", expanded=False):
        st.write(f"**Field:** {sel_attr}")
        st.write(f"**Category:** {category}")
        st.write(f"**Non-zero values:** {(series > 0).sum()} / {len(series)}")
        st.dataframe(series.describe())
        
        # Show sample values
        st.write("**Sample values (first 10 tracts):**")
        sample_df = series.head(10).to_frame(name=sel_label)
        st.dataframe(sample_df)
    
    # Bin into quartiles
    bins, edges = quantile_bins_4(series)
    
    color_map = {0: PALETTE4[0], 1: PALETTE4[1], 2: PALETTE4[2], 3: PALETTE4[3]}
    color_series = pd.Series({uid: color_map[bins.loc[uid]] for uid in series.index})
    
    # Build feature collection
    fc = fc_from_boards_and_values(boards, series, prop_name="demo_value", color_series=color_series)
    
    # Render map
    lat_c, lon_c, zoom_c = boards_bbox({
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "geometry": b["feature_geom"]} for b in boards]
    })
    
    deck = pydeck_choropleth(
        fc, lat_c, lon_c, zoom_c,
        prop_name="demo_value",
        tooltip_label=sel_label,
        opacity=0.90,
        label_layer=None
    )
    
    c_map, c_leg = st.columns([4, 1])
    
    with c_map:
        st.pydeck_chart(deck, use_container_width=True)
    
    with c_leg:
        st.markdown(
            legend_with_ranges_html(sel_label, edges, colors=PALETTE4),
            unsafe_allow_html=True
        )
    
    # Downloads
    c1, c2 = st.columns(2)
    
    with c1:
        dl_df = pd.DataFrame({
            "unit_id": series.index,
            "tract_id": series.index.map(lambda x: normalize_cb_id(x)),
            sel_label: series.values,
            "quartile_bin_0_3": bins.values
        })
        
        st.download_button(
            "📥 Download CSV",
            data=dl_df.to_csv(index=False).encode("utf-8"),
            file_name=f"demographics_{category.lower().replace(' ', '_')}_{sel_label.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with c2:
        png_bytes = save_choropleth_png(boards, series, f"{category}: {sel_label}")
        st.download_button(
            "🖼️ Download PNG",
            data=png_bytes,
            file_name=f"demographics_{category.lower().replace(' ', '_')}.png",
            mime="image/png",
            use_container_width=True
        )
    
    # Methodology
    st.divider()
    st.markdown("#### Methodology")
    st.info(METH_DEMOGRAPHICS)
