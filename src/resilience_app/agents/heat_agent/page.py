from io import BytesIO
import json, math, os, re
from datetime import datetime, timedelta, timezone, date
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from resilience_app.core.shared import *
from resilience_app.services.data_loader import ensure_boards
def page_uhi():
    st.title("🌤️ Urban Heat Island (NRI Heat Wave Risk)")
    boards = ensure_boards()
    if not boards:
        st.stop()

    # NRI field - should exist in Excel, no fallback needed
    requested_field = "HWAV_RISKS"
    field_name = requested_field
    
    # Check if field exists
    found = False
    for b in boards[:10]:
        if get_attr_ci(b.get("attrs", {}), field_name) is not None:
            found = True
            break
    if not found:
        st.error(f"❌ Required NRI field `{field_name}` not found in the merged Excel data. "
                f"Please ensure Risk_Attributes_Table_v4.xlsx contains this column and the merge was successful.")
        st.stop()

    # Pull RAW values
    raw_vals = {}
    for b in boards:
        v_raw = get_attr_ci(b.get('attrs', {}) or {}, field_name)
        raw_vals[b["unit_id"]] = coerce_numeric(v_raw, default=0.0)
    raw_series = pd.Series(raw_vals).sort_index()

    # Quartile bins + numeric range labels
    bins, edges = quantile_bins_4(raw_series)
    color_map = {0: PALETTE4[0], 1: PALETTE4[1], 2: PALETTE4[2], 3: PALETTE4[3]}
    color_series = pd.Series({uid: color_map[int(bins.loc[uid])] for uid in bins.index})

    # Build FC with colors driven by quartile, tooltip shows RAW values
    fc = fc_from_boards_and_values(
        boards,
        raw_series,
        prop_name="uhi_value",
        color_series=color_series
    )

    lat_c, lon_c, zoom_c = boards_bbox({
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "geometry": b["feature_geom"]} for b in boards]
    })
    label_layer = cb_label_layer_from_boards(boards, text_size=10)
    deck = pydeck_choropleth(
        fc, lat_c, lon_c, zoom_c,
        prop_name="uhi_value",
        tooltip_label="Heat Wave Risk (raw)",
        opacity=0.90,
        label_layer=None
    )
    c_map, c_leg = st.columns([4,1])
    with c_map:
        st.pydeck_chart(deck, use_container_width=True)
    with c_leg:
        range_labels = [f"{edges[i]:.2f} – {edges[i+1]:.2f}" for i in range(4)]
        st.markdown(
            legend_html("NRI Heat-Wave Risk (quartiles)", range_labels, PALETTE4),
            unsafe_allow_html=True
        )

    # Debug panel
    with st.expander("🔎 Debug distribution &amp; field check", expanded=False):
        st.write("**Raw value stats:**")
        st.dataframe(raw_series.describe().to_frame(name="raw_stats"))
        counts = bins.value_counts().sort_index().reindex([0,1,2,3], fill_value=0)
        counts.index = [f"Q{i+1}" for i in range(4)]
        st.write("**Quartile bin counts:**")
        st.dataframe(counts.to_frame(name="count").T)

        sample_attrs = boards[0].get("attrs", {}) or {}
        st.write("**Example feature attribute keys (first 60):**")
        st.write(list(sample_attrs.keys())[:60])
        st.write(f"**Field in use:** `{field_name}` → example raw:", get_attr_ci(sample_attrs, field_name))

        if raw_series.nunique() <= 1:
            st.info("All values are identical or nearly identical. Colors collapse to one bin.")

    # Downloads
    c1, c2 = st.columns(2)
    with c1:
        dl_df = pd.DataFrame({
            "unit_id": raw_series.index,
            "heat_risk_raw": raw_series.values,
            "bin_quartile_0..3": bins.values
        })
        dl_df["tract_id"] = dl_df["unit_id"].map(lambda x: normalize_cb_id(x))
        st.download_button(
            "Download CSV",
            data=dl_df.to_csv(index=False).encode("utf-8"),
            file_name=f"uhi_{field_name.lower()}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with c2:
        png_series = pd.Series({k: float(v) for k, v in bins.items()})
        png_bytes = save_choropleth_png(boards, png_series, "UHI: Heat Wave Risk", palette=PALETTE4, breaks=None)
        st.download_button(
            "Download PNG",
            data=png_bytes,
            file_name=f"uhi_{field_name.lower()}.png",
            mime="image/png",
            use_container_width=True
        )
