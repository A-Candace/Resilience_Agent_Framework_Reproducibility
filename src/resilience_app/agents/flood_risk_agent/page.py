from io import BytesIO
import json, math, os, re
from datetime import datetime, timedelta, timezone, date
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from resilience_app.core.shared import *
from resilience_app.services.data_loader import ensure_boards
def list_numeric_candidates(boards, max_show=60):
    """Scan attribute keys and return numeric-looking fields"""
    keys = set()
    samples = []
    for b in boards[:min(len(boards), max_show)]:
        attrs = b.get("attrs", {}) or {}
        for k, v in attrs.items():
            kl = (k or "").lower()
            if any(tok in kl for tok in ["shape", "geom", "globalid", "objectid"]):
                continue
            num = coerce_numeric(v, None)
            if num is not None:
                keys.add(k)
                samples.append((k, num))
    scores = {}
    for k in keys:
        kc = k.lower()
        score = 0
        for token in ["risk", "cfl", "rfl", "hwav", "score", "eal", "alr"]:
            if token in kc:
                score += 1
        scores[k] = score
    ordered = sorted(list(keys), key=lambda x: (-scores.get(x,0), x))
    return ordered, dict(scores)

def pick_field_with_fallback(boards, requested: str):
    """Try requested field; if missing, surface a selectbox"""
    exists = False
    for b in boards:
        if get_attr_ci(b.get("attrs", {}) or {}, requested) is not None:
            exists = True
            break
    if exists:
        return requested, False

    st.warning(f"Requested field `{requested}` not found. Pick a field from your layer that holds numeric risk values.")
    candidates, _ = list_numeric_candidates(boards)
    if not candidates:
        st.error("No numeric-looking fields found in the layer's attributes.")
        return requested, False
    chosen = st.selectbox("Choose field", options=candidates, index=0, key=f"risk_field_fallback_{requested}")
    return chosen, True

def rescale_score_component(series: pd.Series, component_name: str) -> pd.Series:
    """
    Rescale score components that use 3-4 or 4-5 range down to 0-1 range.
    - score_precip: 0 or 4-5 range → subtract 4 from non-zero values to get 0-1
    - score_sf, score_cb: 0 or 3-4 range → subtract 3 from non-zero values to get 0-1
    
    This makes equal-interval binning more meaningful.
    """
    s = series.copy()
    
    if component_name in ['score_sf', 'score_cb']:
        # Subtract 3 from all non-zero values to rescale 3-4 → 0-1
        s = s.apply(lambda x: 0.0 if x <= 0 else x - 3.0)
    elif component_name == 'score_precip':
        # score_precip uses 0 or 4-5 range, subtract 4 from non-zero to get 0-1
        s = s.apply(lambda x: 0.0 if x <= 0 else x - 4.0)
    
    return s

def page_risk_mapping():
    st.title("🗺️ Flood and Heat Risk Mapping")
    boards = ensure_boards()
    if not boards:
        st.stop()

    # Choose which risk field to show
    risk_choice = st.radio(
        "Risk layer:",
        [
            "Coastal Flooding Risk (NRI - CFLD_RISKS)",
            "Riverine Flooding Risk (NRI - RFLD_RISKS)",
            "Heat Wave Risk (NRI - HWAV_RISKS)"
        ],
        horizontal=False
    )
    
    # Skip divider selection
    if "─────" in risk_choice:
        st.info("👆 Please select a risk layer above")
        st.stop()
    
    if "Coastal" in risk_choice:
        requested_field = "CFLD_RISKS"
        pretty          = "NRI Coastal Flood Risk"
        meth_key        = "NRI Coastal"
        risk_type       = "nri"
        allow_fallback  = False
        rescale_component = None
    elif "Riverine" in risk_choice:
        requested_field = "RFLD_RISKS"
        pretty          = "NRI Riverine Flood Risk"
        meth_key        = "NRI Riverine"
        risk_type       = "nri"
        allow_fallback  = False
        rescale_component = None
    elif "Heat Wave" in risk_choice:
        requested_field = "HWAV_RISKS"
        pretty          = "NRI Heat Wave Risk"
        meth_key        = "NRI Heat Wave"
        risk_type       = "heat"
        allow_fallback  = False
        rescale_component = None
    elif "Combined Score" in risk_choice:
        requested_field = "score_total"
        pretty          = "Urban Flooding - Combined Risk"
        meth_key        = "Urban Combined"
        risk_type       = "urban"
        allow_fallback  = True
        rescale_component = None  # Don't rescale total (it's a sum)
    elif "Water Sensor" in risk_choice:
        requested_field = "score_precip"
        pretty          = "Urban Flooding - Precipitation Sensitivity"
        meth_key        = "Urban Precip"
        risk_type       = "urban"
        allow_fallback  = True
        rescale_component = "score_precip"
    elif "Street Flooding" in risk_choice:
        requested_field = "score_sf"
        pretty          = "Urban Flooding - Street Flooding Reports"
        meth_key        = "Urban SF"
        risk_type       = "urban"
        allow_fallback  = True
        rescale_component = "score_sf"
    elif "Catch Basin" in risk_choice:
        requested_field = "score_cb"
        pretty          = "Urban Flooding - Catch Basin Issues"
        meth_key        = "Urban CB"
        risk_type       = "urban"
        allow_fallback  = True
        rescale_component = "score_cb"

    # For NRI fields, use directly; for custom, allow fallback
    if allow_fallback:
        field_name, used_fallback = pick_field_with_fallback(boards, requested_field)
        if used_fallback:
            st.info(f"Using `{field_name}` (custom field selection)")
    else:
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

    # Extract robust numeric RAW values from merged Excel attrs
    raw_vals = {}
    for b in boards:
        v_raw = get_attr_ci(b.get('attrs', {}) or {}, field_name)
        raw_vals[b["unit_id"]] = coerce_numeric(v_raw, default=0.0)
    raw_series = pd.Series(raw_vals).sort_index()
    
    # Apply rescaling if needed for better visualization
    display_series = raw_series.copy()
    use_zero_bin = False
    
    if rescale_component:
        display_series = rescale_score_component(raw_series, rescale_component)
        use_zero_bin = True  # Use special zero-bin binning
        st.info(f"ℹ️ **Visualization Note**: Original scores (3-4 or 4-5 range) rescaled to 0-1. "
                f"**Bin 0 (lightest)** = zero values. **Bins 1-3** = equal intervals for positive values. "
                f"Download CSV contains both original and rescaled values.")

    # Bin by quartiles or special zero-bin method
    if use_zero_bin:
        bins, edges = quantile_bins_4_with_zero(display_series)
    else:
        bins, edges = quantile_bins_4(display_series)

    # Map each quartile to a color
    color_map = {0: PALETTE4[0], 1: PALETTE4[1], 2: PALETTE4[2], 3: PALETTE4[3]}
    color_series = pd.Series({uid: color_map[int(bins.loc[uid])] for uid in bins.index})

    # Build GeoJSON FeatureCollection using DISPLAY values for tooltips
    fc = fc_from_boards_and_values(
        boards,
        display_series,
        prop_name="risk_value",
        color_series=color_series
    )

    # Render map + legend with numeric ranges per quartile
    lat_c, lon_c, zoom_c = boards_bbox({
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "geometry": b["feature_geom"]} for b in boards]
    })
    deck = pydeck_choropleth(
        fc, lat_c, lon_c, zoom_c,
        prop_name="risk_value",
        tooltip_label=f"{pretty} (rescaled)" if rescale_component else f"{pretty} (raw)",
        opacity=0.90,
        label_layer=None
    )

    c_map, c_leg = st.columns([4, 1])
    with c_map:
        st.pydeck_chart(deck, use_container_width=True)
    with c_leg:
        if use_zero_bin:
            legend_title = f"{pretty} (0-1 rescaled)"
            st.markdown(
                legend_with_zero_bin_html(legend_title, edges, colors=PALETTE4),
                unsafe_allow_html=True
            )
        else:
            range_labels = [f"{edges[i]:.2f} – {edges[i+1]:.2f}" for i in range(4)]
            legend_title = f"{pretty} (quartiles)"
            st.markdown(
                legend_html(legend_title, range_labels, PALETTE4),
                unsafe_allow_html=True
            )

    # Debug / QA panel
    with st.expander("🔎 Debug distribution &amp; field check", expanded=False):
        if rescale_component:
            st.write("**Original (raw) value stats:**")
            st.dataframe(raw_series.describe().to_frame(name="original"))
            st.write("**Rescaled (display) value stats:**")
            st.dataframe(display_series.describe().to_frame(name="rescaled"))
            
            # Show bin distribution
            st.write("**Bin distribution:**")
            bin_counts = bins.value_counts().sort_index()
            bin_labels = ["Bin 0 (Zero)", "Bin 1 (Low)", "Bin 2 (Medium)", "Bin 3 (High)"]
            bin_df = pd.DataFrame({
                "Bin": [bin_labels[i] if i in bin_counts.index else bin_labels[i] for i in range(4)],
                "Count": [bin_counts.get(i, 0) for i in range(4)],
                "Percentage": [f"{bin_counts.get(i, 0)/len(bins)*100:.1f}%" for i in range(4)]
            })
            st.dataframe(bin_df)
        else:
            st.write("**Raw value stats:**")
            st.dataframe(raw_series.describe().to_frame(name="raw_stats"))

        counts = bins.value_counts().sort_index().reindex([0, 1, 2, 3], fill_value=0)
        counts.index = [f"Q{i+1}" for i in range(4)]
        st.write("**Quartile bin counts:**")
        st.dataframe(counts.to_frame(name="count").T)

        sample_attrs = boards[0].get("attrs", {}) or {}
        st.write("**Example feature attribute keys (first 60):**")
        st.write(list(sample_attrs.keys())[:60])
        st.write(f"**Field in use:** `{field_name}` → example raw:", get_attr_ci(sample_attrs, field_name))

        if display_series.nunique() <= 1:
            st.info("All values are identical or nearly identical. Coloring still works via safe edges, but variance is minimal.")

    # Downloads - include both original and rescaled if applicable
    c1, c2 = st.columns(2)
    with c1:
        if rescale_component:
            dl_df = pd.DataFrame({
                "unit_id": raw_series.index,
                "risk_original": raw_series.values,
                "risk_rescaled_0_1": display_series.values,
                "bin_0_zero_1_3_intervals": bins.values
            })
        else:
            dl_df = pd.DataFrame({
                "unit_id": raw_series.index,
                "risk_raw": raw_series.values,
                "bin_quartile_0..3": bins.values
            })
        dl_df["tract_id"] = dl_df["unit_id"].map(lambda x: normalize_cb_id(x))
        st.download_button(
            "Download CSV",
            data=dl_df.to_csv(index=False).encode("utf-8"),
            file_name=f"risk_map_{field_name.lower()}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with c2:
        png_series = pd.Series({k: float(v) for k, v in bins.items()})
        png_bytes = save_choropleth_png(boards, png_series, f"Risk Map: {pretty}", palette=PALETTE4, breaks=None)
        st.download_button(
            "Download PNG",
            data=png_bytes,
            file_name=f"risk_map_{field_name.lower()}.png",
            mime="image/png",
            use_container_width=True
        )

    # Methodology + Claude explanation
    st.divider()
    st.markdown("#### Methodology")
    
    # Display appropriate methodology based on risk type
    if risk_type == "nri":
        st.info(METH_RISK[meth_key])
    elif risk_type == "heat":
        st.info(METH_UHI)
    else:  # urban flooding
        st.info(METH_RISK_URBAN.get(meth_key, METH_RISK["My Risk Map"]))

    src = st.session_state.get("boards_source", "Uploaded GeoJSON/Shapefile")
    st.caption(f"**Source**: {src} • **Field used**: `{field_name}`")
    
    if rescale_component:
        st.caption(f"**Note**: Values rescaled from original {requested_field} range to 0-1. "
                   f"Bin 0 = zero values; Bins 1-3 = equal intervals for positive values.")
