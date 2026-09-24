from io import BytesIO
import json, math, os, re
from datetime import datetime, timedelta, timezone, date
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from resilience_app.core.shared import *
from resilience_app.services.data_loader import ensure_boards
def page_ai_query():
    st.title("🎯 Multi-risk Identification Tool")
    st.caption("Use natural language to find census tracts matching multiple criteria")
    
    boards = ensure_boards()
    if not boards:
        st.stop()
    
    st.markdown("### Example Queries:")
    st.markdown("- *Show tracts with highest riverine flooding risk AND highest heat wave risk*")
    st.markdown("- *Find tracts with lowest elevation AND highest coastal flooding risk*")
    st.markdown("- *Show me tracts with highest heat wave risk and lowest income*")
    st.markdown("- *Find tracts with lowest elevation AND highest coastal flooding risk AND lowest income*") 
    st.markdown("- *Show tracts in the red portion (highest interval) for riverine, coastal, and heat wave*")
    st.markdown("- *Find tracts with highest building density and highest coastal flood risk*")
    
    user_query = st.text_area(
        "Enter your query:",
        height=100,
        placeholder="e.g., Show me tracts with highest coastal flood risk and lowest elevation",
        key="query_input"  # Add key to preserve state
    )
    
    if st.button("🔍 Search Tracts", type="primary"):
        if not user_query.strip():
            st.warning("Please enter a query")
            st.stop()
        
        with st.spinner("🤖 Claude is parsing your query..."):
            parsed = parse_user_query_with_claude(user_query, boards)
        
        # Store results in session state
        st.session_state.query_results = {
            "parsed": parsed,
            "query": user_query
        }
    
    # Check if we have stored results
    if "query_results" not in st.session_state:
        st.info("👆 Enter a query above to get started")
        st.stop()
    
    # Use stored results
    parsed = st.session_state.query_results["parsed"]
    original_query = st.session_state.query_results["query"]
    
    st.write("**Original query:**", original_query)
    st.write("**Parsed criteria:**")
    st.json(parsed)
    
    criteria = parsed.get("criteria", [])
    if not criteria:
        st.error("Could not parse query. Try rephrasing with specific field names.")
        if st.button("🔄 Try Again"):
            del st.session_state.query_results
            st.rerun()
        st.stop()
    
    # Compute density if needed
    for crit in criteria:
        if crit.get("field") == "_COMPUTED_DENSITY_":
            for b in boards:
                attrs = b.get("attrs", {})
                ftp = coerce_numeric(get_attr_ci(attrs, "Total_ftp_area"), default=0.0)
                area_sqmi = coerce_numeric(get_attr_ci(attrs, "AREA"), default=0.0)
                area_sqkm = area_sqmi * 2.58999
                if area_sqkm > 0:
                    attrs["_COMPUTED_DENSITY_"] = ftp / area_sqkm
                else:
                    attrs["_COMPUTED_DENSITY_"] = 0.0
    
    matching_tracts = filter_tracts_by_criteria(boards, criteria)
    
    st.success(f"✅ Found **{len(matching_tracts)}** matching tracts (out of {len(boards)})")
    
    if not matching_tracts:
        st.info("No tracts match all criteria. Try relaxing your query.")
        if st.button("🔄 New Search"):
            del st.session_state.query_results
            st.rerun()
        st.stop()
    
    # Build map: grey for all, blue for matching
    fc_features = []
    for b in boards:
        uid = b["unit_id"]
        is_match = any(m["unit_id"] == uid for m in matching_tracts)
        fill_color = [0, 120, 255, 200] if is_match else [200, 200, 200, 100]
        
        fc_features.append({
            "type": "Feature",
            "properties": {
                "unit_id": uid,
                "cb": get_board_display_id(b),
                "matched": "Yes" if is_match else "No",
                "fill_color": fill_color
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
        tooltip={"html": "<b>Tract:</b> {cb}<br><b>Matched:</b> {matched}<br><i>Click for details</i>"}
    )
    
    st.pydeck_chart(deck, use_container_width=True)
    
    # Legend
    st.markdown("""
    <div style="padding:10px; background:#f0f0f0; border-radius:5px; margin:10px 0;">
        <b>Legend:</b> 
        <span style="color:#0078ff;">● Blue</span> = Matching tracts | 
        <span style="color:#c8c8c8;">● Grey</span> = Non-matching tracts
    </div>
    """, unsafe_allow_html=True)
    
    # Show matching tracts with OpenStreetMap
    st.subheader(f"📍 {len(matching_tracts)} Matching Tracts")
    
    # Select a tract to view
    tract_options = {get_board_display_id(m): m for m in matching_tracts}
    
    # Use session state to preserve selection
    if "selected_tract_key" not in st.session_state:
        st.session_state.selected_tract_key = "-- Select --"
    
    selected_display = st.selectbox(
        "Select a tract to view in detail:",
        options=["-- Select --"] + list(tract_options.keys()),
        index=0 if st.session_state.selected_tract_key == "-- Select --" else 
              (["-- Select --"] + list(tract_options.keys())).index(st.session_state.selected_tract_key) 
              if st.session_state.selected_tract_key in tract_options else 0,
        key="tract_selector_dropdown"
    )
    
    # Update session state
    st.session_state.selected_tract_key = selected_display
    
    if selected_display != "-- Select --":
        selected_tract = tract_options[selected_display]
        attrs = selected_tract.get("attrs", {})
        nri_id = attrs.get("NRI_ID", "")
        
        # Get centroid
        centroid = selected_tract["geom"].centroid
        lat, lon = centroid.y, centroid.x
        
        st.markdown(f"### 🗺️ Tract {selected_display}")
        
        # Show attribute values
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**NRI ID:** {nri_id}")
            st.write(f"**Coordinates:** {lat:.6f}, {lon:.6f}")
            st.write("**Criteria values:**")
            for crit in criteria:
                field = crit.get("field")
                val = get_attr_num(attrs, field)
                desc = crit.get("description", field)
                st.write(f"- {desc}: **{val:.2f}**")
        
        with col2:
            st.write("**Quick Links:**")
            osm_url = f"https://www.openstreetmap.org/#map=18/{lat}/{lon}"
            osm_edit = f"https://www.openstreetmap.org/edit?editor=id#map=19/{lat}/{lon}"
            google_maps = f"https://www.google.com/maps/@{lat},{lon},18z"
            
            st.markdown(f"[🗺️ OpenStreetMap]({osm_url})")
            st.markdown(f"[📐 OSM Measurement Tool]({osm_edit})")
            st.markdown(f"[🛰️ Google Maps Satellite]({google_maps})")
        
        # Embed OpenStreetMap
        st.markdown("### 🗺️ Interactive Map View")
        
        try:
            import folium
            from streamlit_folium import st_folium
            
            # Create folium map centered on the tract
            m = folium.Map(
                location=[lat, lon],
                zoom_start=17,
                tiles='OpenStreetMap'
            )
            
            # Add satellite imagery tile layer
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Satellite',
                overlay=False,
                control=True
            ).add_to(m)
            
            # Add additional useful layers
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Satellite with Labels',
                overlay=False,
                control=True
            ).add_to(m)
            
            # Add OpenStreetMap as an alternative (already default, but making it explicit)
            folium.TileLayer(
                tiles='OpenStreetMap',
                name='Street Map',
                overlay=False,
                control=True
            ).add_to(m)
            
            # Add marker for the tract center
            folium.Marker(
                [lat, lon],
                popup=f"<b>Tract {selected_display}</b><br>NRI ID: {nri_id}<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}",
                tooltip=f"Tract {selected_display}",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
            
            # Add tract boundary polygon if available
            try:
                from shapely.geometry import mapping
                tract_geojson = mapping(selected_tract["geom"])
                folium.GeoJson(
                    tract_geojson,
                    style_function=lambda x: {
                        'fillColor': '#ff0000',
                        'color': '#ff0000',
                        'weight': 3,
                        'fillOpacity': 0.2,
                        'dashArray': '5, 5'
                    },
                    tooltip=f"Tract {selected_display} boundary"
                ).add_to(m)
            except Exception:
                pass  # Skip if polygon rendering fails
            
            # Add layer control to toggle between views
            folium.LayerControl(position='topright').add_to(m)
            
            # Add scale bar
            folium.plugins.MeasureControl(
                position='topleft',
                primary_length_unit='meters',
                secondary_length_unit='miles',
                primary_area_unit='sqmeters',
                secondary_area_unit='acres'
            ).add_to(m)
            
            # Add fullscreen button
            folium.plugins.Fullscreen(
                position='topleft',
                title='Fullscreen',
                title_cancel='Exit fullscreen',
                force_separate_button=True
            ).add_to(m)
            
            # Display the map
            st_folium(m, width=700, height=500)
            
            # Add link to full OSM and other mapping tools
            larger_map_url = f"https://www.openstreetmap.org/?mlat={lat}&amp;mlon={lon}#map=18/{lat}/{lon}"
            osm_id_editor_url = f"https://www.openstreetmap.org/edit?editor=id#map=19/{lat}/{lon}"
            google_maps_url = f"https://www.google.com/maps/@{lat},{lon},18z"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'<span><span style="color: rgb(150, 34, 73); font-weight: bold;"><small></span><span style="color: black; font-weight: normal;">🗺️ OpenStreetMap</span><span style="color: rgb(150, 34, 73); font-weight: bold;"></small></span><br><br></span>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<span><span style="color: rgb(150, 34, 73); font-weight: bold;"><small></span><span style="color: black; font-weight: normal;">📐 OSM Measurement Tool</span><span style="color: rgb(150, 34, 73); font-weight: bold;"></small></span><br><br></span>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<span><span style="color: rgb(150, 34, 73); font-weight: bold;"><small></span><span style="color: black; font-weight: normal;">🛰️ Google Maps</span><span style="color: rgb(150, 34, 73); font-weight: bold;"></small></span><br><br></span>', unsafe_allow_html=True)
            
        except ImportError:
            st.warning("⚠️ Folium not installed. Install with: `pip install folium streamlit-folium`")
            
            # Fallback: just show the link
            larger_map_url = f"https://www.openstreetmap.org/?mlat={lat}&amp;mlon={lon}#map=18/{lat}/{lon}"
            st.markdown(f'**[🗺️ View this location on OpenStreetMap]({larger_map_url})**')
        
        st.info("💡 **Tip:** Use the layer control (top-right) to toggle between Street Map, Satellite, and Satellite with Labels. "
                "Use the ruler icon to measure distances. Click 'OSM Measurement Tool' to access the iD Editor with street view (look for the 📷 person icon on the left sidebar).")    # Download matching tracts CSV
    st.divider()
    match_data = []
    for m in matching_tracts:
        attrs = m.get("attrs", {})
        row = {
            "NRI_ID": attrs.get("NRI_ID"),
            "tract_display": get_board_display_id(m),
            "latitude": m["geom"].centroid.y,
            "longitude": m["geom"].centroid.x,
        }
        # Add criteria fields
        for crit in criteria:
            field = crit.get("field")
            row[field] = get_attr_num(attrs, field)
        match_data.append(row)
    
    match_df = pd.DataFrame(match_data)
    
    col_dl, col_new = st.columns([3, 1])
    with col_dl:
        st.download_button(
            "📥 Download Matching Tracts CSV",
            data=match_df.to_csv(index=False).encode("utf-8"),
            file_name="matching_tracts.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_new:
        if st.button("🔄 New Search", use_container_width=True):
            del st.session_state.query_results
            if "selected_tract_key" in st.session_state:
                del st.session_state.selected_tract_key
            st.rerun()
