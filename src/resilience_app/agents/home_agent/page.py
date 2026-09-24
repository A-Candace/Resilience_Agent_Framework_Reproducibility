from io import BytesIO
import json, math, os, re
from datetime import datetime, timedelta, timezone, date
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from resilience_app.core.shared import *
from resilience_app.services.data_loader import ensure_boards
def home_page():
    st.title("🏙️ NYC Resilience AI Agent")
    st.subheader("Census Tract Analysis: Static Urban Features • Risk Maps • Forecasting")
    st.write("This application analyzes NYC census tracts with elevation, slope, building footprint data, and NRI flood risk scores.")
    st.divider()
    st.markdown("- Each map is downloadable as **PNG** and **CSV**.")
    st.markdown("- Risk maps use FEMA NRI data at census tract level.")
    st.markdown("- Color scheme: **Red (high values/risk) → Yellow (low values/risk)**")

def page_landing():
    st.title("🧭 Choose a Domain")
    st.subheader("🤖 AI-Powered Tools")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎯 Multi-risk Identification Tool", use_container_width=True, key="landing_query"):
            st.session_state.page = "query"
            st.rerun()
        st.caption("Find tracts matching complex criteria using natural language")

    with col2:
        if st.button("🤖 Use Agent", use_container_width=True, key="landing_chat"):
            st.session_state.page = "chat"
            st.rerun()
        st.caption("Use the orchestrator agent and its connected tools")
