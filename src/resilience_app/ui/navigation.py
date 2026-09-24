import os
import streamlit as st
from resilience_app.core.shared import HAS_BEDROCK

def sidebar():
    with st.sidebar:
        st.markdown("### NYC Resilience AI Agent")
        st.caption("Census Tract Analysis: Static Features, Risk Maps, and Forecasting")
        st.divider()

        st.button(
            "📈 Forecasting",
            use_container_width=True,
            on_click=lambda: st.session_state.update(page="forecast"),
            key="nav_forecast"
        )

        st.button(
            "🏠 Home",
            use_container_width=True,
            on_click=lambda: st.session_state.update(page="landing"),
            key="nav_home"
        )

        st.button(
            "🏙️ Urban Features",
            use_container_width=True,
            on_click=lambda: st.session_state.update(page="urban"),
            key="nav_urban"
        )

        st.button(
            "🗺️ Flood and Heat Risk Mapping",
            use_container_width=True,
            on_click=lambda: st.session_state.update(page="risk"),
            key="nav_risk"
        )

        st.button(
            "👥 Socio-Demographics",
            use_container_width=True,
            on_click=lambda: st.session_state.update(page="demographics"),
            key="nav_demographics"
        )

        st.button(
            "🎯 Multi-risk Identification Tool",
            use_container_width=True,
            on_click=lambda: st.session_state.update(page="query"),
            key="nav_query"
        )

        st.button(
            "🤖 Use Agent",
            use_container_width=True,
            on_click=lambda: st.session_state.update(page="chat"),
            key="nav_chat"
        )

        with st.expander("🧮 Calculation Tools", expanded=False):
            st.button(
                "🟩 Green Roof Calculator",
                use_container_width=True,
                on_click=lambda: st.session_state.update(page="green_roof"),
                key="nav_green_roof"
            )

            st.button(
                "🌿 Rain Garden Estimator",
                use_container_width=True,
                on_click=lambda: st.session_state.update(page="rain_garden"),
                key="nav_rain_garden"
            )

        st.divider()
        st.markdown("**🚧 Under Construction:**")

        st.button(
            "🌡️ Heat Wave Forecasting",
            use_container_width=True,
            key="nav_heat_forecast_construction",
            disabled=True
        )

        st.button(
            "🛰️ Satellite Datasets",
            use_container_width=True,
            key="nav_satellite_construction",
            disabled=True
        )

        st.button(
            "🌀 Diffusion Modeling",
            use_container_width=True,
            key="nav_diffusion_construction",
            disabled=True
        )

        st.button(
            "📜 Historical Forecasts",
            use_container_width=True,
            key="nav_historical_forecasts_construction",
            disabled=True
        )

        st.divider()

        if HAS_BEDROCK:
            st.success("Claude (Bedrock) connected.")
        else:
            missing = [
                k for k in (
                    "AWS_ACCESS_KEY_ID",
                    "AWS_SECRET_ACCESS_KEY",
                    "AWS_REGION"
                )
                if not os.getenv(k)
            ]
            st.info("Claude: missing " + ", ".join(missing))
