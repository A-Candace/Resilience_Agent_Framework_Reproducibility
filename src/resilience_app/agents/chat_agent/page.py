from __future__ import annotations

import copy
import json
import os
import uuid
from pathlib import Path

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

from agentic.common.settings import get_settings
from resilience_app.core.shared import *
from resilience_app.services.data_loader import ensure_boards


NYC_LATITUDE = 40.7128
NYC_LONGITUDE = -74.0060

FORECAST_ROOT = Path(
    os.getenv(
        "GRID_FLOOD_FORECAST_ROOT",
        "/app/artifacts/flood/forecasting",
    )
)

GRID_GEOJSON_PATH = Path(
    os.getenv(
        "GRID_IMPUTATION_GEOJSON_PATH",
        "/app/artifacts/flood/spatial/"
        "grid_imputation_reference.geojson",
    )
)

DAYPART_HOURS = {
    "overnight": range(0, 6),
    "morning": range(6, 12),
    "afternoon": range(12, 18),
    "evening": range(18, 22),
    "night": range(22, 24),
}


@st.cache_data(show_spinner=False)
def _load_chat_forecast(
    path: str,
    modified_time: float,
) -> pd.DataFrame:
    del modified_time

    frame = pd.read_parquet(path).copy()

    frame["grid_id"] = frame["grid_id"].astype(str)

    frame["forecast_hour"] = pd.to_datetime(
        frame["forecast_hour"],
        utc=True,
        errors="coerce",
    )

    if "predicted_flood_event" in frame.columns:
        frame["forecast_susceptible"] = (
            frame["predicted_flood_event"]
            .fillna(False)
            .astype(bool)
        )
    elif "predicted_flood" in frame.columns:
        frame["forecast_susceptible"] = (
            frame["predicted_flood"]
            .fillna(False)
            .astype(bool)
        )
    else:
        raise ValueError(
            "Forecast artifact contains neither "
            "predicted_flood_event nor predicted_flood."
        )

    return frame


@st.cache_data(show_spinner=False)
def _load_chat_grid(
    path: str,
    modified_time: float,
) -> dict:
    del modified_time

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def _render_forecast_susceptibility_map(
    visualization: dict,
) -> None:
    if (
        not visualization
        or visualization.get("type")
        != "forecast_susceptibility_map"
    ):
        return

    mode = str(
        visualization.get("mode", "today")
    ).strip().lower()

    daypart = str(
        visualization.get("daypart", "")
    ).strip().lower()

    forecast_path = (
        FORECAST_ROOT
        / mode
        / "grid_flood_forecast.parquet"
    )

    if not forecast_path.exists():
        st.warning(
            "The operational forecast map could not be displayed "
            f"because {forecast_path} was not found."
        )
        return

    if not GRID_GEOJSON_PATH.exists():
        st.warning(
            "The operational 1-km grid map could not be displayed "
            f"because {GRID_GEOJSON_PATH} was not found."
        )
        return

    forecast = _load_chat_forecast(
        str(forecast_path),
        forecast_path.stat().st_mtime,
    )

    grid_geojson = _load_chat_grid(
        str(GRID_GEOJSON_PATH),
        GRID_GEOJSON_PATH.stat().st_mtime,
    )

    forecast["forecast_hour_nyc"] = (
        forecast["forecast_hour"]
        .dt.tz_convert("America/New_York")
    )

    if daypart in DAYPART_HOURS:
        allowed_hours = set(
            DAYPART_HOURS[daypart]
        )

        period_frame = forecast[
            forecast[
                "forecast_hour_nyc"
            ].dt.hour.isin(
                allowed_hours
            )
        ].copy()
    else:
        period_frame = forecast.copy()

    if period_frame.empty:
        st.info(
            "No published forecast hours are available "
            "for this requested map period."
        )
        return

    susceptible = period_frame[
        period_frame["forecast_susceptible"]
    ].copy()

    susceptible_grid_ids = set(
        susceptible["grid_id"].astype(str)
    )

    susceptible_hours = (
        susceptible.groupby("grid_id")[
            "forecast_hour"
        ]
        .nunique()
        .to_dict()
    )

    map_geojson = copy.deepcopy(
        grid_geojson
    )

    for feature in map_geojson.get(
        "features",
        [],
    ):
        props = feature.setdefault(
            "properties",
            {},
        )

        grid_id = str(
            props.get("grid_id")
        )

        is_susceptible = (
            grid_id
            in susceptible_grid_ids
        )

        props["grid_id"] = grid_id
        props[
            "forecast_susceptible"
        ] = is_susceptible

        props[
            "susceptible_hour_count"
        ] = int(
            susceptible_hours.get(
                grid_id,
                0,
            )
        )

        props["status"] = (
            "Forecast flood-susceptible"
            if is_susceptible
            else "Not forecast flood-susceptible "
            "during requested period"
        )

    layer = pdk.Layer(
        "GeoJsonLayer",
        data=map_geojson,
        pickable=True,
        stroked=True,
        filled=True,
        get_fill_color=(
            "properties.forecast_susceptible "
            "? [220, 40, 40, 110] "
            ": [180, 180, 180, 20]"
        ),
        get_line_color=(
            "properties.forecast_susceptible "
            "? [150, 0, 0, 180] "
            ": [100, 100, 100, 50]"
        ),
        get_line_width=1,
        line_width_min_pixels=0.5,
    )

    tooltip = {
        "html": """
        <b>Grid:</b> {grid_id}<br/>
        <b>Status:</b> {status}<br/>
        <b>Susceptible forecast hours:</b>
        {susceptible_hour_count}
        """,
        "style": {
            "backgroundColor": "white",
            "color": "black",
            "fontSize": "12px",
        },
    }

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(
            latitude=NYC_LATITUDE,
            longitude=NYC_LONGITUDE,
            zoom=9.7,
            pitch=0,
        ),
        tooltip=tooltip,
        map_style=(
            "mapbox://styles/mapbox/light-v10"
        ),
    )

    title_period = (
        daypart.title()
        if daypart
        else mode.title()
    )

    st.markdown(
        f"#### Forecasted Flood-Susceptible Grids — "
        f"{title_period}"
    )

    st.pydeck_chart(
        deck,
        use_container_width=True,
    )

    st.caption(
        "Red polygons = operational predicted flood "
        "susceptibility during at least one requested "
        "forecast hour. These are model predictions, "
        "not confirmed or observed flooding."
    )


def _render_chat_message(
    message: dict,
) -> None:
    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

        visualization = message.get(
            "visualization"
        )

        if visualization:
            _render_forecast_susceptibility_map(
                visualization
            )


def page_chat() -> None:
    st.title("💬 Chat with Claude")

    st.caption(
        "Ask questions about methodology, data, "
        "or get analysis recommendations"
    )

    boards = ensure_boards()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "orchestrator_session_id" not in st.session_state:
        st.session_state.orchestrator_session_id = (
            str(uuid.uuid4())
        )

    for message in st.session_state.chat_history:
        _render_chat_message(
            message
        )

    prompt = st.chat_input(
        "Ask about NYC resilience data, "
        "methodologies, or analysis..."
    )

    if prompt:
        user_message_record = {
            "role": "user",
            "content": prompt,
        }

        st.session_state.chat_history.append(
            user_message_record
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        context_info = {
            "num_tracts": (
                len(boards)
                if boards
                else 0
            ),
            "available_fields": {
                "static": [
                    "Mean_elevation",
                    "Mean_slope",
                    "Total_ftp_area",
                    "AREA",
                    "Building_density",
                ],
                "nri_risks": [
                    "CFLD_RISKS (coastal)",
                    "RFLD_RISKS (riverine)",
                    "HWAV_RISKS (heat wave)",
                ],
                "custom_risks": [
                    "score_precip",
                    "score_sf",
                    "score_cb",
                    "score_total",
                ],
            },
            "methodologies": {
                "urban_features": METH_URBAN,
                "nri_coastal": METH_RISK[
                    "NRI Coastal"
                ],
                "nri_riverine": METH_RISK[
                    "NRI Riverine"
                ],
                "custom_risk": METH_RISK[
                    "My Risk Map"
                ],
                "heat": METH_UHI,
            },
        }

        system_prompt = (
            "You are an expert assistant for NYC urban "
            "resilience and climate risk analysis. "
            "You help city planners understand flood risk, "
            "heat vulnerability, and infrastructure resilience. "
            "Provide accurate, actionable insights based on "
            "FEMA NRI data and local analysis. "
            "For operational flood forecasts, use predictive "
            "susceptibility language rather than stating that "
            "flooding will definitely occur. "
            "If you need specific information not present in "
            "the provided context, say what information is "
            "needed and suggest authoritative sources. "
            "Be concise but thorough."
        )

        full_prompt = f"""User question: {prompt}

Context about available data:
{json.dumps(context_info, indent=2)}

Provide a helpful, accurate answer.

If the question requires:
- Specific tract-level data → explain what fields to check
- External research → suggest official sources
- Calculations → explain the methodology

Answer:"""

        tool_calls = []
        visualization = None

        with st.chat_message(
            "assistant"
        ):
            with st.spinner(
                "🤔 Claude is thinking..."
            ):
                try:
                    orchestrator_url = (
                        get_settings()
                        .orchestrator_url
                    )

                    response_object = requests.post(
                        f"{orchestrator_url}/v1/agent",
                        json={
                            "message": prompt,
                            "session_id": (
                                st.session_state
                                .orchestrator_session_id
                            ),
                        },
                        timeout=60,
                    )

                    response_object.raise_for_status()

                    agent_result = (
                        response_object.json()
                    )

                    response = agent_result[
                        "response"
                    ]

                    tool_calls = (
                        agent_result.get(
                            "tool_calls"
                        )
                        or []
                    )

                    visualization = (
                        agent_result.get(
                            "visualization"
                        )
                    )

                except Exception as exc:
                    st.warning(
                        "⚠️ Agent orchestrator temporarily "
                        f"unavailable ({exc}); falling back "
                        "to direct Claude call."
                    )

                    response = call_claude(
                        full_prompt,
                        system=system_prompt,
                        temperature=0.3,
                        max_tokens=1500,
                    )

            st.markdown(response)

            if visualization:
                _render_forecast_susceptibility_map(
                    visualization
                )

            if tool_calls:
                with st.expander(
                    f"🔧 Used "
                    f"{len(tool_calls)} tool call(s)"
                ):
                    for call in tool_calls:
                        st.caption(
                            f"`{call.get('name', 'unknown')}`"
                        )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": response,
                "visualization": visualization,
            }
        )

        genai_log(
            full_prompt,
            response,
            meta={
                "span_name": "chat_interaction",
                "user_query": prompt,
            },
        )

    if st.button(
        "🗑️ Clear Chat History"
    ):
        st.session_state.chat_history = []
        st.rerun()