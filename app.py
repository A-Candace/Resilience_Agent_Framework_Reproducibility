import streamlit as st
st.set_page_config(page_title="NYC Resilience AI Agent", page_icon="🌆", layout="wide")

from resilience_app.agents.home_agent import page_landing
from resilience_app.agents.registry import ROUTES
from resilience_app.ui.navigation import sidebar


def main():
    st.session_state.setdefault("page", "landing")
    sidebar()
    ROUTES.get(st.session_state.page, page_landing)()


if __name__ == "__main__":
    main()
