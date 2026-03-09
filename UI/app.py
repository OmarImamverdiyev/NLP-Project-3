from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from UI.components.task_pages import render_task_page
from UI.task_service import task_ids, task_title


def main() -> None:
    st.set_page_config(page_title="NLP Project 3 UI", layout="wide")
    st.title("NLP Project 3 - Streamlit UI")
    st.caption(
        "All tasks are available here. Models/results are stored on disk and reused across Streamlit restarts."
    )

    available_ids = task_ids()
    selected_task = st.sidebar.selectbox(
        "Select Task",
        options=available_ids,
        format_func=task_title,
    )

    st.sidebar.caption("Use 'Run / Use Saved Results' to reuse cached outputs when available.")
    st.sidebar.caption("Use 'Force Rebuild' only when you need retraining or refreshed outputs.")

    render_task_page(selected_task)


if __name__ == "__main__":
    main()
