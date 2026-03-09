from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from UI.task_service import (
    PROJECT_ROOT,
    ensure_task_outputs,
    list_task5_checkpoints,
    task_outputs,
    task_title,
)


def _result_key(task_id: str) -> str:
    return f"result_{task_id}"


def _error_key(task_id: str) -> str:
    return f"error_{task_id}"


def _run_task(task_id: str, force: bool) -> None:
    st.session_state.pop(_error_key(task_id), None)
    try:
        with st.spinner("Running task and checking cache..."):
            result = ensure_task_outputs(task_id=task_id, force_rebuild=force)
        st.session_state[_result_key(task_id)] = result
    except Exception as exc:
        st.session_state[_error_key(task_id)] = str(exc)


def _render_run_controls(task_id: str, force_button: bool = True) -> None:
    run_col, force_col = st.columns(2)
    if run_col.button("Run / Use Saved Results", key=f"run_{task_id}", use_container_width=True):
        _run_task(task_id=task_id, force=False)

    if force_button:
        if force_col.button("Force Rebuild", key=f"force_{task_id}", use_container_width=True):
            _run_task(task_id=task_id, force=True)

    error_text = st.session_state.get(_error_key(task_id))
    if error_text:
        st.error(error_text)

    result = st.session_state.get(_result_key(task_id))
    if result:
        status = result.get("status", "")
        message = result.get("message", "")
        if status == "cache_hit":
            st.info(message)
        else:
            st.success(message)

        command = result.get("command", "")
        if command:
            st.caption(f"Command: `{command}`")

        output_tail = result.get("output_tail", "").strip()
        if output_tail:
            with st.expander("Latest Task Output"):
                st.code(output_tail)


def _render_markdown(path: Path, title: str) -> None:
    st.subheader(title)
    if not path.exists():
        st.warning(f"File not found: {path}")
        return

    st.markdown(path.read_text(encoding="utf-8", errors="ignore"))


def _render_table(path: Path, title: str, sep: str = ",", preview_rows: int = 25) -> None:
    st.subheader(title)
    if not path.exists():
        st.warning(f"File not found: {path}")
        return

    try:
        df = pd.read_csv(path, sep=sep, nrows=preview_rows)
    except Exception as exc:
        st.error(f"Could not load table: {exc}")
        return

    st.dataframe(df, use_container_width=True)


def _render_file_size(path: Path, label: str) -> None:
    if not path.exists():
        st.caption(f"{label}: missing")
        return
    size_mb = path.stat().st_size / (1024 * 1024)
    st.caption(f"{label}: `{path.relative_to(PROJECT_ROOT)}` ({size_mb:.2f} MB)")


def render_task1_page() -> None:
    outputs = task_outputs("task1")
    st.header(task_title("task1"))
    _render_run_controls("task1")

    _render_markdown(outputs["summary"], "Summary")
    _render_table(outputs["word_frequency"], "Word Frequency Preview", sep="\t", preview_rows=30)
    _render_table(outputs["term_document_matrix"], "Term-Document Matrix Preview", sep=",", preview_rows=20)
    _render_table(outputs["word_word_matrix"], "Word-Word Matrix Preview", sep=",", preview_rows=20)


def render_task2_page() -> None:
    outputs = task_outputs("task2")
    st.header(task_title("task2"))
    _render_run_controls("task2")

    _render_file_size(outputs["vectors"], "Word2Vec vectors")
    _render_file_size(outputs["vocab"], "Word2Vec vocabulary")

    _render_markdown(outputs["report"], "Report")
    _render_table(outputs["synonyms"], "Top-5 Synonyms", sep="\t", preview_rows=50)
    _render_table(outputs["equations"], "Vector Equations", sep="\t", preview_rows=20)
    _render_table(outputs["relations"], "Relation Patterns", sep="\t", preview_rows=20)


def render_task3_page() -> None:
    outputs = task_outputs("task3")
    st.header(task_title("task3"))
    _render_run_controls("task3")

    _render_file_size(outputs["vectors"], "GloVe vectors")
    _render_file_size(outputs["vocab"], "GloVe vocabulary")

    _render_markdown(outputs["report"], "Report")
    _render_table(outputs["synonyms"], "Top-5 Synonyms", sep="\t", preview_rows=50)
    _render_table(outputs["similar_word_math"], "Similar-Word Math", sep="\t", preview_rows=20)
    _render_table(outputs["equations"], "Vector Equations", sep="\t", preview_rows=20)
    _render_table(outputs["relations"], "Relation Patterns", sep="\t", preview_rows=20)


def render_task4_page() -> None:
    outputs = task_outputs("task4")
    st.header(task_title("task4"))
    _render_run_controls("task4", force_button=False)
    _render_markdown(outputs["report"], "Comparison Report")


def render_task5_page() -> None:
    outputs = task_outputs("task5")
    st.header(task_title("task5"))
    _render_run_controls("task5")

    checkpoints = list_task5_checkpoints()
    st.subheader("Saved Model Checkpoints")
    if not checkpoints:
        st.warning("No checkpoints found yet. Run Task5 once to create persistent model files.")
    else:
        st.caption(f"Checkpoint files: {len(checkpoints)}")
        preview = "\n".join(str(path.relative_to(PROJECT_ROOT)) for path in checkpoints[:15])
        st.code(preview)

    _render_table(outputs["results"], "Task5 Results", sep=",", preview_rows=50)
    _render_markdown(outputs["report"], "Report")


def render_task_page(task_id: str) -> None:
    if task_id == "task1":
        render_task1_page()
    elif task_id == "task2":
        render_task2_page()
    elif task_id == "task3":
        render_task3_page()
    elif task_id == "task4":
        render_task4_page()
    elif task_id == "task5":
        render_task5_page()
    else:
        st.error(f"Unknown task id: {task_id}")
