from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from UI.components.embedding_visuals import (
    render_task2_visuals,
    render_task3_visuals,
    render_task4_visuals,
)
from UI.task_service import (
    PROJECT_ROOT,
    ensure_task_outputs,
    find_similar_words_for_task,
    list_task5_checkpoints,
    predict_task5_sentiment,
    solve_vector_equation_for_task,
    similarity_example_words,
    task5_inference_status,
    task5_model_options,
    task_outputs,
    task_title,
    vector_arithmetic_examples,
)


def _result_key(task_id: str) -> str:
    return f"result_{task_id}"


def _error_key(task_id: str) -> str:
    return f"error_{task_id}"


def _similarity_result_key(task_id: str) -> str:
    return f"similarity_result_{task_id}"


def _comparison_similarity_result_key(task_id: str) -> str:
    return f"comparison_similarity_result_{task_id}"


def _comparison_vector_result_key(task_id: str) -> str:
    return f"comparison_vector_result_{task_id}"


def _task5_sentiment_result_key() -> str:
    return "task5_sentiment_result"


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
    if run_col.button("Run / Use Saved Results", key=f"run_{task_id}", width="stretch"):
        _run_task(task_id=task_id, force=False)

    if force_button:
        if force_col.button("Force Rebuild", key=f"force_{task_id}", width="stretch"):
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

    st.dataframe(df, width="stretch")


def _render_file_size(path: Path, label: str) -> None:
    if not path.exists():
        st.caption(f"{label}: missing")
        return
    size_mb = path.stat().st_size / (1024 * 1024)
    st.caption(f"{label}: `{path.relative_to(PROJECT_ROOT)}` ({size_mb:.2f} MB)")


def _merge_examples(*groups: list[str], limit: int = 8) -> list[str]:
    merged: list[str] = []
    for group in groups:
        for item in group:
            if item in merged:
                continue
            merged.append(item)
            if len(merged) >= limit:
                return merged
    return merged


def _render_suggestions(suggestions: Any) -> None:
    if isinstance(suggestions, dict):
        for word, options in suggestions.items():
            if not options:
                continue
            st.caption(f"Suggestions for `{word}`: " + ", ".join(f"`{option}`" for option in options))
        return

    if suggestions:
        st.caption("Try one of these: " + ", ".join(f"`{word}`" for word in suggestions))


def _render_similarity_result(result: dict[str, Any] | None, vectors_path: Path, empty_message: str) -> None:
    if not result:
        if not vectors_path.exists():
            st.info("Run this task once to generate vectors, then use this search box.")
        else:
            st.info(empty_message)
        return

    status = result.get("status", "")
    if status == "ok":
        st.caption(
            f"Showing neighbors for `{result['query_word']}` "
            f"from `{result['vocab_size']}` words with dimension `{result['dimension']}`."
        )
        st.dataframe(pd.DataFrame(result["results"]), width="stretch", hide_index=True)
        return

    if status in {"missing_vectors", "empty_query", "word_not_found"}:
        st.warning(result.get("message", "Could not complete the similarity search."))
        _render_suggestions(result.get("suggestions"))
        return

    st.error(result.get("message", "Unexpected error while searching similar words."))


def _render_vector_arithmetic_result(result: dict[str, Any] | None, vectors_path: Path, empty_message: str) -> None:
    if not result:
        if not vectors_path.exists():
            st.info("Run this task once to generate vectors, then use this vector arithmetic tool.")
        else:
            st.info(empty_message)
        return

    equation = result.get("equation", "")
    if equation:
        st.caption(f"Equation: `{equation}`")

    status = result.get("status", "")
    if status == "ok":
        st.caption(
            f"Showing top predictions from `{result['vocab_size']}` words "
            f"with dimension `{result['dimension']}`."
        )
        st.dataframe(pd.DataFrame(result["results"]), width="stretch", hide_index=True)
        return

    if status in {"missing_vectors", "empty_query", "word_not_found", "degenerate_query"}:
        st.warning(result.get("message", "Could not complete the vector arithmetic search."))
        _render_suggestions(result.get("suggestions"))
        return

    st.error(result.get("message", "Unexpected error while running vector arithmetic."))


def _render_similarity_search(task_id: str, model_label: str, vectors_path: Path) -> None:
    st.subheader("Interactive Similarity Search")
    st.caption(f"Type a word from the saved {model_label} vocabulary to see its 5 most similar words.")

    examples = similarity_example_words(task_id)
    if examples:
        st.caption("Example queries: " + ", ".join(f"`{word}`" for word in examples))

    with st.form(key=f"similarity_form_{task_id}"):
        query_word = st.text_input(
            "Query word",
            key=f"similarity_query_{task_id}",
            placeholder="Enter a vocabulary word",
        )
        submitted = st.form_submit_button("Find 5 Similar Words", width="stretch")

    if submitted:
        try:
            st.session_state[_similarity_result_key(task_id)] = find_similar_words_for_task(
                task_id=task_id,
                query_word=query_word,
                limit=5,
            )
        except Exception as exc:
            st.session_state[_similarity_result_key(task_id)] = {
                "status": "error",
                "message": str(exc),
            }

    result = st.session_state.get(_similarity_result_key(task_id))
    _render_similarity_result(
        result=result,
        vectors_path=vectors_path,
        empty_message="Search for a word to see its 5 nearest neighbors.",
    )


def _render_task4_similarity_comparison() -> None:
    st.subheader("Interactive Similarity Comparison")
    st.caption("Search one word once and compare Task 2 on the left with Task 3 on the right.")

    examples = _merge_examples(
        similarity_example_words("task2"),
        similarity_example_words("task3"),
        limit=8,
    )
    if examples:
        st.caption("Example queries: " + ", ".join(f"`{word}`" for word in examples))

    with st.form(key="task4_similarity_comparison_form"):
        query_word = st.text_input(
            "Query word",
            key="task4_similarity_query",
            placeholder="Enter one word to compare across both models",
        )
        submitted = st.form_submit_button("Compare Top 5 Similar Words", width="stretch")

    if submitted:
        for task_id in ("task2", "task3"):
            try:
                st.session_state[_comparison_similarity_result_key(task_id)] = find_similar_words_for_task(
                    task_id=task_id,
                    query_word=query_word,
                    limit=5,
                )
            except Exception as exc:
                st.session_state[_comparison_similarity_result_key(task_id)] = {
                    "status": "error",
                    "message": str(exc),
                }

    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown("### Left: Task 2 - Word2Vec")
        _render_similarity_result(
            result=st.session_state.get(_comparison_similarity_result_key("task2")),
            vectors_path=task_outputs("task2")["vectors"],
            empty_message="Submit a word to compare Word2Vec neighbors.",
        )

    with right_col:
        st.markdown("### Right: Task 3 - GloVe")
        _render_similarity_result(
            result=st.session_state.get(_comparison_similarity_result_key("task3")),
            vectors_path=task_outputs("task3")["vectors"],
            empty_message="Submit a word to compare GloVe neighbors.",
        )


def _render_task4_vector_arithmetic_comparison() -> None:
    st.subheader("Interactive Vector Arithmetic Comparison")
    st.caption("Use the same `A - B + C` expression against both models and compare the top predictions.")

    examples = _merge_examples(
        vector_arithmetic_examples("task2"),
        vector_arithmetic_examples("task3"),
        limit=6,
    )
    if examples:
        st.caption("Example equations: " + ", ".join(f"`{equation}`" for equation in examples))

    with st.form(key="task4_vector_comparison_form"):
        word_a_col, word_b_col, word_c_col = st.columns(3)
        word_a = word_a_col.text_input("Word A", key="task4_vector_word_a", placeholder="moskva")
        word_b = word_b_col.text_input("Word B", key="task4_vector_word_b", placeholder="rusiya")
        word_c = word_c_col.text_input("Word C", key="task4_vector_word_c", placeholder="ukrayna")
        submitted = st.form_submit_button("Compare Vector Arithmetic", width="stretch")

    if submitted:
        if not word_a.strip() or not word_b.strip() or not word_c.strip():
            shared_error = {
                "status": "empty_query",
                "message": "Enter all three words to compare the `A - B + C` expression.",
            }
            st.session_state[_comparison_vector_result_key("task2")] = shared_error
            st.session_state[_comparison_vector_result_key("task3")] = shared_error.copy()
        else:
            for task_id in ("task2", "task3"):
                try:
                    st.session_state[_comparison_vector_result_key(task_id)] = solve_vector_equation_for_task(
                        task_id=task_id,
                        positive_words=[word_a, word_c],
                        negative_words=[word_b],
                        limit=5,
                    )
                except Exception as exc:
                    st.session_state[_comparison_vector_result_key(task_id)] = {
                        "status": "error",
                        "message": str(exc),
                    }

    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown("### Left: Task 2 - Word2Vec")
        _render_vector_arithmetic_result(
            result=st.session_state.get(_comparison_vector_result_key("task2")),
            vectors_path=task_outputs("task2")["vectors"],
            empty_message="Submit an equation to compare Word2Vec predictions.",
        )

    with right_col:
        st.markdown("### Right: Task 3 - GloVe")
        _render_vector_arithmetic_result(
            result=st.session_state.get(_comparison_vector_result_key("task3")),
            vectors_path=task_outputs("task3")["vectors"],
            empty_message="Submit an equation to compare GloVe predictions.",
        )


def _render_task4_saved_results() -> None:
    task2_outputs = task_outputs("task2")
    task3_outputs = task_outputs("task3")

    st.subheader("Saved Result Comparison")
    st.caption("Task 2 is shown on the left and Task 3 is shown on the right so you can compare the stored outputs.")

    overview_left, overview_right = st.columns(2)
    with overview_left:
        st.markdown("### Left: Task 2 - Word2Vec")
        _render_file_size(task2_outputs["vectors"], "Vector file")
        _render_file_size(task2_outputs["vocab"], "Vocabulary file")

    with overview_right:
        st.markdown("### Right: Task 3 - GloVe")
        _render_file_size(task3_outputs["vectors"], "Vector file")
        _render_file_size(task3_outputs["vocab"], "Vocabulary file")

    similarity_left, similarity_right = st.columns(2)
    with similarity_left:
        st.markdown("### Left: Task 2 - Word2Vec")
        _render_table(task2_outputs["synonyms"], "Top-5 Similarity Results", sep="\t", preview_rows=20)

    with similarity_right:
        st.markdown("### Right: Task 3 - GloVe")
        _render_table(task3_outputs["synonyms"], "Top-5 Similarity Results", sep="\t", preview_rows=20)

    equation_left, equation_right = st.columns(2)
    with equation_left:
        st.markdown("### Left: Task 2 - Word2Vec")
        _render_table(task2_outputs["equations"], "Vector Arithmetic Results", sep="\t", preview_rows=15)

    with equation_right:
        st.markdown("### Right: Task 3 - GloVe")
        _render_table(task3_outputs["equations"], "Vector Arithmetic Results", sep="\t", preview_rows=15)

    relation_left, relation_right = st.columns(2)
    with relation_left:
        st.markdown("### Left: Task 2 - Word2Vec")
        _render_table(task2_outputs["relations"], "Relation Pattern Results", sep="\t", preview_rows=15)

    with relation_right:
        st.markdown("### Right: Task 3 - GloVe")
        _render_table(task3_outputs["relations"], "Relation Pattern Results", sep="\t", preview_rows=15)


def _render_task5_sentiment_predictor() -> None:
    st.subheader("Interactive Sentiment Prediction")
    st.caption("Write a sentence, choose one saved Task5 model, and predict its sentiment.")

    status = task5_inference_status()
    if status["status"] == "missing_artifacts":
        st.warning(status["message"])
        return
    if status["status"] == "incompatible_cache":
        st.warning(status["message"])

    model_options = task5_model_options()
    if not model_options:
        st.warning("No Task5 model options are available yet.")
        return

    option_lookup = {option["id"]: option for option in model_options}
    with st.form(key="task5_sentiment_prediction_form"):
        sentence = st.text_area(
            "Sentence",
            key="task5_sentiment_sentence",
            placeholder="Write a sentence to test the model's sentiment prediction",
            height=120,
        )
        selected_model_id = st.selectbox(
            "Model",
            options=[option["id"] for option in model_options],
            key="task5_sentiment_model",
            format_func=lambda option_id: option_lookup[option_id]["label"],
        )
        submitted = st.form_submit_button("Predict Sentiment", width="stretch")

    if submitted:
        try:
            with st.spinner("Predicting sentiment..."):
                result = predict_task5_sentiment(selected_model_id, sentence)
            result["input_sentence"] = sentence.strip()
            st.session_state[_task5_sentiment_result_key()] = result
        except Exception as exc:
            st.session_state[_task5_sentiment_result_key()] = {
                "status": "error",
                "message": str(exc),
                "input_sentence": sentence.strip(),
            }

    result = st.session_state.get(_task5_sentiment_result_key())
    if not result:
        st.info("Submit a sentence to see the predicted sentiment.")
        return

    if result.get("status") == "ok":
        st.success(
            f"Predicted sentiment: `{result['sentiment_label']}` "
            f"with confidence `{result['confidence']:.2%}`"
        )
        st.caption(f"Sentence checked: `{result.get('input_sentence', '')}`")
        st.caption(f"Model used: `{result['feature']} + {result['model']}`")
        st.dataframe(pd.DataFrame(result["scores"]), width="stretch", hide_index=True)
        return

    if result.get("status") in {"empty_query", "missing_checkpoint", "invalid_checkpoint", "incompatible_cache"}:
        if result.get("input_sentence"):
            st.caption(f"Sentence checked: `{result['input_sentence']}`")
        st.warning(result.get("message", "Could not predict sentiment for the provided sentence."))
        return

    st.error(result.get("message", "Unexpected error during sentiment prediction."))


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
    render_task2_visuals()
    _render_similarity_search("task2", "Word2Vec", outputs["vectors"])

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
    render_task3_visuals()
    _render_similarity_search("task3", "GloVe", outputs["vectors"])

    _render_markdown(outputs["report"], "Report")
    _render_table(outputs["synonyms"], "Top-5 Synonyms", sep="\t", preview_rows=50)
    _render_table(outputs["similar_word_math"], "Similar-Word Math", sep="\t", preview_rows=20)
    _render_table(outputs["equations"], "Vector Equations", sep="\t", preview_rows=20)
    _render_table(outputs["relations"], "Relation Patterns", sep="\t", preview_rows=20)


def render_task4_page() -> None:
    outputs = task_outputs("task4")
    st.header(task_title("task4"))
    st.info(
        "Task 4 is a direct UI comparison view: Task 2 (Word2Vec) is shown on the left and "
        "Task 3 (GloVe) is shown on the right so users can compare the same results interactively."
    )
    render_task4_visuals()
    _render_task4_similarity_comparison()
    _render_task4_vector_arithmetic_comparison()
    _render_task4_saved_results()

    with st.expander("Task 4 Written Comparison Report"):
        _render_markdown(outputs["report"], "Comparison Report")


def render_task5_page() -> None:
    outputs = task_outputs("task5")
    st.header(task_title("task5"))
    _render_run_controls("task5")

    _render_task5_sentiment_predictor()

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
