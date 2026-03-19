from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from UI.task_service import _load_normalized_vectors, task_outputs


_MODEL_COLORS = {
    "task2": "#0f766e",
    "task3": "#2563eb",
}

_RELATION_PRESETS: dict[str, dict[str, object]] = {
    "country_capital": {
        "label": "Country -> Capital",
        "description": "Countries and their capitals should produce nearly parallel relation links if the embedding captures the pattern well.",
        "pairs": [
            ("Azərbaycan", "azərbaycan", "Bakı", "bakı"),
            ("Rusiya", "rusiya", "Moskva", "moskva"),
            ("Ukrayna", "ukrayna", "Kiyev", "kiyev"),
            ("Türkiyə", "türkiyə", "Ankara", "ankara"),
            ("ABŞ", "abş", "Vaşinqton", "vaşinqton"),
        ],
    },
    "country_leader": {
        "label": "Country -> Leader",
        "description": "This view checks whether country-to-leader offsets align in a shared semantic direction.",
        "pairs": [
            ("Azərbaycan", "azərbaycan", "Əliyev", "əliyev"),
            ("Rusiya", "rusiya", "Putin", "putin"),
            ("Ukrayna", "ukrayna", "Zelenski", "zelenski"),
            ("Türkiyə", "türkiyə", "Ərdoğan", "ərdoğan"),
            ("ABŞ", "abş", "Tramp", "tramp"),
        ],
    },
    "gender_family": {
        "label": "Gender / Family",
        "description": "These paired terms are useful for checking whether gender-like relations behave consistently across several word pairs.",
        "pairs": [
            ("Kişi", "kişi", "Qadın", "qadın"),
            ("Ata", "ata", "Ana", "ana"),
            ("Oğlan", "oğlan", "Qız", "qız"),
            ("Kral", "kral", "Kraliça", "kraliça"),
            ("Ər", "ər", "Arvad", "arvad"),
        ],
    },
}


def _vectors_path(task_id: str) -> Path:
    return task_outputs(task_id)["vectors"]


def _available_pairs(task_id: str, preset_key: str) -> tuple[list[dict[str, str]], list[str]]:
    preset = _RELATION_PRESETS[preset_key]
    _, _, index, _ = _load_normalized_vectors(_vectors_path(task_id))

    available: list[dict[str, str]] = []
    missing: list[str] = []
    for source_label, source_token, target_label, target_token in preset["pairs"]:
        if source_token in index and target_token in index:
            available.append(
                {
                    "source_label": source_label,
                    "source_token": source_token,
                    "target_label": target_label,
                    "target_token": target_token,
                    "pair_label": f"{source_label} -> {target_label}",
                }
            )
        else:
            missing.append(f"{source_label} -> {target_label}")

    return available, missing


def _project_relation_pairs(task_id: str, preset_key: str) -> dict[str, object]:
    pairs, missing = _available_pairs(task_id, preset_key)
    if len(pairs) < 2:
        return {
            "status": "insufficient_pairs",
            "message": "At least two relation pairs are needed to draw this analogy plot.",
            "available_pairs": pairs,
            "missing_pairs": missing,
        }

    words, vectors, index, _ = _load_normalized_vectors(_vectors_path(task_id))
    del words

    ordered_tokens: list[str] = []
    token_to_label: dict[str, str] = {}
    token_to_side: dict[str, str] = {}
    for pair in pairs:
        for token_key, label_key, side in (
            ("source_token", "source_label", "Source"),
            ("target_token", "target_label", "Target"),
        ):
            token = pair[token_key]
            if token not in token_to_label:
                ordered_tokens.append(token)
                token_to_label[token] = pair[label_key]
                token_to_side[token] = side

    matrix = np.vstack([np.asarray(vectors[index[token]], dtype=np.float32) for token in ordered_tokens])
    centered = matrix - matrix.mean(axis=0, keepdims=True)

    if centered.shape[0] >= 2:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        basis = vh[:2].T
        coords = centered @ basis
    else:
        coords = np.zeros((centered.shape[0], 2), dtype=np.float32)
        coords[:, 0] = centered[:, 0]

    if coords.shape[1] < 2:
        coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0], dtype=np.float32)])

    token_to_coord = {token: coords[i].copy() for i, token in enumerate(ordered_tokens)}
    relation_deltas = np.vstack(
        [
            token_to_coord[pair["target_token"]] - token_to_coord[pair["source_token"]]
            for pair in pairs
        ]
    )
    mean_delta = relation_deltas.mean(axis=0)
    angle = float(np.arctan2(mean_delta[1], mean_delta[0])) if np.linalg.norm(mean_delta) > 0 else 0.0
    rotation = np.array(
        [
            [np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle), np.cos(-angle)],
        ],
        dtype=np.float32,
    )
    coords = coords @ rotation.T

    source_mean_x = float(
        np.mean([coords[ordered_tokens.index(pair["source_token"]), 0] for pair in pairs])
    )
    target_mean_x = float(
        np.mean([coords[ordered_tokens.index(pair["target_token"]), 0] for pair in pairs])
    )
    if target_mean_x < source_mean_x:
        coords[:, 0] *= -1

    coords -= coords.mean(axis=0, keepdims=True)
    max_abs = float(np.abs(coords).max())
    if max_abs > 0:
        coords *= 0.78 / max_abs

    points_rows: list[dict[str, object]] = []
    for idx, token in enumerate(ordered_tokens):
        points_rows.append(
            {
                "token": token,
                "label": token_to_label[token],
                "side": token_to_side[token],
                "x": float(coords[idx, 0]),
                "y": float(coords[idx, 1]),
            }
        )

    point_lookup = {
        row["token"]: row
        for row in points_rows
    }

    lines_rows: list[dict[str, object]] = []
    for pair in pairs:
        source_row = point_lookup[pair["source_token"]]
        target_row = point_lookup[pair["target_token"]]
        lines_rows.append(
            {
                "pair": pair["pair_label"],
                "source_label": pair["source_label"],
                "target_label": pair["target_label"],
                "x": source_row["x"],
                "y": source_row["y"],
                "x2": target_row["x"],
                "y2": target_row["y"],
            }
        )

    lines_df = pd.DataFrame(lines_rows)
    points_df = pd.DataFrame(points_rows)
    relation_strength = _mean_relation_alignment(lines_df)

    return {
        "status": "ok",
        "points": points_df,
        "lines": lines_df,
        "pairs": pairs,
        "missing_pairs": missing,
        "relation_strength": relation_strength,
    }


def _mean_relation_alignment(lines_df: pd.DataFrame) -> float | None:
    if lines_df.empty or len(lines_df) < 2:
        return None

    deltas = np.column_stack(
        [
            lines_df["x2"].to_numpy(dtype=np.float32) - lines_df["x"].to_numpy(dtype=np.float32),
            lines_df["y2"].to_numpy(dtype=np.float32) - lines_df["y"].to_numpy(dtype=np.float32),
        ]
    )
    norms = np.linalg.norm(deltas, axis=1)
    valid = norms > 0
    deltas = deltas[valid]
    norms = norms[valid]
    if len(deltas) < 2:
        return None

    normalized = deltas / norms[:, None]
    cosines: list[float] = []
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            cosines.append(float(np.dot(normalized[i], normalized[j])))
    if not cosines:
        return None
    return float(np.mean(cosines))


def _relation_chart(task_id: str, preset_key: str) -> tuple[alt.Chart | None, dict[str, object]]:
    projection = _project_relation_pairs(task_id, preset_key)
    if projection["status"] != "ok":
        return None, projection

    points_df = projection["points"]
    lines_df = projection["lines"]
    model_color = _MODEL_COLORS[task_id]
    extent = 0.85

    x_scale = alt.Scale(domain=[-extent, extent], nice=False)
    y_scale = alt.Scale(domain=[-extent, extent], nice=False)
    axis = alt.Axis(grid=False, title=None, tickCount=5, labelFontSize=11)

    lines = (
        alt.Chart(lines_df)
        .mark_rule(strokeDash=[7, 6], strokeWidth=1.6, color=model_color, opacity=0.8)
        .encode(
            x=alt.X("x:Q", scale=x_scale, axis=axis),
            y=alt.Y("y:Q", scale=y_scale, axis=axis),
            x2="x2:Q",
            y2="y2:Q",
            tooltip=["pair:N", "source_label:N", "target_label:N"],
        )
    )

    source_points = (
        alt.Chart(points_df[points_df["side"] == "Source"])
        .mark_circle(size=90, color="white", stroke=model_color, strokeWidth=2)
        .encode(
            x=alt.X("x:Q", scale=x_scale, axis=axis),
            y=alt.Y("y:Q", scale=y_scale, axis=axis),
            tooltip=["label:N", "side:N"],
        )
    )

    target_points = (
        alt.Chart(points_df[points_df["side"] == "Target"])
        .mark_circle(size=90, color=model_color, opacity=0.95)
        .encode(
            x=alt.X("x:Q", scale=x_scale, axis=axis),
            y=alt.Y("y:Q", scale=y_scale, axis=axis),
            tooltip=["label:N", "side:N"],
        )
    )

    source_text = (
        alt.Chart(points_df[points_df["side"] == "Source"])
        .mark_text(align="right", dx=-8, dy=-2, fontSize=13, color="#111827")
        .encode(
            x=alt.X("x:Q", scale=x_scale, axis=axis),
            y=alt.Y("y:Q", scale=y_scale, axis=axis),
            text="label:N",
        )
    )

    target_text = (
        alt.Chart(points_df[points_df["side"] == "Target"])
        .mark_text(align="left", dx=8, dy=-2, fontSize=13, color="#111827")
        .encode(
            x=alt.X("x:Q", scale=x_scale, axis=axis),
            y=alt.Y("y:Q", scale=y_scale, axis=axis),
            text="label:N",
        )
    )

    chart = (
        (lines + source_points + target_points + source_text + target_text)
        .properties(height=440)
        .configure_view(stroke="#d1d5db")
    )
    return chart, projection


def _render_relation_plot(task_id: str, preset_key: str, chart_key: str) -> None:
    preset = _RELATION_PRESETS[preset_key]
    chart, projection = _relation_chart(task_id, preset_key)
    if chart is None:
        st.info(projection["message"])
        missing_pairs = projection.get("missing_pairs", [])
        if missing_pairs:
            st.caption("Missing in this vocabulary: " + ", ".join(missing_pairs))
        return

    del chart_key
    st.altair_chart(chart, use_container_width=True)
    relation_strength = projection["relation_strength"]
    if relation_strength is not None:
        st.caption(
            f"{preset['description']} Mean pair-direction cosine in the 2D view: `{relation_strength:.3f}`."
        )
    else:
        st.caption(str(preset["description"]))

    missing_pairs = projection["missing_pairs"]
    if missing_pairs:
        st.caption("Skipped missing pairs: " + ", ".join(missing_pairs))


def _render_relation_tabs(task_id: str, prefix: str) -> None:
    tab_specs = list(_RELATION_PRESETS.items())
    tabs = st.tabs([spec["label"] for _, spec in tab_specs])
    for tab, (preset_key, preset) in zip(tabs, tab_specs):
        with tab:
            st.markdown(f"#### {preset['label']}")
            _render_relation_plot(task_id, preset_key, chart_key=f"{prefix}_{preset_key}")


def render_task2_visuals() -> None:
    st.subheader("Task 2 Relation Plots")
    st.caption("Word2Vec relations projected to 2D. Dashed links represent the same semantic offset across multiple word pairs.")
    _render_relation_tabs("task2", prefix="task2_relation_plot")


def render_task3_visuals() -> None:
    st.subheader("Task 3 Relation Plots")
    st.caption("GloVe relations projected to 2D. If the model captures the analogy well, these links should stay fairly parallel.")
    _render_relation_tabs("task3", prefix="task3_relation_plot")


def render_task4_visuals() -> None:
    st.subheader("Task 4 Relation Comparison")
    st.caption("The same relation families are shown side by side so you can compare Word2Vec and GloVe visually.")

    tab_specs = list(_RELATION_PRESETS.items())
    tabs = st.tabs([spec["label"] for _, spec in tab_specs])
    for tab, (preset_key, preset) in zip(tabs, tab_specs):
        with tab:
            st.markdown(f"#### {preset['label']}")
            left_col, right_col = st.columns(2)
            with left_col:
                st.markdown("##### Word2Vec")
                _render_relation_plot("task2", preset_key, chart_key=f"task4_task2_{preset_key}")
            with right_col:
                st.markdown("##### GloVe")
                _render_relation_plot("task3", preset_key, chart_key=f"task4_task3_{preset_key}")
