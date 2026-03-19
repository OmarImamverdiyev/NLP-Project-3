from __future__ import annotations

import argparse
import csv
import heapq
import math
import subprocess
import sys
import time
from array import array
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from UI.cache_utils import all_files_exist, load_json, path_signature, save_json, stable_hash


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = Path(__file__).resolve().parent
MODEL_CACHE_ROOT = UI_ROOT / "model_cache"
MANIFEST_DIR = MODEL_CACHE_ROOT / "manifests"


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    title: str
    script_path: Path | None
    required_files: tuple[Path, ...]
    model_files: tuple[Path, ...]


TASK_SPECS: dict[str, TaskSpec] = {
    "task1": TaskSpec(
        task_id="task1",
        title="Task1 - Dataset Description and Matrix Construction",
        script_path=PROJECT_ROOT / "Task1" / "task1.py",
        required_files=(
            PROJECT_ROOT / "Task1" / "output" / "dataset_summary.md",
            PROJECT_ROOT / "Task1" / "output" / "word_frequency.tsv",
            PROJECT_ROOT / "Task1" / "output" / "term_document_matrix_visual.csv",
            PROJECT_ROOT / "Task1" / "output" / "word_word_matrix.csv",
        ),
        model_files=(),
    ),
    "task2": TaskSpec(
        task_id="task2",
        title="Task2 - Word2Vec Training and Semantic Analysis",
        script_path=PROJECT_ROOT / "Task2" / "task2.py",
        required_files=(
            PROJECT_ROOT / "Task2" / "output" / "vectors.txt",
            PROJECT_ROOT / "Task2" / "output" / "vocab.txt",
            PROJECT_ROOT / "Task2" / "output" / "task2_report.md",
            PROJECT_ROOT / "Task2" / "output" / "synonyms.tsv",
            PROJECT_ROOT / "Task2" / "output" / "vector_equations.tsv",
            PROJECT_ROOT / "Task2" / "output" / "relation_patterns.tsv",
        ),
        model_files=(
            PROJECT_ROOT / "Task2" / "output" / "vectors.txt",
            PROJECT_ROOT / "Task2" / "output" / "vocab.txt",
        ),
    ),
    "task3": TaskSpec(
        task_id="task3",
        title="Task3 - GloVe Training and Semantic Analysis",
        script_path=PROJECT_ROOT / "Task3" / "task3.py",
        required_files=(
            PROJECT_ROOT / "Task3" / "output" / "vectors.txt",
            PROJECT_ROOT / "Task3" / "output" / "vocab.txt",
            PROJECT_ROOT / "Task3" / "output" / "task3_report.md",
            PROJECT_ROOT / "Task3" / "output" / "synonyms.tsv",
            PROJECT_ROOT / "Task3" / "output" / "vector_equations.tsv",
            PROJECT_ROOT / "Task3" / "output" / "relation_patterns.tsv",
        ),
        model_files=(
            PROJECT_ROOT / "Task3" / "output" / "vectors.txt",
            PROJECT_ROOT / "Task3" / "output" / "vocab.txt",
        ),
    ),
    "task4": TaskSpec(
        task_id="task4",
        title="Task4 - Word2Vec vs GloVe Comparison",
        script_path=None,
        required_files=(
            PROJECT_ROOT / "Task4" / "task4_report.md",
        ),
        model_files=(),
    ),
    "task5": TaskSpec(
        task_id="task5",
        title="Task5 - Text Classification with RNN Variants",
        script_path=PROJECT_ROOT / "Task5" / "task5.py",
        required_files=(
            PROJECT_ROOT / "Task5" / "output" / "task5_results.csv",
            PROJECT_ROOT / "Task5" / "output" / "task5_report.md",
            PROJECT_ROOT / "Task5" / "output" / "model_cache_manifest.json",
        ),
        model_files=(
            PROJECT_ROOT / "Task5" / "output" / "model_cache",
        ),
    ),
}


TASK_OUTPUTS: dict[str, dict[str, Path]] = {
    "task1": {
        "summary": PROJECT_ROOT / "Task1" / "output" / "dataset_summary.md",
        "word_frequency": PROJECT_ROOT / "Task1" / "output" / "word_frequency.tsv",
        "term_document_matrix": PROJECT_ROOT / "Task1" / "output" / "term_document_matrix_visual.csv",
        "word_word_matrix": PROJECT_ROOT / "Task1" / "output" / "word_word_matrix.csv",
    },
    "task2": {
        "report": PROJECT_ROOT / "Task2" / "output" / "task2_report.md",
        "synonyms": PROJECT_ROOT / "Task2" / "output" / "synonyms.tsv",
        "equations": PROJECT_ROOT / "Task2" / "output" / "vector_equations.tsv",
        "relations": PROJECT_ROOT / "Task2" / "output" / "relation_patterns.tsv",
        "vectors": PROJECT_ROOT / "Task2" / "output" / "vectors.txt",
        "vocab": PROJECT_ROOT / "Task2" / "output" / "vocab.txt",
    },
    "task3": {
        "report": PROJECT_ROOT / "Task3" / "output" / "task3_report.md",
        "synonyms": PROJECT_ROOT / "Task3" / "output" / "synonyms.tsv",
        "similar_word_math": PROJECT_ROOT / "Task3" / "output" / "similar_word_math.tsv",
        "equations": PROJECT_ROOT / "Task3" / "output" / "vector_equations.tsv",
        "relations": PROJECT_ROOT / "Task3" / "output" / "relation_patterns.tsv",
        "vectors": PROJECT_ROOT / "Task3" / "output" / "vectors.txt",
        "vocab": PROJECT_ROOT / "Task3" / "output" / "vocab.txt",
    },
    "task4": {
        "report": PROJECT_ROOT / "Task4" / "task4_report.md",
    },
    "task5": {
        "report": PROJECT_ROOT / "Task5" / "output" / "task5_report.md",
        "results": PROJECT_ROOT / "Task5" / "output" / "task5_results.csv",
        "cache_manifest": PROJECT_ROOT / "Task5" / "output" / "model_cache_manifest.json",
        "cache_dir": PROJECT_ROOT / "Task5" / "output" / "model_cache",
    },
}


def task_ids() -> list[str]:
    return ["task1", "task2", "task3", "task4", "task5"]


def task_title(task_id: str) -> str:
    return TASK_SPECS[task_id].title


def task_outputs(task_id: str) -> dict[str, Path]:
    return TASK_OUTPUTS[task_id]


def task_manifest_path(task_id: str) -> Path:
    return MANIFEST_DIR / f"{task_id}.json"


def resolve_dataset_file(project_root: Path = PROJECT_ROOT) -> Path:
    candidates = [
        project_root / "Corpora" / "news" / "content_only.csv",
        project_root / "Corpora" / "News" / "content_only.csv",
        project_root / "corpora" / "news" / "content_only.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for candidate in project_root.rglob("content_only.csv"):
        lowered = str(candidate).lower().replace("\\", "/")
        if "/news/" in lowered:
            return candidate

    raise FileNotFoundError("Could not find dataset file Corpora/news/content_only.csv")


def build_task_signature(task_id: str, run_args: Sequence[str] | None = None) -> str:
    if task_id not in TASK_SPECS:
        raise ValueError(f"Unknown task id: {task_id}")

    spec = TASK_SPECS[task_id]
    args = list(run_args or [])
    payload: dict[str, object] = {
        "task_id": task_id,
        "run_args": args,
    }

    if spec.script_path is not None:
        payload["script"] = path_signature(spec.script_path)

    if task_id in {"task1", "task2", "task3", "task5"}:
        dataset_file = resolve_dataset_file(PROJECT_ROOT)
        payload["dataset"] = path_signature(dataset_file)

    if task_id == "task2":
        payload["word2vec_source"] = path_signature(PROJECT_ROOT / "Task2" / "word2vec" / "word2vec.c")
        payload["word2vec_binary"] = path_signature(PROJECT_ROOT / "Task2" / "word2vec" / "word2vec.exe")
    elif task_id == "task3":
        payload["glove_sources"] = {
            "glove_c": path_signature(PROJECT_ROOT / "Task3" / "glove" / "src" / "glove.c"),
            "cooccur_c": path_signature(PROJECT_ROOT / "Task3" / "glove" / "src" / "cooccur.c"),
        }
    elif task_id == "task5":
        payload["task2_vectors"] = path_signature(PROJECT_ROOT / "Task2" / "output" / "vectors.txt")
        payload["task3_vectors"] = path_signature(PROJECT_ROOT / "Task3" / "output" / "vectors.txt")

    return stable_hash(payload)


def summarize_output(text: str, max_chars: int = 4000) -> str:
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[-max_chars:]


@lru_cache(maxsize=8)
def _load_normalized_vectors_cached(
    resolved_vectors_path: str,
    file_size: int,
    file_mtime_ns: int,
) -> tuple[list[str], list[array], dict[str, int], int]:
    del file_size, file_mtime_ns

    vectors_path = Path(resolved_vectors_path)
    words: list[str] = []
    vectors: list[array] = []
    index: dict[str, int] = {}
    dimension: int | None = None

    def append_vector(parts: list[str]) -> None:
        nonlocal dimension

        if dimension is None:
            if len(parts) < 2:
                raise ValueError(f"Unexpected vector format in {vectors_path}.")
            dimension = len(parts) - 1

        if len(parts) != dimension + 1:
            return

        word = parts[0]
        try:
            vector = array("f", (float(value) for value in parts[1:]))
        except ValueError:
            return

        norm = math.sqrt(sum(component * component for component in vector))
        if norm == 0:
            return

        for i in range(dimension):
            vector[i] /= norm

        index[word] = len(words)
        words.append(word)
        vectors.append(vector)

    with open(vectors_path, "r", encoding="utf-8", errors="ignore") as handle:
        first_parts = handle.readline().strip().split()
        if not first_parts:
            raise ValueError(f"Vector file is empty: {vectors_path}")

        if len(first_parts) == 2 and all(part.isdigit() for part in first_parts):
            dimension = int(first_parts[1])
        else:
            append_vector(first_parts)

        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            append_vector(parts)

    if dimension is None:
        raise ValueError(f"Could not determine vector dimension for {vectors_path}.")

    return words, vectors, index, dimension


def _load_normalized_vectors(vectors_path: Path) -> tuple[list[str], list[array], dict[str, int], int]:
    stat = vectors_path.stat()
    return _load_normalized_vectors_cached(
        str(vectors_path.resolve()),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def _dot(a: array, b: array) -> float:
    score = 0.0
    for x, y in zip(a, b):
        score += x * y
    return score


def _normalize_vector(values: array) -> array | None:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return None

    for i in range(len(values)):
        values[i] /= norm
    return values


def _normalize_query_words(words: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for word in words:
        cleaned = word.strip().lower()
        if cleaned:
            normalized.append(cleaned)
    return normalized


def _format_vector_equation(positive_words: Sequence[str], negative_words: Sequence[str]) -> str:
    if not positive_words:
        return ""

    parts = [positive_words[0]]
    for word in negative_words:
        parts.append(f"- {word}")
    for word in positive_words[1:]:
        parts.append(f"+ {word}")
    return " ".join(parts)


def _top_k_neighbors(
    query_vector: array,
    words: list[str],
    vectors: list[array],
    exclude: set[str],
    limit: int,
) -> list[tuple[str, float]]:
    heap: list[tuple[float, str]] = []
    for word, vector in zip(words, vectors):
        if word in exclude:
            continue
        similarity = _dot(query_vector, vector)
        if len(heap) < limit:
            heapq.heappush(heap, (similarity, word))
        elif similarity > heap[0][0]:
            heapq.heapreplace(heap, (similarity, word))

    return [(word, score) for score, word in sorted(heap, reverse=True)]


def _prefix_suggestions(query_word: str, words: list[str], limit: int = 5) -> list[str]:
    suggestions: list[str] = []
    normalized_query = query_word.lower()
    prefixes = [normalized_query]
    if len(normalized_query) >= 3:
        prefixes.append(normalized_query[:3])

    for prefix in prefixes:
        for word in words:
            if word.startswith(prefix) and word not in suggestions:
                suggestions.append(word)
                if len(suggestions) >= limit:
                    return suggestions

    return suggestions


def search_similar_words(vectors_path: Path, query_word: str, limit: int = 5) -> dict[str, Any]:
    if not vectors_path.exists():
        return {
            "status": "missing_vectors",
            "message": f"Vectors file not found: {vectors_path}",
        }

    normalized_query = query_word.strip().lower()
    if not normalized_query:
        return {
            "status": "empty_query",
            "message": "Enter a word to search.",
        }

    words, vectors, index, dimension = _load_normalized_vectors(vectors_path)
    if normalized_query not in index:
        suggestions = _prefix_suggestions(normalized_query, words)
        return {
            "status": "word_not_found",
            "message": f"`{normalized_query}` is not in the saved vocabulary.",
            "suggestions": suggestions,
        }

    neighbors = _top_k_neighbors(
        query_vector=vectors[index[normalized_query]],
        words=words,
        vectors=vectors,
        exclude={normalized_query},
        limit=limit,
    )

    rows = [
        {
            "rank": rank,
            "word": word,
            "cosine_similarity": round(score, 6),
        }
        for rank, (word, score) in enumerate(neighbors, start=1)
    ]

    return {
        "status": "ok",
        "query_word": normalized_query,
        "results": rows,
        "vocab_size": len(words),
        "dimension": dimension,
    }


def solve_vector_equation(
    vectors_path: Path,
    positive_words: Sequence[str],
    negative_words: Sequence[str],
    limit: int = 5,
) -> dict[str, Any]:
    if not vectors_path.exists():
        return {
            "status": "missing_vectors",
            "message": f"Vectors file not found: {vectors_path}",
        }

    normalized_positive = _normalize_query_words(positive_words)
    normalized_negative = _normalize_query_words(negative_words)
    equation = _format_vector_equation(normalized_positive, normalized_negative)

    if not normalized_positive or not normalized_negative:
        return {
            "status": "empty_query",
            "message": "Enter words for the full vector arithmetic expression.",
            "equation": equation,
        }

    words, vectors, index, dimension = _load_normalized_vectors(vectors_path)
    missing_words = [word for word in [*normalized_positive, *normalized_negative] if word not in index]
    if missing_words:
        return {
            "status": "word_not_found",
            "message": "These words are not in the saved vocabulary: " + ", ".join(f"`{word}`" for word in missing_words),
            "missing_words": missing_words,
            "suggestions": {word: _prefix_suggestions(word, words) for word in missing_words},
            "equation": equation,
        }

    query_vector = array("f", [0.0] * dimension)
    for word in normalized_positive:
        source_vector = vectors[index[word]]
        for i in range(dimension):
            query_vector[i] += source_vector[i]

    for word in normalized_negative:
        source_vector = vectors[index[word]]
        for i in range(dimension):
            query_vector[i] -= source_vector[i]

    normalized_query_vector = _normalize_vector(query_vector)
    if normalized_query_vector is None:
        return {
            "status": "degenerate_query",
            "message": "The resulting vector has zero magnitude. Try a different equation.",
            "equation": equation,
        }

    neighbors = _top_k_neighbors(
        query_vector=normalized_query_vector,
        words=words,
        vectors=vectors,
        exclude=set(normalized_positive) | set(normalized_negative),
        limit=limit,
    )

    rows = [
        {
            "rank": rank,
            "word": word,
            "cosine_similarity": round(score, 6),
        }
        for rank, (word, score) in enumerate(neighbors, start=1)
    ]

    return {
        "status": "ok",
        "equation": equation,
        "positive_words": normalized_positive,
        "negative_words": normalized_negative,
        "results": rows,
        "vocab_size": len(words),
        "dimension": dimension,
    }


def find_similar_words_for_task(task_id: str, query_word: str, limit: int = 5) -> dict[str, Any]:
    if task_id not in {"task2", "task3"}:
        raise ValueError(f"Interactive similarity search is not supported for {task_id}.")

    return search_similar_words(TASK_OUTPUTS[task_id]["vectors"], query_word=query_word, limit=limit)


def solve_vector_equation_for_task(
    task_id: str,
    positive_words: Sequence[str],
    negative_words: Sequence[str],
    limit: int = 5,
) -> dict[str, Any]:
    if task_id not in {"task2", "task3"}:
        raise ValueError(f"Interactive vector arithmetic is not supported for {task_id}.")

    return solve_vector_equation(
        TASK_OUTPUTS[task_id]["vectors"],
        positive_words=positive_words,
        negative_words=negative_words,
        limit=limit,
    )


def similarity_example_words(task_id: str, limit: int = 5) -> list[str]:
    if task_id not in {"task2", "task3"}:
        return []

    synonyms_path = TASK_OUTPUTS[task_id]["synonyms"]
    if not synonyms_path.exists():
        return []

    examples: list[str] = []
    with open(synonyms_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            query_word = (row.get("query_word") or "").strip()
            if not query_word or query_word in examples:
                continue
            examples.append(query_word)
            if len(examples) >= limit:
                break

    return examples


def vector_arithmetic_examples(task_id: str, limit: int = 5) -> list[str]:
    if task_id not in {"task2", "task3"}:
        return []

    equations_path = TASK_OUTPUTS[task_id]["equations"]
    if not equations_path.exists():
        return []

    examples: list[str] = []
    with open(equations_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            equation = (row.get("equation") or "").strip()
            if not equation or equation in examples:
                continue
            examples.append(equation)
            if len(examples) >= limit:
                break

    return examples


def _file_cache_key(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns)


def _task5_metadata() -> dict[str, Any]:
    return _load_task5_metadata_cached(
        *_file_cache_key(TASK_OUTPUTS["task5"]["cache_manifest"]),
        *_file_cache_key(TASK_OUTPUTS["task5"]["results"]),
    )


@lru_cache(maxsize=2)
def _load_task5_metadata_cached(
    manifest_path_str: str,
    manifest_size: int,
    manifest_mtime_ns: int,
    results_path_str: str,
    results_size: int,
    results_mtime_ns: int,
) -> dict[str, Any]:
    del manifest_size, manifest_mtime_ns, results_size, results_mtime_ns

    import pandas as pd
    from Task5 import task5 as task5_module

    manifest_path = Path(manifest_path_str)
    results_path = Path(results_path_str)
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise FileNotFoundError(f"Could not load Task5 manifest from {manifest_path}.")

    args_payload = manifest.get("args")
    if not isinstance(args_payload, dict):
        raise ValueError("Task5 manifest is missing the training arguments.")

    dataset_info = manifest.get("dataset_file")
    word2vec_info = manifest.get("word2vec_file")
    glove_info = manifest.get("glove_file")
    if not isinstance(dataset_info, dict) or not isinstance(word2vec_info, dict) or not isinstance(glove_info, dict):
        raise ValueError("Task5 manifest is missing one or more source file entries.")

    dataset_file = Path(str(dataset_info["path"]))
    word2vec_file = Path(str(word2vec_info["path"]))
    glove_file = Path(str(glove_info["path"]))
    args_namespace = argparse.Namespace(**args_payload)
    current_signature = task5_module.build_cache_signature(
        args=args_namespace,
        dataset_file=dataset_file,
        word2vec_file=word2vec_file,
        glove_file=glove_file,
    )

    label_to_index_raw = manifest.get("label_to_index")
    if not isinstance(label_to_index_raw, dict):
        raise ValueError("Task5 manifest is missing the label mapping.")

    label_to_index = {int(label): int(index) for label, index in label_to_index_raw.items()}
    index_to_label = {index: label for label, index in label_to_index.items()}

    results_df = pd.read_csv(results_path)
    supported_models: dict[str, dict[str, tuple[str, bool]]] = {}
    for feature_name in results_df["feature"].dropna().astype(str).unique().tolist():
        supported_models[feature_name] = {
            model_name: (architecture, bidirectional)
            for model_name, architecture, bidirectional in task5_module.model_settings_for_feature(feature_name)
        }

    sorted_rows = (
        results_df.sort_values(["f1", "accuracy"], ascending=[False, False]).to_dict(orient="records")
        if not results_df.empty
        else []
    )

    options: list[dict[str, Any]] = []
    cache_dir = TASK_OUTPUTS["task5"]["cache_dir"]
    for row in sorted_rows:
        feature_name = str(row.get("feature", "")).strip()
        model_name = str(row.get("model", "")).strip()
        model_settings = supported_models.get(feature_name, {})
        if model_name not in model_settings:
            continue

        checkpoint_path = cache_dir / task5_module.checkpoint_file_name(feature_name, model_name)
        if not checkpoint_path.exists():
            continue

        architecture, bidirectional = model_settings[model_name]
        option_id = f"{feature_name}::{model_name}"
        options.append(
            {
                "id": option_id,
                "label": f"{feature_name} + {model_name} (F1 {float(row.get('f1', 0.0)):.4f})",
                "feature": feature_name,
                "model": model_name,
                "architecture": architecture,
                "bidirectional": bidirectional,
                "checkpoint_path": checkpoint_path,
                "f1": float(row.get("f1", 0.0)),
                "accuracy": float(row.get("accuracy", 0.0)),
            }
        )

    return {
        "manifest": manifest,
        "args": {key: int(value) if key not in {"learning_rate"} else float(value) for key, value in args_payload.items()},
        "dataset_file": dataset_file,
        "word2vec_file": word2vec_file,
        "glove_file": glove_file,
        "saved_signature": str(manifest.get("cache_signature", "")),
        "current_signature": current_signature,
        "signature_match": str(manifest.get("cache_signature", "")) == current_signature,
        "options": options,
        "label_to_index": label_to_index,
        "index_to_label": index_to_label,
    }


def task5_model_options() -> list[dict[str, Any]]:
    try:
        return _task5_metadata()["options"]
    except FileNotFoundError:
        return []


def task5_inference_status() -> dict[str, Any]:
    try:
        metadata = _task5_metadata()
    except FileNotFoundError:
        return {
            "status": "missing_artifacts",
            "message": "Run Task5 once to generate model checkpoints before using sentiment prediction.",
        }

    if not metadata["options"]:
        return {
            "status": "missing_artifacts",
            "message": "No Task5 model checkpoints were found for the UI prediction tool.",
        }

    if not metadata["signature_match"]:
        return {
            "status": "incompatible_cache",
            "message": (
                "Saved Task5 checkpoints were generated by an older training pipeline. "
                "Run Task5 again to refresh the model cache before using sentence prediction."
            ),
        }

    return {
        "status": "ready",
        "message": "Task5 sentiment prediction is ready.",
    }


@lru_cache(maxsize=2)
def _load_task5_training_docs_cached(
    dataset_path_str: str,
    dataset_size: int,
    dataset_mtime_ns: int,
    sample_size: int,
    seed: int,
) -> tuple[str, ...]:
    del dataset_size, dataset_mtime_ns

    from Task5 import task5 as task5_module

    dataset = task5_module.load_sentiment_dataset(
        dataset_file=Path(dataset_path_str),
        sample_size=sample_size,
        random_seed=seed,
    )
    train_docs, _val_docs, _test_docs, _y_train, _y_val, _y_test = task5_module.split_data(
        docs=dataset.docs,
        labels=dataset.labels,
        random_seed=seed,
    )
    return tuple(train_docs)


@lru_cache(maxsize=1)
def _load_task5_dense_artifacts_cached(
    dataset_path_str: str,
    dataset_size: int,
    dataset_mtime_ns: int,
    sample_size: int,
    seed: int,
    bow_features: int,
) -> tuple[Any, Any, Any]:
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from Task5 import task5 as task5_module

    train_docs = _load_task5_training_docs_cached(
        dataset_path_str,
        dataset_size,
        dataset_mtime_ns,
        sample_size,
        seed,
    )

    count_vectorizer = CountVectorizer(
        max_features=bow_features,
        min_df=3,
        max_df=0.95,
        lowercase=False,
        preprocessor=None,
        token_pattern=None,
        tokenizer=task5_module.tokenize,
    )
    x_train_count = count_vectorizer.fit_transform(train_docs).astype(np.float32)

    tfidf_vectorizer = TfidfVectorizer(
        vocabulary=count_vectorizer.vocabulary_,
        min_df=3,
        max_df=0.95,
        lowercase=False,
        preprocessor=None,
        token_pattern=None,
        tokenizer=task5_module.tokenize,
    )
    tfidf_vectorizer.fit_transform(train_docs)
    term_strength = task5_module.compute_term_strength_from_pmi(x_train_count.tocsr())
    return count_vectorizer, tfidf_vectorizer, term_strength


@lru_cache(maxsize=2)
def _load_task5_embedding_artifacts_cached(
    dataset_path_str: str,
    dataset_size: int,
    dataset_mtime_ns: int,
    sample_size: int,
    seed: int,
    vectors_path_str: str,
    vectors_size: int,
    vectors_mtime_ns: int,
) -> tuple[dict[str, int], Any]:
    del vectors_size, vectors_mtime_ns

    from Task5 import task5 as task5_module

    train_docs = _load_task5_training_docs_cached(
        dataset_path_str,
        dataset_size,
        dataset_mtime_ns,
        sample_size,
        seed,
    )
    task5_module.set_seed(seed)
    token_to_id, embedding_matrix = task5_module.load_pretrained_embeddings(Path(vectors_path_str))
    token_to_id, embedding_matrix = task5_module.extend_embedding_vocabulary(
        train_docs=list(train_docs),
        token_to_id=token_to_id,
        embedding_matrix=embedding_matrix,
    )
    return token_to_id, embedding_matrix


def _build_task5_inference_feature_set(feature_name: str, text: str, metadata: dict[str, Any]) -> Any:
    import numpy as np
    from Task5 import task5 as task5_module

    normalized_text = task5_module.normalize_text(text)
    args = metadata["args"]
    dataset_key = _file_cache_key(metadata["dataset_file"])

    if feature_name in {"Count Vectorizer", "TF-IDF", "PMI"}:
        count_vectorizer, tfidf_vectorizer, term_strength = _load_task5_dense_artifacts_cached(
            *dataset_key,
            int(args["sample_size"]),
            int(args["seed"]),
            int(args["bow_features"]),
        )
        if feature_name == "Count Vectorizer":
            x = count_vectorizer.transform([normalized_text]).toarray().astype(np.float32)
        elif feature_name == "TF-IDF":
            x = tfidf_vectorizer.transform([normalized_text]).toarray().astype(np.float32)
        else:
            x_count = count_vectorizer.transform([normalized_text]).astype(np.float32)
            x = x_count.multiply(term_strength).toarray().astype(np.float32)

        lengths = np.full(1, x.shape[1], dtype=np.int64)
        return task5_module.FeatureSet(
            train_x=x,
            val_x=x,
            test_x=x,
            train_len=lengths,
            val_len=lengths,
            test_len=lengths,
            embedding_matrix=None,
            is_token_feature=False,
        )

    if feature_name == "Word2Vec":
        vectors_file = metadata["word2vec_file"]
    elif feature_name == "GloVe":
        vectors_file = metadata["glove_file"]
    else:
        raise ValueError(f"Unsupported Task5 feature set: {feature_name}")

    token_to_id, embedding_matrix = _load_task5_embedding_artifacts_cached(
        *dataset_key,
        int(args["sample_size"]),
        int(args["seed"]),
        *_file_cache_key(vectors_file),
    )
    sequences, lengths = task5_module.docs_to_token_sequences(
        [normalized_text],
        token_to_id=token_to_id,
        max_len=int(args["max_len"]),
    )
    return task5_module.FeatureSet(
        train_x=sequences,
        val_x=sequences,
        test_x=sequences,
        train_len=lengths,
        val_len=lengths,
        test_len=lengths,
        embedding_matrix=embedding_matrix,
        is_token_feature=True,
    )


def predict_task5_sentiment(model_id: str, text: str) -> dict[str, Any]:
    cleaned_text = text.strip()
    if not cleaned_text:
        return {
            "status": "empty_query",
            "message": "Enter a sentence to predict its sentiment.",
        }

    metadata = _task5_metadata()
    if not metadata["signature_match"]:
        return {
            "status": "incompatible_cache",
            "message": (
                "Saved Task5 checkpoints were generated by an older training pipeline. "
                "Run Task5 again to refresh the model cache before using sentence prediction."
            ),
            "saved_signature": metadata["saved_signature"],
            "current_signature": metadata["current_signature"],
        }

    option_lookup = {option["id"]: option for option in metadata["options"]}
    if model_id not in option_lookup:
        raise ValueError(f"Unknown Task5 model option: {model_id}")

    import torch
    from Task5 import task5 as task5_module

    option = option_lookup[model_id]
    checkpoint_path = Path(option["checkpoint_path"])
    if not checkpoint_path.exists():
        return {
            "status": "missing_checkpoint",
            "message": f"Checkpoint file not found: {checkpoint_path}",
        }

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        return {
            "status": "invalid_checkpoint",
            "message": f"Checkpoint is invalid or missing model weights: {checkpoint_path}",
        }

    if checkpoint.get("cache_signature") != metadata["saved_signature"]:
        return {
            "status": "incompatible_cache",
            "message": (
                "The selected Task5 checkpoint does not match the current saved manifest. "
                "Run Task5 again to refresh the model cache."
            ),
        }

    feature_set = _build_task5_inference_feature_set(option["feature"], cleaned_text, metadata)
    model = task5_module.build_classifier(
        architecture=option["architecture"],
        bidirectional=bool(option["bidirectional"]),
        feature_set=feature_set,
        num_classes=int(checkpoint.get("num_classes", len(metadata["index_to_label"]))),
        hidden_size=int(checkpoint.get("hidden_size", metadata["args"]["hidden_size"])),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    if feature_set.is_token_feature:
        x_tensor = torch.from_numpy(feature_set.train_x.astype("int64"))
    else:
        x_tensor = torch.from_numpy(feature_set.train_x.astype("float32"))
    lengths_tensor = torch.from_numpy(feature_set.train_len.astype("int64"))

    with torch.no_grad():
        logits = model(x_tensor, lengths_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()

    predicted_index = int(torch.argmax(logits, dim=1).item())
    predicted_label = int(metadata["index_to_label"].get(predicted_index, predicted_index))
    scores = [
        {
            "sentiment_label": task5_module.sentiment_label_name(int(metadata["index_to_label"].get(index, index))),
            "original_label": int(metadata["index_to_label"].get(index, index)),
            "probability": round(float(probability), 6),
        }
        for index, probability in enumerate(probabilities)
    ]
    scores.sort(key=lambda row: row["probability"], reverse=True)

    return {
        "status": "ok",
        "feature": option["feature"],
        "model": option["model"],
        "predicted_label": predicted_label,
        "sentiment_label": task5_module.sentiment_label_name(predicted_label),
        "confidence": round(float(probabilities[predicted_index]), 6),
        "scores": scores,
    }


def list_task5_checkpoints() -> list[Path]:
    cache_dir = TASK_OUTPUTS["task5"]["cache_dir"]
    if not cache_dir.exists() or not cache_dir.is_dir():
        return []
    return sorted(cache_dir.glob("*.pt"))


def ensure_task_outputs(
    task_id: str,
    force_rebuild: bool = False,
    run_args: Sequence[str] | None = None,
) -> dict[str, str]:
    if task_id not in TASK_SPECS:
        raise ValueError(f"Unknown task id: {task_id}")

    spec = TASK_SPECS[task_id]
    args = list(run_args or [])
    signature = build_task_signature(task_id=task_id, run_args=args)
    manifest_path = task_manifest_path(task_id)
    existing_manifest = load_json(manifest_path)

    if not force_rebuild and all_files_exist(spec.required_files):
        payload = {
            "task_id": task_id,
            "signature": signature,
            "command": "(reuse-existing-outputs)",
            "run_args": args,
            "updated_at_epoch": int(time.time()),
            "required_files": [str(path) for path in spec.required_files],
        }

        if existing_manifest is None or existing_manifest.get("signature") != signature:
            save_json(manifest_path, payload)
            return {
                "status": "cache_hit",
                "message": "Existing outputs found. Cache manifest refreshed.",
                "command": payload["command"],
                "output_tail": "",
            }

        return {
            "status": "cache_hit",
            "message": "Saved outputs and model files reused.",
            "command": "(reused from disk)",
            "output_tail": "",
        }

    if spec.script_path is None:
        missing = [str(path) for path in spec.required_files if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Task does not have an executable script and required files are missing: " + ", ".join(missing)
            )

        save_json(
            manifest_path,
            {
                "task_id": task_id,
                "signature": signature,
                "command": "(static-task)",
                "run_args": args,
                "updated_at_epoch": int(time.time()),
                "required_files": [str(path) for path in spec.required_files],
            },
        )
        return {
            "status": "cache_hit",
            "message": "Static task outputs loaded.",
            "command": "(static-task)",
            "output_tail": "",
        }

    command = [sys.executable, str(spec.script_path), *args]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="ignore",
    )

    merged_output = "\n".join(part for part in [completed.stdout, completed.stderr] if part)
    output_tail = summarize_output(merged_output)
    if completed.returncode != 0:
        raise RuntimeError(
            "Task execution failed.\n"
            f"Command: {' '.join(command)}\n"
            f"Exit code: {completed.returncode}\n"
            f"Output:\n{output_tail}"
        )

    missing_after_run = [str(path) for path in spec.required_files if not path.exists()]
    if missing_after_run:
        raise RuntimeError("Task completed but expected output files are missing: " + ", ".join(missing_after_run))

    save_json(
        manifest_path,
        {
            "task_id": task_id,
            "signature": signature,
            "command": " ".join(command),
            "run_args": args,
            "updated_at_epoch": int(time.time()),
            "required_files": [str(path) for path in spec.required_files],
        },
    )

    status = "ran" if not force_rebuild else "retrained"
    message = "Task executed and outputs were refreshed."
    return {
        "status": status,
        "message": message,
        "command": " ".join(command),
        "output_tail": output_tail,
    }
