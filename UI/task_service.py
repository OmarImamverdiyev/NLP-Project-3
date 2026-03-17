from __future__ import annotations

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


def find_similar_words_for_task(task_id: str, query_word: str, limit: int = 5) -> dict[str, Any]:
    if task_id not in {"task2", "task3"}:
        raise ValueError(f"Interactive similarity search is not supported for {task_id}.")

    return search_similar_words(TASK_OUTPUTS[task_id]["vectors"], query_word=query_word, limit=limit)


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
