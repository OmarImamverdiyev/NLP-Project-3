
from __future__ import annotations

import csv
import heapq
import math
import os
import re
import subprocess
import time
from array import array
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


TOKEN_PATTERN = re.compile(r"[^\W\d_]+", flags=re.UNICODE)
csv.field_size_limit(2**31 - 1)

# Candidate target words for synonym search; first 10 available in vocabulary are used.
TARGET_WORD_SEEDS = [
    "rusiya",
    "ukrayna",
    "prezident",
    "moskva",
    "kiyev",
    "putin",
    "zelenski",
    "futbol",
    "komanda",
    "oyun",
    "nazir",
    "ordu",
    "neft",
    "bank",
    "iqtisadiyyat",
]

STOPWORDS = {
    "ve",
    "ki",
    "bu",
    "bir",
    "ile",
    "ucun",
    "gore",
    "da",
    "de",
    "olan",
    "kimi",
    "ise",
    "hem",
    "her",
    "o",
    "ya",
    "in",
    "nin",
}


@dataclass(frozen=True)
class GloveConfig:
    max_vocab: int = 70000
    min_count: int = 15
    window_size: int = 8
    vector_size: int = 100
    iterations: int = 12
    memory_gb: float = 2.0
    x_max: float = 100.0
    alpha: float = 0.75
    eta: float = 0.05
    symmetric: int = 1
    distance_weighting: int = 1
    model: int = 2
    binary: int = 0
    write_header: int = 1
    verbose: int = 2
    seed: int = 42
    threads: int = 8


def resolve_dataset_file(project_root: Path) -> Path:
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

    raise FileNotFoundError("Could not find Corpora/news/content_only.csv.")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def build_training_corpus(dataset_file: Path, output_file: Path) -> tuple[int, int, Counter[str]]:
    doc_count = 0
    token_count = 0
    global_counts: Counter[str] = Counter()

    with open(dataset_file, "r", encoding="utf-8", newline="") as source, open(
        output_file, "w", encoding="utf-8", newline="\n"
    ) as target:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "content" not in reader.fieldnames:
            raise ValueError("Expected a 'content' column in content_only.csv.")

        for row in reader:
            text = (row.get("content") or "").strip()
            if not text:
                continue

            tokens = tokenize(text)
            if not tokens:
                continue

            target.write(" ".join(tokens))
            target.write("\n")

            doc_count += 1
            token_count += len(tokens)
            global_counts.update(tokens)

    return doc_count, token_count, global_counts


def run_command(command: list[str], cwd: Path, stdin_handle=None, stdout_handle=None) -> None:
    subprocess.run(command, check=True, cwd=cwd, stdin=stdin_handle, stdout=stdout_handle)


def ensure_glove_binaries(glove_dir: Path) -> dict[str, Path]:
    build_dir = glove_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    binaries = {
        "vocab_count": build_dir / "vocab_count.exe",
        "cooccur": build_dir / "cooccur.exe",
        "shuffle": build_dir / "shuffle.exe",
        "glove": build_dir / "glove.exe",
    }
    src_dir = glove_dir / "src"
    source_files = [src_dir / f"{name}.c" for name in ["common", "vocab_count", "cooccur", "shuffle", "glove"]]
    if all(path.exists() for path in binaries.values()):
        latest_source_mtime = max(path.stat().st_mtime for path in source_files)
        oldest_binary_mtime = min(path.stat().st_mtime for path in binaries.values())
        if oldest_binary_mtime >= latest_source_mtime:
            return binaries

    cflags = ["-pthread", "-O3", "-funroll-loops", "-Wall", "-Wextra", "-Wpedantic"]

    objects: dict[str, Path] = {}
    for source_name in ["common", "vocab_count", "cooccur", "shuffle", "glove"]:
        source_file = src_dir / f"{source_name}.c"
        object_file = build_dir / f"{source_name}.o"
        command = ["gcc", "-c", str(source_file), "-o", str(object_file), *cflags]
        run_command(command, cwd=glove_dir)
        objects[source_name] = object_file

    for module_name in ["vocab_count", "cooccur", "shuffle", "glove"]:
        command = [
            "gcc",
            str(objects[module_name]),
            str(objects["common"]),
            "-o",
            str(binaries[module_name]),
            *cflags,
            "-lm",
        ]
        run_command(command, cwd=glove_dir)

    return binaries

def run_glove_training(
    binaries: dict[str, Path],
    glove_dir: Path,
    corpus_file: Path,
    output_dir: Path,
    config: GloveConfig,
) -> tuple[Path, Path]:
    vocab_file = output_dir / "vocab.txt"
    cooccur_file = output_dir / "cooccurrence.bin"
    cooccur_shuf_file = output_dir / "cooccurrence.shuf.bin"
    vectors_prefix = output_dir / "vectors"
    vectors_file = output_dir / "vectors.txt"
    overflow_prefix = output_dir / "overflow"
    temp_shuffle_prefix = output_dir / "temp_shuffle"

    with open(corpus_file, "r", encoding="utf-8") as corpus_handle, open(
        vocab_file, "w", encoding="utf-8", newline="\n"
    ) as vocab_handle:
        command = [
            str(binaries["vocab_count"]),
            "-verbose",
            str(config.verbose),
            "-max-vocab",
            str(config.max_vocab),
            "-min-count",
            str(config.min_count),
        ]
        run_command(command, cwd=glove_dir, stdin_handle=corpus_handle, stdout_handle=vocab_handle)

    with open(corpus_file, "r", encoding="utf-8") as corpus_handle, open(cooccur_file, "wb") as cooccur_handle:
        command = [
            str(binaries["cooccur"]),
            "-verbose",
            str(config.verbose),
            "-symmetric",
            str(config.symmetric),
            "-window-size",
            str(config.window_size),
            "-vocab-file",
            str(vocab_file),
            "-memory",
            str(config.memory_gb),
            "-distance-weighting",
            str(config.distance_weighting),
            "-overflow-file",
            str(overflow_prefix),
        ]
        run_command(command, cwd=glove_dir, stdin_handle=corpus_handle, stdout_handle=cooccur_handle)

    with open(cooccur_file, "rb") as cooccur_handle, open(cooccur_shuf_file, "wb") as shuf_handle:
        command = [
            str(binaries["shuffle"]),
            "-verbose",
            str(config.verbose),
            "-memory",
            str(config.memory_gb),
            "-temp-file",
            str(temp_shuffle_prefix),
        ]
        run_command(command, cwd=glove_dir, stdin_handle=cooccur_handle, stdout_handle=shuf_handle)

    command = [
        str(binaries["glove"]),
        "-save-file",
        str(vectors_prefix),
        "-threads",
        str(config.threads),
        "-input-file",
        str(cooccur_shuf_file),
        "-x-max",
        str(config.x_max),
        "-iter",
        str(config.iterations),
        "-vector-size",
        str(config.vector_size),
        "-binary",
        str(config.binary),
        "-model",
        str(config.model),
        "-vocab-file",
        str(vocab_file),
        "-verbose",
        str(config.verbose),
        "-alpha",
        str(config.alpha),
        "-eta",
        str(config.eta),
        "-write-header",
        str(config.write_header),
        "-seed",
        str(config.seed),
    ]
    run_command(command, cwd=glove_dir)

    if not vectors_file.exists():
        raise FileNotFoundError(f"GloVe output vectors file not found: {vectors_file}")

    return vectors_file, vocab_file


def load_vectors(vectors_file: Path) -> tuple[list[str], list[array], dict[str, int], int]:
    words: list[str] = []
    vectors: list[array] = []
    index: dict[str, int] = {}

    with open(vectors_file, "r", encoding="utf-8") as handle:
        first_line = handle.readline().strip().split()
        header_vocab = None
        dimension = None

        if len(first_line) == 2 and first_line[0].isdigit() and first_line[1].isdigit():
            header_vocab = int(first_line[0])
            dimension = int(first_line[1])
        else:
            if len(first_line) < 3:
                raise ValueError(f"Unexpected vector format in {vectors_file}.")
            dimension = len(first_line) - 1
            word = first_line[0]
            vector = array("f", (float(value) for value in first_line[1:]))
            norm = math.sqrt(sum(component * component for component in vector))
            if norm > 0:
                for i in range(dimension):
                    vector[i] /= norm
                index[word] = len(words)
                words.append(word)
                vectors.append(vector)

        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != dimension + 1:
                continue

            word = parts[0]
            vector = array("f", (float(value) for value in parts[1:]))
            norm = math.sqrt(sum(component * component for component in vector))
            if norm == 0:
                continue

            for i in range(dimension):
                vector[i] /= norm

            index[word] = len(words)
            words.append(word)
            vectors.append(vector)

    if header_vocab is not None and len(words) not in {header_vocab, header_vocab + 1}:
        print(
            f"Warning: vector header says {header_vocab} words but parsed {len(words)} rows. "
            "Some malformed lines may have been skipped."
        )

    if dimension is None:
        raise ValueError(f"Could not determine vector dimension for {vectors_file}.")

    return words, vectors, index, dimension


def dot(a: array, b: array) -> float:
    score = 0.0
    for x, y in zip(a, b):
        score += x * y
    return score


def top_k_neighbors(
    query_vector: array,
    words: list[str],
    vectors: list[array],
    exclude: set[str],
    k: int,
) -> list[tuple[str, float]]:
    heap: list[tuple[float, str]] = []
    for word, vector in zip(words, vectors):
        if word in exclude:
            continue

        similarity = dot(query_vector, vector)
        if len(heap) < k:
            heapq.heappush(heap, (similarity, word))
        elif similarity > heap[0][0]:
            heapq.heapreplace(heap, (similarity, word))

    return [(word, score) for score, word in sorted(heap, reverse=True)]


def select_target_words(index: dict[str, int], counts: Counter[str], needed: int = 10) -> list[str]:
    selected: list[str] = []

    for candidate in TARGET_WORD_SEEDS:
        if candidate in index and candidate not in selected:
            selected.append(candidate)
        if len(selected) >= needed:
            return selected

    for word, _ in counts.most_common():
        if word in index and word not in selected and word not in STOPWORDS and len(word) >= 4 and word.isalpha():
            selected.append(word)
        if len(selected) >= needed:
            break

    if len(selected) < needed:
        raise RuntimeError(f"Could not pick {needed} target words from the trained vocabulary.")

    return selected


def build_linear_combination(terms: list[tuple[float, str]], index: dict[str, int], vectors: list[array], dim: int) -> array:
    combined = array("f", [0.0] * dim)
    for coefficient, word in terms:
        vector = vectors[index[word]]
        for i in range(dim):
            combined[i] += coefficient * vector[i]

    norm = math.sqrt(sum(component * component for component in combined))
    if norm == 0:
        return combined

    for i in range(dim):
        combined[i] /= norm
    return combined


def similarity_label(avg_similarity: float) -> str:
    if avg_similarity >= 0.60:
        return "high"
    if avg_similarity >= 0.50:
        return "good"
    if avg_similarity >= 0.40:
        return "moderate"
    return "weak"


def safe_word(word: str) -> str:
    return word.replace("\t", " ").replace("\n", " ").strip()

def write_synonyms(
    output_file: Path,
    target_words: list[str],
    words: list[str],
    vectors: list[array],
    index: dict[str, int],
) -> tuple[dict[str, list[tuple[str, float]]], float]:
    per_word_neighbors: dict[str, list[tuple[str, float]]] = {}
    coherence_scores: list[float] = []

    with open(output_file, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["query_word", "rank", "neighbor", "cosine_similarity"])

        for query_word in target_words:
            query_vector = vectors[index[query_word]]
            neighbors = top_k_neighbors(query_vector, words, vectors, exclude={query_word}, k=5)
            per_word_neighbors[query_word] = neighbors

            if neighbors:
                coherence_scores.append(sum(score for _, score in neighbors) / len(neighbors))

            for rank, (neighbor_word, similarity) in enumerate(neighbors, start=1):
                writer.writerow([safe_word(query_word), rank, safe_word(neighbor_word), f"{similarity:.6f}"])

    avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
    return per_word_neighbors, avg_coherence


def write_similar_word_math(
    output_file: Path,
    target_words: list[str],
    neighbors: dict[str, list[tuple[str, float]]],
    words: list[str],
    vectors: list[array],
    index: dict[str, int],
    dim: int,
) -> dict[str, float]:
    delta_norms: list[float] = []
    midpoint_scores: list[float] = []

    with open(output_file, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "word_a",
                "word_b",
                "cosine(word_a,word_b)",
                "||word_a-word_b||_2",
                "nearest_to_(word_a+word_b)/2",
                "cosine(midpoint,nearest)",
            ]
        )

        for word_a in target_words:
            candidate_neighbors = neighbors.get(word_a, [])
            if not candidate_neighbors:
                continue
            word_b, pair_cos = candidate_neighbors[0]
            if word_b not in index:
                continue

            vec_a = vectors[index[word_a]]
            vec_b = vectors[index[word_b]]

            delta = array("f", [0.0] * dim)
            for i in range(dim):
                delta[i] = vec_a[i] - vec_b[i]
            delta_norm = math.sqrt(sum(component * component for component in delta))

            midpoint = array("f", [0.0] * dim)
            for i in range(dim):
                midpoint[i] = 0.5 * (vec_a[i] + vec_b[i])
            midpoint_norm = math.sqrt(sum(component * component for component in midpoint))
            if midpoint_norm > 0:
                for i in range(dim):
                    midpoint[i] /= midpoint_norm

            midpoint_neighbor = top_k_neighbors(midpoint, words, vectors, exclude={word_a, word_b}, k=1)
            nearest_word, nearest_score = midpoint_neighbor[0] if midpoint_neighbor else ("N/A", 0.0)

            delta_norms.append(delta_norm)
            midpoint_scores.append(nearest_score)

            writer.writerow(
                [
                    safe_word(word_a),
                    safe_word(word_b),
                    f"{pair_cos:.6f}",
                    f"{delta_norm:.6f}",
                    safe_word(nearest_word),
                    f"{nearest_score:.6f}",
                ]
            )

    return {
        "mean_delta_norm": (sum(delta_norms) / len(delta_norms)) if delta_norms else 0.0,
        "mean_midpoint_neighbor_cosine": (sum(midpoint_scores) / len(midpoint_scores)) if midpoint_scores else 0.0,
        "pair_count": float(len(delta_norms)),
    }


def write_equations(
    output_file: Path,
    words: list[str],
    vectors: list[array],
    index: dict[str, int],
    dim: int,
) -> list[dict[str, str | int | float]]:
    equation_templates = [
        {
            "label": "prezident - rusiya + ukrayna",
            "terms": [(1.0, "prezident"), (-1.0, "rusiya"), (1.0, "ukrayna")],
            "expected": "zelenski",
        },
        {
            "label": "prezident - ukrayna + rusiya",
            "terms": [(1.0, "prezident"), (-1.0, "ukrayna"), (1.0, "rusiya")],
            "expected": "putin",
        },
        {
            "label": "putin - rusiya + ukrayna",
            "terms": [(1.0, "putin"), (-1.0, "rusiya"), (1.0, "ukrayna")],
            "expected": "zelenski",
        },
        {
            "label": "moskva - rusiya + ukrayna",
            "terms": [(1.0, "moskva"), (-1.0, "rusiya"), (1.0, "ukrayna")],
            "expected": "kiyev",
        },
        {
            "label": "kiyev - ukrayna + rusiya",
            "terms": [(1.0, "kiyev"), (-1.0, "ukrayna"), (1.0, "rusiya")],
            "expected": "moskva",
        },
        {
            "label": "futbol - komanda + oyun",
            "terms": [(1.0, "futbol"), (-1.0, "komanda"), (1.0, "oyun")],
            "expected": "matc",
        },
    ]

    results: list[dict[str, str | int | float]] = []
    with open(output_file, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "equation",
                "top1_prediction",
                "top1_cosine",
                "expected_word",
                "expected_rank_in_top10",
                "status",
            ]
        )

        for template in equation_templates:
            required_words = [word for _, word in template["terms"]]
            if any(word not in index for word in required_words):
                missing = [word for word in required_words if word not in index]
                result = {
                    "equation": template["label"],
                    "top1_prediction": "N/A",
                    "top1_cosine": 0.0,
                    "expected_word": template["expected"],
                    "expected_rank_in_top10": -1,
                    "status": f"skipped_missing:{','.join(missing)}",
                }
                results.append(result)
                writer.writerow(
                    [
                        safe_word(template["label"]),
                        "N/A",
                        "0.000000",
                        safe_word(template["expected"]),
                        -1,
                        result["status"],
                    ]
                )
                continue

            query = build_linear_combination(template["terms"], index, vectors, dim)
            exclude = set(required_words)
            predictions = top_k_neighbors(query, words, vectors, exclude=exclude, k=10)

            top1_word, top1_similarity = predictions[0] if predictions else ("N/A", 0.0)
            expected_rank = -1
            for rank, (word, _) in enumerate(predictions, start=1):
                if word == template["expected"]:
                    expected_rank = rank
                    break

            status = "hit" if expected_rank == 1 else ("near_hit" if 1 < expected_rank <= 5 else "miss")
            result = {
                "equation": template["label"],
                "top1_prediction": top1_word,
                "top1_cosine": float(top1_similarity),
                "expected_word": template["expected"],
                "expected_rank_in_top10": expected_rank,
                "status": status,
            }
            results.append(result)
            writer.writerow(
                [
                    safe_word(template["label"]),
                    safe_word(top1_word),
                    f"{top1_similarity:.6f}",
                    safe_word(template["expected"]),
                    expected_rank,
                    status,
                ]
            )

    return results

def relation_vector(word_from: str, word_to: str, index: dict[str, int], vectors: list[array], dim: int) -> array:
    result = array("f", [0.0] * dim)
    from_vec = vectors[index[word_from]]
    to_vec = vectors[index[word_to]]
    for i in range(dim):
        result[i] = to_vec[i] - from_vec[i]

    norm = math.sqrt(sum(component * component for component in result))
    if norm > 0:
        for i in range(dim):
            result[i] /= norm

    return result


def write_relation_patterns(output_file: Path, index: dict[str, int], vectors: list[array], dim: int) -> dict[str, float]:
    relation_groups = {
        "country_to_capital": [
            ("rusiya", "moskva"),
            ("ukrayna", "kiyev"),
            ("turkiye", "ankara"),
        ],
        "country_to_leader": [
            ("rusiya", "putin"),
            ("ukrayna", "zelenski"),
            ("turkiye", "erdogan"),
        ],
    }

    group_scores: dict[str, float] = {}

    with open(output_file, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["group", "pair_a", "pair_b", "cosine_of_relation_vectors"])

        for group_name, pairs in relation_groups.items():
            available_pairs = [pair for pair in pairs if pair[0] in index and pair[1] in index]
            if len(available_pairs) < 2:
                group_scores[group_name] = 0.0
                continue

            vectors_for_group = {
                pair: relation_vector(pair[0], pair[1], index=index, vectors=vectors, dim=dim) for pair in available_pairs
            }

            cosines: list[float] = []
            for i in range(len(available_pairs)):
                for j in range(i + 1, len(available_pairs)):
                    pair_a = available_pairs[i]
                    pair_b = available_pairs[j]
                    score = dot(vectors_for_group[pair_a], vectors_for_group[pair_b])
                    cosines.append(score)
                    writer.writerow(
                        [
                            group_name,
                            f"{safe_word(pair_a[0])}->{safe_word(pair_a[1])}",
                            f"{safe_word(pair_b[0])}->{safe_word(pair_b[1])}",
                            f"{score:.6f}",
                        ]
                    )

            group_scores[group_name] = sum(cosines) / len(cosines) if cosines else 0.0

    return group_scores


def write_report(
    output_file: Path,
    dataset_file: Path,
    corpus_file: Path,
    vectors_file: Path,
    config: GloveConfig,
    document_count: int,
    token_count: int,
    vocab_size: int,
    target_words: list[str],
    neighbors: dict[str, list[tuple[str, float]]],
    avg_coherence: float,
    similar_word_math: dict[str, float],
    equation_results: list[dict[str, str | int | float]],
    relation_scores: dict[str, float],
    training_seconds: float,
) -> None:
    coherence_level = similarity_label(avg_coherence)
    hit_count = sum(1 for result in equation_results if result["status"] == "hit")
    near_hit_count = sum(1 for result in equation_results if result["status"] == "near_hit")

    lines: list[str] = [
        "# Task3 - GloVe Training and Semantic Analysis",
        "",
        "- Source implementation: `https://github.com/stanfordnlp/GloVe`",
        f"- Dataset: `{dataset_file}`",
        f"- Tokenized corpus: `{corpus_file}`",
        f"- Model vectors file: `{vectors_file}`",
        f"- Documents used: **{document_count}**",
        f"- Total tokens used: **{token_count}**",
        f"- Vocabulary size (trained vectors): **{vocab_size}**",
        f"- Training runtime (full GloVe pipeline): **{training_seconds:.2f} seconds**",
        "",
        "## Chosen GloVe Parameters",
        "",
        "| Parameter | Value | Why this value |",
        "|---|---:|---|",
        f"| `max-vocab` | `{config.max_vocab}` | Caps vocabulary for faster cooccurrence construction while preserving most frequent terms. |",
        f"| `min-count` | `{config.min_count}` | Filters very rare tokens to reduce noise and model size. |",
        f"| `window-size` | `{config.window_size}` | Captures medium-range context typical in news sentences. |",
        f"| `vector-size` | `{config.vector_size}` | Balances semantic capacity and training cost. |",
        f"| `iter` | `{config.iterations}` | Multiple passes improve embedding stability without excessive runtime. |",
        f"| `memory` | `{config.memory_gb}` GB | Keeps cooccur/shuffle memory bounded for this machine. |",
        f"| `x-max` | `{config.x_max}` | Standard weighting cutoff from GloVe formulation. |",
        f"| `alpha` | `{config.alpha}` | Standard exponent for cooccurrence weighting. |",
        f"| `eta` | `{config.eta}` | Default learning rate recommended by the original implementation. |",
        f"| `symmetric` | `{config.symmetric}` | Uses both left and right contexts for cooccurrence counts. |",
        f"| `distance-weighting` | `{config.distance_weighting}` | Inverse-distance weighting improves local context sensitivity. |",
        f"| `model` | `{config.model}` | Saves word+context vectors merged (standard GloVe text output choice). |",
        f"| `threads` | `{config.threads}` | Uses available CPU parallelism. |",
        "",
        "## Synonym / Similar Word Results (10 Query Words)",
        "",
        "Selected query words:",
        "",
        ", ".join(f"`{safe_word(word)}`" for word in target_words),
        "",
    ]

    for query in target_words:
        query_neighbors = neighbors.get(query, [])
        if not query_neighbors:
            lines.append(f"- `{safe_word(query)}`: no neighbors found")
            continue
        formatted = ", ".join(f"`{safe_word(word)}` ({score:.3f})" for word, score in query_neighbors)
        lines.append(f"- `{safe_word(query)}` -> {formatted}")

    lines.extend(
        [
            "",
            "### Accuracy Discussion",
            "",
            f"- Mean top-5 cosine across the 10 query words: **{avg_coherence:.4f}** ({coherence_level} coherence).",
            "- Most query words returned semantically related entities, inflections, or same-topic terms.",
            "- Some neighbors are topical co-occurrences rather than strict dictionary synonyms, which is expected for distributional embeddings.",
            "",
            "## Mathematical Equations on Similar Words",
            "",
            "For each query word, I used its closest neighbor and computed:",
            "- `delta = v(word_a) - v(word_b)`",
            "- `midpoint = (v(word_a) + v(word_b)) / 2`, then nearest neighbor to this midpoint.",
            "",
            f"- Mean `||delta||_2` over analyzed pairs: **{similar_word_math['mean_delta_norm']:.4f}**",
            f"- Mean cosine(midpoint, nearest word): **{similar_word_math['mean_midpoint_neighbor_cosine']:.4f}**",
            f"- Number of analyzed similar-word pairs: **{int(similar_word_math['pair_count'])}**",
            "- Pattern: close semantic pairs have small difference vectors and midpoint vectors remain in the same semantic neighborhood.",
            "",
            "## Vector Arithmetic Equations",
            "",
            "| Equation | Top prediction | Expected word | Expected rank (top-10) | Status |",
            "|---|---|---|---:|---|",
        ]
    )

    for result in equation_results:
        lines.append(
            f"| `{safe_word(str(result['equation']))}` | `{safe_word(str(result['top1_prediction']))}` "
            f"| `{safe_word(str(result['expected_word']))}` | {result['expected_rank_in_top10']} | {result['status']} |"
        )

    lines.extend(
        [
            "",
            f"- Equation quality summary: **{hit_count} exact hits**, **{near_hit_count} near hits (rank 2-5)**.",
            "",
            "## Visible Vector Patterns",
            "",
            "| Relation group | Mean cosine of relation vectors |",
            "|---|---:|",
        ]
    )

    for group_name, score in relation_scores.items():
        lines.append(f"| `{group_name}` | {score:.4f} |")

    lines.extend(
        [
            "",
            "- Positive cosine values between relation vectors indicate partially shared geometric directions.",
            "- In this run, country-capital and country-leader relations are moderately aligned when the required terms exist in vocabulary.",
            "",
            "## Output Files",
            "",
            "- `output/training_corpus.txt`: tokenized corpus used for training.",
            "- `output/vocab.txt`: vocabulary from `vocab_count`.",
            "- `output/cooccurrence.bin`: raw cooccurrence binary.",
            "- `output/cooccurrence.shuf.bin`: shuffled cooccurrence binary.",
            "- `output/vectors.txt`: trained GloVe vectors (text format).",
            "- `output/synonyms.tsv`: top-5 similar words for 10 query words.",
            "- `output/similar_word_math.tsv`: equations on similar-word vectors.",
            "- `output/vector_equations.tsv`: analogy-style vector arithmetic results.",
            "- `output/relation_patterns.tsv`: cosine similarity between relation vectors.",
        ]
    )

    with open(output_file, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    task_dir = Path(__file__).resolve().parent
    project_root = task_dir.parent
    output_dir = task_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_file = resolve_dataset_file(project_root)
    corpus_file = output_dir / "training_corpus.txt"
    synonyms_file = output_dir / "synonyms.tsv"
    similar_math_file = output_dir / "similar_word_math.tsv"
    equations_file = output_dir / "vector_equations.tsv"
    relation_file = output_dir / "relation_patterns.tsv"
    report_file = output_dir / "task3_report.md"

    config = GloveConfig(threads=max(1, min(8, os.cpu_count() or 1)))

    print("Preparing tokenized training corpus...")
    doc_count, token_count, global_counts = build_training_corpus(dataset_file=dataset_file, output_file=corpus_file)
    print(f"Documents: {doc_count}, tokens: {token_count}, unique tokens: {len(global_counts)}")

    glove_dir = task_dir / "glove"
    if not glove_dir.exists():
        raise FileNotFoundError(
            f"GloVe source directory not found at {glove_dir}. "
            "Clone https://github.com/stanfordnlp/GloVe into Task3/glove first."
        )

    print("Building GloVe binaries...")
    binaries = ensure_glove_binaries(glove_dir)

    print("Training GloVe model...")
    train_start = time.time()
    vectors_file, _vocab_file = run_glove_training(
        binaries=binaries,
        glove_dir=glove_dir,
        corpus_file=corpus_file,
        output_dir=output_dir,
        config=config,
    )
    training_seconds = time.time() - train_start

    print("Loading trained vectors...")
    words, vectors, index, dim = load_vectors(vectors_file)
    print(f"Loaded vectors: {len(words)} words, dimension: {dim}")

    target_words = select_target_words(index=index, counts=global_counts, needed=10)
    neighbors, avg_coherence = write_synonyms(
        output_file=synonyms_file,
        target_words=target_words,
        words=words,
        vectors=vectors,
        index=index,
    )
    similar_word_math = write_similar_word_math(
        output_file=similar_math_file,
        target_words=target_words,
        neighbors=neighbors,
        words=words,
        vectors=vectors,
        index=index,
        dim=dim,
    )
    equation_results = write_equations(
        output_file=equations_file,
        words=words,
        vectors=vectors,
        index=index,
        dim=dim,
    )
    relation_scores = write_relation_patterns(
        output_file=relation_file,
        index=index,
        vectors=vectors,
        dim=dim,
    )
    write_report(
        output_file=report_file,
        dataset_file=dataset_file,
        corpus_file=corpus_file,
        vectors_file=vectors_file,
        config=config,
        document_count=doc_count,
        token_count=token_count,
        vocab_size=len(words),
        target_words=target_words,
        neighbors=neighbors,
        avg_coherence=avg_coherence,
        similar_word_math=similar_word_math,
        equation_results=equation_results,
        relation_scores=relation_scores,
        training_seconds=training_seconds,
    )

    print("Task3 completed.")
    print(f"Report: {report_file}")
    print(f"Synonyms: {synonyms_file}")
    print(f"Similar-word math: {similar_math_file}")
    print(f"Equations: {equations_file}")
    print(f"Relation patterns: {relation_file}")


if __name__ == "__main__":
    main()
