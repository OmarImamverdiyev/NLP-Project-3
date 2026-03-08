from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


TOKEN_PATTERN = re.compile(r"[^\W\d_]+", flags=re.UNICODE)

# Windows-safe upper bound for large text rows in CSV.
csv.field_size_limit(2**31 - 1)

FREQUENT_WORD_MIN_COUNT = 10
RARE_WORD_MAX_COUNT = 1

TOP_TERMS_FOR_TD_VIS = 20
TOP_DOCS_FOR_TD_VIS = 20

TOP_TERMS_FOR_WW = 40
WINDOW_SIZE = 2


def resolve_dataset_file(project_root: Path) -> Path:
    candidates = [
        project_root / "Corpora" / "News" / "content_only.csv",
        project_root / "corpora" / "news" / "content_only.csv",
        project_root / "coprpora" / "news" / "content_only.csv",
    ]
    for path in candidates:
        if path.exists():
            return path

    for path in project_root.rglob("content_only.csv"):
        lowered = str(path).lower().replace("\\", "/")
        if "/news/" in lowered:
            return path

    raise FileNotFoundError("Could not find news dataset CSV (content_only.csv).")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def load_documents(csv_path: Path) -> list[str]:
    documents: list[str] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "content" not in reader.fieldnames:
            raise ValueError("Expected a 'content' column in content_only.csv.")

        for row in reader:
            text = (row.get("content") or "").strip()
            if text:
                documents.append(text)

    return documents


def write_counter_tsv(path: Path, counter: Counter[str], header: tuple[str, str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        for word, count in counter.most_common():
            writer.writerow((word, count))


def write_list(path: Path, items: Iterable[str]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(f"{item}\n")


def matrix_to_text(
    row_labels: list[str],
    col_labels: list[str],
    matrix: list[list[int]],
    top_left_name: str,
) -> str:
    row_header_width = max(len(top_left_name), max((len(r) for r in row_labels), default=0))
    col_widths: list[int] = []

    for col_index, col_name in enumerate(col_labels):
        max_value_width = max((len(str(row[col_index])) for row in matrix), default=1)
        col_widths.append(max(len(col_name), max_value_width))

    header = f"{top_left_name:<{row_header_width}} | " + " ".join(
        f"{col_labels[idx]:>{col_widths[idx]}}" for idx in range(len(col_labels))
    )
    separator = "-" * len(header)

    lines = [header, separator]
    for row_name, row_values in zip(row_labels, matrix):
        row_text = f"{row_name:<{row_header_width}} | " + " ".join(
            f"{str(row_values[idx]):>{col_widths[idx]}}" for idx in range(len(col_labels))
        )
        lines.append(row_text)

    return "\n".join(lines)


def write_matrix_csv(path: Path, row_name: str, row_labels: list[str], col_labels: list[str], matrix: list[list[int]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([row_name, *col_labels])
        for label, values in zip(row_labels, matrix):
            writer.writerow([label, *values])


def build_term_document_visual_matrix(
    tokens_by_doc: list[list[str]],
    global_counts: Counter[str],
) -> tuple[list[str], list[str], list[list[int]]]:
    terms = [word for word, _ in global_counts.most_common(TOP_TERMS_FOR_TD_VIS)]
    doc_count = min(TOP_DOCS_FOR_TD_VIS, len(tokens_by_doc))
    docs = [f"doc_{idx + 1}" for idx in range(doc_count)]

    first_doc_counters = [Counter(tokens_by_doc[idx]) for idx in range(doc_count)]
    matrix = [[first_doc_counters[col_idx].get(term, 0) for col_idx in range(doc_count)] for term in terms]
    return terms, docs, matrix


def write_sparse_term_document(path: Path, tokens_by_doc: list[list[str]]) -> int:
    non_zero = 0
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["doc_id", "term", "count"])

        for doc_id, tokens in enumerate(tokens_by_doc, start=1):
            counts = Counter(tokens)
            for term, count in sorted(counts.items()):
                writer.writerow([doc_id, term, count])
                non_zero += 1

    return non_zero


def build_word_word_matrix(tokens_by_doc: list[list[str]], vocab: list[str]) -> tuple[list[str], list[list[int]]]:
    vocab_set = set(vocab)
    cooc: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for tokens in tokens_by_doc:
        for i, wi in enumerate(tokens):
            if wi not in vocab_set:
                continue

            upper = min(len(tokens), i + WINDOW_SIZE + 1)
            for j in range(i + 1, upper):
                wj = tokens[j]
                if wj not in vocab_set:
                    continue

                cooc[wi][wj] += 1
                cooc[wj][wi] += 1

    matrix = [[cooc[row_word].get(col_word, 0) for col_word in vocab] for row_word in vocab]
    return vocab, matrix


def write_summary_markdown(
    path: Path,
    dataset_path: Path,
    document_count: int,
    total_tokens: int,
    distinct_words: int,
    top_words: list[tuple[str, int]],
    frequent_word_count: int,
    rare_word_count: int,
    td_shape: tuple[int, int],
    td_nonzero: int,
    ww_shape: tuple[int, int],
) -> None:
    lines = [
        "# Task1 - Dataset Description and Matrix Construction",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Number of documents (size): **{document_count}**",
        f"- Total token count (size): **{total_tokens}**",
        f"- Number of distinct words: **{distinct_words}**",
        f"- Number of frequent words (frequency >= {FREQUENT_WORD_MIN_COUNT}): **{frequent_word_count}**",
        f"- Number of rare words (frequency <= {RARE_WORD_MAX_COUNT}): **{rare_word_count}**",
        "",
        "## Top Word Frequencies",
        "",
        "| Word | Frequency |",
        "|---|---:|",
    ]
    for word, count in top_words:
        lines.append(f"| {word} | {count} |")

    lines.extend(
        [
            "",
            "## Matrix Shapes",
            "",
            f"- Term-document (full sparse): `{td_shape[0]} x {td_shape[1]}` with `{td_nonzero}` non-zero entries",
            f"- Word-word (top-{TOP_TERMS_FOR_WW} terms): `{ww_shape[0]} x {ww_shape[1]}`",
            "",
            "## Notes",
            "",
            f"- Term-document matrix is saved in sparse form at `output/term_document_sparse.tsv`.",
            f"- Matrix-form visual term-document matrix uses top {TOP_TERMS_FOR_TD_VIS} terms and first {TOP_DOCS_FOR_TD_VIS} documents.",
            f"- Word-word matrix uses a symmetric co-occurrence window of {WINDOW_SIZE} tokens and top {TOP_TERMS_FOR_WW} frequent words.",
        ]
    )

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    output_dir = script_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_file = resolve_dataset_file(project_root)
    documents = load_documents(dataset_file)
    tokens_by_doc = [tokenize(doc) for doc in documents]

    global_counts: Counter[str] = Counter()
    for tokens in tokens_by_doc:
        global_counts.update(tokens)

    document_count = len(tokens_by_doc)
    total_tokens = sum(len(tokens) for tokens in tokens_by_doc)
    distinct_words = len(global_counts)

    frequent_words = sorted(word for word, count in global_counts.items() if count >= FREQUENT_WORD_MIN_COUNT)
    rare_words = sorted(word for word, count in global_counts.items() if count <= RARE_WORD_MAX_COUNT)

    write_counter_tsv(output_dir / "word_frequency.tsv", global_counts, ("word", "frequency"))
    write_list(output_dir / "frequent_words.txt", frequent_words)
    write_list(output_dir / "rare_words.txt", rare_words)

    td_nonzero = write_sparse_term_document(output_dir / "term_document_sparse.tsv", tokens_by_doc)
    td_terms, td_docs, td_matrix = build_term_document_visual_matrix(tokens_by_doc, global_counts)

    write_matrix_csv(output_dir / "term_document_matrix_visual.csv", "term", td_terms, td_docs, td_matrix)
    td_text = matrix_to_text(td_terms, td_docs, td_matrix, "term")
    with open(output_dir / "term_document_matrix_visual.txt", "w", encoding="utf-8") as handle:
        handle.write(td_text)

    ww_terms = [word for word, _ in global_counts.most_common(TOP_TERMS_FOR_WW)]
    ww_terms, ww_matrix = build_word_word_matrix(tokens_by_doc, ww_terms)
    write_matrix_csv(output_dir / "word_word_matrix.csv", "word", ww_terms, ww_terms, ww_matrix)
    ww_text = matrix_to_text(ww_terms, ww_terms, ww_matrix, "word")
    with open(output_dir / "word_word_matrix.txt", "w", encoding="utf-8") as handle:
        handle.write(ww_text)

    write_summary_markdown(
        path=output_dir / "dataset_summary.md",
        dataset_path=dataset_file,
        document_count=document_count,
        total_tokens=total_tokens,
        distinct_words=distinct_words,
        top_words=global_counts.most_common(20),
        frequent_word_count=len(frequent_words),
        rare_word_count=len(rare_words),
        td_shape=(document_count, distinct_words),
        td_nonzero=td_nonzero,
        ww_shape=(len(ww_terms), len(ww_terms)),
    )

    print("Task1 completed.")
    print(f"Dataset path: {dataset_file}")
    print(f"Documents: {document_count}")
    print(f"Total tokens: {total_tokens}")
    print(f"Distinct words: {distinct_words}")
    print(f"Frequent words (>= {FREQUENT_WORD_MIN_COUNT}): {len(frequent_words)}")
    print(f"Rare words (<= {RARE_WORD_MAX_COUNT}): {len(rare_words)}")
    print(f"Term-document matrix shape: {document_count} x {distinct_words}")
    print(f"Word-word matrix shape: {len(ww_terms)} x {len(ww_terms)}")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
