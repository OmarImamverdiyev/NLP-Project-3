# Task1

This folder contains Task1 for the main dataset in `Corpora/News/`.

## Run

From project root:

```powershell
./venv/Scripts/python.exe Task1/task1.py
```

## Outputs (`Task1/output/`)

- `dataset_summary.md`: dataset size, distinct words, frequent/rare counts, top frequencies, matrix shapes.
- `word_frequency.tsv`: full word frequency table.
- `frequent_words.txt`: words with frequency >= 10.
- `rare_words.txt`: words with frequency <= 1.
- `term_document_sparse.tsv`: full term-document matrix in sparse form (`doc_id`, `term`, `count`).
- `term_document_matrix_visual.csv`: matrix-form term-document visualization (top terms x first docs).
- `term_document_matrix_visual.txt`: readable matrix text form.
- `word_word_matrix.csv`: word-word co-occurrence matrix (top 40 words, window size 2).
- `word_word_matrix.txt`: readable matrix text form.
