# Task5

This folder contains Task5:

- Sentiment classification on `Sentiment140_v2.csv`
- Dense-model comparison for `Count Vectorizer`, `TF-IDF`, and `PMI`
- Sequence-model comparison for `Word2Vec` and `GloVe` using `RNN`, `Bidirectional RNN`, and `LSTM`
- Result table written to `output/task5_results.csv` and `output/task5_report.md`

Run:

```bash
python task5.py
```

Model persistence:

- First run saves checkpoints in `output/model_cache/*.pt`.
- `output/model_cache_manifest.json` stores the cache signature.
- Later runs with the same data/config reuse cached checkpoints and skip retraining.
