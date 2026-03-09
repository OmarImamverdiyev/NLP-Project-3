# Task5

This folder contains Task5:

- Text classification with `RNN`, `Bidirectional RNN`, and `LSTM`
- Feature comparison across `Count Vectorizer`, `TF-IDF`, `PMI`, `Word2Vec`, and `GloVe`
- Result table written to `output/task5_results.csv` and `output/task5_report.md`

Run:

```bash
python task5.py
```

Model persistence:

- First run saves checkpoints in `output/model_cache/*.pt`.
- `output/model_cache_manifest.json` stores the cache signature.
- Later runs with the same data/config reuse cached checkpoints and skip retraining.
