# Task5 - Sentiment Classification with Dense and Recurrent Models

## Setup

- Dataset: `D:\GitHub Repos\NLP-Projects\NLP-Project-3\Task5\Sentiment140_v2.csv`
- Source dataset: `Task5/Sentiment140_v2.csv`
- Samples used in this run: **200000**
- Sampling mode: **full dataset**
- Gold labels are read directly from the `polarity` column and remapped internally to contiguous class IDs for PyTorch training.
- Label mapping: **0 (negative) -> 0, 4 (positive) -> 1**
- Class distribution: **0 (negative): 100000, 4 (positive): 100000**
- Train/validation/test split: **140000 / 30000 / 30000**
- A tweet-aware tokenizer is used across all feature sets, preserving hashtags, mentions, contractions, emoticons, emojis, elongated words, and repeated punctuation.
- Non-sequential feature sets (`Count Vectorizer`, `TF-IDF`, `PMI`) use dense classifiers so vocabulary dimensions are not treated as a fake time axis.
- Sequence feature sets (`Word2Vec`, `GloVe`) extend the pretrained vocabularies with frequent tweet tokens and map any remaining unseen items to `<unk>` instead of silently dropping them.
- Pretrained sequence embeddings are fine-tuned during sentiment training (`freeze=False`) so the model can adapt them to Sentiment140.
- Sequence feature sets use packed recurrent models and pool the true forward/backward hidden states.

## Training Configuration

- BOW vocabulary size (`Count`, `TF-IDF`, `PMI`): **200**
- Max token length (`Word2Vec`, `GloVe`): **80**
- Extra tweet-token embeddings added per pretrained vocabulary (max): **30000**
- Minimum train frequency before adding a new tweet-token embedding: **2**
- Batch size: **128**
- Epochs per model: **3**
- Hidden size: **64**
- Learning rate: **0.001**

## Results Table

| Feature | Model | Accuracy | Precision | Recall | F1 | Train Time (s) |
|---|---|---:|---:|---:|---:|---:|
| Count Vectorizer | Linear | 0.7125 | 0.7125 | 0.7125 | 0.7124 | 7.17 |
| Count Vectorizer | MLP | 0.7202 | 0.7206 | 0.7202 | 0.7201 | 9.80 |
| Count Vectorizer | Deep MLP | 0.7213 | 0.7214 | 0.7213 | 0.7212 | 10.99 |
| TF-IDF | Linear | 0.7117 | 0.7117 | 0.7117 | 0.7117 | 7.55 |
| TF-IDF | MLP | 0.7177 | 0.7184 | 0.7177 | 0.7175 | 10.13 |
| TF-IDF | Deep MLP | 0.7222 | 0.7224 | 0.7222 | 0.7221 | 11.52 |
| PMI | Linear | 0.7124 | 0.7124 | 0.7124 | 0.7124 | 7.50 |
| PMI | MLP | 0.7192 | 0.7194 | 0.7192 | 0.7192 | 9.38 |
| PMI | Deep MLP | 0.7212 | 0.7212 | 0.7212 | 0.7212 | 11.27 |
| Word2Vec | RNN | 0.7907 | 0.7907 | 0.7907 | 0.7907 | 232.40 |
| Word2Vec | Bidirectional RNN | 0.7900 | 0.7900 | 0.7900 | 0.7900 | 255.38 |
| Word2Vec | LSTM | 0.8015 | 0.8025 | 0.8015 | 0.8014 | 262.03 |
| GloVe | RNN | 0.7894 | 0.7897 | 0.7894 | 0.7893 | 146.07 |
| GloVe | Bidirectional RNN | 0.7930 | 0.7931 | 0.7930 | 0.7930 | 190.87 |
| GloVe | LSTM | 0.8021 | 0.8021 | 0.8021 | 0.8021 | 190.07 |

## Best Combination

- Best by macro F1: **LSTM + GloVe**
- Accuracy: **0.8021**
- Precision: **0.8021**
- Recall: **0.8021**
- F1: **0.8021**

## Output Files

- `output/task5_results.csv`
- `output/task5_report.md`
- `output/model_cache/*.pt` (saved model checkpoints for each feature/model combination)
- `output/model_cache_manifest.json` (cache signature and checkpoint index)