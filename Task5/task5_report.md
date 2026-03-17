# Task5 - Text Classification with RNN Variants

## Setup

- Dataset: `D:\GitHub Repos\NLP-Projects\NLP-Project-3\Corpora\news\content_only.csv`
- Source corpus: `Corpora/News/content_only.csv`
- Documents used in this run (after filtering + sampling): **6000**
- Labeling note: this dataset has no gold label column, so pseudo-topic labels were created using TF-IDF + KMeans.
- Number of pseudo-topic classes: **6**

## Training Configuration

- BOW vocabulary size (`Count`, `TF-IDF`, `PMI`): **200**
- Max token length (`Word2Vec`, `GloVe`): **80**
- Batch size: **128**
- Epochs per model: **3**
- Hidden size: **64**
- Learning rate: **0.001**

## Results Table

| Feature | Model | Accuracy | Precision | Recall | F1 | Train Time (s) |
|---|---|---:|---:|---:|---:|---:|
| Count Vectorizer | RNN | 0.3765 | 0.3648 | 0.5247 | 0.3481 | 71.62 |
| Count Vectorizer | Bidirectional RNN | 0.5081 | 0.4553 | 0.6141 | 0.4790 | 115.01 |
| Count Vectorizer | LSTM | 0.4489 | 0.4144 | 0.5565 | 0.4225 | 69.32 |
| TF-IDF | RNN | 0.3597 | 0.3921 | 0.5085 | 0.3409 | 65.16 |
| TF-IDF | Bidirectional RNN | 0.4631 | 0.4174 | 0.5436 | 0.4233 | 116.50 |
| TF-IDF | LSTM | 0.4017 | 0.3772 | 0.5384 | 0.3785 | 86.89 |
| PMI | RNN | 0.5186 | 0.4631 | 0.6057 | 0.4949 | 61.91 |
| PMI | Bidirectional RNN | 0.5471 | 0.5101 | 0.6719 | 0.5428 | 115.96 |
| PMI | LSTM | 0.4885 | 0.4511 | 0.5752 | 0.4722 | 71.71 |
| Word2Vec | RNN | 0.8365 | 0.8503 | 0.8859 | 0.8662 | 31.07 |
| Word2Vec | Bidirectional RNN | 0.7425 | 0.8185 | 0.8232 | 0.8206 | 66.57 |
| Word2Vec | LSTM | 0.9027 | 0.9108 | 0.9343 | 0.9213 | 47.81 |
| GloVe | RNN | 0.8250 | 0.8483 | 0.8781 | 0.8620 | 32.00 |
| GloVe | Bidirectional RNN | 0.7490 | 0.7944 | 0.8303 | 0.8042 | 59.79 |
| GloVe | LSTM | 0.8905 | 0.8902 | 0.9288 | 0.9075 | 43.17 |

## Best Combination

- Best by macro F1: **Bidirectional RNN + GloVe**
- Accuracy: **0.6033**
- Precision: **0.6718**
- Recall: **0.7033**
- F1: **0.6504**

## Output Files

- `output/task5_results.csv`
- `output/task5_report.md`
