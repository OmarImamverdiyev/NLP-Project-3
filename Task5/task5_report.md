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
| Count Vectorizer | RNN | 0.2644 | 0.2895 | 0.3643 | 0.2242 | 4.76 |
| Count Vectorizer | Bidirectional RNN | 0.1978 | 0.2977 | 0.3546 | 0.1701 | 7.43 |
| Count Vectorizer | LSTM | 0.2611 | 0.3253 | 0.3564 | 0.2110 | 5.20 |
| TF-IDF | RNN | 0.3411 | 0.2266 | 0.3000 | 0.2007 | 6.24 |
| TF-IDF | Bidirectional RNN | 0.3367 | 0.1870 | 0.2433 | 0.1649 | 10.59 |
| TF-IDF | LSTM | 0.2333 | 0.0389 | 0.1667 | 0.0631 | 11.90 |
| PMI | RNN | 0.2433 | 0.2821 | 0.3671 | 0.2090 | 4.75 |
| PMI | Bidirectional RNN | 0.1900 | 0.3210 | 0.3420 | 0.1639 | 8.50 |
| PMI | LSTM | 0.2578 | 0.3071 | 0.3470 | 0.2591 | 6.31 |
| Word2Vec | RNN | 0.5167 | 0.6080 | 0.7398 | 0.5656 | 1.74 |
| Word2Vec | Bidirectional RNN | 0.4922 | 0.5772 | 0.7102 | 0.5067 | 3.32 |
| Word2Vec | LSTM | 0.3756 | 0.6017 | 0.6457 | 0.4491 | 2.21 |
| GloVe | RNN | 0.4744 | 0.5887 | 0.6908 | 0.5797 | 1.64 |
| GloVe | Bidirectional RNN | 0.6033 | 0.6718 | 0.7033 | 0.6504 | 3.10 |
| GloVe | LSTM | 0.3833 | 0.5713 | 0.6687 | 0.4556 | 1.87 |

## Best Combination

- Best by macro F1: **Bidirectional RNN + GloVe**
- Accuracy: **0.6033**
- Precision: **0.6718**
- Recall: **0.7033**
- F1: **0.6504**

## Output Files

- `output/task5_results.csv`
- `output/task5_report.md`
