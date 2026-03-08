# Task4 - Comparison of GloVe and Word2Vec Results

## Data and sources used

This comparison is based on outputs already produced in:

- `Task2/output/*` (Word2Vec)
- `Task3/output/*` (GloVe)

Both models were trained on the same corpus (`Corpora/news/content_only.csv`), so the comparison focuses on model behavior and chosen hyperparameters.

## 1) Training and model characteristics

| Metric | Word2Vec (Task2) | GloVe (Task3) | Comparison |
|---|---:|---:|---|
| Training runtime | 228.51 s | 127.37 s | GloVe is faster (~1.79x) |
| Vocabulary size | 48,780 | 38,455 | Word2Vec covers more tokens |
| Vector dimension | 150 | 100 | Word2Vec uses larger vectors |
| Vector file size | 69,891,197 bytes | 36,906,383 bytes | GloVe output is smaller (~47.2%) |
| Tokens used | 13,480,722 | 13,480,722 | Same training data size |

## 2) Semantic similarity quality

| Metric | Word2Vec | GloVe | Better |
|---|---:|---:|---|
| Mean top-5 cosine (all 10 query words in each task) | 0.7034 | 0.7218 | GloVe |

Important note: the 10 query words are not identical between Task2 and Task3.  
For a fair overlap-only check, shared query words are `prezident`, `rusiya`, and `ukrayna`:

| Shared query word | Word2Vec mean top-5 cosine | GloVe mean top-5 cosine |
|---|---:|---:|
| prezident | 0.7540 | 0.7741 |
| rusiya | 0.7726 | 0.7756 |
| ukrayna | 0.7939 | 0.7733 |

On the shared subset, results are very close overall, with GloVe slightly better on two words and Word2Vec better on one.

## 3) Vector arithmetic (analogy-style equations)

| Metric | Word2Vec | GloVe |
|---|---:|---:|
| Exact hits | 3/6 | 3/6 |
| Near hits (rank 2-5) | 1/6 | 0/6 |
| Misses | 2/6 | 3/6 |

Shared equations across both tasks:

- Capital transfer equation (`moskva - rusiya + ukrayna`) is solved by both models (`kiyev` top-1).
- Sports equation (`futbol - komanda + oyun`) is missed by both; Word2Vec places expected target at rank 9, while GloVe does not return it in top-10.

## 4) Relation-vector consistency

For the shared relation pairs available in both tasks:

| Relation-pair comparison | Word2Vec cosine | GloVe cosine | Better |
|---|---:|---:|---|
| country->capital: `(rusiya->moskva)` vs `(ukrayna->kiyev)` | 0.4388 | 0.6252 | GloVe |
| country->leader: `(rusiya->putin)` vs `(ukrayna->zelenski)` | 0.6724 | 0.7483 | GloVe |

GloVe shows stronger alignment of these relation directions in this project run.

## Final conclusion

For this corpus and these settings, **GloVe is the better overall choice**: it trains faster, produces smaller embeddings, and gives slightly stronger semantic and relation-geometry scores.  
**Word2Vec remains competitive** with larger vocabulary coverage and slightly better behavior on some difficult analogy cases.
