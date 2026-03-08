# Task3 - GloVe Training and Semantic Analysis

- Source implementation: `https://github.com/stanfordnlp/GloVe`
- Dataset: `D:\GitHub Repos\NLP-Projects\NLP-Project-3\Corpora\news\content_only.csv`
- Tokenized corpus: `D:\GitHub Repos\NLP-Projects\NLP-Project-3\Task3\output\training_corpus.txt`
- Model vectors file: `D:\GitHub Repos\NLP-Projects\NLP-Project-3\Task3\output\vectors.txt`
- Documents used: **97997**
- Total tokens used: **13480722**
- Vocabulary size (trained vectors): **38455**
- Training runtime (full GloVe pipeline): **127.37 seconds**

## Chosen GloVe Parameters

| Parameter | Value | Why this value |
|---|---:|---|
| `max-vocab` | `70000` | Caps vocabulary for faster cooccurrence construction while preserving most frequent terms. |
| `min-count` | `15` | Filters very rare tokens to reduce noise and model size. |
| `window-size` | `8` | Captures medium-range context typical in news sentences. |
| `vector-size` | `100` | Balances semantic capacity and training cost. |
| `iter` | `12` | Multiple passes improve embedding stability without excessive runtime. |
| `memory` | `2.0` GB | Keeps cooccur/shuffle memory bounded for this machine. |
| `x-max` | `100.0` | Standard weighting cutoff from GloVe formulation. |
| `alpha` | `0.75` | Standard exponent for cooccurrence weighting. |
| `eta` | `0.05` | Default learning rate recommended by the original implementation. |
| `symmetric` | `1` | Uses both left and right contexts for cooccurrence counts. |
| `distance-weighting` | `1` | Inverse-distance weighting improves local context sensitivity. |
| `model` | `2` | Saves word+context vectors merged (standard GloVe text output choice). |
| `threads` | `8` | Uses available CPU parallelism. |

## Synonym / Similar Word Results (10 Query Words)

Selected query words:

`rusiya`, `ukrayna`, `prezident`, `moskva`, `kiyev`, `putin`, `zelenski`, `futbol`, `komanda`, `oyun`

- `rusiya` -> `ukrayna` (0.838), `rusiyanın` (0.801), `rublurub` (0.758), `federasiyası` (0.754), `moskva` (0.726)
- `ukrayna` -> `rusiya` (0.838), `münaqişəsində` (0.776), `qrivnasıuah` (0.772), `zlotısıpln` (0.741), `zelenski` (0.739)
- `prezident` -> `i̇lham` (0.867), `cənab` (0.787), `prezidenti` (0.772), `əliyev` (0.723), `seçkisi` (0.721)
- `moskva` -> `müxbiri` (0.808), `rusiya` (0.726), `nən` (0.677), `rusiyanın` (0.673), `bürosu` (0.665)
- `kiyev` -> `xarkov` (0.684), `odessa` (0.682), `tehran` (0.661), `sumı` (0.652), `meri` (0.651)
- `putin` -> `vladimir` (0.921), `putinlə` (0.862), `putinin` (0.833), `zelenski` (0.772), `putinə` (0.732)
- `zelenski` -> `volodimir` (0.932), `zelenskinin` (0.828), `vladimir` (0.776), `putin` (0.772), `zelenskiyə` (0.770)
- `futbol` -> `federasiyaları` (0.678), `cüdo` (0.675), `voleybol` (0.665), `basketbol` (0.646), `karate` (0.625)
- `komanda` -> `qərargah` (0.647), `yarışacaq` (0.632), `təliminin` (0.594), `yarışda` (0.547), `yığma` (0.536)
- `oyun` -> `meydançaları` (0.758), `i̇lk` (0.670), `günündə` (0.592), `qarşılaşma` (0.585), `pen` (0.547)

### Accuracy Discussion

- Mean top-5 cosine across the 10 query words: **0.7218** (high coherence).
- Most query words returned semantically related entities, inflections, or same-topic terms.
- Some neighbors are topical co-occurrences rather than strict dictionary synonyms, which is expected for distributional embeddings.

## Mathematical Equations on Similar Words

For each query word, I used its closest neighbor and computed:
- `delta = v(word_a) - v(word_b)`
- `midpoint = (v(word_a) + v(word_b)) / 2`, then nearest neighbor to this midpoint.

- Mean `||delta||_2` over analyzed pairs: **0.6173**
- Mean cosine(midpoint, nearest word): **0.7944**
- Number of analyzed similar-word pairs: **10**
- Pattern: close semantic pairs have small difference vectors and midpoint vectors remain in the same semantic neighborhood.

## Vector Arithmetic Equations

| Equation | Top prediction | Expected word | Expected rank (top-10) | Status |
|---|---|---|---:|---|
| `prezident - rusiya + ukrayna` | `cənab` | `zelenski` | 9 | miss |
| `prezident - ukrayna + rusiya` | `i̇lham` | `putin` | -1 | miss |
| `putin - rusiya + ukrayna` | `zelenski` | `zelenski` | 1 | hit |
| `moskva - rusiya + ukrayna` | `kiyev` | `kiyev` | 1 | hit |
| `kiyev - ukrayna + rusiya` | `moskva` | `moskva` | 1 | hit |
| `futbol - komanda + oyun` | `meydançaları` | `matc` | -1 | miss |

- Equation quality summary: **3 exact hits**, **0 near hits (rank 2-5)**.

## Visible Vector Patterns

| Relation group | Mean cosine of relation vectors |
|---|---:|
| `country_to_capital` | 0.6252 |
| `country_to_leader` | 0.7483 |

- Positive cosine values between relation vectors indicate partially shared geometric directions.
- In this run, country-capital and country-leader relations are moderately aligned when the required terms exist in vocabulary.

## Output Files

- `output/training_corpus.txt`: tokenized corpus used for training.
- `output/vocab.txt`: vocabulary from `vocab_count`.
- `output/cooccurrence.bin`: raw cooccurrence binary.
- `output/cooccurrence.shuf.bin`: shuffled cooccurrence binary.
- `output/vectors.txt`: trained GloVe vectors (text format).
- `output/synonyms.tsv`: top-5 similar words for 10 query words.
- `output/similar_word_math.tsv`: equations on similar-word vectors.
- `output/vector_equations.tsv`: analogy-style vector arithmetic results.
- `output/relation_patterns.tsv`: cosine similarity between relation vectors.