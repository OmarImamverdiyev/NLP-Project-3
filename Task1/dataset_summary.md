# Task1 - Dataset Description and Matrix Construction

- Dataset: `D:\GitHub Repos\NLP-Projects\NLP-Project-3\Corpora\News\content_only.csv`
- Number of documents (size): **97997**
- Total token count (size): **13480722**
- Number of distinct words: **221456**
- Number of frequent words (frequency >= 10): **48779**
- Number of rare words (frequency <= 1): **86784**

## Top Word Frequencies

| Word | Frequency |
|---|---:|
| və | 361463 |
| ki | 179232 |
| bu | 161810 |
| ilə | 122482 |
| apa | 102447 |
| azərbaycan | 90016 |
| də | 76792 |
| verir | 70810 |
| xəbər | 69536 |
| üçün | 67433 |
| görə | 59795 |
| bir | 59668 |
| edib | 54841 |
| barədə | 54536 |
| üzrə | 54039 |
| qeyd | 52914 |
| baş | 49216 |
| da | 49057 |
| dövlət | 48957 |
| olan | 44792 |

## Matrix Shapes

- Term-document (full sparse): `97997 x 221456` with `9518204` non-zero entries
- Word-word (top-40 terms): `40 x 40`

## Notes

- Term-document matrix is saved in sparse form at `output/term_document_sparse.tsv`.
- Matrix-form visual term-document matrix uses top 20 terms and first 20 documents.
- Word-word matrix uses a symmetric co-occurrence window of 2 tokens and top 40 frequent words.