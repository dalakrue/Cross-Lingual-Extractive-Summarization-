# src/fasttext/ — FastText training + feature extraction

## Files

| File | What it does |
|---|---|
| `fasttext_features.py` | Baseline: trains FastText (64-dim) on training articles only, converts every sentence to a 64-dim vector. |
| `improved_features.py` | Improved: same 64 FastText features **plus 5 sentence features** — `sentence_position`, `relative_position`, `word_count`, `character_count`, `normalized_word_count` (69 features total). |

## How to run

```bash
# Baseline (reads data/sentence_labels.csv)
python src/fasttext/fasttext_features.py
# Writes: results/sentence_vectors.csv, results/article_split.csv, results/fasttext.model

# Improved (reads data/sentence_labels.csv + results/article_split.csv)
python src/fasttext/improved_features.py
# Writes: results/improved_sentence_vectors.csv, results/improved_fasttext.model
```

## Important details

- **Article-level split, not sentence-level.** Articles are split 80/20
  (40 train / 10 test) with `random_state=42`. Sentences from the same
  article never appear in both train and test — otherwise the model would
  memorize articles and the scores would be inflated.
- **FastText is trained only on training articles.** It can still vectorize
  unseen test sentences because FastText uses subword (character n-gram)
  information.
- **Settings:** `vector_size=64`, `window=4`, `min_count=1`,
  `min_n=3`, `max_n=6`, `sg=1` (skip-gram), `epochs=50`, `seed=7`.
- **Sentence vector = mean of its word vectors.** Empty sentences become
  a zero vector.
- **Tokenization:** `re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", text.lower())`
  in the baseline. The improved pipeline uses the same pattern without
  lowercasing in both its training and inference code, so the two stay
  consistent — do not change one without the other.
