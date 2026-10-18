# notebooks/ — Original Colab notebook, cleaned

## Files

| File | What it is |
|---|---|
| `NLP_Extractive_Summarization.ipynb` | The team's original Colab exploration notebook, cleaned so it can run outside Colab. |

## What was fixed vs the original

- First cell typo `whafrom google.colab import drive` → `from google.colab import drive`.
- Hard-coded Google Drive paths (`/content/drive/MyDrive/NLP Project/...`)
  replaced with relative paths (`../data/...`, `../src/...`, `../results/...`).
- `%run` paths updated to the new repo layout (`../src/fasttext/...`,
  `../src/training/...`, `../src/summarization/...`,
  `../src/evaluation/...`).

## What to know

- The notebook is exploration history — the runnable pipeline is the
  `src/` scripts (see root README). Use the notebook to see how results
  were explored, not as the main entry point.
- Early cells in the notebook use a sentence-level random split for a
  quick experiment. The final pipeline uses the article-level split
  (see `src/fasttext/README.md`) — that is the split the reported
  numbers come from.
