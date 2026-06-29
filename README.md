# Translingual

Master's thesis research on transferring NLP resources from one language to another, applied to cross-lingual sentiment analysis. The goal is to leverage labelled resources available in a high-resource language to perform sentiment analysis in a lower-resource language.

## Stack
- Python, Jupyter notebooks
- Poetry for dependency management (`pyproject.toml`)

## Structure
- `code_memoire_raw/`: original analysis notebooks
  - `Normalization.ipynb`: text normalization
  - `Sentiment analysis (step 1).ipynb` and `(step 2 & 3).ipynb`: the sentiment analysis pipeline
- `code_memoire_transform/`: refactored version of the notebooks plus `models.py`

## Status
Academic research code (2022). Provided for reference and reproducibility.
