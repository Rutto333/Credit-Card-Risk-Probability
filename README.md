# Home Credit Risk Stability — Inference Notebook

Inference pipeline for the [Home Credit – Credit Risk Model Stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability) Kaggle competition. Loads pre-trained models and generates a submission file predicting loan default probability.

## What it does

1. **Data loading & preprocessing** — Reads test data from Parquet files (base + depth 0/1/2 tables) using Polars, casting columns to appropriate dtypes.
2. **Feature aggregation** — Aggregates numerical, date, and categorical features (max, last, mean, median, count, etc.) per `case_id` across related tables.
3. **Feature engineering** — Joins all depth tables onto the base table and derives extra features (e.g. month/weekday of decision date).
4. **Memory optimization** — Converts to pandas and downcasts dtypes to reduce memory usage.
5. **Model loading** — Loads pre-trained **LightGBM** and **CatBoost** model ensembles (saved with `joblib`) from prior training notebooks.
6. **Ensembling** — Combines all models into a custom `VotingModel` that averages predicted probabilities.
7. **Prediction & submission** — Predicts default probability (`score`) for each `case_id` and writes `submission.csv` in the competition's expected format.

## Requirements

- `numpy`, `pandas`, `polars`, `scikit-learn`, `lightgbm`, `catboost`, `joblib`

## Inputs expected

- `/kaggle/input/home-credit-credit-risk-model-stability/` — competition test data
- `/kaggle/input/home-credit-lgb-train-copy/` — pre-trained LightGBM models
- `/kaggle/input/home-credit-cat-train-copy/` — pre-trained CatBoost models

## Output

- `submission.csv` — `case_id`, `score` (predicted probability of default)
