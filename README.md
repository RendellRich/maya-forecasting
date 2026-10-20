# Maya Forecasting System (Streamlit app)

Day-ahead forecasts of PacifiCorp East (PACE) demand, Milford, Utah renewable output and net load.
Maui Hiva Innovations capstone, Purdue University.

> **Branch `v1.1-model-upgrade`** (Oct 20, 2026): same app and 24-hour setup, retrained demand model with two new
> inputs: weekday / holiday / Utah school-break flags, and population-weighted forecast temperature over eight PACE
> load centers (bias-corrected to ERA5). Demand error, Maya vs PACE's own day-ahead forecast:
>
> | Period | PACE | v1 | v1.1 |
> |---|---|---|---|
> | Backtest 2024-25 (MWh/h) | 254.7 | 186.7 | 171.9 |
> | Forward test 2026 (MWh/h) | 347.8 | 211.2 | 186.5 |
>
> `main` (the live app) is unchanged. Retrain with `cd training && python final_models_v1_1.py`
> (copy `master_hourly_cleaned_ALL_YEARS_v2.csv` into `training/data` first; it is in the Maya v2 repo).

## Run locally

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

1. Put this folder in a GitHub repository (files at the repository root).
2. On share.streamlit.io choose **Create app**, pick the repository and branch, and set the main file to `streamlit_app.py`.

## Files

| Path | What it is |
|---|---|
| `streamlit_app.py` | The app |
| `final_forecasts_hourly.csv` | Out-of-sample hourly forecasts, Feb 2024 to Sep 2026 |
| `final_metrics.csv` | Accuracy by period and series |
| `scenario_features_2026.csv` | Day-ahead model inputs for 2026, used by the live what-if tab |
| model files | Production gradient boosting models (demand, renewable) and model card |
| `prep_features.py` | Rebuilds `scenario_features_2026.csv` from the master dataset |
| `training/` | v1.1 training script, input helpers and the weather inputs |

`scikit-learn` is pinned to the version the models were saved with.
