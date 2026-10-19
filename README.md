# Maya Forecasting System (Streamlit app)

Day-ahead forecasts of PacifiCorp East (PACE) demand, Milford, Utah renewable output and net load.
Maui Hiva Innovations capstone, Purdue University.

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

`scikit-learn` is pinned to the version the models were saved with.
