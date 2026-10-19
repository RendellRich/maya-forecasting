"""Permutation importance of the production GBMs on 2026 test hours -> feature_importance.csv.
Scrambles one input at a time and records how much the mean absolute error rises."""
import numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.inspection import permutation_importance
ROOT = Path(__file__).parent
def fpath(n):
    for s in ("data", "models", ""):
        if (ROOT / s / n).exists(): return ROOT / s / n
F = pd.read_csv(fpath("scenario_features_2026.csv"), parse_dates=["timestamp_utc"]).set_index("timestamp_utc")
A = pd.read_csv(fpath("final_forecasts_hourly.csv"), parse_dates=["timestamp_utc"]).set_index("timestamp_utc").reindex(F.index)
NAMES = {"lag24": "Same hour 24 hours earlier", "lag48": "Same hour 2 days earlier", "lag168": "Same hour one week earlier",
         "roll7": "7-day average, same hour", "Temperature (C)": "Forecast temperature", "GHI (W/m2)": "Forecast sunlight (GHI)",
         "DNI (W/m2)": "Forecast direct sunlight (DNI)", "DHI (W/m2)": "Forecast scattered sunlight (DHI)",
         "Cloud Cover (%)": "Forecast cloud cover", "Clear-sky Index": "Clear-sky index", "is_weekend": "Weekend or weekday",
         "Installed Solar DC (MW)": "Installed solar capacity"}
GROUP = {"hour_sin": "Hour of day", "hour_cos": "Hour of day", "doy_sin": "Time of year", "doy_cos": "Time of year", "month": "Time of year",
         "Wind Speed 10m (m/s)": "Forecast wind", "Wind Speed 80m (m/s)": "Forecast wind", "Wind Speed 100m (m/s)": "Forecast wind"}
rows = []
for model, file, pre, target in [("Demand", "demand_gbm.joblib", "d_", "Demand Actual (MWh)"),
                                 ("Renewable output", "renewable_gbm.joblib", "r_", "Renewable Actual (MWh)")]:
    m = joblib.load(fpath(file))
    Z = pd.DataFrame(index=F.index)
    for c in m.feature_names_in_:
        Z[c] = F[pre + c] if c in ("lag24", "lag48", "lag168", "roll7") else (F["fc_" + c] if "fc_" + c in F else F[c])
    ok = Z.drop(columns=["Clear-sky Index"], errors="ignore").notna().all(axis=1) & A[target].notna()
    Z, y = Z[ok], A.loc[ok, target]
    mae = float(np.abs(m.predict(Z) - y).mean())
    pi = permutation_importance(m, Z, y, scoring="neg_mean_absolute_error", n_repeats=5, random_state=0)
    s = pd.Series(pi.importances_mean, index=Z.columns)
    for c, v in s.items():
        rows.append(dict(model=model, input=GROUP.get(c, NAMES.get(c, c)), added_error_mwh=v, normal_mae_mwh=mae))
out = pd.DataFrame(rows).groupby(["model", "input"], as_index=False).agg(added_error_mwh=("added_error_mwh", "sum"), normal_mae_mwh=("normal_mae_mwh", "first"))
out["added_error_mwh"] = out["added_error_mwh"].clip(lower=0).round(1)
out["normal_mae_mwh"] = out["normal_mae_mwh"].round(1)
out.sort_values(["model", "added_error_mwh"], ascending=[True, False]).to_csv(ROOT / "data" / "feature_importance.csv" if (ROOT / "data").exists() else ROOT / "feature_importance.csv", index=False)
print(out.sort_values(["model", "added_error_mwh"], ascending=[True, False]).to_string(index=False))
