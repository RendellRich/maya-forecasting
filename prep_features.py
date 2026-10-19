"""Builds data/scenario_features_2026.csv: the exact day-ahead feature rows the production GBMs use for 2026,
so the app can rerun the models live with adjusted weather. Run from the master_v2 folder."""
import numpy as np, pandas as pd, sys
OUT = sys.argv[1]
df = pd.read_csv("master_hourly_cleaned_ALL_YEARS_v2.csv")
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"]).dt.tz_localize(None)
df = df.set_index("timestamp_utc").sort_index()
fc = pd.read_csv("ecmwf_dayahead_forecast_2024-2026.csv", parse_dates=["timestamp_utc"]).set_index("timestamp_utc").reindex(df.index)
ts = df.index; year = ts.year.values
df["hour_sin"] = np.sin(2*np.pi*ts.hour/24); df["hour_cos"] = np.cos(2*np.pi*ts.hour/24)
df["doy_sin"] = np.sin(2*np.pi*ts.dayofyear/365.25); df["doy_cos"] = np.cos(2*np.pi*ts.dayofyear/365.25)
df["month"] = ts.month; df["is_weekend"] = (ts.dayofweek >= 5).astype(int)
df["Installed Solar DC (MW)"] = np.where(year <= 2020, 377.1, 476.1)*1.25
cs = df["Clear-sky GHI (W/m2)"]
fc["Wind Speed 80m (m/s)"] = fc["Wind Speed 10m (m/s)"].clip(lower=0.01)*8**(np.log(fc["Wind Speed 100m (m/s)"].clip(lower=0.01)/fc["Wind Speed 10m (m/s)"].clip(lower=0.01))/np.log(10))
fc["Clear-sky Index"] = np.where(cs >= 20, (fc["GHI (W/m2)"]/cs).clip(0, 1.5), np.nan)
fc["Installed Solar DC (MW)"] = df["Installed Solar DC (MW)"]
CAL = ["hour_sin","hour_cos","doy_sin","doy_cos","month","is_weekend"]
WX_D = ["GHI (W/m2)","Temperature (C)","Wind Speed 10m (m/s)","Wind Speed 100m (m/s)","Wind Speed 80m (m/s)"]
WX_R = ["GHI (W/m2)","Temperature (C)","DNI (W/m2)","DHI (W/m2)","Installed Solar DC (MW)","Cloud Cover (%)","Clear-sky Index"]
def lags(y, p):
    o = pd.DataFrame(index=y.index)
    o[p+"lag24"], o[p+"lag48"], o[p+"lag168"] = y.shift(24), y.shift(48), y.shift(168)
    o[p+"roll7"] = pd.concat([y.shift(24*k) for k in range(1, 8)], axis=1).mean(axis=1)
    return o
X = pd.DataFrame(index=ts)
for c in sorted(set(WX_D + WX_R)): X["fc_"+c] = fc[c]
for c in CAL: X[c] = df[c]
X["Clear-sky GHI (W/m2)"] = cs
X = pd.concat([X, lags(df["Demand (MWh)"].astype(float), "d_"), lags(df["Site Renewable Output Modeled (MWh)"].astype(float), "r_")], axis=1)
X = X[(year == 2026) & (ts <= df["Demand (MWh)"].last_valid_index())]
X.index.name = "timestamp_utc"
X.round(4).to_csv(OUT)
print(X.shape, X.index.min(), X.index.max())
