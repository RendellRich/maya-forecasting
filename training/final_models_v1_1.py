"""
v1.1 production models (Oct 20): the Oct 18 v1 models plus two input upgrades found while building Maya v2.
  1. Calendar: day of week, holidays (federal + Utah Pioneer Day, Christmas Eve, day after Thanksgiving), bridge days,
     Utah school breaks.
  2. Population-weighted temperature over eight PACE load centers (Salt Lake City, Ogden, Provo, Logan, Cedar City,
     Rexburg, Casper, Rock Springs). ERA5 for training; ECMWF day-ahead forecasts (previous_day1) for testing, shifted by
     their trailing 60-day bias vs ERA5 (by hour of day, ending 8 days earlier so it is known at forecast time), because
     ECMWF runs about 0.9 C colder than ERA5 at these points.
Everything else (24 h minimum lag, GBM + LSTM blend, renewable model, periods) is unchanged from v1.
Run from this folder: python final_models_v1_1.py   (inputs in ./data, outputs written to the repo root)

Original v1 notes:

DEMAND: average of GBM (HistGradientBoostingRegressor, setup D3b) and LSTM (see lstm_comparison.py).
RENEWABLE: GBM (setup R2 features) trained on ECMWF forecast weather where forecast history exists.
Every forecast in the output is day-ahead and out-of-sample: target-hour weather = ECMWF forecast issued
the day before; target lags >= 24 h; the model producing each hour never trained on that hour.

Forecast periods
  A  2024-02-04 .. 2025-12-31  (backtest)
     demand:    blend trained on 2019, 2021-2023 (reanalysis weather)
     renewable: 2024 -> GBM trained on reanalysis 2019-2023
                2025 -> GBM trained on ECMWF forecasts Feb-Dec 2024
  B  2026-01-01 .. last hour with demand data (forward test)
     demand:    blend trained on 2019, 2021-2025      <- saved as the production demand model
     renewable: GBM trained on ECMWF forecasts Feb 2024 - Dec 2025   <- saved as production renewable model

Net load = demand - Milford-area renewable output. Renewable share = renewable / demand.
Inputs: master_hourly_cleaned_ALL_YEARS_v2.csv, ecmwf_dayahead_forecast_2024-2026.csv
Outputs: final_forecasts_hourly.csv, final_metrics.csv, models/ (demand_gbm.joblib, demand_lstm.pt,
         renewable_gbm.joblib, model_card.json)
"""
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

torch.manual_seed(42)
np.random.seed(42)
WIN, GAP, LSTM_EPOCHS = 168, 24, 11
HERE = os.path.dirname(os.path.abspath(__file__))
DATA, ROOT = os.path.join(HERE, "data"), os.path.dirname(HERE)
os.chdir(DATA)
os.makedirs("models", exist_ok=True)

df = pd.read_csv("master_hourly_cleaned_ALL_YEARS_v2.csv")
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"]).dt.tz_localize(None)
df = df.set_index("timestamp_utc").sort_index()
fc = pd.read_csv("ecmwf_dayahead_forecast_2024-2026.csv", parse_dates=["timestamp_utc"]).set_index("timestamp_utc").reindex(df.index)
year = df.index.year.values
ts = df.index

df["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
df["doy_sin"] = np.sin(2 * np.pi * ts.dayofyear / 365.25)
df["doy_cos"] = np.cos(2 * np.pi * ts.dayofyear / 365.25)
df["month"] = ts.month
df["is_weekend"] = (ts.dayofweek >= 5).astype(int)
df["Installed Solar DC (MW)"] = np.where(year <= 2020, 377.1, 476.1) * 1.25
cs = df["Clear-sky GHI (W/m2)"]
df["Clear-sky Index"] = np.where(cs >= 20, (df["GHI (W/m2)"] / cs).clip(0, 1.5), np.nan)
fc["Wind Speed 80m (m/s)"] = fc["Wind Speed 10m (m/s)"].clip(lower=0.01) * 8 ** (
    np.log(fc["Wind Speed 100m (m/s)"].clip(lower=0.01) / fc["Wind Speed 10m (m/s)"].clip(lower=0.01)) / np.log(10))
fc["Clear-sky Index"] = np.where(cs >= 20, (fc["GHI (W/m2)"] / cs).clip(0, 1.5), np.nan)
fc["Installed Solar DC (MW)"] = df["Installed Solar DC (MW)"]
CAL = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "month", "is_weekend"]

# ---- v1.1 additions: calendar + population-weighted weather ----
from pandas.tseries.holiday import USFederalHolidayCalendar
CITY_W = {"slc": 0.38, "ogden": 0.14, "provo": 0.14, "logan": 0.05, "cedar": 0.04, "rexburg": 0.07, "casper": 0.10, "rocksp": 0.08}


def calendar_extra(idx):
    loc = idx - pd.Timedelta(hours=7)            # Mountain Standard clock
    d = loc.normalize()
    fed = USFederalHolidayCalendar().holidays("2018-01-01", "2030-12-31")
    yrs = range(2018, 2031)
    extra = [pd.Timestamp(f"{y}-07-24") for y in yrs] + [pd.Timestamp(f"{y}-12-24") for y in yrs] + \
            [h + pd.Timedelta(days=1) for h in fed if h.month == 11 and h.day >= 22]
    hol = fed.union(pd.DatetimeIndex(extra))
    e = pd.DataFrame(index=idx)
    e["dow"] = loc.dayofweek
    e["holiday"] = d.isin(hol).astype(int)
    e["bridge"] = (d.isin(hol - pd.Timedelta(days=1)) | d.isin(hol + pd.Timedelta(days=1))).astype(int)
    md = loc.month * 100 + loc.day
    e["school_out"] = (((md >= 605) & (md <= 820)) | (md >= 1222) | (md <= 102)).astype(int)
    return e


def city_temp(raw):
    w = pd.Series(CITY_W) / sum(CITY_W.values())
    return pd.Series(raw[[f"{k}_temperature_2m" for k in w.index]].values @ w.values, index=raw.index)


def forecast_bias(f, a, win=60, delay=8):
    d, out = f - a, pd.Series(np.nan, index=f.index)
    for h in range(24):
        sel = f.index.hour == h
        out[sel] = d[sel].rolling(win, min_periods=20).mean().shift(delay).values
    return out.ffill().fillna(0.0)


def read_raw(name):
    c = pd.read_csv(name, index_col=0, parse_dates=True)
    c.index = c.index.tz_localize(None) if c.index.tz is not None else c.index
    return c


CAL_EXTRA, CITY_COLS = ["dow", "holiday", "bridge", "school_out"], ["pw_temp"]
a = city_temp(read_raw("cities_era5_2019_2026.csv.gz")).reindex(df.index)
f = city_temp(read_raw("cities_ecmwf_dayahead_2024_2026.csv.gz")).reindex(df.index)
f = f - forecast_bias(f, a)
df = df.join(calendar_extra(ts))
df["pw_temp"], fc["pw_temp"] = a.fillna(f), f      # ERA5 runs ~5 days behind: newest hours use the corrected forecast
CAL_V1 = list(CAL)
CAL = CAL + CAL_EXTRA

D, R = "Demand (MWh)", "Site Renewable Output Modeled (MWh)"
WX_D = ["GHI (W/m2)", "Temperature (C)", "Wind Speed 10m (m/s)", "Wind Speed 100m (m/s)", "Wind Speed 80m (m/s)"] + CITY_COLS
SEQ_D = ["Temperature (C)", "GHI (W/m2)", "pw_temp"]
WX_R = ["GHI (W/m2)", "Temperature (C)", "DNI (W/m2)", "DHI (W/m2)", "Installed Solar DC (MW)", "Cloud Cover (%)", "Clear-sky Index"]
GBM_ARGS = dict(max_iter=300, max_depth=6, learning_rate=0.05, random_state=42)

last_demand = df[D].last_valid_index()
A = (ts >= "2024-02-04") & (year <= 2025)
B = (year == 2026) & (ts <= last_demand)


def lags(y):
    o = pd.DataFrame(index=y.index)
    o["lag24"], o["lag48"], o["lag168"] = y.shift(24), y.shift(48), y.shift(168)
    o["roll7"] = pd.concat([y.shift(24 * k) for k in range(1, 8)], axis=1).mean(axis=1)
    return o


def with_fc(frame, cols):
    out = frame.copy()
    for c in cols:
        out[c] = fc[c]
    return out


# ------------------------------- DEMAND ------------------------------------------------------------
y = df[D].astype(float)
G_act = pd.concat([df[WX_D + CAL], lags(y)], axis=1)
G_fc = with_fc(G_act, WX_D)
seq_df = pd.concat([y.rename("y"), df[SEQ_D], df[["hour_sin", "hour_cos", "doy_sin", "doy_cos"]]], axis=1)
now_act = df[WX_D + CAL]
now_fc = with_fc(now_act, WX_D)
seq_ok = seq_df.notna().all(axis=1).astype(int).rolling(WIN).sum().shift(GAP).fillna(0).values == WIN
train_ok = y.notna().values & G_act.notna().all(axis=1).values & seq_ok
pred_ok = G_fc.notna().all(axis=1).values & seq_ok


class Net(nn.Module):
    def __init__(self, n_seq, n_now, hidden=48):
        super().__init__()
        self.lstm = nn.LSTM(n_seq, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_now, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, seq, now):
        _, (h, _) = self.lstm(seq)
        return self.head(torch.cat([h[-1], now], dim=1)).squeeze(1)


def fit_demand(train_years):
    tr = np.where(train_ok & np.isin(year, train_years))[0]
    g = HistGradientBoostingRegressor(**GBM_ARGS).fit(G_act.iloc[tr], y.values[tr])
    sc = dict(mu_s=seq_df.iloc[tr].mean(), sd_s=seq_df.iloc[tr].std().replace(0, 1),
              mu_n=now_act.iloc[tr].mean(), sd_n=now_act.iloc[tr].std().replace(0, 1))
    S_arr = torch.tensor(((seq_df - sc["mu_s"]) / sc["sd_s"]).fillna(0).values, dtype=torch.float32)
    N_act = torch.tensor(((now_act - sc["mu_n"]) / sc["sd_n"]).fillna(0).values, dtype=torch.float32)
    mu, sd = float(sc["mu_s"]["y"]), float(sc["sd_s"]["y"])
    yt = torch.tensor((y.values - mu) / sd, dtype=torch.float32)
    torch.manual_seed(42)
    net = Net(S_arr.shape[1], N_act.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    off = torch.arange(-WIN + 1, 1)
    for ep in range(LSTM_EPOCHS):
        net.train()
        perm = np.random.RandomState(ep).permutation(tr)
        for b in range(0, len(perm), 256):
            bi = perm[b:b + 256]
            seq = S_arr[(torch.tensor(bi - GAP)[:, None] + off[None, :])]
            loss = nn.functional.l1_loss(net(seq, N_act[bi]), yt[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"  demand LSTM (train {train_years[0]}-{train_years[-1]}) epoch {ep + 1}/{LSTM_EPOCHS}", flush=True)
    return g, net, sc


def predict_demand(g, net, sc, mask):
    ids = np.where(mask & pred_ok)[0]
    S_arr = torch.tensor(((seq_df - sc["mu_s"]) / sc["sd_s"]).fillna(0).values, dtype=torch.float32)
    N_fc = torch.tensor(((now_fc - sc["mu_n"]) / sc["sd_n"]).fillna(0).values, dtype=torch.float32)
    mu, sd = float(sc["mu_s"]["y"]), float(sc["sd_s"]["y"])
    off = torch.arange(-WIN + 1, 1)
    net.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(ids), 2048):
            bi = ids[b:b + 2048]
            out.append(net(S_arr[(torch.tensor(bi - GAP)[:, None] + off[None, :])], N_fc[bi]).numpy())
    pl = np.concatenate(out) * sd + mu
    pg = g.predict(G_fc.iloc[ids])
    return pd.Series((pg + pl) / 2, index=ts[ids]), pd.Series(pg, index=ts[ids]), pd.Series(pl, index=ts[ids])


print("Demand, period A model (train 2019, 2021-2023)")
gA, nA, sA = fit_demand([2019, 2021, 2022, 2023])
dA, dA_g, dA_l = predict_demand(gA, nA, sA, A)
print("Demand, period B / production model (train 2019, 2021-2025)")
gB, nB, sB = fit_demand([2019, 2021, 2022, 2023, 2024, 2025])
dB, dB_g, dB_l = predict_demand(gB, nB, sB, B)
joblib.dump(gB, os.path.join(ROOT, "demand_gbm.joblib"))
torch.save({"state_dict": nB.state_dict(), "n_seq": len(seq_df.columns), "n_now": len(now_act.columns),
            "scaler": {k: v.to_dict() for k, v in sB.items()}, "seq_cols": list(seq_df.columns),
            "now_cols": list(now_act.columns), "window": WIN, "gap": GAP}, "models/demand_lstm.pt")  # not used by the app; kept with the training data

# ------------------------------- RENEWABLE ---------------------------------------------------------
yr = df[R].astype(float)
H_act = pd.concat([df[WX_R + CAL_V1], lags(yr)], axis=1)
H_fc = with_fc(H_act, WX_R)
need = [c for c in H_act.columns if c != "Clear-sky Index"]
ok_act = H_act[need].notna().all(axis=1).values & yr.notna().values
ok_fc = H_fc[need].notna().all(axis=1).values


def fit_r(X, mask):
    return HistGradientBoostingRegressor(**GBM_ARGS).fit(X[mask], yr.values[mask])


fc_hist = ok_fc & yr.notna().values & np.asarray(ts >= "2024-02-04")
r24 = fit_r(H_act, ok_act & np.isin(year, range(2019, 2024)))
r25 = fit_r(H_fc, fc_hist & (year == 2024))
r26 = fit_r(H_fc, fc_hist & np.isin(year, [2024, 2025]))
joblib.dump(r26, os.path.join(ROOT, "renewable_gbm.joblib"))
ren = pd.Series(np.nan, index=ts)
for mdl, m in [(r24, A & (year == 2024)), (r25, A & (year == 2025)), (r26, B)]:
    ids = np.where(m & ok_fc)[0]
    ren.iloc[ids] = np.clip(mdl.predict(H_fc.iloc[ids]), 0, None)

# ------------------------------- ASSEMBLE ----------------------------------------------------------
out = pd.DataFrame(index=ts[A | B])
out["period"] = np.where(out.index.year == 2026, "B forward test 2026", "A backtest 2024-25")
out["Demand Actual (MWh)"] = df[D]
out["Demand PACE Forecast (MWh)"] = df["Day-Ahead Demand Forecast (MWh)"]
out["Demand Forecast (MWh)"] = pd.concat([dA, dB])
out["Demand Forecast GBM (MWh)"] = pd.concat([dA_g, dB_g])
out["Demand Forecast LSTM (MWh)"] = pd.concat([dA_l, dB_l])
out["Renewable Actual (MWh)"] = df[R]
out["Renewable Forecast (MWh)"] = ren
out["Renewable Persistence (MWh)"] = df[R].shift(24)
out["Solar Actual (MWh)"] = df["Solar Output Modeled (MWh)"]
out["Net Load Actual (MWh)"] = out["Demand Actual (MWh)"] - out["Renewable Actual (MWh)"]
out["Net Load Forecast (MWh)"] = out["Demand Forecast (MWh)"] - out["Renewable Forecast (MWh)"]
out["Net Load Baseline (MWh)"] = out["Demand PACE Forecast (MWh)"] - out["Renewable Persistence (MWh)"]
out["Renewable Share Actual (%)"] = out["Renewable Actual (MWh)"] / out["Demand Actual (MWh)"] * 100
out["Renewable Share Forecast (%)"] = out["Renewable Forecast (MWh)"] / out["Demand Forecast (MWh)"] * 100
out["Renewable Target Calibration"] = df["Renewable Target Calibration"]
out.index.name = "timestamp_utc"
out.round(3).reset_index().to_csv(os.path.join(ROOT, "final_forecasts_hourly.csv"), index=False)

rows = []
for per in ["A backtest 2024-25", "B forward test 2026"]:
    o = out[out.period == per]
    for name, a, p, b in [("Demand", "Demand Actual (MWh)", "Demand Forecast (MWh)", "Demand PACE Forecast (MWh)"),
                          ("Demand (GBM only)", "Demand Actual (MWh)", "Demand Forecast GBM (MWh)", "Demand PACE Forecast (MWh)"),
                          ("Demand (LSTM only)", "Demand Actual (MWh)", "Demand Forecast LSTM (MWh)", "Demand PACE Forecast (MWh)"),
                          ("Renewable", "Renewable Actual (MWh)", "Renewable Forecast (MWh)", "Renewable Persistence (MWh)"),
                          ("Net load", "Net Load Actual (MWh)", "Net Load Forecast (MWh)", "Net Load Baseline (MWh)")]:
        k = o[[a, p, b]].dropna()
        mae, bmae = (k[p] - k[a]).abs().mean(), (k[b] - k[a]).abs().mean()
        rmse, brmse = np.sqrt(((k[p] - k[a]) ** 2).mean()), np.sqrt(((k[b] - k[a]) ** 2).mean())
        rows.append(dict(period=per, series=name, hours=len(k), start=str(k.index.min()), end=str(k.index.max()),
                         MAE=round(mae, 2), baseline_MAE=round(bmae, 2), MAE_improvement_pct=round((bmae - mae) / bmae * 100, 1),
                         RMSE=round(rmse, 2), baseline_RMSE=round(brmse, 2), RMSE_improvement_pct=round((brmse - rmse) / brmse * 100, 1),
                         MAPE_pct=round(((k[p] - k[a]).abs() / k[a].clip(lower=1)).mean() * 100, 2)))
met = pd.DataFrame(rows)
met.to_csv(os.path.join(ROOT, "final_metrics.csv"), index=False)
print(met[["period", "series", "hours", "MAE", "baseline_MAE", "MAE_improvement_pct", "MAPE_pct"]].to_string(index=False))

json.dump({
    "demand": {"type": "average of HistGradientBoostingRegressor and LSTM", "trained_on": "2019, 2021-2025 (2020 excluded, COVID)",
               "version": "v1.1 (Oct 20, 2026)", "inputs": "target-hour weather forecast (GHI, temperature, wind 10/80/100 m at Milford; population-weighted temperature over 8 PACE load centers, bias-corrected to ERA5) + calendar (hour, season, weekday, holidays, bridge days, Utah school breaks) + demand lags 24/48/168 h + 7-day same-hour mean; LSTM: past 168 h ending 24 h before target",
               "files": ["demand_gbm.joblib", "demand_lstm.pt"]},
    "renewable": {"type": "HistGradientBoostingRegressor", "trained_on": "ECMWF day-ahead forecast weather, Feb 2024 - Dec 2025",
                  "inputs": "target-hour forecast GHI, DNI, DHI, temperature, cloud cover, clear-sky index + installed capacity + calendar + output lags",
                  "files": ["renewable_gbm.joblib"]},
    "weather_inputs_in_production": "ECMWF IFS 0.25 deg day-ahead forecast (Open-Meteo previous-runs API, ecmwf_ifs025, *_previous_day1), point 38.4167 N, -113.0167 E",
    "metrics_file": "final_metrics.csv",
}, open(os.path.join(ROOT, "model_card.json"), "w"), indent=2)
print("Saved: final_forecasts_hourly.csv, final_metrics.csv, models/")
