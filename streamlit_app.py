"""
Maya Forecasting System: Streamlit app
Day-ahead forecasts of PACE demand, Milford-area renewable output and net load.
Maui Hiva Innovations capstone, Purdue University.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent


def fpath(name):
    """Finds a data or model file whether it sits in data/, models/ or next to this script."""
    for sub in ("data", "models", ""):
        f = ROOT / sub / name
        if f.exists():
            return f
    return ROOT / name

SOLAR_AC_MW = 476.1
LOCAL_TZ = "America/Denver"
C_ACTUAL, C_MAYA, C_BASE, C_SCEN = "#1f1f1f", "#2a78d6", "#eb6834", "#0f8a6a"

st.set_page_config(page_title="Maya Forecasting System", page_icon="⚡", layout="wide")


# ----------------------------------------------------------------------------- data
@st.cache_data
def load_forecasts():
    df = pd.read_csv(fpath("final_forecasts_hourly.csv"), parse_dates=["timestamp_utc"])
    df = df.set_index("timestamp_utc")
    df = df[df.index <= df["Demand Actual (MWh)"].last_valid_index()]
    local = df.index.tz_localize("UTC").tz_convert(LOCAL_TZ)
    df["local_hour"] = local.hour
    df["local_month"] = local.month
    df["local_date"] = local.date
    df["Baseload (MWh)"] = df["Renewable Actual (MWh)"] - df["Solar Actual (MWh)"]
    return df


@st.cache_data
def load_metrics():
    return pd.read_csv(fpath("final_metrics.csv"))


@st.cache_data
def load_features():
    return pd.read_csv(fpath("scenario_features_2026.csv"), parse_dates=["timestamp_utc"]).set_index("timestamp_utc")


@st.cache_resource
def load_models():
    return (joblib.load(fpath("demand_gbm.joblib")),
            joblib.load(fpath("renewable_gbm.joblib")),
            json.loads((fpath("model_card.json")).read_text()))


df = load_forecasts()
met = load_metrics()

SERIES = {
    "Demand": ("Demand Actual (MWh)", "Demand Forecast (MWh)", "Demand PACE Forecast (MWh)", "PACE day-ahead forecast"),
    "Renewable output": ("Renewable Actual (MWh)", "Renewable Forecast (MWh)", "Renewable Persistence (MWh)", "Persistence (same hour yesterday)"),
    "Net load": ("Net Load Actual (MWh)", "Net Load Forecast (MWh)", "Net Load Baseline (MWh)", "PACE forecast minus persistence"),
}
PERIODS = {
    "2024-25 backtest": "A backtest 2024-25",
    "2026 forward test": "B forward test 2026",
}


def mae(a, b):
    k = pd.concat([a, b], axis=1).dropna()
    return float((k.iloc[:, 0] - k.iloc[:, 1]).abs().mean()), len(k)


def fmt(x, d=0):
    return f"{x:,.{d}f}"


# ----------------------------------------------------------------------------- header
st.title("Maya Forecasting System")
st.caption("Day-ahead forecasts of PacifiCorp East (PACE) demand, Milford, Utah renewable output and net load. "
           "Every forecast uses only information available the day before: ECMWF weather forecasts and history at least 24 hours old.")

with st.sidebar:
    st.header("Settings")
    period_label = st.radio("Evaluation period", list(PERIODS), index=0,
                            help="2024-25 is the representative backtest. 2026 runs January to early September and is summer-heavy.")
    period = PERIODS[period_label]
    st.divider()
    st.markdown("**How to read this**")
    st.markdown("- **Actual**: what happened\n- **Maya**: our day-ahead forecast\n- **Baseline**: what we compare against")
    st.caption("Renewable output is modeled from pvlib tracker physics and calibrated monthly to EIA-923 reported generation.")

P = df[df["period"] == period]

tab_over, tab_explore, tab_miss, tab_solar, tab_whatif, tab_drivers, tab_data = st.tabs(
    ["Overview", "Forecast explorer", "Where PACE misses", "Added solar", "Live what-if model", "What drives the forecast", "Data and method"])

# ----------------------------------------------------------------------------- overview
with tab_over:
    m = met[met["period"] == period].set_index("series")
    c1, c2, c3, c4 = st.columns(4)
    for col, name, label in [(c1, "Demand", "Demand error vs PACE"), (c2, "Renewable", "Renewable error vs persistence"),
                             (c3, "Net load", "Net load error vs baseline")]:
        r = m.loc[name]
        col.metric(label, f"{fmt(r.MAE, 1)} MWh", f"{r.MAE_improvement_pct:.1f}% better than baseline ({fmt(r.baseline_MAE, 1)})")
    c4.metric("Demand MAPE", f"{m.loc['Demand'].MAPE_pct:.1f}%", f"{fmt(m.loc['Demand'].hours)} hours scored", delta_color="off")

    st.subheader("Daily demand: actual vs Maya vs PACE")
    d = P.groupby("local_date")[["Demand Actual (MWh)", "Demand Forecast (MWh)", "Demand PACE Forecast (MWh)"]].sum(min_count=24)
    fig = go.Figure()
    fig.add_scatter(x=d.index, y=d.iloc[:, 0], name="Actual", line=dict(color=C_ACTUAL, width=1.6))
    fig.add_scatter(x=d.index, y=d.iloc[:, 1], name="Maya", line=dict(color=C_MAYA, width=1.4))
    fig.add_scatter(x=d.index, y=d.iloc[:, 2], name="PACE", line=dict(color=C_BASE, width=1.2, dash="dot"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="MWh per day",
                      legend=dict(orientation="h", y=1.08), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Model comparison")
    cmp = m.loc[["Demand", "Demand (GBM only)", "Demand (LSTM only)", "Renewable", "Net load"],
                ["hours", "MAE", "baseline_MAE", "MAE_improvement_pct", "RMSE", "MAPE_pct"]]
    cmp.columns = ["Hours", "MAE (MWh)", "Baseline MAE", "Improvement %", "RMSE (MWh)", "MAPE %"]
    cmp.index = ["Demand (blend)", "Demand, GBM only", "Demand, LSTM only", "Renewable output", "Net load"]
    st.dataframe(cmp, width="stretch")
    st.caption("The demand forecast averages a gradient boosting model and an LSTM. The blend beats either one alone.")

# ----------------------------------------------------------------------------- explorer
with tab_explore:
    c1, c2, c3 = st.columns([1.2, 1, 1])
    series = c1.selectbox("Series", list(SERIES))
    res = c2.radio("Resolution", ["Daily", "Hourly"], horizontal=True)
    dates = sorted(P["local_date"].unique())
    rng = c3.date_input("Date range (local)", (dates[0], dates[-1]), min_value=dates[0], max_value=dates[-1])
    a, p, b, blabel = SERIES[series]
    if isinstance(rng, tuple) and len(rng) == 2:
        S = P[(P["local_date"] >= rng[0]) & (P["local_date"] <= rng[1])]
        if res == "Daily":
            S = S.groupby("local_date")[[a, p, b]].sum(min_count=24)
            x = S.index
        else:
            x = S.index.tz_localize("UTC").tz_convert(LOCAL_TZ)
        fig = go.Figure()
        fig.add_scatter(x=x, y=S[a], name="Actual", line=dict(color=C_ACTUAL, width=1.8))
        fig.add_scatter(x=x, y=S[p], name="Maya", line=dict(color=C_MAYA, width=1.5))
        fig.add_scatter(x=x, y=S[b], name=blabel, line=dict(color=C_BASE, width=1.2, dash="dot"))
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                          yaxis_title="MWh" + (" per day" if res == "Daily" else ""), legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, width="stretch")
        m1, n1 = mae(S[a], S[p])
        m2, _ = mae(S[a], S[b])
        k1, k2, k3 = st.columns(3)
        k1.metric("Maya MAE in range", f"{fmt(m1, 1)} MWh")
        k2.metric("Baseline MAE in range", f"{fmt(m2, 1)} MWh")
        k3.metric("Maya improvement", f"{(m2 - m1) / m2 * 100:.1f}%" if m2 else "n/a", f"{fmt(n1)} {'days' if res == 'Daily' else 'hours'}", delta_color="off")
    else:
        st.info("Pick a start and end date.")

# ----------------------------------------------------------------------------- PACE misses
with tab_miss:
    st.markdown("PACE's own day-ahead forecast is good most of the year but runs high in summer, especially in the evening. "
                "Positive values mean the forecast was **too high**.")
    E = P.dropna(subset=["Demand Actual (MWh)", "Demand Forecast (MWh)", "Demand PACE Forecast (MWh)"]).copy()
    E["PACE bias"] = E["Demand PACE Forecast (MWh)"] - E["Demand Actual (MWh)"]
    E["Maya bias"] = E["Demand Forecast (MWh)"] - E["Demand Actual (MWh)"]
    E["PACE abs"] = E["PACE bias"].abs()
    E["Maya abs"] = E["Maya bias"].abs()

    mon = E.groupby("local_month")[["PACE abs", "Maya abs", "PACE bias", "Maya bias"]].mean()
    mon.index = pd.to_datetime(mon.index, format="%m").strftime("%b")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Average error by month")
        fig = go.Figure()
        fig.add_bar(x=mon.index, y=mon["PACE abs"], name="PACE", marker_color=C_BASE)
        fig.add_bar(x=mon.index, y=mon["Maya abs"], name="Maya", marker_color=C_MAYA)
        fig.update_layout(barmode="group", height=340, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="MAE (MWh)",
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("Average bias by hour (local)")
        hr = E.groupby("local_hour")[["PACE bias", "Maya bias"]].mean()
        fig = go.Figure()
        fig.add_scatter(x=hr.index, y=hr["PACE bias"], name="PACE", line=dict(color=C_BASE, width=2))
        fig.add_scatter(x=hr.index, y=hr["Maya bias"], name="Maya", line=dict(color=C_MAYA, width=2))
        fig.add_hline(y=0, line_color="#999", line_width=1)
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Hour of day", yaxis_title="Forecast minus actual (MWh)",
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")

    st.subheader("PACE bias heatmap: month by hour")
    hm = E.pivot_table(index="local_month", columns="local_hour", values="PACE bias", aggfunc="mean")
    hm.index = pd.to_datetime(hm.index, format="%m").strftime("%b")
    lim = float(np.nanmax(np.abs(hm.values)))
    fig = go.Figure(go.Heatmap(z=hm.values, x=hm.columns, y=hm.index, colorscale="RdBu_r", zmin=-lim, zmax=lim,
                               colorbar=dict(title="MWh")))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Hour of day (local)")
    st.plotly_chart(fig, width="stretch")

    big = 500
    k1, k2, k3 = st.columns(3)
    k1.metric(f"Hours PACE misses by over {big} MWh", f"{(E['PACE abs'] > big).mean() * 100:.1f}%")
    k2.metric(f"Hours Maya misses by over {big} MWh", f"{(E['Maya abs'] > big).mean() * 100:.1f}%")
    over = E.loc[E["PACE abs"] > big, "PACE bias"]
    k3.metric("Share of PACE's big misses that are over-forecasts", f"{(over > 0).mean() * 100:.0f}%" if len(over) else "n/a")

# ----------------------------------------------------------------------------- added solar
with tab_solar:
    st.markdown("How would more solar in the Milford area change what the rest of PACE's system must supply? "
                "Added solar follows the same hourly shape as the existing 476 MW of Milford solar.")
    c1, c2 = st.columns([1, 1])
    added = c1.slider("Added solar capacity (MW AC)", 0, 3000, 500, 50)
    months = c2.multiselect("Months", list(range(1, 13)), default=[6, 7, 8],
                            format_func=lambda x: pd.Timestamp(2026, x, 1).strftime("%b"))
    Q = P[P["local_month"].isin(months)].dropna(subset=["Net Load Actual (MWh)", "Solar Actual (MWh)"]).copy()
    if Q.empty:
        st.info("Pick at least one month with data in this period.")
    else:
        Q["Added solar (MWh)"] = Q["Solar Actual (MWh)"] / SOLAR_AC_MW * added
        Q["Net load with added solar"] = Q["Net Load Actual (MWh)"] - Q["Added solar (MWh)"]
        Q["Share with added solar"] = (Q["Renewable Actual (MWh)"] + Q["Added solar (MWh)"]) / Q["Demand Actual (MWh)"] * 100
        prof = Q.groupby("local_hour")[["Demand Actual (MWh)", "Net Load Actual (MWh)", "Net load with added solar"]].mean()
        fig = go.Figure()
        fig.add_scatter(x=prof.index, y=prof.iloc[:, 0], name="Demand", line=dict(color=C_ACTUAL, width=1.5, dash="dot"))
        fig.add_scatter(x=prof.index, y=prof.iloc[:, 1], name="Net load today", line=dict(color=C_BASE, width=2))
        fig.add_scatter(x=prof.index, y=prof.iloc[:, 2], name=f"Net load with +{added:,} MW", line=dict(color=C_SCEN, width=2.4))
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Hour of day (local)",
                          yaxis_title="Average MWh", legend=dict(orientation="h", y=1.08), hovermode="x unified")
        st.plotly_chart(fig, width="stretch")
        base_peak, new_peak = prof.iloc[:, 1].max(), prof.iloc[:, 2].max()
        mid = prof.loc[9:16]  # midday hours, local
        base_min, new_min = mid.iloc[:, 1].min(), mid.iloc[:, 2].min()
        ramp_base = base_peak - base_min
        ramp_new = new_peak - new_min
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Average evening peak", f"{fmt(new_peak)} MWh", f"{fmt(new_peak - base_peak)} MWh", delta_color="inverse")
        k2.metric("Average midday low (9 am to 4 pm)", f"{fmt(new_min)} MWh", f"{fmt(new_min - base_min)} MWh", delta_color="off")
        k3.metric("Midday-to-peak ramp", f"{fmt(ramp_new)} MWh", f"{fmt(ramp_new - ramp_base)} MWh", delta_color="inverse")
        k4.metric("Milford renewable share", f"{Q['Share with added solar'].mean():.1f}%",
                  f"{Q['Share with added solar'].mean() - Q['Renewable Share Actual (%)'].mean():+.1f} pts")
        st.caption("Solar stops producing before the evening peak, so added solar lowers the midday trough much more than the peak "
                   "and makes the evening ramp steeper. That is why storage is the natural next question.")

# ----------------------------------------------------------------------------- live what-if
with tab_whatif:
    st.markdown("This tab **reruns the production models live**. Pick a 2026 day, change the weather forecast, and see how "
                "the demand and renewable forecasts respond.")
    try:
        g_dem, g_ren, card = load_models()
        F = load_features()
    except Exception as e:  # keeps the rest of the app usable if a model file fails to load
        st.error(f"Could not load the models: {e}")
        F = None
    if F is not None:
        F_local = F.index.tz_localize("UTC").tz_convert(LOCAL_TZ)
        cnt = pd.Series(1, index=F_local.date).groupby(level=0).size()
        days = sorted(cnt[cnt == 24].index)  # full local days only
        c1, c2, c3 = st.columns(3)
        day = c1.selectbox("Day (local)", days, index=len(days) - 1, format_func=lambda d: d.strftime("%a %b %d, %Y"))
        dT = c2.slider("Temperature change (°F)", -15, 15, 0, 1)
        sun = c3.slider("Sunshine vs forecast (%)", 30, 130, 100, 5, help="Scales forecast irradiance. Below 100% means cloudier than forecast.")
        X = F[F_local.date == day].copy()
        fact = sun / 100

        def demand_X(X, dT, fact):
            Z = pd.DataFrame(index=X.index)
            Z["GHI (W/m2)"] = X["fc_GHI (W/m2)"] * fact
            Z["Temperature (C)"] = X["fc_Temperature (C)"] + dT * 5 / 9
            for c in ["Wind Speed 10m (m/s)", "Wind Speed 100m (m/s)", "Wind Speed 80m (m/s)"]:
                Z[c] = X["fc_" + c]
            for c in ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "month", "is_weekend"]:
                Z[c] = X[c]
            for c in ["lag24", "lag48", "lag168", "roll7"]:
                Z[c] = X["d_" + c]
            for c in ["dow", "holiday", "bridge", "school_out"]:      # v1.1 calendar inputs
                if c in X:
                    Z[c] = X[c]
            if "fc_pw_temp" in X:                                     # v1.1 population-weighted temperature
                Z["pw_temp"] = X["fc_pw_temp"] + dT * 5 / 9
            return Z[list(g_dem.feature_names_in_)]

        def renew_X(X, dT, fact):
            Z = pd.DataFrame(index=X.index)
            Z["GHI (W/m2)"] = X["fc_GHI (W/m2)"] * fact
            Z["Temperature (C)"] = X["fc_Temperature (C)"] + dT * 5 / 9
            Z["DNI (W/m2)"] = X["fc_DNI (W/m2)"] * fact
            Z["DHI (W/m2)"] = X["fc_DHI (W/m2)"]
            Z["Installed Solar DC (MW)"] = X["fc_Installed Solar DC (MW)"]
            Z["Cloud Cover (%)"] = (X["fc_Cloud Cover (%)"] + (1 - fact) * 100).clip(0, 100)
            cs = X["Clear-sky GHI (W/m2)"]
            Z["Clear-sky Index"] = np.where(cs >= 20, (Z["GHI (W/m2)"] / cs).clip(0, 1.5), np.nan)
            for c in ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "month", "is_weekend"]:
                Z[c] = X[c]
            for c in ["lag24", "lag48", "lag168", "roll7"]:
                Z[c] = X["r_" + c]
            return Z[list(g_ren.feature_names_in_)]

        base_d = g_dem.predict(demand_X(X, 0, 1.0))
        new_d = g_dem.predict(demand_X(X, dT, fact))
        base_r = np.clip(g_ren.predict(renew_X(X, 0, 1.0)), 0, None)
        new_r = np.clip(g_ren.predict(renew_X(X, dT, fact)), 0, None)
        A = df.reindex(X.index)
        hrs = X.index.tz_localize("UTC").tz_convert(LOCAL_TZ).hour

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Demand")
            fig = go.Figure()
            fig.add_scatter(x=hrs, y=A["Demand Actual (MWh)"], name="Actual", line=dict(color=C_ACTUAL, width=1.6))
            fig.add_scatter(x=hrs, y=A["Demand PACE Forecast (MWh)"], name="PACE", line=dict(color=C_BASE, width=1.2, dash="dot"))
            fig.add_scatter(x=hrs, y=base_d, name="Model, forecast weather", line=dict(color=C_MAYA, width=1.5))
            fig.add_scatter(x=hrs, y=new_d, name="Model, your scenario", line=dict(color=C_SCEN, width=2.4))
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Hour (local)", yaxis_title="MWh",
                              legend=dict(orientation="h", y=1.15), hovermode="x unified")
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.subheader("Milford renewable output")
            fig = go.Figure()
            fig.add_scatter(x=hrs, y=A["Renewable Actual (MWh)"], name="Actual (modeled)", line=dict(color=C_ACTUAL, width=1.6))
            fig.add_scatter(x=hrs, y=base_r, name="Model, forecast weather", line=dict(color=C_MAYA, width=1.5))
            fig.add_scatter(x=hrs, y=new_r, name="Model, your scenario", line=dict(color=C_SCEN, width=2.4))
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Hour (local)", yaxis_title="MWh",
                              legend=dict(orientation="h", y=1.15), hovermode="x unified")
            st.plotly_chart(fig, width="stretch")
        k1, k2, k3 = st.columns(3)
        k1.metric("Daily demand, your scenario", f"{fmt(new_d.sum())} MWh", f"{fmt(new_d.sum() - base_d.sum())} MWh vs forecast weather", delta_color="off")
        k2.metric("Daily renewable, your scenario", f"{fmt(new_r.sum())} MWh", f"{fmt(new_r.sum() - base_r.sum())} MWh", delta_color="off")
        k3.metric("Peak net load, your scenario", f"{fmt((new_d - new_r).max())} MWh",
                  f"{fmt((new_d - new_r).max() - (base_d - base_r).max())} MWh", delta_color="off")
        st.caption("Uses the production gradient boosting models (trained through 2025). The LSTM half of the demand blend needs a full "
                   "168-hour weather sequence, so it is left out here; the blend shown elsewhere in the app is slightly more accurate.")

# ----------------------------------------------------------------------------- drivers
with tab_drivers:
    st.markdown("Which inputs matter? We scrambled one input column at a time and measured how much worse the forecast got. "
                "A big jump means the model leans on that input; almost no change means it barely uses it.")
    imp = pd.read_csv(fpath("feature_importance.csv"))
    which = st.radio("Model", ["Demand", "Renewable output"], horizontal=True, key="drivers_model")
    I = imp[imp["model"] == which].sort_values("added_error_mwh")
    normal = float(I["normal_mae_mwh"].iloc[0])
    top2 = set(I.nlargest(2, "added_error_mwh")["input"])
    fig = go.Figure(go.Bar(
        x=I["added_error_mwh"], y=I["input"], orientation="h",
        marker_color=[C_MAYA if n in top2 else "#b9b6aa" for n in I["input"]],
        text=[f"+{v:,.1f}" for v in I["added_error_mwh"]], textposition="outside",
        hovertemplate="%{y}: +%{x:,.1f} MWh error if scrambled<extra></extra>"))
    fig.update_layout(height=60 + 34 * len(I), margin=dict(l=10, r=40, t=10, b=10),
                      xaxis_title=f"Extra error (MWh) when scrambled. Normal error: {normal:,.1f} MWh")
    st.plotly_chart(fig, width="stretch")
    if which == "Demand":
        st.markdown("**Two inputs do most of the work.** Demand at the same hour yesterday sets the starting level, and the "
                    "forecast temperature says how far tomorrow will move from it, mostly through air conditioning on hot days. "
                    "That pairing is why Maya catches the summer evenings PACE over-forecasts. Wind, rain, humidity, snow and cloud "
                    "cover add almost nothing for demand.")
    else:
        st.markdown("**Sunlight is everything for renewable output.** The forecast of sunlight reaching the ground (GHI) is by far "
                    "the strongest input, followed by direct and scattered sunlight. Day-ahead cloud forecasts are the main source "
                    "of renewable error.")
    st.caption("Permutation test on the production gradient boosting models, January to early September 2026. The LSTM half of "
               "the demand blend cannot be scored column by column the same way. Built by compute_importance.py.")

# ----------------------------------------------------------------------------- data and method
with tab_data:
    st.subheader("How the forecasts are made")
    st.markdown("""
- **Demand:** average of a gradient boosting model and an LSTM. Inputs are next-day ECMWF weather forecasts (temperature, irradiance, wind),
  calendar features and demand from 24, 48 and 168 hours earlier. Trained on 2019 and 2021 to 2025; 2020 is excluded because of COVID.
- **Renewable output:** gradient boosting on forecast irradiance (GHI, DNI, DHI), cloud cover, temperature, clear-sky index and
  output lags. Trained on past ECMWF forecasts, not reanalysis, so it learns how much to trust a forecast.
- **Net load:** demand minus Milford-area renewable output.
- **Out-of-sample:** no model was scored on hours it trained on. 2024-25 uses models trained through 2023.
""")
    st.subheader("Data sources")
    st.markdown("EIA-930 (PACE demand and PACE forecast) · ERA5 and ERA5-Land (Copernicus) · CAMS McClear clear-sky · "
                "ECMWF IFS day-ahead forecasts via Open-Meteo · EIA-923 monthly plant generation · EPA eGRID")
    st.subheader("Download")
    c1, c2 = st.columns(2)
    c1.download_button("Hourly forecasts (CSV)", (fpath("final_forecasts_hourly.csv")).read_bytes(),
                       "final_forecasts_hourly.csv", "text/csv")
    c2.download_button("Accuracy metrics (CSV)", (fpath("final_metrics.csv")).read_bytes(),
                       "final_metrics.csv", "text/csv")
    with st.expander("Preview hourly data"):
        st.dataframe(P.drop(columns=["local_date"]).head(500), width="stretch")
