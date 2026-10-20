"""v1.1 input helpers shared by prep_features.py (the app's what-if rows). Same definitions as final_models_v1_1.py."""
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

CITY_W = {"slc": 0.38, "ogden": 0.14, "provo": 0.14, "logan": 0.05, "cedar": 0.04, "rexburg": 0.07, "casper": 0.10, "rocksp": 0.08}


def calendar_extra(idx):
    loc = idx - pd.Timedelta(hours=7)
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


def _temp(raw):
    w = pd.Series(CITY_W) / sum(CITY_W.values())
    return pd.Series(raw[[f"{k}_temperature_2m" for k in w.index]].values @ w.values, index=raw.index)


def _read(name):
    c = pd.read_csv(name, index_col=0, parse_dates=True)
    c.index = c.index.tz_localize(None) if c.index.tz is not None else c.index
    return c


def pw_temp_inputs(idx, win=60, delay=8):
    """(actual, forecast) population-weighted temperature; the forecast is shifted by its trailing bias vs ERA5."""
    a = _temp(_read("cities_era5_2019_2026.csv.gz")).reindex(idx)
    f = _temp(_read("cities_ecmwf_dayahead_2024_2026.csv.gz")).reindex(idx)
    d, b = f - a, pd.Series(np.nan, index=idx)
    for h in range(24):
        sel = idx.hour == h
        b[sel] = d[sel].rolling(win, min_periods=20).mean().shift(delay).values
    f = f - b.ffill().fillna(0.0)
    return a.fillna(f), f
