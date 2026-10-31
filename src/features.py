"""Shared, leakage-safe feature builder used by BOTH capacity_model.py (emz) and delay_model.py (rach).

Every feature for a prediction issued at UTC time T uses only:
  - weather_timeline rows with obs_utc <= T
  - flights scheduled at least `buffer_h` hours before T (treated as completed)

Typical use:
    issues = make_issues(flights, horizons=[3, 6, 12])        # one row per flight x horizon
    X = build_features(issues, flights, timeline)               # weather + congestion + calendar
    # join X back to labels on (flight_id, side, horizon_h)

flight_id is shared by both ends of a flight between two selected airports (e.g. ATL->DFW),
so every join keys on (flight_id, side).

Data-quality handling baked in (verified on the provided CSV, ATL/DFW subset):
  - wpgt (gust) is 0 in ~98% of rows = "no gust reported", so 0 -> NaN and gust features use
    max(gust, wind speed).
  - snow depth is always 0 at ATL/DFW -> not used; snow comes from wx_group instead.
  - coco 0 is not a valid code -> wx_group 'unknown'.
"""
import numpy as np
import pandas as pd

WX_GROUPS = ["storm", "rain", "fog", "snow", "freezing"]


def _ns(s):
    """pandas 2/3 mix datetime resolutions (ns vs us); merge_asof refuses mismatches."""
    return pd.to_datetime(s).astype("datetime64[ns]")


def make_issues(flights, horizons=(3, 6, 12)):
    """One row per flight x horizon with issued_utc = sched_utc - h."""
    base = flights[["flight_id", "side", "airport", "sched_utc"]].copy()
    base["sched_utc"] = _ns(base["sched_utc"])
    out = []
    for h in horizons:
        b = base.copy()
        b["horizon_h"] = h
        b["issued_utc"] = b["sched_utc"] - pd.Timedelta(hours=h)
        out.append(b)
    return pd.concat(out, ignore_index=True)


def _timeline_rolling(timeline):
    """Per-airport rolling windows over the observation timeline (computed at each obs time)."""
    tl = timeline.copy()
    tl["obs_utc"] = _ns(tl["obs_utc"])
    tl["gust_kt"] = tl["wpgt_kt"].where(tl["wpgt_kt"] > 0)
    tl["gust_or_wind_kt"] = np.fmax(tl["gust_kt"], tl["wspd_kt"])
    if "gust_crosswind_kt" in tl:
        tl["gust_xw_kt"] = np.fmax(tl["gust_crosswind_kt"].where(tl["wpgt_kt"] > 0), tl["crosswind_kt"])
    else:
        tl["gust_xw_kt"] = tl["crosswind_kt"]
    for g in WX_GROUPS:
        tl[f"is_{g}"] = (tl["wx_group"] == g).astype(float)
    parts = []
    for ap, g in tl.groupby("airport"):
        g = g.sort_values("obs_utc").set_index("obs_utc")
        r = pd.DataFrame(index=g.index)
        for w in ("6h", "12h"):
            roll = g.rolling(w)
            r[f"xw_max_{w}"] = roll["crosswind_kt"].max()
            r[f"gust_xw_max_{w}"] = roll["gust_xw_kt"].max()
            r[f"wind_max_{w}"] = roll["gust_or_wind_kt"].max()
            r[f"prcp_sum_{w}"] = roll["prcp_mm"].sum()
            r[f"prcp_max_{w}"] = roll["prcp_mm"].max()
            for grp in WX_GROUPS:
                r[f"n_{grp}_{w}"] = roll[f"is_{grp}"].sum()
        first = lambda x: x[0]  # noqa: E731
        r["pres_chg_3h"] = g["pres_hpa"] - g["pres_hpa"].rolling("3h").apply(first, raw=True)
        r["pres_chg_6h"] = g["pres_hpa"] - g["pres_hpa"].rolling("6h").apply(first, raw=True)
        r["n_obs_12h"] = g["temp_c"].rolling("12h").count()
        last = g[["temp_c", "prcp_mm", "wspd_kt", "pres_hpa", "crosswind_kt", "headwind_kt",
                  "tailwind_kt", "wind_dir_known", "wx_group"]].add_prefix("last_")
        parts.append(pd.concat([last, r], axis=1).assign(airport=ap).reset_index())
    out = pd.concat(parts, ignore_index=True)
    out["obs_utc"] = _ns(out["obs_utc"])
    return out.sort_values("obs_utc")


def weather_asof(issues, timeline):
    """Latest timeline state at or before issued_utc, per airport. Adds obs_age_h."""
    roll = _timeline_rolling(timeline)
    left = issues.copy()
    left["issued_utc"] = _ns(left["issued_utc"])
    left = left.sort_values("issued_utc")
    m = pd.merge_asof(left, roll, left_on="issued_utc", right_on="obs_utc",
                      by="airport", direction="backward")
    m["obs_age_h"] = (m["issued_utc"] - m["obs_utc"]).dt.total_seconds() / 3600
    # last obs from yesterday is stale; let the model know rather than silently trusting it
    m["last_obs_stale"] = (m["obs_age_h"] > 12).astype(int)
    return m.drop(columns=["obs_utc"])


def congestion_asof(issues, flights, lookback_h=12, buffer_h=3):
    """Share of stressed flights at the airport among flights scheduled in
    [T - buffer - lookback, T - buffer]. buffer_h=3 so that a flight's delay outcome is very
    likely known by T (a flight scheduled 1h before T could still be sitting on the ground)."""
    f = flights[["airport", "sched_utc", "stress_flag"]].copy()
    f["sched_utc"] = _ns(f["sched_utc"])
    parts = []
    for ap, g in f.groupby("airport"):
        s = g.sort_values("sched_utc").set_index("sched_utc")["stress_flag"].astype(float)
        r = s.rolling(f"{lookback_h}h").agg(["sum", "count"])
        parts.append(pd.DataFrame({"airport": ap, "t": r.index, "cong_sum": r["sum"].values,
                                   "cong_n": r["count"].values}))
    cong = pd.concat(parts).sort_values("t")
    cong["t"] = _ns(cong["t"])
    left = issues[["flight_id", "side", "horizon_h", "airport", "issued_utc"]].copy()
    left["tc"] = _ns(left["issued_utc"]) - pd.Timedelta(hours=buffer_h)
    left = left.sort_values("tc")
    m = pd.merge_asof(left, cong, left_on="tc", right_on="t", by="airport", direction="backward")
    stale = (m["tc"] - m["t"]) > pd.Timedelta(hours=lookback_h)
    m.loc[stale, ["cong_sum", "cong_n"]] = 0
    m["cong_rate_12h"] = m["cong_sum"] / m["cong_n"].replace(0, np.nan)
    return m[["flight_id", "side", "horizon_h", "cong_rate_12h", "cong_n"]]


def calendar_features(flights):
    c = flights[["flight_id", "side"]].copy()
    sl = pd.to_datetime(flights["sched_local"])
    c["hour_local"] = sl.dt.hour
    c["dow"] = sl.dt.dayofweek
    c["month"] = sl.dt.month
    c["is_dep"] = (flights["side"] == "dep").astype(int)
    return c


def build_features(issues, flights, timeline, with_congestion=True):
    X = weather_asof(issues, timeline)
    X = X.merge(calendar_features(flights), on=["flight_id", "side"], how="left")
    if with_congestion:
        X = X.merge(congestion_asof(issues, flights), on=["flight_id", "side", "horizon_h"], how="left")
    X["last_wx_group"] = X["last_wx_group"].fillna("unknown").astype("category")
    X["last_wind_dir_known"] = X["last_wind_dir_known"].astype(float)
    X["airport"] = X["airport"].astype("category")
    return X.sort_values(["horizon_h", "flight_id", "side"]).reset_index(drop=True)


def feature_columns(X):
    """Model inputs: everything except keys, minus columns that are entirely empty
    (e.g. headwind/tailwind when no wind direction is available yet)."""
    drop = {"flight_id", "side", "sched_utc", "issued_utc", "horizon_h"}  # side is in is_dep
    return [c for c in X.columns if c not in drop and X[c].notna().any()]
