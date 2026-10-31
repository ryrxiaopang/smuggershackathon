#!/usr/bin/env python3
"""Flight-level delay-risk classifier (branch: rach).

P(flight delayed > 45 min) issued 3, 6 and 12 h before its scheduled time,
for any airport/side present in the input (ATL/DFW by default).

Inputs (auto-detected, parquet preferred):
  data/processed/flight_sides.parquet + data/processed/weather_timeline.parquet   (shared contract)
  data/raw/Flight_Data_With_Weather_Final.csv                                    (prototype fallback)

Outputs:
  outputs/flight_risk_preds.parquet   flight_id, airport, side, sched_local, horizon_h, p_delay45, y_true, top_factors
  outputs/delay_metrics.json          test metrics per horizon, baselines, lift, per-airport rates, generalisation
  outputs/delay_failures.csv          one high-confidence miss + one false alarm with their weather timeline
  outputs/delay_model_bundle.joblib   fitted models/calibrators/thresholds for re-scoring

LEAKAGE RULES (enforced in build_features, verified in tests/test_delay_leakage.py)
  * issue time T = sched_utc - h.  Weather features use ONLY timeline rows with obs_utc <= T.
  * congestion feature uses only flights at the same airport whose outcome was fully known
    (complete_utc < T) and that completed on the same local day as T; never the flight itself.
  * the flight's own delay / cause columns are never features (only labels).

Usage:  python src/delay_model.py [--sides P --timeline P | --raw CSV] [--out outputs]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HORIZONS = (3, 6, 12)
DELAY_MIN_THRESHOLD = 45
TARGET_RECALL = 0.60
KT_PER_KMH = 1 / 1.852
CAL_FEATS = ["hour_local", "dow", "month", "side", "airport", "carrier", "distance"]
CAT_FEATS = ["side", "airport", "carrier"]
CONG_FEATS = ["cong_share", "cong_n"]
MAX_VARS = {  # timeline column -> short name used in feature names
    "crosswind_kt": "crosswind",
    "gust_crosswind_kt": "gust_crosswind",
    "wspd_kt": "wspd",
    "prcp_mm": "prcp",
    "snow_mm": "snow",
}
LAST_VARS = {
    "crosswind_kt": "crosswind_last", "gust_crosswind_kt": "gust_crosswind_last",
    "tailwind_kt": "tailwind_last", "wspd_kt": "wspd_last", "wpgt_kt": "wpgt_last",
    "prcp_mm": "prcp_last", "snow_mm": "snow_last", "temp_c": "temp_last",
    "pres_hpa": "pres_last", "coco": "coco_last",
}
WINDOWS = (6, 12)
COUNT_GROUPS = ("storm", "fog", "snow")
WX_FEATS = (
    list(LAST_VARS.values())
    + ["obs_age_h", "pres_chg_3h"]
    + [f"{s}_{w}h_max" for w in WINDOWS for s in MAX_VARS.values()]
    + [f"{g}_count_{w}h" for w in WINDOWS for g in COUNT_GROUPS]
)


# ----------------------------------------------------------------------------------------------
# Config / helpers
# ----------------------------------------------------------------------------------------------
def runway_headings() -> dict:
    try:
        import config  # type: ignore
        for name in ("RUNWAY_HEADINGS", "RUNWAY_HEADING", "RUNWAYS", "HEADINGS"):
            if hasattr(config, name):
                return {str(k): float(v) for k, v in dict(getattr(config, name)).items()}
    except Exception:
        pass
    return {"ATL": 80.0, "DFW": 180.0}


def coco_to_group(coco: pd.Series) -> pd.Series:
    """Meteostat condition code -> coarse group (same buckets as the shared contract)."""
    c = pd.to_numeric(coco, errors="coerce")
    g = pd.Series(np.nan, index=c.index, dtype=object)
    g[c.between(1, 2)] = "clear"
    g[c.between(3, 4)] = "cloud"
    g[c.between(5, 6)] = "fog"
    g[c.between(7, 9) | c.between(17, 18)] = "rain"
    g[c.between(10, 13) | c.between(19, 20)] = "freezing"
    g[c.between(14, 16) | c.between(21, 22)] = "snow"
    g[c.between(23, 27)] = "storm"
    g[c == 0] = "cloud"
    return g


def add_wind(tl: pd.DataFrame, headings: dict) -> pd.DataFrame:
    """Crosswind/tailwind vs runway heading. Unknown direction -> crosswind = full speed (upper bound)."""
    tl = tl.copy()
    hd = tl["airport"].map(headings)
    spd = tl["wspd_kt"]
    gust = np.fmax(tl["wpgt_kt"], tl["wspd_kt"])  # gust is never below the steady wind
    known = tl["wdir_deg"].notna() & hd.notna()
    th = np.radians(tl["wdir_deg"] - hd)
    head = spd * np.cos(th)
    tl["wind_dir_known"] = known
    tl["headwind_kt"] = np.where(known, head, np.nan)
    tl["tailwind_kt"] = np.where(known, np.fmax(0, -head), spd)
    tl["crosswind_kt"] = np.where(known, np.abs(spd * np.sin(th)), spd)
    tl["gust_crosswind_kt"] = np.where(known, np.abs(gust * np.sin(th)), gust)
    return tl


def to_utc_naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s


STATE_TZ = {
    **{s: "America/New_York" for s in "CT DE DC FL GA ME MD MA NH NJ NY NC OH PA RI SC VT VA WV MI".split()},
    **{s: "America/Chicago" for s in "AL AR IL IA KS LA MN MS MO NE ND OK SD TN TX WI KY IN".split()},
    **{s: "America/Denver" for s in "CO ID MT NM UT WY".split()},
    "AZ": "America/Phoenix",
    **{s: "America/Los_Angeles" for s in "CA NV OR WA".split()},
    "AK": "America/Anchorage", "HI": "Pacific/Honolulu",
    "PR": "America/Puerto_Rico", "VI": "America/Puerto_Rico",
}
TZ_OVERRIDE = {"ATL": "America/New_York", "DFW": "America/Chicago", "ELP": "America/Denver"}


def _tz_convert_groups(values: pd.Series, tzs: pd.Series, to_utc: bool) -> pd.Series:
    out = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    for tz, idx in tzs.groupby(tzs).groups.items():
        v = values.loc[idx]
        if to_utc:
            r = v.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
        else:
            r = v.dt.tz_localize("UTC").dt.tz_convert(tz)
        out.loc[idx] = r.dt.tz_localize(None)
    return out


def local_to_utc(local: pd.Series, tzs: pd.Series) -> pd.Series:
    return _tz_convert_groups(local, tzs, True)


def utc_to_local(utc: pd.Series, tzs: pd.Series) -> pd.Series:
    return _tz_convert_groups(utc, tzs, False)


# ----------------------------------------------------------------------------------------------
# Input adapters
# ----------------------------------------------------------------------------------------------
RAW_COLS = [
    "FL_DATE", "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST", "ORIGIN_STATE_ABR", "DEST_STATE_ABR",
    "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "DEP_DELAY", "ARR_DELAY", "CANCELLED", "DIVERTED",
    "DISTANCE", "NAS_DELAY", "WEATHER_DELAY",
    *[f"{v}_{e}" for e in ("ORIGIN", "DEST") for v in ("temp", "prcp", "wspd", "wpgt", "pres", "snow", "coco")],
]


def _hhmm_to_min(t: pd.Series) -> pd.Series:
    t = pd.to_numeric(t, errors="coerce")
    return (t // 100) * 60 + (t % 100)  # 2400 -> 1440 (next-day midnight)


def load_raw(csv: Path, airports: list[str] | None, wind_csv: Path | None, headings: dict):
    """Prototype adapter: raw BTS+Meteostat CSV -> (flight_sides, weather_timeline) in contract shape.

    Known data quirk (verified by the team): the weather attached to a flight is the observation at UTC
    hour = LOCAL scheduled hour. Origin weather is keyed by the dep hour, destination weather by the ARR hour
    (verified: dest rows agree 100% with origin-side rows only when keyed on arr hour), both on FL_DATE.
    We relabel each observation with its true obs_utc instead of pretending it is concurrent with the flight.
    """
    df = pd.read_csv(csv, usecols=RAW_COLS, low_memory=False)
    airports = airports or ["ATL", "DFW"]
    df = df[df.ORIGIN.isin(airports) | df.DEST.isin(airports)].reset_index(drop=True)
    df["date"] = pd.to_datetime(df.FL_DATE)

    st = pd.concat([df[["ORIGIN", "ORIGIN_STATE_ABR"]].set_axis(["ap", "st"], axis=1),
                    df[["DEST", "DEST_STATE_ABR"]].set_axis(["ap", "st"], axis=1)])
    ap_tz = st.groupby("ap")["st"].agg(lambda s: s.mode().iat[0]).map(STATE_TZ).fillna("America/Chicago").to_dict()
    ap_tz.update(TZ_OVERRIDE)
    o_tz, d_tz = df.ORIGIN.map(ap_tz), df.DEST.map(ap_tz)

    dep_min, arr_min = _hhmm_to_min(df.CRS_DEP_TIME), _hhmm_to_min(df.CRS_ARR_TIME)
    dep_local = df.date + pd.to_timedelta(dep_min, unit="m")
    dep_utc = local_to_utc(dep_local, o_tz)
    # arrival: pick the date (FL_DATE or +1) whose UTC instant is closest to dep_utc + scheduled elapsed
    est = dep_utc + pd.to_timedelta(df.CRS_ELAPSED_TIME, unit="m")
    c0 = local_to_utc(df.date + pd.to_timedelta(arr_min, unit="m"), d_tz)
    c1 = local_to_utc(df.date + pd.Timedelta(days=1) + pd.to_timedelta(arr_min, unit="m"), d_tz)
    arr_utc = c0.where((c0 - est).abs() <= (c1 - est).abs(), c1)

    df["flight_id"] = (df.FL_DATE + "_" + df.OP_UNIQUE_CARRIER + df.OP_CARRIER_FL_NUM.fillna(0).astype(int).astype(str)
                       + "_" + df.ORIGIN + "-" + df.DEST + "_" + df.CRS_DEP_TIME.fillna(0).astype(int).astype(str).str.zfill(4))
    dup = df.groupby("flight_id").cumcount()
    df.loc[dup > 0, "flight_id"] += "_" + dup[dup > 0].astype(str)

    arr_delay_f = df.ARR_DELAY.fillna(0)
    cancelled = df.CANCELLED.fillna(0).astype(bool)
    stress = cancelled | ((df.NAS_DELAY.fillna(0) + df.WEATHER_DELAY.fillna(0)) >= 15)
    rows = []
    for ap in airports:
        for side, mask, sched_utc, tzs, delay_col, other in (
            ("dep", df.ORIGIN == ap, dep_utc, o_tz, "DEP_DELAY", "DEST"),
            ("arr", df.DEST == ap, arr_utc, d_tz, "ARR_DELAY", "ORIGIN"),
        ):
            d = df[mask]
            su = sched_utc[mask]
            # time at which this flight's outcome (incl. delay-cause codes, known at arrival) is fully known
            if side == "dep":
                comp = dep_utc[mask] + pd.to_timedelta(d.CRS_ELAPSED_TIME, unit="m") + pd.to_timedelta(arr_delay_f[mask], unit="m")
            else:
                comp = su + pd.to_timedelta(arr_delay_f[mask], unit="m")
            comp = comp.where(~cancelled[mask], su)  # a cancellation is known by the scheduled time
            sl = utc_to_local(su, tzs[mask])
            rows.append(pd.DataFrame({
                "flight_id": d.flight_id, "airport": ap, "side": side, "other_airport": d[other],
                "carrier": d.OP_UNIQUE_CARRIER, "sched_local": sl, "sched_utc": su,
                "date_local": sl.dt.normalize(), "hour_local": sl.dt.hour,
                "block_local": (sl.dt.hour // 6) * 6, "distance": d.DISTANCE,
                "cancelled": cancelled[mask], "diverted": d.DIVERTED.fillna(0).astype(bool),
                "delay_min": d[delay_col], "nas_delay": d.NAS_DELAY, "weather_delay": d.WEATHER_DELAY,
                "stress_flag": stress[mask], "complete_utc": comp,
            }))
    sides = pd.concat(rows, ignore_index=True)
    sides = sides[sides.sched_utc.notna()].reset_index(drop=True)
    unl = sides.cancelled | sides.diverted
    sides["delayed_45"] = np.where(unl | sides.delay_min.isna(), np.nan, (sides.delay_min > DELAY_MIN_THRESHOLD).astype(float))

    # ---- weather timeline (true obs_utc), origin rows keyed on dep hour, dest rows on arr hour
    parts = []
    for e, hh in (("ORIGIN", (dep_min // 60) % 24), ("DEST", (arr_min // 60) % 24)):
        t = pd.DataFrame({
            "airport": df[e], "obs_utc": df.date + pd.to_timedelta(hh, unit="h"),
            "temp_c": df[f"temp_{e}"], "prcp_mm": df[f"prcp_{e}"], "wspd_kt": df[f"wspd_{e}"] * KT_PER_KMH,
            "wpgt_kt": df[f"wpgt_{e}"] * KT_PER_KMH, "pres_hpa": df[f"pres_{e}"], "snow_mm": df[f"snow_{e}"],
            "coco": df[f"coco_{e}"],
        })
        parts.append(t[t.airport.isin(airports)])
    tl = pd.concat(parts).drop_duplicates(["airport", "obs_utc"]).sort_values(["airport", "obs_utc"]).reset_index(drop=True)
    tl["wx_group"] = coco_to_group(tl.coco)
    tl["wdir_deg"] = np.nan
    if wind_csv is not None and Path(wind_csv).exists():
        w = pd.read_csv(wind_csv)
        w["airport"] = w.station.astype(str).str.replace(r"^K", "", regex=True)
        w["obs_utc"] = to_utc_naive(w.obs_utc).dt.floor("h")
        w = w.drop_duplicates(["airport", "obs_utc"], keep="last")
        tl = tl.merge(w[["airport", "obs_utc", "drct"]], on=["airport", "obs_utc"], how="left")
        tl["wdir_deg"] = tl.pop("drct")
    tl = add_wind(tl, headings)
    return sides, tl


def load_parquet(sides_p: Path, tl_p: Path, headings: dict):
    sides = pd.read_parquet(sides_p)
    tl = pd.read_parquet(tl_p)
    if "wx_group" not in tl and "coco" in tl:
        tl["wx_group"] = coco_to_group(tl.coco)
    if "crosswind_kt" not in tl:
        tl = add_wind(tl, headings)
    return sides, tl


def prepare_sides(sides: pd.DataFrame) -> pd.DataFrame:
    s = sides.copy()
    s["sched_utc"] = to_utc_naive(s["sched_utc"])
    dl = pd.to_datetime(s["date_local"])
    s["date_local"] = dl.dt.tz_localize(None) if getattr(dl.dt, "tz", None) is not None else dl
    for c in ("cancelled", "diverted", "stress_flag"):
        s[c] = s[c].fillna(0).astype(bool)
    if "delayed_45" not in s:
        s["delayed_45"] = np.where(s.cancelled | s.diverted | s.delay_min.isna(), np.nan, (s.delay_min > DELAY_MIN_THRESHOLD).astype(float))
    if "complete_utc" in s:
        s["complete_utc"] = to_utc_naive(s["complete_utc"])
    else:  # conservative fallback from the shared contract: actual dep/arr + approximate block time
        block = pd.to_timedelta(s.distance.fillna(500) / 7.5 + 30 + 30, unit="m")
        dm = pd.to_timedelta(s.delay_min.fillna(0), unit="m")
        comp = np.where(s.side == "dep", s.sched_utc + dm + block, s.sched_utc + dm)
        s["complete_utc"] = pd.Series(comp, index=s.index).where(~s.cancelled, s.sched_utc)
    return s.reset_index(drop=True)


def assign_split(s: pd.DataFrame, train_end: str, val_end: str) -> pd.Series:
    d = s["date_local"]
    return pd.Series(np.select([d <= pd.Timestamp(train_end), d <= pd.Timestamp(val_end)], ["train", "val"], "test"), index=s.index)


# ----------------------------------------------------------------------------------------------
# Point-in-time features
# ----------------------------------------------------------------------------------------------
def build_weather_store(tl: pd.DataFrame) -> dict:
    """Per airport: hourly grid of 'last value', rolling maxima/counts and obs age, indexed by the hour <= T."""
    tl = tl.copy()
    tl["obs_utc"] = to_utc_naive(tl["obs_utc"])
    for c in list(LAST_VARS) + list(MAX_VARS):
        if c not in tl:
            tl[c] = np.nan
    if "wx_group" not in tl:
        tl["wx_group"] = coco_to_group(tl["coco"])
    store = {}
    for ap, g in tl.groupby("airport"):
        g = g.sort_values("obs_utc").assign(_h=lambda x: x.obs_utc.dt.floor("h")).drop_duplicates("_h", keep="last").set_index("_h")
        idx = pd.date_range(g.index.min(), g.index.max() + pd.Timedelta(hours=48), freq="h")
        g = g.assign(_obs=1.0).reindex(idx)
        observed = g["_obs"].notna()
        F = pd.DataFrame(index=idx)
        cols = list(dict.fromkeys(list(LAST_VARS) + list(MAX_VARS)))
        ff = g[cols].ffill()
        for src, name in LAST_VARS.items():
            F[name] = ff[src]
        obs_time = pd.Series(idx.where(observed.values), index=idx).ffill()
        F["_last_obs_ns"] = np.where(obs_time.isna().values, -1, obs_time.values.astype("datetime64[ns]").astype("int64"))
        for w in WINDOWS:
            for src, short in MAX_VARS.items():
                F[f"{short}_{w}h_max"] = g[src].rolling(w, min_periods=1).max()
            for grp in COUNT_GROUPS:
                flag = (g["wx_group"] == grp)
                if grp == "snow":
                    flag = flag | (g["snow_mm"].fillna(0) > 0)
                F[f"{grp}_count_{w}h"] = flag.astype(float).where(observed).rolling(w, min_periods=1).sum()
        pres_t = pd.Series(idx.where(g["pres_hpa"].notna().values), index=idx).ffill()
        chg = ff["pres_hpa"] - ff["pres_hpa"].shift(3)
        F["pres_chg_3h"] = chg.where(pres_t != pres_t.shift(3))  # NaN if no newer pressure obs than 3h earlier
        store[ap] = (idx[0].value, F)
    return store


def weather_features(sides: pd.DataFrame, T: pd.Series, store: dict) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=sides.index, columns=WX_FEATS)
    hour_ns = 3_600_000_000_000
    for ap, ix in sides.groupby("airport").groups.items():
        if ap not in store:
            continue
        start_ns, F = store[ap]
        t = T.loc[ix]
        pos = ((t.dt.floor("h").astype("int64").values - start_ns) // hour_ns).astype(int)
        ok = pos >= 0
        pos = np.clip(pos, 0, len(F) - 1)
        vals = F.iloc[pos]
        last_ns = vals["_last_obs_ns"].values
        age = np.where(last_ns >= 0, (t.astype("int64").values - last_ns) / hour_ns, np.nan)
        block = vals.drop(columns="_last_obs_ns").reindex(columns=WX_FEATS).copy()
        block["obs_age_h"] = age
        arr = block.values.astype(float)
        arr[~ok] = np.nan
        out.loc[ix, WX_FEATS] = arr
    return out


def congestion_features(sides: pd.DataFrame, T: pd.Series, tz_of: dict) -> pd.DataFrame:
    """Share of stress_flag among flights at the same airport that COMPLETED (outcome known) before T, same local day."""
    share = pd.Series(np.nan, index=sides.index)
    n = pd.Series(np.nan, index=sides.index)
    for ap, ix in sides.groupby("airport").groups.items():
        d = sides.loc[ix]
        comp = d["complete_utc"].values.astype("datetime64[ns]").astype("int64")
        order = np.argsort(comp)
        comp_s = comp[order]
        cs = np.concatenate([[0.0], np.cumsum(d["stress_flag"].values.astype(float)[order])])
        t = T.loc[ix]
        tz = tz_of.get(ap, "UTC")
        day0 = t.dt.tz_localize("UTC").dt.tz_convert(tz).dt.normalize().dt.tz_convert("UTC").dt.tz_localize(None)
        tq, d0 = t.values.astype("datetime64[ns]").astype("int64"), day0.values.astype("datetime64[ns]").astype("int64")
        lo, hi = np.searchsorted(comp_s, d0, "left"), np.searchsorted(comp_s, tq, "left")
        cnt, s = (hi - lo).astype(float), cs[hi] - cs[lo]
        self_in = (comp >= d0) & (comp < tq)  # defensive: never count the flight itself
        cnt -= self_in
        s -= self_in * d["stress_flag"].values.astype(float)
        share.loc[ix] = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
        n.loc[ix] = cnt
    return pd.DataFrame({"cong_share": share, "cong_n": n})


def build_features(sides: pd.DataFrame, store: dict, h: int, vocab: dict, tz_of: dict) -> tuple[pd.DataFrame, pd.Series]:
    T = sides["sched_utc"] - pd.Timedelta(hours=h)
    X = pd.DataFrame(index=sides.index)
    X["hour_local"] = sides["hour_local"].astype(int)
    X["dow"] = sides["date_local"].dt.dayofweek
    X["month"] = sides["date_local"].dt.month
    X["distance"] = sides["distance"].astype(float)
    for c, src in (("side", "side"), ("airport", "airport"), ("carrier", "carrier")):
        X[c] = pd.Categorical(sides[src], categories=vocab[c])
    X = pd.concat([X, weather_features(sides, T, store), congestion_features(sides, T, tz_of)], axis=1)
    return X[CAL_FEATS + WX_FEATS + CONG_FEATS], T


# ----------------------------------------------------------------------------------------------
# Modelling
# ----------------------------------------------------------------------------------------------
def _lgbm(seed: int):
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        n_estimators=800, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, reg_lambda=5.0, class_weight="balanced", random_state=seed, n_jobs=4,
        verbosity=-1, deterministic=True, force_row_wise=True)


def fit_model(Xtr, ytr, Xva, yva, feats, seed=42):
    """LightGBM (class-weighted, early-stopped on validation AUC) + Platt calibration fit on validation."""
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    m = _lgbm(seed)
    m.fit(Xtr[feats], ytr, eval_set=[(Xva[feats], yva)], eval_metric="auc",
          callbacks=[lgb.early_stopping(60, verbose=False)])
    marg = m.predict(Xva[feats], raw_score=True)
    cal = LogisticRegression(C=1e6, max_iter=1000).fit(marg.reshape(-1, 1), yva)
    return {"model": m, "cal": cal, "feats": list(feats)}


def predict(bundle, X) -> tuple[np.ndarray, np.ndarray]:
    marg = bundle["model"].predict(X[bundle["feats"]], raw_score=True)
    return bundle["cal"].predict_proba(marg.reshape(-1, 1))[:, 1], marg


def isotonic_cv_brier(bundle_marg_va, yva, seed=0):
    """Diagnostic: 2-fold CV Brier on validation for Platt vs isotonic (why Platt was kept)."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, 2, len(yva))
    res = {"platt": [], "isotonic": []}
    for k in (0, 1):
        tr, te = fold != k, fold == k
        pl = LogisticRegression(C=1e6, max_iter=1000).fit(bundle_marg_va[tr].reshape(-1, 1), yva[tr])
        res["platt"].append(np.mean((pl.predict_proba(bundle_marg_va[te].reshape(-1, 1))[:, 1] - yva[te]) ** 2))
        iso = IsotonicRegression(out_of_bounds="clip").fit(bundle_marg_va[tr], yva[tr])
        res["isotonic"].append(np.mean((iso.predict(bundle_marg_va[te]) - yva[te]) ** 2))
    return {k: float(np.mean(v)) for k, v in res.items()}


def scores(y, p) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    y = np.asarray(y)
    if len(y) == 0 or y.min() == y.max():
        return {"n": int(len(y)), "pr_auc": None, "roc_auc": None, "brier": None}
    return {"n": int(len(y)), "pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p))}


def threshold_for_recall(y, p, target=TARGET_RECALL) -> float:
    """Highest threshold whose validation recall is >= target."""
    from sklearn.metrics import precision_recall_curve
    _, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= target)[0]
    return float(thr[ok[-1]]) if len(ok) else float(np.min(p))


def op_point(y, p, thr) -> dict:
    """Operating-point metrics, all as rates per 100 flights."""
    y = np.asarray(y).astype(int)
    flag = np.asarray(p) >= thr
    tp, fp, fn = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum()), int((~flag & (y == 1)).sum())
    n = max(len(y), 1)
    return {"threshold": float(thr), "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
            "n_flights": int(len(y)), "delayed_per_100": 100 * y.mean() if len(y) else None, "flagged_per_100": 100 * flag.sum() / n,
            "caught_per_100": 100 * tp / n, "false_alarms_per_100": 100 * fp / n, "missed_per_100": 100 * fn / n}


def top_frac_point(y, p, frac) -> dict:
    """Precision/recall if ops only review the top `frac` of flights by risk (small, actionable list)."""
    y = np.asarray(y).astype(int)
    k = max(int(np.ceil(frac * len(y))), 1)
    top = np.argsort(-np.asarray(p), kind="stable")[:k]
    tp = int(y[top].sum())
    return {"frac": frac, "flagged_per_100": 100 * k / len(y), "precision": tp / k, "recall": tp / max(int(y.sum()), 1),
            "lift_over_base_rate": (tp / k) / max(y.mean(), 1e-9)}


def block_bootstrap(y, preds: dict, dates, B=200, seed=0):
    from sklearn.metrics import average_precision_score, roc_auc_score
    codes, uniq = pd.factorize(pd.Series(dates))
    groups = [np.where(codes == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    out = {k: {"pr_auc": [], "roc_auc": []} for k in preds}
    for _ in range(B):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if y[idx].min() == y[idx].max():
            continue
        for k, p in preds.items():
            out[k]["pr_auc"].append(average_precision_score(y[idx], p[idx]))
            out[k]["roc_auc"].append(roc_auc_score(y[idx], p[idx]))
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    res = {k: {m: ci(v) for m, v in d.items()} for k, d in out.items()}
    if "model" in preds and "calendar" in preds:
        for m in ("pr_auc", "roc_auc"):
            res[f"lift_{m}_model_minus_calendar"] = ci(np.array(out["model"][m]) - np.array(out["calendar"][m]))
    return res


# ----------------------------------------------------------------------------------------------
# Explainability
# ----------------------------------------------------------------------------------------------
def _fmt_num(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.0f}" if abs(v) >= 10 else f"{v:.1f}".rstrip("0").rstrip(".")


def _fmt_feature(name: str, v) -> str:
    if isinstance(v, str):
        return f"{name}={v}"
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return f"{name}=n/a"
    if name == "cong_share":
        return f"{name}={100 * v:.0f}%"
    if name == "coco_last":
        return f"{name}={coco_to_group(pd.Series([v])).iat[0]}({v:.0f})"
    if name in ("hour_local", "dow", "month", "cong_n", "coco_last") or "_count_" in name:
        return f"{name}={v:.0f}"
    unit = ("kt" if any(k in name for k in ("crosswind", "wspd", "wpgt", "tailwind")) else
            "mm" if ("prcp" in name or "snow" in name) else "hPa" if "pres" in name else
            "C" if "temp" in name else "h" if name == "obs_age_h" else "mi" if name == "distance" else "")
    return f"{name}={_fmt_num(float(v))}{unit}"


def top_factors(bundle, X: pd.DataFrame, k=3) -> tuple[list[str], np.ndarray]:
    """Exact TreeSHAP (LightGBM pred_contrib) in log-odds, mapped to calibrated-probability points.

    Contributions are accumulated in order of |SHAP| and pushed through the Platt calibrator so that they
    sum exactly to p_delay45 - p_reference (the calibrated probability of the model's average output).
    """
    feats = bundle["feats"]
    contrib = bundle["model"].predict(X[feats], pred_contrib=True)
    sh, base = contrib[:, :-1], contrib[:, -1]
    a, b = float(bundle["cal"].coef_[0, 0]), float(bundle["cal"].intercept_[0])
    sig = lambda z: 1 / (1 + np.exp(-z))
    order = np.argsort(-np.abs(sh), axis=1)
    sh_sorted = np.take_along_axis(sh, order, axis=1)
    path = sig(a * (base[:, None] + np.cumsum(sh_sorted, axis=1)) + b)
    prev = np.concatenate([sig(a * base + b)[:, None], path[:, :-1]], axis=1)
    pp = path - prev  # probability-point contribution of the j-th largest factor
    vals = X[feats]
    strs = []
    for i in range(len(X)):
        parts = []
        for j in range(k):
            f = feats[order[i, j]]
            v = vals.iloc[i][f]
            parts.append(f"{_fmt_feature(f, v)} ({pp[i, j]:+.2f})")
        strs.append("; ".join(parts))
    return strs, sig(a * base + b)


# ----------------------------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------------------------
def label_report(s: pd.DataFrame) -> dict:
    rep = {"total_flight_sides": int(len(s)), "excluded_cancelled": int(s.cancelled.sum()),
           "excluded_diverted": int((s.diverted & ~s.cancelled).sum()),
           "excluded_missing_delay": int((s.delayed_45.isna() & ~s.cancelled & ~s.diverted).sum()),
           "labelled": int(s.delayed_45.notna().sum())}
    rep["excluded_total"] = rep["total_flight_sides"] - rep["labelled"]
    lab = s[s.delayed_45.notna()]
    rep["base_rate_per_100_by_airport_side"] = {
        f"{ap}|{sd}": {sp: {"n": int(len(g)), "delayed_per_100": round(100 * g.delayed_45.mean(), 2)} for sp, g in gg.groupby("split")}
        for (ap, sd), gg in lab.groupby(["airport", "side"])}
    rep["by_airport_side_excluded"] = {f"{ap}|{sd}": int(g.cancelled.sum() + (g.diverted & ~g.cancelled).sum() + (g.delayed_45.isna() & ~g.cancelled & ~g.diverted).sum())
                                       for (ap, sd), g in s.groupby(["airport", "side"])}
    return rep


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    headings = runway_headings()
    sp, tp = Path(args.sides), Path(args.timeline)
    if not args.raw and sp.exists() and tp.exists():
        sides, tl = load_parquet(sp, tp, headings)
        source = f"parquet:{sp.name}+{tp.name}"
    else:
        raw = Path(args.raw or "data/raw/Flight_Data_With_Weather_Final.csv")
        sides, tl = load_raw(raw, args.airports, Path(args.wind) if args.wind else None, headings)
        source = f"raw_csv:{raw.name}"
    sides = prepare_sides(sides)
    if args.airports:
        sides = sides[sides.airport.isin(args.airports)].reset_index(drop=True)
    sides["split"] = assign_split(sides, args.train_end, args.val_end)
    airports = sorted(sides.airport.unique())
    tz_of = {**{ap: TZ_OVERRIDE.get(ap) for ap in airports}}
    tz_of = {k: (v or "America/Chicago") for k, v in tz_of.items()}
    vocab = {"side": sorted(sides.side.unique()), "airport": airports, "carrier": sorted(sides.carrier.dropna().unique())}
    store = build_weather_store(tl)
    wind_known = float(tl.loc[tl.airport.isin(airports), "wind_dir_known"].mean()) if "wind_dir_known" in tl else None
    lab_rep = label_report(sides)
    print(f"[data] source={source} sides={len(sides)} labelled={lab_rep['labelled']} excluded={lab_rep['excluded_total']} "
          f"wind_dir_known={wind_known}", flush=True)

    sets = {"calendar": CAL_FEATS, "cal_wx": CAL_FEATS + WX_FEATS, "full": CAL_FEATS + WX_FEATS + CONG_FEATS}
    metrics = {"meta": {"source": source, "airports": airports, "horizons": list(args.horizons), "split": {
        "train_end": args.train_end, "val_end": args.val_end}, "target_recall": TARGET_RECALL,
        "wind_dir_known_share": wind_known,
        "weather_note": "Weather is OBSERVATIONS (not forecasts) with obs_utc <= issue time T = sched_utc - horizon.",
        "rates": "All rates are per 100 flights of the stated airport/side; no raw delay minutes are reported.",
        "label_report": lab_rep}, "by_horizon": {}}
    preds_all, bundles, failure_pool = [], {}, {}

    for h in args.horizons:
        X, T = build_features(sides, store, h, vocab, tz_of)
        y = sides["delayed_45"]
        lab = y.notna()
        idx = {sp_: sides.index[(sides.split == sp_) & lab] for sp_ in ("train", "val", "test")}
        Xtr, Xva, Xte = X.loc[idx["train"]], X.loc[idx["val"]], X.loc[idx["test"]]
        ytr, yva, yte = (y.loc[idx[k]].astype(int).values for k in ("train", "val", "test"))
        fitted = {k: fit_model(Xtr, ytr, Xva, yva, f, args.seed) for k, f in sets.items()}
        pva = {k: predict(b, Xva)[0] for k, b in fitted.items()}
        pte = {k: predict(b, Xte)[0] for k, b in fitted.items()}
        val_pr = {k: scores(yva, pva[k])["pr_auc"] for k in fitted}
        main_key = args.main_set if args.main_set != "auto" else max(("cal_wx", "full"), key=lambda k: val_pr[k])
        main = fitted[main_key]
        thr = {k: threshold_for_recall(yva, pva[k]) for k in fitted}
        base_p = float(ytr.mean())

        res = {"main_feature_set": main_key, "n_train": len(ytr), "n_val": len(yva), "n_test": len(yte),
               "val_pr_auc_by_set": val_pr, "test": {}, "operating_point": {}, "best_iteration": {k: int(b["model"].best_iteration_ or b["model"].n_estimators) for k, b in fitted.items()}}
        res["test"]["base_rate"] = {"p": base_p, "pr_auc": float(yte.mean()), "roc_auc": 0.5, "brier": float(np.mean((base_p - yte) ** 2))}
        for k in fitted:
            res["test"][{"calendar": "calendar_only", "cal_wx": "calendar_plus_weather", "full": "calendar_weather_congestion"}[k]] = scores(yte, pte[k])
        res["test"]["model"] = scores(yte, pte[main_key])
        cal_s, m_s = res["test"]["calendar_only"], res["test"]["model"]
        res["weather_lift"] = {
            "pr_auc_model_vs_calendar": m_s["pr_auc"] - cal_s["pr_auc"], "pr_auc_rel_pct": 100 * (m_s["pr_auc"] / cal_s["pr_auc"] - 1),
            "roc_auc_model_vs_calendar": m_s["roc_auc"] - cal_s["roc_auc"], "brier_model_vs_calendar": m_s["brier"] - cal_s["brier"],
            "pr_auc_x_over_base_rate": m_s["pr_auc"] / yte.mean(), "pr_auc_calendar_x_over_base_rate": cal_s["pr_auc"] / yte.mean(),
            "pure_weather_lift_cal_wx_vs_calendar_pr_auc": res["test"]["calendar_plus_weather"]["pr_auc"] - cal_s["pr_auc"]}
        res["bootstrap_95ci_by_test_day"] = block_bootstrap(yte, {"model": pte[main_key], "calendar": pte["calendar"]},
                                                           sides.loc[idx["test"], "date_local"].values, seed=args.seed)
        # operating point (threshold chosen on validation for recall ~0.6)
        ops = {"model": op_point(yte, pte[main_key], thr[main_key]), "calendar_only": op_point(yte, pte["calendar"], thr["calendar"]),
               "base_rate_flag_all": op_point(yte, np.ones_like(yte, float), 0.5)}
        ops["top_risk_lists"] = {f"top_{int(100 * fr)}pct": {"model": top_frac_point(yte, pte[main_key], fr),
                                                             "calendar_only": top_frac_point(yte, pte["calendar"], fr)} for fr in (0.05, 0.10)}
        ops["val_recall_at_threshold"] = op_point(yva, pva[main_key], thr[main_key])["recall"]
        ap_arr = sides.loc[idx["test"], "airport"].values
        sd_arr = sides.loc[idx["test"], "side"].values
        ops["by_airport"] = {ap: {**scores(yte[ap_arr == ap], pte[main_key][ap_arr == ap]),
                                  "calendar_pr_auc": scores(yte[ap_arr == ap], pte["calendar"][ap_arr == ap])["pr_auc"],
                                  "base_rate_per_100": 100 * yte[ap_arr == ap].mean(), **{f"op_{k}": v for k, v in op_point(yte[ap_arr == ap], pte[main_key][ap_arr == ap], thr[main_key]).items()}}
                             for ap in airports if (ap_arr == ap).any()}
        ops["by_airport_side"] = {f"{ap}|{sd}": {**scores(yte[m], pte[main_key][m]), "calendar_pr_auc": scores(yte[m], pte["calendar"][m])["pr_auc"],
                                                  **op_point(yte[m], pte[main_key][m], thr[main_key])}
                                  for ap in airports for sd in ("dep", "arr") for m in [(ap_arr == ap) & (sd_arr == sd)] if m.sum() > 0}
        res["operating_point"] = ops
        # slices: where does the weather help / fail?
        adv = ((Xte["storm_count_12h"].fillna(0) > 0) | (Xte["prcp_12h_max"].fillna(0) >= 2.5) | (Xte["gust_crosswind_12h_max"].fillna(0) >= 25)).values \
            if "gust_crosswind_12h_max" in Xte else np.zeros(len(Xte), bool)
        hb = (Xte["hour_local"].values // 6) * 6
        sl = {"adverse_wx_12h(storm|prcp>=2.5mm|gust_xw>=25kt)": adv, "calm_wx_12h": ~adv, **{f"hour_block_{b}": hb == b for b in (0, 6, 12, 18)}}
        res["slices"] = {}
        for name, m in sl.items():
            if m.sum() >= 150 and 10 <= yte[m].sum() < m.sum() - 10:
                res["slices"][name] = {"n": int(m.sum()), "delayed_per_100": 100 * yte[m].mean(), "pr_auc_model": scores(yte[m], pte[main_key][m])["pr_auc"],
                                       "pr_auc_calendar": scores(yte[m], pte["calendar"][m])["pr_auc"],
                                       "recall_at_thr": op_point(yte[m], pte[main_key][m], thr[main_key])["recall"],
                                       "precision_at_thr": op_point(yte[m], pte[main_key][m], thr[main_key])["precision"]}
        # calibration (reliability) on test + calibrator choice diagnostic
        q = pd.qcut(pte[main_key], 10, duplicates="drop")
        res["calibration_test_deciles"] = [{"mean_pred": float(g["p"].mean()), "obs_rate": float(g["y"].mean()), "n": int(len(g))}
                                           for _, g in pd.DataFrame({"p": pte[main_key], "y": yte, "q": q}).groupby("q", observed=True)]
        res["calibrator_val_cv_brier"] = isotonic_cv_brier(main["model"].predict(Xva[main["feats"]], raw_score=True), yva)
        metrics["by_horizon"][f"h{h}"] = res

        # cross-airport generalisation (feature set without airport identity)
        if len(airports) >= 2:
            gen = {}
            nf = lambda f: [c for c in f if c != "airport"]
            for a_tr in airports:
                for a_te in airports:
                    tr_i = idx["train"][sides.loc[idx["train"], "airport"] == a_tr]
                    va_i = idx["val"][sides.loc[idx["val"], "airport"] == a_tr]
                    te_i = idx["test"][sides.loc[idx["test"], "airport"] == a_te]
                    if min(len(tr_i), len(va_i), len(te_i)) < 100:
                        continue
                    ytr_, yva_, yte_ = (y.loc[i].astype(int).values for i in (tr_i, va_i, te_i))
                    r = {"n_test": int(len(te_i)), "test_base_rate_per_100": 100 * yte_.mean(), "train_base_rate_per_100": 100 * ytr_.mean()}
                    for tag, fs in (("model", nf(sets[main_key])), ("calendar_only", nf(CAL_FEATS))):
                        b = fit_model(X.loc[tr_i], ytr_, X.loc[va_i], yva_, fs, args.seed)
                        r[tag] = scores(yte_, predict(b, X.loc[te_i])[0])
                    gen[f"train_{a_tr}->test_{a_te}" + ("(in-airport reference)" if a_tr == a_te else "")] = r
            res["generalisation"] = gen

        # explanations + predictions for the test period
        tf, p_ref = top_factors(main, Xte)
        res["reference_probability_p_ref"] = float(np.mean(p_ref))
        te = sides.loc[idx["test"]]
        pr = pd.DataFrame({"flight_id": te.flight_id.values, "airport": te.airport.values, "side": te.side.values,
                           "sched_local": te.sched_local.values, "horizon_h": h, "p_delay45": pte[main_key],
                           "y_true": yte.astype(float), "top_factors": tf})
        preds_all.append(pr)
        failure_pool[h] = (pr.assign(_idx=idx["test"], _T=T.loc[idx["test"]].values), te)
        bundles[h] = {"horizon_h": h, "feature_set": main_key, "feats": main["feats"], "model": main["model"], "cal": main["cal"],
                      "threshold": thr[main_key], "vocab": vocab}
        print(f"[h={h}] test n={len(yte)} base={100*yte.mean():.1f}/100 | PR-AUC model {m_s['pr_auc']:.3f} cal-only {cal_s['pr_auc']:.3f} "
              f"| ROC {m_s['roc_auc']:.3f} vs {cal_s['roc_auc']:.3f} | main={main_key}", flush=True)

    preds = pd.concat(preds_all, ignore_index=True)
    preds.to_parquet(out / "flight_risk_preds.parquet", index=False)
    write_failures(out / "delay_failures.csv", failure_pool, tl, args.failure_horizon if args.failure_horizon in failure_pool else args.horizons[0])
    # flat aliases (shared contract: metric name -> value) next to the nested detail
    flat = {}
    for hk, r in metrics["by_horizon"].items():
        for name, key in (("model", "model"), ("calendar", "calendar_only"), ("base_rate", "base_rate")):
            for m in ("pr_auc", "roc_auc", "brier"):
                flat[f"{hk}_{name}_{m}"] = r["test"][key][m]
        flat[f"{hk}_precision_at_recall60"] = r["operating_point"]["model"]["precision"]
        flat[f"{hk}_recall_at_recall60_thr"] = r["operating_point"]["model"]["recall"]
        flat[f"{hk}_weather_lift_pr_auc"] = r["weather_lift"]["pr_auc_model_vs_calendar"]
    metrics["flat"] = flat
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if hasattr(o, "item"):
            o = o.item()
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o if isinstance(o, (int, float, str, bool, type(None))) else str(o)
    with open(out / "delay_metrics.json", "w") as f:
        json.dump(clean(metrics), f, indent=2)
    import joblib
    joblib.dump(bundles, out / "delay_model_bundle.joblib")
    print(f"[done] wrote {out}/flight_risk_preds.parquet ({len(preds)} rows), delay_metrics.json, delay_failures.csv", flush=True)
    return metrics


def write_failures(path: Path, pool: dict, tl: pd.DataFrame, h: int):
    """Highest-confidence miss (delayed, lowest p) and worst false alarm (on time, highest p), with the timeline the model saw."""
    pr, te = pool[h]
    pr = pr.reset_index(drop=True)
    te = te.reset_index(drop=True)
    miss = pr[pr.y_true == 1].p_delay45.idxmin()
    fa = pr[pr.y_true == 0].p_delay45.idxmax()
    tl = tl.assign(obs_utc=to_utc_naive(tl["obs_utc"]))
    rows = []
    for kind, i in (("high_confidence_miss", miss), ("false_alarm", fa)):
        r, f = pr.loc[i], te.loc[i]
        T = pd.Timestamp(r["_T"])
        w = tl[(tl.airport == r.airport) & (tl.obs_utc >= T - pd.Timedelta(hours=12)) & (tl.obs_utc <= f.sched_utc)].sort_values("obs_utc").copy()
        w.insert(0, "visible_at_T", w.obs_utc <= T)
        meta = {"case": kind, "flight_id": r.flight_id, "airport": r.airport, "side": r.side, "sched_local": r.sched_local, "horizon_h": h,
                "issue_utc": T, "p_delay45": r.p_delay45, "y_true": r.y_true, "top_factors": r.top_factors,
                "post_hoc_delay_min": f.delay_min, "post_hoc_weather_delay": f.weather_delay, "post_hoc_nas_delay": f.nas_delay}
        if w.empty:
            rows.append(pd.DataFrame([meta]))
        else:
            rows.append(pd.concat([pd.DataFrame([meta] * len(w)).reset_index(drop=True), w.reset_index(drop=True)], axis=1))
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sides", default="data/processed/flight_sides.parquet")
    ap.add_argument("--timeline", default="data/processed/weather_timeline.parquet")
    ap.add_argument("--raw", default=None, help="force the raw-CSV prototype path")
    ap.add_argument("--wind", default="data/external/iem_wind_ATL_DFW.csv", help="IEM wind direction file (raw path only)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--airports", nargs="*", default=None, help="default: all in parquet, ATL DFW for raw")
    ap.add_argument("--horizons", nargs="*", type=int, default=list(HORIZONS))
    ap.add_argument("--train-end", default="2024-06-30")
    ap.add_argument("--val-end", default="2024-12-31")
    ap.add_argument("--main-set", default="auto", choices=["auto", "cal_wx", "full"])
    ap.add_argument("--failure-horizon", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    if a.raw and not a.airports:
        a.airports = ["ATL", "DFW"]
    return a


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*eval_set.*")
    warnings.filterwarnings("ignore", category=UserWarning)
    run(parse_args())
