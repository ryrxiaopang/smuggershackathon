"""
src/capacity_model.py — capacity-stress proxy forecast for ATL and DFW,
3/6/12h ahead.

Target: for each (airport, local block of block_h hours) the share of
non-diverted flights whose stress_flag is True. stress_flag was defined
upstream as `cancelled OR nas_delay + weather_delay >= 15`. This is our
honest proxy for Arrival Acceptance Rate — the sample (~6 ATL arrivals/
day) cannot support a true AAR calculation, so we never call it that.

Model choice: ONE LightGBM regressor with horizon_h as a feature, not
three per-horizon models. With ~4k airport-blocks in train, splitting by
horizon leaves ≤1.4k rows each — the shared model keeps every tree in
touch with every horizon and lets the splits on horizon_h move the
weight between recent-weather and calendar features as horizon grows.

Run:
    python src/capacity_model.py                 # block_h = 6 (default)
    python src/capacity_model.py --block-h 3     # 3h blocks for noise comparison
Set --block-h to 'both' to run 6 then 3 and compare in the metrics file.

Reads  : data/processed/flight_sides.parquet, weather_timeline.parquet
Writes : outputs/capacity_preds.parquet       (contract columns)
         outputs/capacity_metrics.json        (model vs baselines + generalisation)
         outputs/capacity_importance.csv      (top-10 gain)
         outputs/capacity_failures.csv        (2 worst test errors + weather window)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

AIRPORT_TZ = {"ATL": "America/New_York", "DFW": "America/Chicago"}
AIRPORTS   = ("ATL", "DFW")
HORIZONS   = (3, 6, 12)

FLIGHTS   = Path("data/processed/flight_sides.parquet")
WEATHER   = Path("data/processed/weather_timeline.parquet")
OUT_DIR   = Path("outputs")

TRAIN_END  = pd.Timestamp("2024-06-30 23:59:59")
VAL_END    = pd.Timestamp("2024-12-31 23:59:59")
TEST_START = pd.Timestamp("2025-01-01")
TEST_END   = pd.Timestamp("2025-07-31 23:59:59")

TARGET_LABEL = ("capacity-stress proxy "
                "(share of flights with ATC/airport-capacity or weather delay >=15 min or cancelled)")


# ================================================================= blocks
def build_blocks(flights: pd.DataFrame, block_h: int) -> pd.DataFrame:
    """Collapse non-diverted flights to one row per (airport, date, block)."""
    f = flights[~flights["diverted"].fillna(False).astype(bool)].copy()
    if block_h == 6:
        f["block_h_start"] = f["block_local"].astype(int)
    else:
        f["block_h_start"] = (f["hour_local"].astype(int) // block_h) * block_h
    g = f.groupby(["airport", "date_local", "block_h_start"], sort=False)
    blocks = g.agg(
        stress_rate=("stress_flag", "mean"),
        n_flights  =("stress_flag", "size"),
    ).reset_index()
    return blocks


def attach_block_times(blocks: pd.DataFrame, block_h: int) -> pd.DataFrame:
    """Add block_start_local (naive) and block_start_utc / block_end_utc per airport tz."""
    out = []
    for airport, g in blocks.groupby("airport", sort=False):
        dates = pd.to_datetime(g["date_local"]).to_numpy()
        offs  = pd.to_timedelta(g["block_h_start"].astype(int).to_numpy(), unit="h")
        local_naive = pd.DatetimeIndex(dates + offs)
        local = local_naive.tz_localize(
            ZoneInfo(AIRPORT_TZ[airport]),
            nonexistent="shift_forward",
            ambiguous=False,     # pick post-fallback hour on DST ambiguity
        )
        utc = local.tz_convert("UTC").tz_localize(None)
        g = g.copy()
        g["block_start_local"] = local_naive
        g["block_start_utc"]   = utc
        g["block_end_utc"]     = utc + pd.Timedelta(hours=block_h)
        out.append(g)
    return pd.concat(out, ignore_index=True)


# ================================================================= weather features
_ROLL_MAX_MEAN = ["crosswind_kt", "gust_crosswind_kt", "wspd_kt"]
_ROLL_SUM      = ["prcp_mm", "snow_mm"]
_WX_FLAGS      = ["storm", "rain", "fog", "snow"]


def precompute_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """Add rolling features at every hourly observation; one row per (airport, obs_utc)."""
    weather = weather.sort_values(["airport", "obs_utc"]).reset_index(drop=True)
    parts = []
    for airport, g in weather.groupby("airport", sort=False):
        g = g.set_index("obs_utc").copy()
        for c in _ROLL_MAX_MEAN:
            g[f"{c}_max6h"]   = g[c].rolling("6h").max()
            g[f"{c}_mean6h"]  = g[c].rolling("6h").mean()
            g[f"{c}_max12h"]  = g[c].rolling("12h").max()
            g[f"{c}_mean12h"] = g[c].rolling("12h").mean()
        for c in _ROLL_SUM:
            g[f"{c}_sum6h"]  = g[c].rolling("6h").sum()
            g[f"{c}_sum12h"] = g[c].rolling("12h").sum()
        g["pres_chg_3h"] = g["pres_hpa"] - g["pres_hpa"].shift(3)
        g["pres_chg_6h"] = g["pres_hpa"] - g["pres_hpa"].shift(6)
        for wx in _WX_FLAGS:
            flag = (g["wx_group"] == wx).astype(int)
            g[f"cnt_{wx}_6h"]  = flag.rolling("6h").sum()
            g[f"cnt_{wx}_12h"] = flag.rolling("12h").sum()
        wdk = g["wind_dir_known"].astype(float)
        g["wdk_share_6h"]  = wdk.rolling("6h").mean()
        g["wdk_share_12h"] = wdk.rolling("12h").mean()
        g = g.reset_index()
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


# ================================================================= predictions table
def expand_horizons(blocks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        r = blocks.copy()
        r["horizon_h"]  = h
        r["issued_utc"] = r["block_start_utc"] - pd.Timedelta(hours=h)
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def merge_weather_features(preds: pd.DataFrame, wf: pd.DataFrame) -> pd.DataFrame:
    """merge_asof backward on obs_utc <= issued_utc, per airport."""
    out = []
    for airport, g in preds.groupby("airport", sort=False):
        w = wf[wf["airport"] == airport].sort_values("obs_utc")
        gs = g.sort_values("issued_utc")
        merged = pd.merge_asof(
            gs, w.drop(columns=["airport"]),
            left_on="issued_utc", right_on="obs_utc",
            direction="backward",
        )
        out.append(merged)
    preds_f = pd.concat(out, ignore_index=True)
    preds_f["age_last_obs_h"] = (preds_f["issued_utc"] - preds_f["obs_utc"]).dt.total_seconds() / 3600.0
    return preds_f


def merge_prev_block(preds: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    """Previous block's realised stress_rate, only when that block ENDED before issued_utc."""
    key = blocks[["airport", "block_end_utc", "stress_rate"]].rename(
        columns={"block_end_utc": "prev_end_utc", "stress_rate": "prev_block_stress"}
    )
    out = []
    for airport, g in preds.groupby("airport", sort=False):
        b = key[key["airport"] == airport].sort_values("prev_end_utc")
        gs = g.sort_values("issued_utc")
        merged = pd.merge_asof(
            gs, b.drop(columns=["airport"]),
            left_on="issued_utc", right_on="prev_end_utc",
            direction="backward",
        )
        out.append(merged)
    return pd.concat(out, ignore_index=True)


def add_calendar(preds: pd.DataFrame) -> pd.DataFrame:
    bs = pd.DatetimeIndex(preds["block_start_local"])
    preds = preds.copy()
    preds["month"]      = bs.month.astype("int16")
    preds["dow"]        = bs.dayofweek.astype("int16")
    preds["block_hour"] = bs.hour.astype("int16")
    preds["airport_ATL"] = (preds["airport"] == "ATL").astype("int8")
    preds["airport_DFW"] = (preds["airport"] == "DFW").astype("int8")
    return preds


# ================================================================= baselines & metrics
def add_hist_baseline(preds: pd.DataFrame, block_h: int) -> pd.DataFrame:
    """Historical mean on train: (airport, month, block_h_start) -> stress_rate."""
    train = preds[preds["block_start_utc"] <= TRAIN_END]
    hist = (train.groupby(["airport", "month", "block_h_start"])["y_true"]
                 .mean().reset_index(name="baseline_hist"))
    out = preds.merge(hist, on=["airport", "month", "block_h_start"], how="left")
    # Fallback: overall train mean per airport if group missing
    global_mean = train.groupby("airport")["y_true"].mean().to_dict()
    out["baseline_hist"] = out["baseline_hist"].fillna(out["airport"].map(global_mean))
    return out


def rmse(y, p): return float(np.sqrt(mean_squared_error(y, p)))
def mae (y, p): return float(mean_absolute_error(y, p))


# ================================================================= training
FEATURE_COLS = [
    "horizon_h", "airport_ATL", "airport_DFW",
    "month", "dow", "block_hour",
    "age_last_obs_h",
    "temp_c", "prcp_mm", "wspd_kt", "wpgt_kt", "pres_hpa", "snow_mm",
    "wdir_deg", "crosswind_kt", "gust_crosswind_kt",
    "crosswind_kt_max6h","crosswind_kt_mean6h","crosswind_kt_max12h","crosswind_kt_mean12h",
    "gust_crosswind_kt_max6h","gust_crosswind_kt_mean6h","gust_crosswind_kt_max12h","gust_crosswind_kt_mean12h",
    "wspd_kt_max6h","wspd_kt_mean6h","wspd_kt_max12h","wspd_kt_mean12h",
    "prcp_mm_sum6h","prcp_mm_sum12h","snow_mm_sum6h","snow_mm_sum12h",
    "pres_chg_3h","pres_chg_6h",
    "cnt_storm_6h","cnt_rain_6h","cnt_fog_6h","cnt_snow_6h",
    "cnt_storm_12h","cnt_rain_12h","cnt_fog_12h","cnt_snow_12h",
    "wdk_share_6h","wdk_share_12h",
    "prev_block_stress",
]

LGB_PARAMS = dict(
    objective="regression",
    metric="rmse",
    num_leaves=31,
    learning_rate=0.05,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    min_data_in_leaf=20,
    verbosity=-1,
)


def train_lgb(train_df: pd.DataFrame, val_df: pd.DataFrame) -> lgb.Booster:
    dtr = lgb.Dataset(train_df[FEATURE_COLS], label=train_df["y_true"])
    dvl = lgb.Dataset(val_df[FEATURE_COLS],   label=val_df["y_true"], reference=dtr)
    return lgb.train(
        LGB_PARAMS, dtr, num_boost_round=1200,
        valid_sets=[dvl], valid_names=["val"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )


# ================================================================= failures
def save_failures(test: pd.DataFrame, weather_raw: pd.DataFrame,
                  block_h: int, out_path: Path) -> list[dict]:
    """Pick the 2 largest |residuals| and dump the weather hours around them."""
    t = test.copy()
    t["abs_err"] = (t["y_true"] - t["y_pred"]).abs()
    # Dedupe by (airport, block_start_local) so the 2 cases cover 2 distinct situations
    # rather than the same block at two horizons.
    t = (t.sort_values("abs_err", ascending=False)
           .drop_duplicates(["airport", "block_start_local"])
           .head(2)
           .reset_index(drop=True))
    worst = t

    cols = ["obs_utc","temp_c","prcp_mm","wspd_kt","wpgt_kt","pres_hpa",
            "snow_mm","wx_group","wdir_deg","wind_dir_known",
            "crosswind_kt","gust_crosswind_kt"]
    rows = []
    notes = []
    for i, r in worst.iterrows():
        case_id   = i + 1
        airport   = r["airport"]
        T         = pd.Timestamp(r["issued_utc"])
        S         = pd.Timestamp(r["block_start_utc"])
        lo        = T - pd.Timedelta(hours=12)
        hi        = S + pd.Timedelta(hours=block_h) + pd.Timedelta(hours=6)
        w = weather_raw[(weather_raw["airport"] == airport)
                         & (weather_raw["obs_utc"] >= lo)
                         & (weather_raw["obs_utc"] <= hi)].sort_values("obs_utc")
        for _, wr in w.iterrows():
            rows.append({
                "case_id":           case_id,
                "airport":           airport,
                "issued_utc":        T,
                "block_start_local": r["block_start_local"],
                "block_start_utc":   S,
                "horizon_h":         int(r["horizon_h"]),
                "y_true":            float(r["y_true"]),
                "y_pred":            float(r["y_pred"]),
                "abs_err":           float(r["abs_err"]),
                **{c: wr[c] for c in cols},
            })
        notes.append(_failure_note(r, w, S, T))

    df = pd.DataFrame(rows)
    df.insert(len(df.columns), "note", df["case_id"].map(dict(enumerate(notes, 1))))
    df.to_csv(out_path, index=False)

    return [{"case_id": i+1,
             "airport": worst.loc[i, "airport"],
             "issued_utc": worst.loc[i, "issued_utc"].isoformat(),
             "block_start_local": worst.loc[i, "block_start_local"].isoformat(),
             "horizon_h": int(worst.loc[i, "horizon_h"]),
             "y_true": float(worst.loc[i, "y_true"]),
             "y_pred": float(worst.loc[i, "y_pred"]),
             "note": notes[i]} for i in range(len(worst))]


def _failure_note(row, window, block_start, issued) -> str:
    err = row["y_true"] - row["y_pred"]
    direction = "under-called" if err > 0 else "over-called"
    before = window[window["obs_utc"] <= issued]
    after  = window[(window["obs_utc"] >  issued) & (window["obs_utc"] <= block_start + pd.Timedelta(hours=6))]
    bad = {"storm","rain","snow","fog","freezing"}
    before_bad = before["wx_group"].isin(bad).sum() if len(before) else 0
    after_bad  = after ["wx_group"].isin(bad).sum() if len(after)  else 0
    if direction == "under-called":
        if after_bad > before_bad:
            return ("model under-called stress — weather at issue time looked benign but "
                    f"{after_bad} bad-wx hours arrived between T and block start")
        return ("model under-called stress — weather features at T didn't flag what the block saw "
                "(small block, lucky run of flights avoided bad wx)")
    if before_bad > after_bad:
        return ("model over-called stress — bad-wx signals before T cleared out before the block, "
                "so flights ran smoother than the features predicted")
    return ("model over-called stress — weather and persistence both pointed higher than realised; "
            "small block sampling noise")


# ================================================================= one full run
def run(block_h: int, flights: pd.DataFrame, weather_raw: pd.DataFrame,
        save_outputs: bool) -> dict:
    print(f"\n=== block_h={block_h} ===")
    blocks = build_blocks(flights, block_h)
    n_before = len(blocks)
    blocks = blocks[blocks["n_flights"] >= 3].reset_index(drop=True)
    dropped = n_before - len(blocks)
    print(f"blocks: kept {len(blocks):,} / {n_before:,} (dropped {dropped} with n_flights<3)")

    blocks = attach_block_times(blocks, block_h)
    preds = expand_horizons(blocks)
    preds = preds.rename(columns={"stress_rate": "y_true"})

    wf = precompute_weather(weather_raw)
    preds = merge_weather_features(preds, wf)
    preds = merge_prev_block(preds, blocks)
    preds = add_calendar(preds)
    preds = add_hist_baseline(preds, block_h)

    # Split by block_start_utc (NOT issued_utc — target time is what defines split)
    tr_mask = preds["block_start_utc"] <= TRAIN_END
    vl_mask = (preds["block_start_utc"] > TRAIN_END) & (preds["block_start_utc"] <= VAL_END)
    te_mask = (preds["block_start_utc"] >= TEST_START) & (preds["block_start_utc"] <= TEST_END)
    train, val, test = preds[tr_mask], preds[vl_mask], preds[te_mask]
    print(f"splits: train={len(train):,}  val={len(val):,}  test={len(test):,}")

    model = train_lgb(train, val)
    print(f"best_iter={model.best_iteration}")
    test = test.copy()
    test["y_pred"] = model.predict(test[FEATURE_COLS], num_iteration=model.best_iteration)
    # persistence baseline = last completed block's stress_rate (= prev_block_stress)
    test["baseline_persist"] = test["prev_block_stress"].fillna(test["baseline_hist"])
    test["y_pred"] = test["y_pred"].clip(0.0, 1.0)
    test["baseline_persist"] = test["baseline_persist"].clip(0.0, 1.0)
    test["baseline_hist"]    = test["baseline_hist"].clip(0.0, 1.0)

    # ---- metrics ----
    def pack(df):
        return {
            "n":               int(len(df)),
            "model_rmse":      rmse(df["y_true"], df["y_pred"]),
            "model_mae":       mae (df["y_true"], df["y_pred"]),
            "persist_rmse":    rmse(df["y_true"], df["baseline_persist"]),
            "persist_mae":     mae (df["y_true"], df["baseline_persist"]),
            "hist_rmse":       rmse(df["y_true"], df["baseline_hist"]),
            "hist_mae":        mae (df["y_true"], df["baseline_hist"]),
        }

    metrics = {
        "target": TARGET_LABEL,
        "block_h": block_h,
        "n_blocks_before_filter": int(n_before),
        "n_blocks_after_filter":  int(len(blocks)),
        "n_dropped_lt3":          int(dropped),
        "target_std_test":        float(test["y_true"].std()),
        "overall": pack(test),
        "by_airport": {a: pack(test[test["airport"] == a]) for a in AIRPORTS},
        "by_horizon": {int(h): pack(test[test["horizon_h"] == h]) for h in HORIZONS},
        "by_airport_horizon": {
            f"{a}_h{h}": pack(test[(test["airport"] == a) & (test["horizon_h"] == h)])
            for a in AIRPORTS for h in HORIZONS
        },
    }

    # ---- generalisation: train on one airport, score on the other ----
    gen = {}
    for tr_ap, te_ap in [("ATL", "DFW"), ("DFW", "ATL")]:
        tr_sub = train[train["airport"] == tr_ap]
        vl_sub = val  [val  ["airport"] == tr_ap]
        if len(tr_sub) < 100 or len(vl_sub) < 20:
            gen[f"{tr_ap}->{te_ap}"] = {"skipped": "not enough rows"}
            continue
        m2 = train_lgb(tr_sub, vl_sub)
        te_sub = test[test["airport"] == te_ap].copy()
        pred = m2.predict(te_sub[FEATURE_COLS], num_iteration=m2.best_iteration)
        pred = np.clip(pred, 0.0, 1.0)
        gen[f"{tr_ap}->{te_ap}"] = {
            "n":    int(len(te_sub)),
            "rmse": rmse(te_sub["y_true"], pred),
            "mae":  mae (te_sub["y_true"], pred),
        }
    metrics["generalisation"] = gen

    if not save_outputs:
        return metrics

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- capacity_preds.parquet (contract) ----
    out_cols = ["airport", "block_start_local", "horizon_h", "issued_utc",
                "y_true", "y_pred", "baseline_persist", "baseline_hist"]
    test[out_cols].to_parquet(OUT_DIR / "capacity_preds.parquet", index=False)

    # ---- top-10 feature importances (gain) ----
    imp = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance_gain": model.feature_importance(importance_type="gain"),
    }).sort_values("importance_gain", ascending=False).head(10).reset_index(drop=True)
    imp.to_csv(OUT_DIR / "capacity_importance.csv", index=False)
    metrics["top_importance_gain"] = imp.to_dict(orient="records")

    # ---- failure cases ----
    metrics["failure_cases"] = save_failures(test, weather_raw, block_h,
                                             OUT_DIR / "capacity_failures.csv")

    return metrics


# ================================================================= cli
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-h", default="6",
                    help="6, 3, or 'both' (runs 6 as primary, 3 as comparison)")
    args = ap.parse_args()

    flights = pd.read_parquet(FLIGHTS)
    weather = pd.read_parquet(WEATHER)
    # normalise datetime precision to ns across both frames (parquet may give us)
    for col in ("sched_local", "sched_utc", "date_local", "wx_obs_utc"):
        if col in flights.columns:
            flights[col] = pd.to_datetime(flights[col]).astype("datetime64[ns]")
    weather["obs_utc"] = pd.to_datetime(weather["obs_utc"]).astype("datetime64[ns]")

    choice = args.block_h.lower()
    if choice == "both":
        primary   = run(6, flights, weather, save_outputs=True)
        secondary = run(3, flights, weather, save_outputs=False)
        comparison = {
            "block_h_6_overall_rmse": primary["overall"]["model_rmse"],
            "block_h_3_overall_rmse": secondary["overall"]["model_rmse"],
            "block_h_6_target_std":   primary["target_std_test"],
            "block_h_3_target_std":   secondary["target_std_test"],
            "less_noisy_block_h":     6 if primary["target_std_test"] <= secondary["target_std_test"] else 3,
            "comment": ("6h blocks pool more flights per block, so the stress_rate target "
                        "is less noisy; 3h blocks are closer to real-time but give higher "
                        "variance and are harder to beat the persistence baseline on."),
        }
        primary["block_size_comparison"] = comparison
        primary["secondary_block_h_3"]   = secondary
        out_metrics = primary
    else:
        bh = int(choice)
        out_metrics = run(bh, flights, weather, save_outputs=True)

    (OUT_DIR / "capacity_metrics.json").write_text(json.dumps(out_metrics, indent=2, default=str))
    print(f"\nwrote {OUT_DIR / 'capacity_preds.parquet'}")
    print(f"wrote {OUT_DIR / 'capacity_metrics.json'}")
    print(f"wrote {OUT_DIR / 'capacity_importance.csv'}")
    print(f"wrote {OUT_DIR / 'capacity_failures.csv'}")
    print(f"\n== TEST (overall) ==")
    o = out_metrics["overall"]
    print(f"  model   rmse={o['model_rmse']:.4f}  mae={o['model_mae']:.4f}")
    print(f"  persist rmse={o['persist_rmse']:.4f}  mae={o['persist_mae']:.4f}")
    print(f"  hist    rmse={o['hist_rmse']:.4f}  mae={o['hist_mae']:.4f}")


if __name__ == "__main__":
    main()
