"""Never-seen-airport test: fit both models once on ATL + DFW, freeze them, and score other airports
without retraining.

    python src/score_new_airports.py                      # ORD DEN SEA (default)
    python src/score_new_airports.py --airports KLAX PHX  # any airports in the CSV (FAA or ICAO codes)
    python src/score_new_airports.py --rebuild            # rebuild the processed data first

How it stays honest
  - Both models are fitted on ATL + DFW only: train 2023-01 -> 2024-06, early stopping and calibration on
    validation 2024-07 -> 2024-12. Then they are frozen.
  - The new airports are scored on the 2025 test period only. None of their data is used to fit or tune.
  - Airport identity is left out of the features (a new airport has no value for it). Everything else is
    the same feature set and the same model settings as src/delay_model.py and src/capacity_model.py.
  - Weather comes from the flight CSV for every airport here (the organiser hourly file only covers
    ATL/DFW), so training and scoring see the same kind of weather data.
  - Baselines: delay -> guess the average, and a timetable-only model frozen the same way;
    congestion -> "same as last block", and that airport's own 2023-24 average for the block and month
    (history a local ops desk would have; the model itself uses none of it).
  - ATL and DFW are scored too, with the same frozen models, as the "seen airports" reference.

Writes outputs/new_airport_metrics.json (shown in the dashboard, tab 3).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import config  # noqa: E402
from airports import ensure as ensure_airports  # noqa: E402
import capacity_model as cm  # noqa: E402
import delay_model as dm  # noqa: E402

SEEN = ["ATL", "DFW"]
NS = "datetime64[ns]"
OUT = config.OUTPUTS / "new_airport_metrics.json"


def load(codes, rebuild=False):
    """Processed flight sides + timeline (flight-CSV weather) for these airports, built with the pipeline."""
    out_dir = config.PROCESSED / f"flightwx_{'_'.join(codes)}"
    fs_p, tl_p = out_dir / config.FLIGHT_SIDES.name, out_dir / config.WEATHER_TIMELINE.name
    if rebuild or not fs_p.exists():
        subprocess.run([sys.executable, str(ROOT / "src" / "pipeline.py"), "--airports", *codes,
                        "--out-dir", str(out_dir), "--no-hourly"], check=True)
    fs, tl = pd.read_parquet(fs_p), pd.read_parquet(tl_p)
    for c in ("sched_local", "sched_utc", "date_local", "wx_obs_utc"):
        fs[c] = pd.to_datetime(fs[c]).astype(NS)
    tl["obs_utc"] = pd.to_datetime(tl["obs_utc"]).astype(NS)
    return fs, tl


def top_share(y, p, frac=0.10):
    k = max(int(len(y) * frac), 1)
    return float(np.asarray(y)[np.argsort(-np.asarray(p), kind="stable")[:k]].mean())


# ---------------------------------------------------------------------------------- delay model
def delay_test(seen, new, codes):
    feats_full = [f for f in dm.CAL_FEATS + dm.WX_FEATS + dm.CONG_FEATS if f != "airport"]
    feats_cal = [f for f in dm.CAL_FEATS if f != "airport"]
    tz_of = dict(config.AIRPORT_TZ)
    tr = dm.prepare_sides(seen[0])
    nw = dm.prepare_sides(new[0])
    for s in (tr, nw):
        s["split"] = dm.assign_split(s, config.TRAIN_END, config.VAL_END)
    vocab = {"side": sorted(tr["side"].unique()), "airport": SEEN, "carrier": sorted(tr["carrier"].dropna().unique())}
    st_tr, st_nw = dm.build_weather_store(seen[1]), dm.build_weather_store(new[1])
    out = {}
    for h in config.HORIZONS_H:
        Xtr, _ = dm.build_features(tr, st_tr, h, vocab, tz_of)
        Xnw, _ = dm.build_features(nw, st_nw, h, vocab, tz_of)
        y = tr["delayed_45"]
        lab = y.notna()
        itr, iva = tr.index[(tr["split"] == "train") & lab], tr.index[(tr["split"] == "val") & lab]
        fit = lambda f: dm.fit_model(Xtr.loc[itr], y.loc[itr].astype(int).values,  # noqa: E731
                                     Xtr.loc[iva], y.loc[iva].astype(int).values, f)
        full, cal = fit(feats_full), fit(feats_cal)  # fitted once, then frozen
        res = {}
        for name, s, X in [(ap, nw, Xnw) for ap in codes] + [(ap, tr, Xtr) for ap in SEEN]:
            m = (s["airport"] == name) & (s["split"] == "test") & s["delayed_45"].notna()
            if m.sum() < 50:
                continue
            yt = s.loc[m, "delayed_45"].astype(int).values
            pf, pc = dm.predict(full, X.loc[m])[0], dm.predict(cal, X.loc[m])[0]
            sf, sc = dm.scores(yt, pf), dm.scores(yt, pc)
            res[name] = {"seen_in_training": name in SEEN, "flights": int(m.sum()), "late_per_100": 100 * float(yt.mean()),
                         "pr_auc_model": sf["pr_auc"], "pr_auc_timetable": sc["pr_auc"], "pr_auc_guess": float(yt.mean()),
                         "roc_auc_model": sf["roc_auc"], "roc_auc_timetable": sc["roc_auc"],
                         "top10_late_share": top_share(yt, pf), "top10_lift": top_share(yt, pf) / max(yt.mean(), 1e-9)}
        out[f"h{h}"] = res
        print(f"[delay h={h}] " + " | ".join(f"{k} PR-AUC {v['pr_auc_model']:.3f} vs {v['pr_auc_timetable']:.3f}"
                                             for k, v in res.items()))
    return out


# ---------------------------------------------------------------------------------- capacity model
def capacity_table(fs, tl):
    blocks = cm.build_blocks(fs, config.BLOCK_HOURS)
    blocks = blocks[blocks["n_flights"] >= 3].reset_index(drop=True)
    blocks = cm.attach_block_times(blocks, config.BLOCK_HOURS)
    p = cm.expand_horizons(blocks).rename(columns={"stress_rate": "y_true"})
    p = cm.merge_weather_features(p, cm.precompute_weather(tl))
    p = cm.merge_prev_block(p, blocks)
    p = cm.add_calendar(p)
    return cm.add_hist_baseline(p, config.BLOCK_HOURS)


def capacity_test(seen, new, codes):
    import lightgbm as lgb
    cm.AIRPORT_TZ.update(config.AIRPORT_TZ)  # the module ships with ATL/DFW only
    feats = [c for c in cm.FEATURE_COLS if not c.startswith("airport_")]
    P, N = capacity_table(*seen), capacity_table(*new)
    tr = P[P["block_start_utc"] <= cm.TRAIN_END]
    va = P[(P["block_start_utc"] > cm.TRAIN_END) & (P["block_start_utc"] <= cm.VAL_END)]
    model = lgb.train(cm.LGB_PARAMS, lgb.Dataset(tr[feats], label=tr["y_true"]), num_boost_round=1200,
                      valid_sets=[lgb.Dataset(va[feats], label=va["y_true"])], valid_names=["val"],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    rmse = lambda y, p: 100 * float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))  # noqa: E731
    out = {}
    for name, df in [(ap, N) for ap in codes] + [(ap, P) for ap in SEEN]:
        te = df[(df["airport"] == name) & (df["block_start_utc"] >= cm.TEST_START) & (df["block_start_utc"] <= cm.TEST_END)].copy()
        if len(te) < 30:
            continue
        te["pred"] = np.clip(model.predict(te[feats], num_iteration=model.best_iteration), 0, 1)
        te["persist"] = te["prev_block_stress"].fillna(te["baseline_hist"]).clip(0, 1)
        te["hist"] = te["baseline_hist"].clip(0, 1)
        out[name] = {"seen_in_training": name in SEEN, "blocks": int(te["block_start_utc"].nunique()), "rows": int(len(te)),
                     "rmse_model": rmse(te["y_true"], te["pred"]), "rmse_hist": rmse(te["y_true"], te["hist"]),
                     "rmse_persist": rmse(te["y_true"], te["persist"]),
                     "by_horizon": {f"h{h}": {"rmse_model": rmse(g["y_true"], g["pred"]), "rmse_hist": rmse(g["y_true"], g["hist"]),
                                              "rmse_persist": rmse(g["y_true"], g["persist"])}
                                    for h, g in te.groupby("horizon_h")}}
        v = out[name]
        print(f"[congestion] {name}: RMSE {v['rmse_model']:.1f} vs historical {v['rmse_hist']:.1f} vs same-as-last {v['rmse_persist']:.1f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--airports", nargs="+", default=["ORD", "DEN", "SEA"])
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    codes = [c for c in ensure_airports(a.airports) if c not in SEEN]
    print(f"Fitting on {SEEN} (frozen), scoring never-seen airports {codes} on the 2025 test period")
    seen, new = load(SEEN, a.rebuild), load(codes, a.rebuild)
    metrics = {"trained_on": SEEN, "never_seen": codes, "test_period": f"{config.TEST_START} to {config.TEST_END}",
               "weather_source": "flight CSV for all airports (same kind of data in training and scoring)",
               "delay": delay_test(seen, new, codes), "capacity": capacity_test(seen, new, codes)}
    config.OUTPUTS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
