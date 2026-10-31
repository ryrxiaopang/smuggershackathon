"""Leakage + correctness tests for src/delay_model.py (run: python tests/test_delay_leakage.py [raw_csv]).

Brute-force reference implementations (plain pandas filters on obs_utc <= T / complete_utc < T) are compared with
the vectorised features; plus a perturbation test that scrambles every weather row after T and checks nothing moves.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import delay_model as dm  # noqa: E402

CSV = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data/raw/Flight_Data_With_Weather_Final.csv")
HEAD = dm.runway_headings()
sides, tl = dm.load_raw(Path(CSV), ["ATL", "DFW"], None, HEAD)
sides = dm.prepare_sides(sides)
vocab = {"side": sorted(sides.side.unique()), "airport": ["ATL", "DFW"], "carrier": sorted(sides.carrier.unique())}
tz_of = {"ATL": "America/New_York", "DFW": "America/Chicago"}
store = dm.build_weather_store(tl)
rng = np.random.default_rng(0)
HR = pd.Timedelta(hours=1)


def test_no_label_columns_in_features():
    banned = {"delay_min", "nas_delay", "weather_delay", "delayed_45", "cancelled", "diverted", "stress_flag", "complete_utc", "dep_delay", "arr_delay"}
    feats = set(dm.CAL_FEATS + dm.WX_FEATS + dm.CONG_FEATS)
    assert not feats & banned, feats & banned


def test_time_conversion_atl_dfw_elapsed():
    """ATL<->DFW flights appear as dep at one airport and arr at the other; UTC gap must equal CRS_ELAPSED (tz sanity)."""
    a = sides[sides.other_airport.isin(["ATL", "DFW"])]
    dep = a[a.side == "dep"].set_index("flight_id")
    arr = a[a.side == "arr"].set_index("flight_id")
    j = dep.join(arr, lsuffix="_d", rsuffix="_a", how="inner")
    gap = (j.sched_utc_a - j.sched_utc_d).dt.total_seconds() / 60
    assert len(j) > 100 and gap.between(60, 200).all(), gap.describe()
    off_atl = (sides[sides.airport == "ATL"].sched_local - sides[sides.airport == "ATL"].sched_utc).dt.total_seconds() / 3600
    assert set(np.round(-off_atl.unique())) <= {4.0, 5.0}


def _brute_weather(ap, T, h_floor):
    g = tl[(tl.airport == ap) & (tl.obs_utc <= T)].sort_values("obs_utc")  # <-- the leakage rule, spelled out
    out = {}
    if g.empty:
        return None
    last = g.iloc[-1]
    out["wspd_last"], out["prcp_last"], out["pres_last"] = last.wspd_kt, last.prcp_mm, last.pres_hpa
    out["obs_age_h"] = (T - last.obs_utc) / HR
    for w in dm.WINDOWS:
        win = g[g.obs_utc.dt.floor("h") > h_floor - w * HR]
        out[f"gust_crosswind_{w}h_max"] = win.gust_crosswind_kt.max() if len(win) else np.nan
        out[f"prcp_{w}h_max"] = win.prcp_mm.max() if len(win) else np.nan
        out[f"storm_count_{w}h"] = float((win.wx_group == "storm").sum()) if len(win) else np.nan
        out[f"fog_count_{w}h"] = float((win.wx_group == "fog").sum()) if len(win) else np.nan
    prev = g[g.obs_utc.dt.floor("h") <= h_floor - 3 * HR]
    out["pres_chg_3h"] = (last.pres_hpa - prev.iloc[-1].pres_hpa) if len(prev) and prev.iloc[-1].obs_utc != last.obs_utc else np.nan
    return out


def test_weather_features_match_bruteforce_and_use_only_past():
    for h in dm.HORIZONS:
        X, T = dm.build_features(sides, store, h, vocab, tz_of)
        for i in rng.choice(len(sides), 150, replace=False):
            ref = _brute_weather(sides.airport[i], T[i], T[i].floor("h"))
            if ref is None:
                continue
            for k, v in ref.items():
                got = X.at[i, k]
                assert (np.isnan(v) and np.isnan(got)) or np.isclose(v, got, atol=1e-6), (h, i, k, v, got)


def test_perturbing_future_weather_changes_nothing():
    for h in dm.HORIZONS:
        for i in rng.choice(len(sides), 12, replace=False):
            row = sides.loc[[i]]
            T = row.sched_utc.iloc[0] - h * HR
            tl2 = tl.copy()
            fut = (tl2.airport == row.airport.iloc[0]) & (tl2.obs_utc > T)
            for c in ("wspd_kt", "wpgt_kt", "prcp_mm", "snow_mm", "crosswind_kt", "gust_crosswind_kt", "pres_hpa", "temp_c"):
                tl2.loc[fut, c] = 9999.0
            tl2.loc[fut, "wx_group"] = "storm"
            a = dm.build_features(row, dm.build_weather_store(tl), h, vocab, tz_of)[0][dm.WX_FEATS]
            b = dm.build_features(row, dm.build_weather_store(tl2), h, vocab, tz_of)[0][dm.WX_FEATS]
            pd.testing.assert_frame_equal(a, b)


def test_congestion_matches_bruteforce():
    for h in (3, 12):
        X, T = dm.build_features(sides, store, h, vocab, tz_of)
        for i in rng.choice(len(sides), 150, replace=False):
            ap, t = sides.airport[i], T[i]
            day0 = t.tz_localize("UTC").tz_convert(tz_of[ap]).normalize().tz_convert("UTC").tz_localize(None)
            d = sides[(sides.airport == ap) & (sides.index != i) & (sides.complete_utc >= day0) & (sides.complete_utc < t)]
            exp = d.stress_flag.mean() if len(d) else np.nan
            got = X.at[i, "cong_share"]
            assert (np.isnan(exp) and np.isnan(got)) or np.isclose(exp, got), (h, i, exp, got)
            assert X.at[i, "cong_n"] == len(d)


def test_congestion_ignores_flights_completing_after_T():
    h = 6
    i = int(rng.integers(len(sides)))
    row = sides.loc[[i]]
    T = row.sched_utc.iloc[0] - h * HR
    s2 = sides.copy()
    late = (s2.airport == row.airport.iloc[0]) & (s2.complete_utc >= T)
    s2.loc[late, "stress_flag"] = ~s2.loc[late, "stress_flag"]  # flip every not-yet-known outcome
    a = dm.congestion_features(sides, sides.sched_utc - h * HR, tz_of).loc[[i]]
    b = dm.congestion_features(s2, s2.sched_utc - h * HR, tz_of).loc[[i]]
    pd.testing.assert_frame_equal(a, b)


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e)[:400])
    sys.exit(1 if fails else 0)
