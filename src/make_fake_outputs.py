"""Generate SYNTHETIC outputs matching the data contract so the dashboard runs before real
models exist. Never overwrites a real (non-synthetic) file unless --force.
Usage: python src/make_fake_outputs.py [--force] [--seed 7]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src import dash_config as C

AIRPORTS = ["ATL", "DFW"]
DAILY_VOL = {"ATL": 14.6, "DFW": 12.5}      # sampled flights/day in the provided CSV
DEP_SHARE = {"ATL": 0.61, "DFW": 0.66}
MAIN_CARRIER = {"ATL": "DL", "DFW": "AA"}
OTHER_CARRIERS = ["WN", "UA", "NK", "F9", "B6", "AS"]
COUNTERPARTS = ["LAX", "ORD", "JFK", "MCO", "DEN", "LGA", "CLT", "LAS", "PHX", "SEA", "BOS", "MIA"]
STORM_DAYS = {"ATL": ["2025-01-10", "2025-03-15", "2025-04-06", "2025-05-20", "2025-06-12", "2025-07-08"],
              "DFW": ["2025-01-21", "2025-03-04", "2025-04-28", "2025-05-28", "2025-06-17", "2025-07-15"]}
COCO = {"clear": 1, "cloudy": 3, "fog": 5, "rain": 8, "showers": 17, "freezing": 10, "snow": 15, "thunderstorm": 25}


def _sig(x):
    return 1 / (1 + np.exp(-x))


def _decay(h):
    return max(0.3, 1 - 0.075 * (h - 3))


def _r(x):
    x = float(x)
    return None if np.isnan(x) else round(x, 4)


def _is_real(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        if path.suffix == ".parquet":
            return "is_synthetic" not in pq.read_schema(path).names
        if path.suffix == ".json":
            return not json.loads(path.read_text()).get("is_synthetic", False)
    except Exception:
        return True
    return True


def make_weather(rng):
    idx = pd.date_range("2024-12-31 00:00", "2025-08-01 23:00", freq="h", tz="UTC")
    n = len(idx)
    hr, month, doy = idx.hour.to_numpy(), idx.month.to_numpy(), idx.dayofyear.to_numpy()
    frames, sev_true, sev_obs = [], {}, {}
    for ap in AIRPORTS:
        hdg = C.RUNWAY_HEADING[ap]
        e = rng.normal(0, 2.2, n)
        w = np.empty(n)
        w[0] = 8
        for i in range(1, n):
            w[i] = 0.85 * w[i - 1] + 0.15 * 9 + e[i]
        w = np.clip(w + 2.5 * np.sin((hr - 14) / 24 * 2 * np.pi), 0, None)
        storm = np.zeros(n, bool)
        for d in STORM_DAYS[ap]:
            storm |= np.asarray((idx >= pd.Timestamp(f"{d} 16:00", tz="UTC")) & (idx < pd.Timestamp(f"{d} 23:00", tz="UTC")))
        w[storm] += rng.uniform(8, 14, storm.sum())
        gust = w + np.abs(rng.normal(5, 3, n))
        gust[storm] += 10
        drct = (200 + np.cumsum(rng.normal(0, 12, n))) % 360
        prcp = np.where(rng.random(n) < 0.05, rng.exponential(1.5, n), 0.0)
        prcp[storm] = rng.exponential(8, storm.sum())
        group = np.where(rng.random(n) < 0.55, "clear", "cloudy").astype(object)
        group[(rng.random(n) < 0.015) & np.isin(hr, [9, 10, 11, 12, 13])] = "fog"
        group[prcp > 0] = "rain"
        group[(prcp > 0) & (prcp < 1) & (rng.random(n) < 0.3)] = "showers"
        group[(prcp > 0) & np.isin(month, [1, 2]) & (rng.random(n) < 0.2)] = "freezing"
        thunder = storm & (rng.random(n) < 0.75)
        group[thunder] = "thunderstorm"
        known = (rng.random(n) > 0.10) & (w >= 3)
        known[storm & (hr == 16)] = False                     # guarantee an upper-bound demo case
        s = np.abs(np.sin(np.radians(drct - hdg)))
        true_gxw = gust * s
        xw = np.where(known, w * s, w)
        gxw = np.where(known, true_gxw, gust)
        tw = np.where(known, np.clip(-w * np.cos(np.radians(drct - hdg)), 0, None), w)
        temp = 17 + 10 * np.sin((doy - 110) / 365 * 2 * np.pi) + rng.normal(0, 2, n)
        base = 0.30 * thunder + 0.08 * (prcp > 2) + 0.15 * (group == "freezing") + 0.05 * (group == "fog")
        sev_true[ap] = pd.Series(0.02 * np.clip(true_gxw - 12, 0, None) + base, index=idx)
        sev_obs[ap] = pd.Series(0.02 * np.clip(gxw - 12, 0, None) + base, index=idx)
        frames.append(pd.DataFrame({
            "airport": ap, "obs_utc": idx, "runway_heading": hdg, "drct": np.where(known, np.round(drct), np.nan),
            "wspd_kt": w.round(1), "gust_kt": gust.round(1), "crosswind_kt": xw.round(1),
            "gust_crosswind_kt": gxw.round(1), "tailwind_kt": tw.round(1), "wind_dir_known": known,
            "wind_source": np.where(known, "IEM ASOS", "fallback: upper bound"),
            "prcp_mm": prcp.round(1), "temp_c": temp.round(1), "coco": [COCO[g] for g in group],
            "wx_group": group.astype(str), "is_synthetic": True}))
    return pd.concat(frames, ignore_index=True), sev_true, sev_obs


def make_capacity(rng, sev_true, sev_obs):
    B = C.BLOCK_HOURS
    starts = pd.date_range(f"{C.TEST_START} 00:00", f"{C.TEST_END} {24 - B:02d}:00", freq=f"{B}h", tz="UTC")
    out = []
    for ap in AIRPORTS:
        rt = sev_true[ap].rolling(3, min_periods=1).mean()
        ro = sev_obs[ap].rolling(3, min_periods=1).mean()
        hind = rt.reindex(starts + pd.Timedelta(hours=B - 1)).to_numpy()
        lh = starts.tz_convert(C.AIRPORT_TZ[ap]).hour.to_numpy()
        diurnal = np.where((lh >= 15) & (lh <= 21), 0.05, 0.0)
        active = np.where((lh >= 6) & (lh <= 21), 1.0, 0.08)
        n = rng.poisson(DAILY_VOL[ap] * active / (active.mean() * 24 / B))
        true_rate = np.clip(0.07 + 0.9 * hind + diurnal + rng.normal(0, 0.03, len(starts)), 0.01, 0.95)
        actual = np.where(n > 0, rng.binomial(n, true_rate) / np.maximum(n, 1), np.nan)
        p50, p80, p90 = np.quantile(true_rate, [0.5, 0.8, 0.9])
        for h in C.HORIZONS:
            issue = starts - pd.Timedelta(hours=h)
            lag = np.nan_to_num(ro.reindex(issue).to_numpy())
            pred = np.clip(0.085 + diurnal + 0.9 * _decay(h) * lag + rng.normal(0, 0.02, len(starts)), 0.01, 0.95)
            out.append(pd.DataFrame({
                "airport": ap, "block_start_utc": starts, "block_end_utc": starts + pd.Timedelta(hours=B),
                "horizon_h": h, "issue_time_utc": issue, "stress_rate_pred": pred.round(4),
                "stress_rate_actual": actual, "n_flights": n, "hist_p50": p50, "hist_p80": p80, "hist_p90": p90,
                "model": "synthetic", "is_synthetic": True}))
    return pd.concat(out, ignore_index=True)


def _factors(wc, ev, ar, t, grp, gx, kn):
    items = []
    if wc > 0.05 and pd.notna(gx):
        known = (kn is True) or (kn == 1)
        items.append((wc, f"wx obs {t:%H:%MZ}: {grp}, gust xwind {'' if known else '≤'}{gx:.0f}kt"))
    if ev:
        items.append((0.45, "evening bank (15–21 local)"))
    if ar:
        items.append((0.25, "arrival side"))
    items.sort(key=lambda x: -x[0])
    return "; ".join(f"{txt} (+{c:.2f} log-odds)" for c, txt in items[:3]) or "no elevated factors"


def make_flights(rng, sev_true, sev_obs, wx):
    days = pd.date_range(C.TEST_START, C.TEST_END, freq="D")
    out = []
    for ap in AIRPORTS:
        tz = C.AIRPORT_TZ[ap]
        nd = rng.poisson(DAILY_VOL[ap], len(days))
        N = int(nd.sum())
        off = np.round(rng.uniform(5.5, 23.0, N) * 12) / 12
        local = pd.Series(pd.to_datetime(np.repeat(days.values, nd)) + pd.to_timedelta(off, unit="h"))
        utc = local.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
        ok = utc.notna().to_numpy()
        local, utc, N = local[ok].reset_index(drop=True), utc[ok].reset_index(drop=True), int(ok.sum())
        rt = sev_true[ap].rolling(3, min_periods=1).mean()
        ro = sev_obs[ap].rolling(3, min_periods=1).mean()
        hind = np.nan_to_num(rt.reindex(pd.DatetimeIndex(utc.dt.floor("h"))).to_numpy())
        lh = local.dt.hour.to_numpy()
        evening = ((lh >= 15) & (lh <= 21)).astype(float)
        side = np.where(rng.random(N) < DEP_SHARE[ap], "DEP", "ARR")
        arr = (side == "ARR").astype(float)
        carrier = np.where(rng.random(N) < 0.7, MAIN_CARRIER[ap], rng.choice(OTHER_CARRIERS, N))
        cp = rng.choice([c for c in COUNTERPARTS + AIRPORTS if c != ap], N)
        fid = [f"{c}{k}" for c, k in zip(carrier, rng.integers(100, 3000, N))]
        actual = (rng.random(N) < _sig(-2.4 + 4.0 * hind + 0.5 * evening + 0.3 * arr + rng.normal(0, 0.4, N))).astype(int)
        wxa = wx[wx.airport == ap].set_index("obs_utc")
        for h in C.HORIZONS:
            iidx = pd.DatetimeIndex((utc - pd.Timedelta(hours=h)).dt.floor("h"))
            lag = np.nan_to_num(ro.reindex(iidx).to_numpy())
            wc = 3.0 * _decay(h) * lag
            p = _sig(-2.3 + wc + 0.45 * evening + 0.25 * arr + rng.normal(0, 0.15, N))
            ob = wxa.reindex(iidx)
            fac = [_factors(*a) for a in zip(wc, evening, arr, iidx, ob["wx_group"].to_numpy(),
                                              ob["gust_crosswind_kt"].to_numpy(), ob["wind_dir_known"].to_numpy())]
            out.append(pd.DataFrame({
                "flight_id": fid, "airport": ap, "side": side, "carrier": carrier, "counterpart": cp,
                "FL_DATE": local.dt.normalize(), "sched_local": local, "sched_utc": utc, "horizon_h": h,
                "issue_time_utc": iidx, "p_delay45": p.round(4), "top_factors": fac,
                "actual_delay45": actual, "model": "synthetic", "is_synthetic": True}))
    return pd.concat(out, ignore_index=True)


def make_metrics(fl, cap):
    h0 = 6 if 6 in C.HORIZONS else C.HORIZONS[0]
    f = fl[fl.horizon_h == h0]
    y, p = f.actual_delay45.to_numpy(), f.p_delay45.to_numpy()
    lh = f.sched_local.dt.hour.to_numpy()
    cal = _sig(-2.3 + 0.45 * ((lh >= 15) & (lh <= 21)) + 0.25 * (f.side.to_numpy() == "ARR"))
    br = y.mean()
    models = [
        {"name": "Base rate (always prior)", "horizon_h": h0, "auc": 0.5, "pr_auc": _r(br), "brier": _r(brier_score_loss(y, np.full_like(p, br)))},
        {"name": "Calendar-only", "horizon_h": h0, "auc": _r(roc_auc_score(y, cal)), "pr_auc": _r(average_precision_score(y, cal)), "brier": _r(brier_score_loss(y, cal))},
        {"name": "Model (calendar + lagged obs)", "horizon_h": h0, "auc": _r(roc_auc_score(y, p)), "pr_auc": _r(average_precision_score(y, p)), "brier": _r(brier_score_loss(y, p))},
    ]
    q = pd.qcut(p, 10, duplicates="drop")
    cb = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q", observed=True).agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    gen = []
    for test in AIRPORTS:
        g = fl[(fl.horizon_h == h0) & (fl.airport == test)]
        a, pr = roc_auc_score(g.actual_delay45, g.p_delay45), average_precision_score(g.actual_delay45, g.p_delay45)
        other = [x for x in AIRPORTS if x != test][0]
        gen.append({"train": test, "test": test, "auc": _r(a), "pr_auc": _r(pr)})
        gen.append({"train": other, "test": test, "auc": _r(a - 0.02), "pr_auc": _r(pr - 0.01)})
    fm = {"task": "Flight >45-min delay risk", "is_synthetic": True, "test_period": f"{C.TEST_START}..{C.TEST_END}",
          "base_rate": _r(br), "primary_metric": "pr_auc", "models": models,
          "calibration": {"bins_pred": cb.pred.round(4).tolist(), "bins_obs": cb.obs.round(4).tolist(), "bins_n": cb.n.astype(int).tolist()},
          "generalisation": gen,
          "failure_cases": [
              {"date": "2025-05-28", "airport": "DFW", "what": "Low predicted risk at h=12, then a late-afternoon thunderstorm line.",
               "why": "Storm formed after issue time; observations cannot see the future — needs TAF/convective forecasts."},
              {"date": "2025-03-21", "airport": "ATL", "what": "Many >45-min delays in fair weather.",
               "why": "Upstream disruption (late aircraft / NAS) — no local weather signal to learn from."}]}
    B = C.BLOCK_HOURS
    rows = []
    for h in C.HORIZONS:
        g = cap[cap.horizon_h == h].sort_values(["airport", "block_start_utc"]).copy()
        steps = int(np.ceil(h / B)) + 1      # last block that ENDED before issue time (no leakage)
        g["persist"] = g.groupby("airport").stress_rate_actual.transform(lambda s: s.ffill().shift(steps))
        g["clim"] = g.groupby("airport").stress_rate_actual.transform("mean")
        g = g[(g.n_flights > 0)].dropna(subset=["stress_rate_actual", "persist"])
        ev = ((g.stress_rate_actual >= 0.5) & (g.n_flights >= 2)).astype(int)
        for name, col in (("Climatology (airport mean)", "clim"), ("Persistence (last completed block)", "persist"), ("Model", "stress_rate_pred")):
            auc = roc_auc_score(ev, g[col]) if ev.nunique() == 2 and g[col].nunique() > 1 else np.nan
            rows.append({"name": name, "horizon_h": h, "mae_per100": _r(100 * (g[col] - g.stress_rate_actual).abs().mean()), "auc_stress_event": _r(auc)})
    cm = {"task": "Capacity stress (block delay-rate proxy)", "is_synthetic": True,
          "test_period": f"{C.TEST_START}..{C.TEST_END}", "primary_metric": "mae_per100",
          "stress_definition": "Share of sampled flights at the airport in a 3h block with >45 min delay, cancellation or diversion (proxy: no true AAR in sample data).",
          "models": rows, "failure_cases": []}
    return fm, cm


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite real files too")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    C.OUTPUTS.mkdir(parents=True, exist_ok=True)
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    wx, st_, so = make_weather(rng)
    cap = make_capacity(rng, st_, so)
    fl = make_flights(rng, st_, so, wx)
    fm, cm = make_metrics(fl, cap)
    targets = {C.WEATHER_TIMELINE: wx, C.CAPACITY_PREDS: cap, C.FLIGHT_PREDS: fl,
               C.OUTPUTS / "flight_risk_metrics.json": fm, C.OUTPUTS / "capacity_metrics.json": cm}
    for path, obj in targets.items():
        if _is_real(path) and not a.force:
            print(f"skip (real file exists): {path}")
            continue
        if isinstance(obj, pd.DataFrame):
            obj.to_parquet(path, index=False)
        else:
            path.write_text(json.dumps(obj, indent=2))
        print(f"wrote SYNTHETIC {path}")


if __name__ == "__main__":
    main()