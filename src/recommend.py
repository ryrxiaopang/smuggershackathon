"""Pure rule-based recommendations: no ML, no LLM, no file I/O.

Every rule returns Rec(level, action, reason, evidence). Evidence lists the exact
values and timestamps used. Only data available at decision time (obs/issue <= now)
is used.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from src.dash_config import BLOCK_HOURS, RULES, RUNWAY_HEADING, coco_to_group

LEVELS = ("NORMAL", "WATCH", "ACTION")
_RANK = {lvl: i for i, lvl in enumerate(LEVELS)}


class Rec(NamedTuple):
    level: str
    action: str
    reason: str
    evidence: list


# ---------- helpers ----------
def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _fmt(ts) -> str:
    if ts is None or (not isinstance(ts, str) and pd.isna(ts)):
        return ""
    return _utc(ts).strftime("%Y-%m-%d %H:%MZ")


def _py(v):
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else round(float(v), 3)
    return v


def _ev(signal, value, unit="", ts=None, source="", threshold=None) -> dict:
    return {"signal": signal, "value": _py(value), "unit": unit,
            "obs_or_valid_time": _fmt(ts), "source": source, "threshold": threshold}


def overall_level(recs) -> str:
    recs = list(recs)
    return max((r.level for r in recs), key=_RANK.get) if recs else "NORMAL"


def normalise_weather(wx: pd.DataFrame) -> pd.DataFrame:
    w = wx.copy()
    if "wx_group" not in w.columns:
        w["wx_group"] = w["coco"].map(coco_to_group) if "coco" in w.columns else "unknown"
    w["wx_group"] = w["wx_group"].fillna("unknown").astype(str)
    if "crosswind_kt" not in w.columns:
        w["crosswind_kt"] = np.nan
    if "gust_crosswind_kt" not in w.columns:
        w["gust_crosswind_kt"] = w["crosswind_kt"]
    if "wind_dir_known" not in w.columns:
        w["wind_dir_known"] = w["drct"].notna() if "drct" in w.columns else False
    w["wind_dir_known"] = w["wind_dir_known"].fillna(False).astype(bool)
    if "prcp_mm" not in w.columns:
        w["prcp_mm"] = 0.0
    return w.sort_values("obs_utc").reset_index(drop=True)


def select_forecasts(df: pd.DataFrame, now_utc, time_col: str) -> pd.DataFrame:
    """For each target time keep the forecast with the smallest horizon >= lead time,
    then drop anything issued after `now` (leakage guard)."""
    if df is None or df.empty:
        return df
    now = _utc(now_utc)
    out = df
    if "horizon_h" in df.columns and df["horizon_h"].notna().any():
        hs = np.sort(df["horizon_h"].dropna().unique().astype(float))
        lead = ((df[time_col] - now).dt.total_seconds() / 3600).clip(lower=0).to_numpy()
        i = np.searchsorted(hs, lead - 1e-9, side="left")
        need = np.where(i < len(hs), hs[np.minimum(i, len(hs) - 1)], np.nan)
        out = df[df["horizon_h"].astype(float).to_numpy() == need]
    if "issue_time_utc" in out.columns:
        out = out[out["issue_time_utc"] <= now]
    return out


def window_blocks(cap, now_utc, horizon_h):
    if cap is None or cap.empty:
        return cap
    now = _utc(now_utc)
    s = select_forecasts(cap, now, "block_start_utc")
    return s[(s["block_start_utc"] > now - pd.Timedelta(hours=BLOCK_HOURS))
             & (s["block_start_utc"] < now + pd.Timedelta(hours=horizon_h))].sort_values("block_start_utc")


def window_flights(fl, now_utc, horizon_h):
    if fl is None or fl.empty:
        return fl
    now = _utc(now_utc)
    s = select_forecasts(fl, now, "sched_utc")
    return s[(s["sched_utc"] > now) & (s["sched_utc"] <= now + pd.Timedelta(hours=horizon_h))]


def _wx_window(wx, now, lookback_h):
    if wx is None or wx.empty:
        return wx
    return wx[(wx["obs_utc"] <= now) & (wx["obs_utc"] > now - pd.Timedelta(hours=lookback_h))]


# ---------- rules ----------
def rule_stress(blocks: pd.DataFrame, now_utc, cfg=RULES) -> Rec:
    now = _utc(now_utc)
    if blocks is None or blocks.empty:
        return Rec("NORMAL", "No action from this rule.", "No capacity forecast available for this window.", [])
    b = blocks
    if "issue_time_utc" in b.columns:
        b = b[b["issue_time_utc"] <= now]
    b = b.dropna(subset=["stress_rate_pred"])
    wc, ac = cfg["stress_watch_col"], cfg["stress_action_col"]
    if b.empty or wc not in b.columns or ac not in b.columns:
        return Rec("NORMAL", "No action from this rule.", "No usable forecast/percentiles at decision time.", [])
    r = b.loc[b["stress_rate_pred"].idxmax()]
    pred, p80, p90 = float(r["stress_rate_pred"]), float(r[wc]), float(r[ac])
    start = r["block_start_utc"]
    end = r["block_end_utc"] if "block_end_utc" in r.index and pd.notna(r["block_end_utc"]) \
        else start + pd.Timedelta(hours=BLOCK_HOURS)
    block = f"{_fmt(start)}–{_utc(end):%H:%MZ}"
    h = r.get("horizon_h")
    src = "capacity_preds.parquet"
    if h is not None and pd.notna(h):
        src += f" (horizon {int(h)}h, issued {_fmt(r.get('issue_time_utc'))})"
    ev = [_ev("predicted stress_rate (peak block)", pred, "share of flights", start, src),
          _ev("airport historical p80", p80, "share of flights", None, "train period", "WATCH if above"),
          _ev("airport historical p90", p90, "share of flights", None, "train period", "ACTION if above")]
    txt = f"Predicted stress {pred*100:.0f}/100 flights for block {block}"
    if pred > p90:
        return Rec("ACTION", f"Consider reducing arrival rate / ground delay program for block {block}.",
                   f"{txt} is above the airport's 90th percentile ({p90*100:.0f}/100).", ev)
    if pred > p80:
        return Rec("WATCH", f"Pre-brief ATC flow coordinator; review arrival rate for block {block}.",
                   f"{txt} is above the airport's 80th percentile ({p80*100:.0f}/100).", ev)
    return Rec("NORMAL", "No action from this rule.",
               f"{txt} is within the normal band (≤ p80 {p80*100:.0f}/100).", ev)


def rule_crosswind(wx: pd.DataFrame, now_utc, cfg=RULES) -> Rec:
    now = _utc(now_utc)
    lb = cfg["wx_lookback_h"]
    w = _wx_window(wx, now, lb)
    if w is None or w.empty:
        return Rec("NORMAL", "No action from this rule.", f"No observations in last {lb}h (see weather rule).", [])
    gx = w["gust_crosswind_kt"].fillna(w["crosswind_kt"]) if "gust_crosswind_kt" in w.columns else w["crosswind_kt"]
    if not gx.notna().any():
        return Rec("NORMAL", "No action from this rule.", "No crosswind values in lookback window.", [])
    r = w.loc[gx.idxmax()]
    v = float(gx.loc[r.name])
    known = bool(r.get("wind_dir_known", False))
    hdg = r.get("runway_heading", RUNWAY_HEADING.get(str(r.get("airport", ""))))
    drct = r.get("drct", np.nan)
    src = "weather_timeline.parquet (METAR)"
    ev = [_ev("gust_crosswind_kt" + ("" if known else " (UPPER BOUND)"), v, "kt", r["obs_utc"], src,
              f"WATCH ≥{cfg['crosswind_watch_kt']:.0f}, ACTION ≥{cfg['crosswind_action_kt']:.0f}"),
          _ev("crosswind_kt", r.get("crosswind_kt", np.nan), "kt", r["obs_utc"], src),
          _ev("wind_dir_known", known, "", r["obs_utc"], src),
          _ev("wind direction", drct if (drct is not None and pd.notna(drct)) else "unknown", "deg", r["obs_utc"], "IEM ASOS"),
          _ev("runway_heading", hdg, "deg", None, "brief assumption")]
    if known and v >= cfg["crosswind_action_kt"]:
        return Rec("ACTION", "Review runway configuration with ATC flow coordinator; expect reduced arrival rate.",
                   f"Gust crosswind {v:.0f} kt ≥ {cfg['crosswind_action_kt']:.0f} kt relative to runway {hdg}°.", ev)
    if not known and v >= cfg["crosswind_unknown_dir_kt"]:
        return Rec("WATCH", "Verify actual wind direction (METAR/ATIS) — crosswind is an upper bound.",
                   f"Wind direction unknown; full gust {v:.0f} kt used as crosswind upper bound.", ev)
    if known and v >= cfg["crosswind_watch_kt"]:
        return Rec("WATCH", "Monitor crosswind trend; brief crews on gusty crosswind.",
                   f"Gust crosswind {v:.0f} kt ≥ {cfg['crosswind_watch_kt']:.0f} kt.", ev)
    return Rec("NORMAL", "No action from this rule.",
               f"Max gust crosswind {'≤' if not known else ''}{v:.0f} kt in last {lb}h is below thresholds.", ev)


def rule_weather_obs(wx: pd.DataFrame, now_utc, cfg=RULES) -> Rec:
    now = _utc(now_utc)
    lb = cfg["wx_lookback_h"]
    w = _wx_window(wx, now, lb)
    src = "weather_timeline.parquet (METAR)"
    if w is None or w.empty:
        return Rec("WATCH", "Check METAR/ATIS manually — weather feed gap.",
                   f"No observations in the {lb}h before decision time.",
                   [_ev("observations in lookback", 0, "count", now, src)])
    actions, reasons, ev = [], [], []
    conv = w[w["wx_group"].isin(cfg["convective_groups"])]
    if not conv.empty:
        actions.append("Coordinate with ATC flow coordinator on convective holding/reroutes")
        reasons.append(f"convective weather in {len(conv)} obs")
        ev += [_ev("wx_group", r["wx_group"], "", r["obs_utc"], src) for _, r in conv.tail(3).iterrows()]
    wint = w[w["wx_group"].isin(cfg["winter_groups"])]
    if not wint.empty:
        actions.append("Pre-arrange de-icing capacity; expect slower turns")
        reasons.append(f"winter precipitation in {len(wint)} obs")
        ev += [_ev("wx_group", r["wx_group"], "", r["obs_utc"], src) for _, r in wint.tail(3).iterrows()]
    psum = float(w["prcp_mm"].fillna(0).sum())
    if psum >= cfg["precip_watch_mm"]:
        actions.append("Expect slower runway ops / longer taxi in heavy rain")
        reasons.append(f"{psum:.1f} mm precipitation")
    ev.append(_ev(f"precipitation total last {lb}h", psum, "mm", w["obs_utc"].max(), src, f"WATCH ≥{cfg['precip_watch_mm']}"))
    if actions:
        return Rec("WATCH", "; ".join(actions) + ".", f"Observed in last {lb}h: " + ", ".join(reasons) + ".", ev)
    last = w.iloc[-1]
    ev.insert(0, _ev("latest wx_group", last["wx_group"], "", last["obs_utc"], src))
    return Rec("NORMAL", "No action from this rule.", f"No adverse weather observed in last {lb}h.", ev)


def rule_flight_share(flights: pd.DataFrame, now_utc, horizon_h, cfg=RULES) -> Rec:
    now = _utc(now_utc)
    f = window_flights(flights, now, horizon_h)
    thr = cfg["p_delay_threshold"]
    src = "flight_risk_preds.parquet"
    n = 0 if f is None else len(f)
    if n < cfg["min_flights"]:
        return Rec("NORMAL", "No action from this rule.",
                   f"Only {n} sampled flights in next {horizon_h}h — share not meaningful (sample data).",
                   [_ev("flights in window (sample)", n, "flights", now, src, f"min {cfg['min_flights']}")])
    hi = f[f["p_delay45"] >= thr].sort_values("p_delay45", ascending=False)
    share = len(hi) / n
    ev = [_ev("flights in window (sample)", n, "flights", now, src),
          _ev(f"share with p_delay45 ≥ {thr:.2f}", share, "share", now, src,
              f"WATCH ≥{cfg['share_watch']:.0%}, ACTION ≥{cfg['share_action']:.0%}")]
    for _, r in hi.head(5).iterrows():
        issued = f", issued {_fmt(r['issue_time_utc'])}" if "issue_time_utc" in r.index else ""
        ev.append(_ev(f"{r.get('flight_id', '?')} {r.get('side', '')} p_delay45", r["p_delay45"], "prob",
                      r["sched_utc"], src + issued))
    reason = f"{len(hi)} of {n} flights ({share:.0%}) in next {horizon_h}h have p(>45 min) ≥ {thr:.2f}."
    if share >= cfg["share_action"]:
        return Rec("ACTION", "Prioritise gate/crew buffers; pre-position reserve crew for flagged flights.", reason, ev)
    if share >= cfg["share_watch"]:
        return Rec("WATCH", "Prioritise gate/crew buffers for flagged flights.", reason, ev)
    return Rec("NORMAL", "No action from this rule.", reason, ev)


def evaluate_all(cap, wx, flights, now_utc, horizon_h, cfg=RULES):
    now = _utc(now_utc)
    return [
        ("Capacity stress forecast", rule_stress(window_blocks(cap, now, horizon_h), now, cfg)),
        ("Crosswind vs runway heading", rule_crosswind(wx, now, cfg)),
        (f"Observed weather (last {cfg['wx_lookback_h']}h)", rule_weather_obs(wx, now, cfg)),
        ("At-risk flight share", rule_flight_share(flights, now, horizon_h, cfg)),
    ]


# ---------- failure-case explanation (rules, not ML) ----------
def _adverse(w, cfg):
    out = []
    for _, r in w.iterrows():
        tags = []
        g = r.get("wx_group")
        if g in cfg["convective_groups"] or g in cfg["winter_groups"]:
            tags.append(str(g))
        p = r.get("prcp_mm")
        if p is not None and pd.notna(p) and p >= cfg["precip_heavy_mm_h"]:
            tags.append(f"{p:.1f} mm/h precip")
        gx = r.get("gust_crosswind_kt")
        if gx is not None and pd.notna(gx) and gx >= cfg["crosswind_action_kt"]:
            tags.append(f"gust xwind {'' if bool(r.get('wind_dir_known', False)) else '≤'}{gx:.0f} kt")
        if tags:
            out.append(f"{', '.join(tags)} @ {_fmt(r['obs_utc'])}")
    return out


def explain_miss(block: pd.Series, wx: pd.DataFrame, cfg=RULES) -> str:
    start = _utc(block["block_start_utc"])
    end = start + pd.Timedelta(hours=BLOCK_HOURS)
    issue = block.get("issue_time_utc")
    if issue is None or pd.isna(issue):
        issue = start - pd.Timedelta(hours=float(block.get("horizon_h", 3)))
    issue = _utc(issue)
    lb = pd.Timedelta(hours=cfg["wx_lookback_h"])
    pre = _adverse(wx[(wx["obs_utc"] > issue - lb) & (wx["obs_utc"] <= issue)], cfg)
    dur = _adverse(wx[(wx["obs_utc"] >= start) & (wx["obs_utc"] < end)], cfg)
    n = int(block.get("n_flights", 0) or 0)
    err = float(block["stress_rate_actual"]) - float(block["stress_rate_pred"])
    if err > 0:
        if dur and not pre:
            return f"Missed: weather developed after issue time ({dur[0]}); observation-only inputs cannot see it — needs TAF/forecast feeds."
        if pre:
            return f"Missed despite adverse obs at issue time ({pre[0]}) — model under-weights this signal."
        return f"Missed with no local weather signal before or during block — likely non-weather cause (carrier, late aircraft, upstream NAS) or tiny sample (n={n})."
    if pre and not dur:
        return f"False alarm: adverse obs at issue time ({pre[0]}) cleared before the block."
    if pre:
        return f"False alarm: weather present but flights coped — small sample (n={n}) or flow measures absorbed it."
    return f"False alarm without weather signal — time-of-day effect overstated; small sample (n={n})."