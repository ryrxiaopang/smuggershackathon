"""Predictive Airport Capacity & Delay-Risk — decision support for an airline ops duty manager.
Numbers come from model files + pure rules (src/recommend.py). The human decides."""
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src import dash_config as C
from src import recommend as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
import theme  # noqa: E402  styling + animated landing hero

st.set_page_config(page_title="Airport Capacity & Delay-Risk", page_icon="✈", layout="wide")
theme.inject_css()
LEVEL_COLOR = theme.LEVEL_COLOR
LEVEL_ICON = {"NORMAL": "🟢", "WATCH": "🟠", "ACTION": "🔴"}
LEVEL_MEANING = {"NORMAL": "business as usual", "WATCH": "be ready, keep an eye on it", "ACTION": "act now to get ahead of it"}
AIRPORT_NAMES = {"ATL": "Atlanta (ATL)", "DFW": "Dallas/Fort Worth (DFW)"}
FLIGHT_SIDES = C.PROCESSED / "flight_sides.parquet"
CAP_FAIL = C.OUTPUTS / "capacity_failures.csv"
DELAY_FAIL = C.OUTPUTS / "delay_failures.csv"
HIST_COLS = ("hist_p50", "hist_watch", "hist_action")

# plain-language names for the delay model's features (src/delay_model.py), used in "why" strings
FEATURE_WORDS = {
    "hour_local": "time of day", "dow": "day of week", "month": "month", "side": "departure/arrival",
    "airport": "airport", "carrier": "airline", "distance": "route length", "cong_share": "share of disrupted flights earlier today",
    "cong_n": "flights already flown today", "obs_age_h": "age of latest weather report", "pres_chg_3h": "pressure change (3h)",
    "crosswind_last": "crosswind now", "gust_crosswind_last": "gust crosswind now", "tailwind_last": "tailwind now",
    "wspd_last": "wind now", "wpgt_last": "gusts now", "prcp_last": "rain now", "snow_last": "snow now",
    "temp_last": "temperature now", "pres_last": "air pressure now", "coco_last": "weather type now",
}
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MODEL_NAMES = {"base_rate": "Guess the average", "calendar_only": "Timetable only", "calendar_plus_weather": "Timetable + weather",
               "calendar_weather_congestion": "+ earlier delays today", "model": "Final model"}
WX_COLS = {"local": "time (local)", "obs_utc": "time (UTC)", "drct": "wind from (°)", "wspd_kt": "wind (kt)",
           "gust_kt": "gust (kt)", "tailwind_kt": "tailwind (kt)", "prcp_mm": "rain (mm)", "wx_group": "weather",
           "wind_source": "wind source"}
EVIDENCE_COLS = {"signal": "what was checked", "value": "value", "local": "time (local)",
                 "obs_or_valid_time": "time (UTC)", "source": "source", "threshold": "rule"}


# ---------------- loading ----------------
def ensure_outputs():
    """Stop with instructions if a model output is missing. Never writes fake files: they would land on the
    same paths as the real outputs (run src/make_fake_outputs.py by hand for a UI-only demo)."""
    missing = [p for p in (C.CAPACITY_PREDS, C.FLIGHT_PREDS, C.WEATHER_TIMELINE) if not p.exists()]
    if missing or not list(C.OUTPUTS.glob("*_metrics.json")):
        st.error("Model outputs missing: " + ", ".join(str(p.relative_to(ROOT)) for p in missing) +
                 ". Run `python src/capacity_model.py` and `python src/delay_model.py` first.")
        st.stop()


def _to_utc(s):
    s = pd.to_datetime(s, errors="coerce", utc=False)
    if s.dt.tz is None:
        return s.dt.tz_localize("UTC")
    return s.dt.tz_convert("UTC")


@st.cache_data(show_spinner=False)
def load_parquet(path, mtime):
    df = pd.read_parquet(path)
    for c in df.columns:
        if c.endswith("_utc"):
            df[c] = _to_utc(df[c])
    if "airport" in df.columns:
        df["airport"] = df["airport"].astype(str).str.upper()
    return df


@st.cache_data(show_spinner=False)
def load_csv(path, mtime):
    df = pd.read_csv(path)
    if "airport" in df.columns:
        df["airport"] = df["airport"].astype(str).str.upper()
    return df


@st.cache_data(show_spinner=False)
def load_metrics(paths_mtimes):
    out = {}
    for p, _ in paths_mtimes:
        try:
            out[Path(p).name] = json.loads(Path(p).read_text())
        except Exception as e:  # show, don't crash
            out[Path(p).name] = {"task": Path(p).name, "error": str(e)}
    return out


# ---------------- adapters: real file formats -> internal names used by rules ----------------
def _first(d, names):
    low = {c.lower(): c for c in d.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def local_to_utc(s, airports):
    """Local wall-clock times -> UTC using each row's airport time zone."""
    s = pd.to_datetime(s, errors="coerce")
    if s.dt.tz is not None:
        return s.dt.tz_convert("UTC")
    aps = pd.Series(airports, index=s.index).astype(str).str.upper()
    parts = []
    for a in aps.unique():
        m = aps == a
        parts.append(s[m].dt.tz_localize(C.AIRPORT_TZ.get(a, "UTC"), ambiguous="NaT",
                                         nonexistent="shift_forward").dt.tz_convert("UTC"))
    return pd.concat(parts).reindex(s.index) if parts else s.dt.tz_localize("UTC")


def infer_block_hours(d):
    try:
        g = d[(d["airport"] == d["airport"].iloc[0]) & (d["horizon_h"] == d["horizon_h"].iloc[0])]
        diffs = g["block_start_utc"].drop_duplicates().sort_values().diff().dropna().dt.total_seconds() / 3600
        v = float(diffs.mode().iloc[0])
        return v if 0 < v <= 24 else float(C.BLOCK_HOURS)
    except Exception:
        return float(C.BLOCK_HOURS)


def train_block_thresholds(sides, block_h):
    """Per airport: realised share of stressed flights per block in the TRAIN period (blocks with >=3
    flights, as in capacity_model.py), at the median and the WATCH / ACTION quantiles."""
    if sides is None or sides.empty or "stress_flag" not in sides.columns:
        return None
    f = sides[(sides["diverted"].fillna(0).astype(int) == 0)
              & (pd.to_datetime(sides["date_local"]) <= pd.Timestamp(C.TRAIN_END))].copy()
    f["blk"] = (f["hour_local"].astype(int) // int(block_h)) * int(block_h)
    b = f.groupby(["airport", "date_local", "blk"])["stress_flag"].agg(["mean", "size"])
    b = b[b["size"] >= 3].reset_index()
    qs = (0.5, C.RULES["stress_watch_q"], C.RULES["stress_action_q"])
    return b.groupby("airport")["mean"].quantile(list(qs)).unstack().set_axis(list(HIST_COLS), axis=1)


def adapt_capacity(raw, sides=None):
    d = raw.copy()
    for a, b in (("y_pred", "stress_rate_pred"), ("y_true", "stress_rate_actual"), ("issued_utc", "issue_time_utc")):
        if a in d.columns and b not in d.columns:
            d = d.rename(columns={a: b})
    if "block_start_utc" not in d.columns:
        d["block_start_utc"] = local_to_utc(d["block_start_local"], d["airport"])
    d = d.dropna(subset=["block_start_utc"])
    d["horizon_h"] = pd.to_numeric(d["horizon_h"], errors="coerce")
    if "issue_time_utc" not in d.columns:
        d["issue_time_utc"] = d["block_start_utc"] - pd.to_timedelta(d["horizon_h"], unit="h")
    bh = infer_block_hours(d)
    if "block_end_utc" not in d.columns:
        d["block_end_utc"] = d["block_start_utc"] + pd.Timedelta(hours=bh)
    src = "train period (from file)"
    q = train_block_thresholds(sides, bh) if not all(c in d.columns for c in HIST_COLS) else None
    if q is not None and set(d["airport"]) <= set(q.index):
        for c in HIST_COLS:
            d[c] = d["airport"].map(q[c])
        src = (f"how disrupted this airport's {bh:g}h blocks really were in 2023 – mid-2024 (training period): "
               f"WATCH above the worst {1 - C.RULES['stress_watch_q']:.0%}, ACTION above the worst "
               f"{1 - C.RULES['stress_action_q']:.0%}")
    elif not all(c in d.columns for c in HIST_COLS):
        basis = "baseline_hist" if "baseline_hist" in d.columns else "stress_rate_pred"
        for q, c in zip((0.5, 0.8, 0.9), HIST_COLS):
            d[c] = d.groupby("airport")[basis].transform(lambda s, q=q: s.quantile(q))
        src = ("per-airport quantile of baseline_hist (train-period historical mean)" if basis == "baseline_hist"
               else "ESTIMATED from predictions (no historical baseline in file)")
    d["hist_source"] = src
    return d, bh, src


def _norm_id(s, numeric):
    return pd.to_numeric(s, errors="coerce").astype("Int64") if numeric else s.astype(str)


def adapt_flights(raw, sides):
    d = raw.copy()
    d["side"] = d["side"].astype(str).str.strip().str.lower()
    numeric = bool(pd.to_numeric(d["flight_id"], errors="coerce").notna().all())
    d["flight_id"] = _norm_id(d["flight_id"], numeric)
    joined = False
    if sides is not None and not sides.empty and {"flight_id", "side"} <= set(sides.columns):
        s = sides.copy()
        s["side"] = s["side"].astype(str).str.strip().str.lower()
        s["flight_id"] = _norm_id(s["flight_id"], numeric)
        s = s.drop_duplicates(["flight_id", "side"])
        keep = ["flight_id", "side"] + [c for c in s.columns if c not in d.columns]
        d = d.merge(s[keep], on=["flight_id", "side"], how="left", validate="many_to_one")  # never on flight_id alone
        joined = True
    for a, b in (("y_true", "actual_delay45"), ("issue_utc", "issue_time_utc")):
        if a in d.columns and b not in d.columns:
            d = d.rename(columns={a: b})
    if "sched_utc" not in d.columns:
        d["sched_utc"] = local_to_utc(d["sched_local"], d["airport"])
    d = d.dropna(subset=["sched_utc"])
    if "horizon_h" in d.columns:
        d["horizon_h"] = pd.to_numeric(d["horizon_h"], errors="coerce")
    issue_nominal = "issue_time_utc" not in d.columns
    if issue_nominal:
        d["issue_time_utc"] = d["sched_utc"] - pd.to_timedelta(d["horizon_h"], unit="h")
    cc = _first(d, ["carrier", "OP_UNIQUE_CARRIER", "op_carrier", "airline"])
    d["carrier"] = d[cc].astype(str) if cc else ""
    cp = _first(d, ["counterpart", "other_airport", "counterpart_airport"])
    oc, dc = _first(d, ["ORIGIN"]), _first(d, ["DEST"])
    if cp:
        d["counterpart"] = d[cp]
    elif oc and dc:
        d["counterpart"] = np.where(d["side"] == "dep", d[dc], d[oc])
    else:
        d["counterpart"] = ""
    fn = _first(d, ["OP_CARRIER_FL_NUM", "flight_number", "fl_num"])
    num = pd.to_numeric(d[fn], errors="coerce").astype("Int64").astype(str).replace("<NA>", "") if fn else None
    d["flight_label"] = (d["carrier"] + num) if fn else d["flight_id"].astype(str)
    d["side"] = d["side"].str.upper()
    return d, joined, issue_nominal


# ---------------- generic display helpers ----------------
def feature_words(name):
    if name in FEATURE_WORDS:
        return FEATURE_WORDS[name]
    m = re.fullmatch(r"(\w+?)_(\d+)h_max", name)  # e.g. gust_crosswind_6h_max
    if m:
        return f"max {FEATURE_WORDS.get(m.group(1) + '_last', m.group(1).replace('_', ' ')).replace(' now', '')} in last {m.group(2)}h"
    m = re.fullmatch(r"(\w+?)_count_(\d+)h", name)  # e.g. storm_count_12h
    if m:
        return f"{m.group(1)} reports in last {m.group(2)}h"
    return name.replace("_", " ")


def explain_factors(s):
    """'hour_local=7 (-0.03); carrier=AA (+0.02)' -> 'time of day 07:00 ↓ · airline AA ↑' (↑ raises the risk)."""
    out = []
    for part in str(s or "").split(";"):
        m = re.match(r"\s*([\w]+)=(.*?)\s*\(([+-][\d.]+)\)\s*$", part)
        if not m:
            if part.strip():
                out.append(part.strip())
            continue
        name, val, eff = m.group(1), m.group(2), float(m.group(3))
        if name == "hour_local" and val.isdigit():
            val = f"{int(val):02d}:00"
        elif name == "dow" and val.isdigit() and int(val) < 7:
            val = DOW[int(val)]
        elif name == "month" and val.isdigit() and 1 <= int(val) <= 12:
            val = dt.date(2000, int(val), 1).strftime("%b")
        out.append(f"{feature_words(name)} {val} {'↑' if eff > 0 else '↓'}")
    return " · ".join(out)



def local(s, tz):
    return s.dt.tz_convert(tz).dt.tz_localize(None)


def badge(level):
    return theme.badge(level)


def _safe(df):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str)
    return df


def show_obj(obj):
    """Render arbitrary JSON (dict of dicts, list of dicts, dict of scalars) as a table."""
    if obj is None or obj == {} or obj == []:
        st.caption("(empty)")
    elif isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
        st.dataframe(_safe(pd.json_normalize(obj)), hide_index=True, width="stretch")
    elif isinstance(obj, dict) and all(isinstance(v, dict) for v in obj.values()):
        df = pd.json_normalize(list(obj.values()))
        df.insert(0, "key", list(obj.keys()))
        st.dataframe(_safe(df), hide_index=True, width="stretch")
    elif isinstance(obj, dict) and all(not isinstance(v, (dict, list)) for v in obj.values()):
        st.dataframe(pd.DataFrame({"metric": list(obj.keys()), "value": [str(v) for v in obj.values()]}),
                     hide_index=True, width="stretch")
    else:
        st.json(obj, expanded=False)


def _plain_value(v, unit):
    """0.47 + 'share of flights' -> '47 of 100 flights'; 18.2 + 'kt' -> '18 kt'."""
    if v is None or v == "" or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        if unit == "share of flights":
            return f"{v * 100:.0f} of 100 flights"
        if unit in ("share", "prob"):
            return f"{v:.0%}"
        if unit == "kt":
            return f"{v:.0f} kt"
        if unit == "deg":
            return f"{v:.0f}°"
        if unit == "mm":
            return f"{v:.1f} mm"
        return f"{v:g} {unit}".strip()
    return f"{v} {unit}".strip() if unit and unit not in ("deg",) else (f"{v}°" if unit == "deg" and str(v)[:1].isdigit() else str(v))


def evidence_df(ev, tz):
    if not ev:
        return pd.DataFrame()
    d = pd.DataFrame(ev)
    d["value"] = [_plain_value(v, u) for v, u in zip(d["value"], d["unit"])]
    d = d.drop(columns="unit")
    t = pd.to_datetime(d["obs_or_valid_time"].str.replace("Z", "", regex=False), errors="coerce")
    d.insert(d.columns.get_loc("obs_or_valid_time") + 1, "local",
             t.dt.tz_localize("UTC").dt.tz_convert(tz).dt.strftime("%m-%d %H:%M").fillna(""))
    d = d.astype(str).replace({"None": "", "nan": "", "NaT": ""})
    return d[[c for c in EVIDENCE_COLS if c in d.columns]].rename(columns=EVIDENCE_COLS)


def safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def safe_ap(y, p):
    return average_precision_score(y, p) if len(np.unique(y)) == 2 else np.nan


def hnum(k):
    digits = re.sub(r"\D", "", str(k))
    return int(digits) if digits else 0


# ---------------- metric renderers (real JSON formats) ----------------
CAP_MODELS = [("model", "Model"), ("persist", "Persistence baseline"), ("hist", "Historical baseline")]


def _cap_scale(m):
    vals = [v for v in (m.get("overall") or {}).values() if isinstance(v, (int, float))]
    vals = [v for k, v in (m.get("overall") or {}).items() if k.endswith(("_mae", "_rmse"))] or vals
    return 100.0 if vals and max(vals) <= 1.5 else 1.0


def cap_rows(d, scale):
    rows = []
    for p, label in CAP_MODELS:
        if f"{p}_mae" in d or f"{p}_rmse" in d:
            rows.append({"model": label,
                         "MAE": round(float(d.get(f"{p}_mae", np.nan)) * scale, 2),
                         "RMSE": round(float(d.get(f"{p}_rmse", np.nan)) * scale, 2),
                         "n blocks": d.get("n")})
    return rows


def importance_df(obj):
    if isinstance(obj, dict) and obj and all(isinstance(v, (int, float)) for v in obj.values()):
        return pd.DataFrame({"feature": list(obj.keys()), "gain": list(obj.values())})
    if isinstance(obj, list) and obj and isinstance(obj[0], (list, tuple)):
        return pd.DataFrame([x[:2] for x in obj], columns=["feature", "gain"])
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        df = pd.DataFrame(obj)
        num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        txt = [c for c in df.columns if df[c].dtype == object]
        if num and txt:
            return df[[txt[0], num[0]]].set_axis(["feature", "gain"], axis=1)
    return None


def calib_df(obj):
    try:
        if isinstance(obj, dict) and not any(isinstance(v, list) for v in obj.values()):
            df = pd.DataFrame.from_dict(obj, orient="index")
        else:
            df = pd.DataFrame(obj)
    except Exception:
        return None, None, None
    cols = list(df.columns)
    pc = next((c for c in cols if "pred" in str(c).lower() or str(c).lower() in ("p_mean", "mean_p")), None)
    oc = next((c for c in cols if c != pc and any(k in str(c).lower() for k in ("obs", "true", "actual", "rate", "frac"))), None)
    return df, pc, oc


def render_delay(name, m, horizon):
    bh = m.get("by_horizon", {}) or {}
    st.markdown(f"### Flight >45-min delay risk · `{name}`" + ("  ⚠ synthetic" if m.get("is_synthetic") else ""))
    long = []
    for k, v in bh.items():
        for model, vals in ((v or {}).get("test") or {}).items():
            if isinstance(vals, dict):
                long.append({"horizon_h": hnum(k), "model": model,
                             **{kk: vv for kk, vv in vals.items() if isinstance(vv, (int, float))}})
    if not long:
        st.info("No by_horizon.*.test entries found.")
        return
    L = pd.DataFrame(long)
    L["model"] = L["model"].map(lambda x: MODEL_NAMES.get(x, x))
    order = list(dict.fromkeys(L["model"]))
    key = f"h{horizon}" if f"h{horizon}" in bh else sorted(bh, key=hnum)[0]
    test = bh[key].get("test", {})
    base = test.get("base_rate")
    br = base.get("pr_auc") if isinstance(base, dict) else base
    st.caption(f"Horizon {hnum(key)}h · test set unseen in training"
               + (f" · base rate (prevalence) {br:.1%} = PR-AUC of a no-skill model" if isinstance(br, (int, float)) else ""))
    cur = L[L["horizon_h"] == hnum(key)].drop(columns="horizon_h")
    c1, c2 = st.columns(2)
    with c1:
        show = cur.drop(columns=[c for c in ("p",) if c in cur.columns]).round(3)
        st.dataframe(show.rename(columns={"model": "model", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "brier": "Brier",
                                          "n": "flights"}), hide_index=True, width="stretch")
        if "pr_auc" in cur.columns:
            fig = px.bar(cur, x="model", y="pr_auc", category_orders={"model": order}, height=300,
                         color_discrete_sequence=["#6366f1"])
            if isinstance(br, (int, float)):
                fig.add_hline(y=br, line_dash="dot", line_color="grey")
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_title="", yaxis_title="PR-AUC (higher is better)")
            st.plotly_chart(fig, width="stretch")
    with c2:
        if "pr_auc" in L.columns:
            fig = px.line(L.sort_values("horizon_h"), x="horizon_h", y="pr_auc", color="model", markers=True,
                          category_orders={"model": order}, height=330, title="PR-AUC by horizon (skill decay)")
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), xaxis=dict(tickvals=sorted(L["horizon_h"].unique())))
            st.plotly_chart(fig, width="stretch")
    h = bh[key]
    a, b = st.columns(2)
    with a:
        st.markdown("**Weather lift**")
        show_obj(h.get("weather_lift"))
    with b:
        st.markdown("**Operating point**")
        show_obj(h.get("operating_point"))
    df, pc, oc = calib_df(h.get("calibration_test_deciles"))
    if df is not None and pc and oc:
        fig = go.Figure()
        mx = float(max(df[pc].max(), df[oc].max())) * 1.05
        fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(dash="dash", color="grey"), name="perfect"))
        fig.add_trace(go.Scatter(x=df[pc], y=df[oc], mode="lines+markers", name="test deciles"))
        fig.update_layout(height=320, title=f"Reliability from {name} (h={hnum(key)})", xaxis_title=str(pc),
                          yaxis_title=str(oc), margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
    elif h.get("calibration_test_deciles") is not None:
        st.markdown("**Calibration (test deciles)**")
        show_obj(h.get("calibration_test_deciles"))
    st.markdown("**Generalisation (train on one airport → test on the other)**")
    show_obj(h.get("generalisation"))
    with st.expander("Slices"):
        show_obj(h.get("slices"))


def render_capacity(name, m, horizon):
    scale = _cap_scale(m)
    unit = "disrupted flights per 100" if scale == 100 else "file units"
    st.markdown(f"### Capacity stress · `{name}`" + ("  ⚠ synthetic" if m.get("is_synthetic") else ""))
    st.caption(f"Errors in {unit} (lower is better). Test set unseen in training.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Overall**")
        st.dataframe(pd.DataFrame(cap_rows(m.get("overall") or {}, scale)), hide_index=True, width="stretch")
        rows = []
        for ap_, d in (m.get("by_airport") or {}).items():
            rows += [{"airport": ap_, **r} for r in cap_rows(d, scale)]
        if rows:
            st.markdown("**By airport** (rates, comparable across airports)")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with c2:
        rows = []
        for k, d in (m.get("by_horizon") or {}).items():
            rows += [{"horizon_h": hnum(k), **r} for r in cap_rows(d, scale)]
        if rows:
            H = pd.DataFrame(rows).sort_values("horizon_h")
            fig = px.bar(H, x=H["horizon_h"].astype(str) + "h", y="MAE", color="model", barmode="group", height=330,
                         title=f"MAE by horizon ({unit})")
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), xaxis_title="horizon")
            st.plotly_chart(fig, width="stretch")
    gen = m.get("generalisation") or {}
    if gen:
        st.markdown("**Generalisation (train → test airport)**")
        rows = []
        for k, d in gen.items():
            tr, te = (k.split("->") + [""])[:2]
            in_ap = (m.get("by_airport") or {}).get(te.strip(), {}).get("model_mae")
            rows.append({"train → test": k, "n": d.get("n"),
                         "MAE": round(float(d.get("mae", np.nan)) * scale, 2),
                         "RMSE": round(float(d.get("rmse", np.nan)) * scale, 2),
                         "in-airport model MAE (reference)": round(float(in_ap) * scale, 2) if in_ap is not None else None})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    imp = importance_df(m.get("top_importance_gain"))
    if imp is not None:
        imp = imp.sort_values("gain").tail(15)
        fig = px.bar(imp, x="gain", y="feature", orientation="h", height=380, title="Top features (LightGBM gain)")
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
    elif m.get("top_importance_gain") is not None:
        with st.expander("Top importance (gain)"):
            show_obj(m.get("top_importance_gain"))
    with st.expander("Block size comparison"):
        show_obj(m.get("block_size_comparison"))
    if m.get("failure_cases"):
        with st.expander("Failure cases listed in capacity_metrics.json"):
            show_obj(m.get("failure_cases"))


def render_legacy(name, m):
    st.markdown(f"### {m.get('task', name)}" + ("  ⚠ synthetic" if m.get("is_synthetic") else ""))
    if "error" in m:
        st.error(f"Could not read {name}: {m['error']}")
        return
    dm = pd.DataFrame(m.get("models", []))
    if not dm.empty:
        st.dataframe(dm, hide_index=True, width="stretch")
    if m.get("generalisation"):
        show_obj(m["generalisation"])
    for fc in m.get("failure_cases", []) or []:
        if isinstance(fc, dict):
            st.markdown(f"- ❌ **{fc.get('date', '')} · {fc.get('airport', '')}** — {fc.get('what', '')}  \n  _Why:_ {fc.get('why', '')}")


def is_delay_metrics(m):
    return isinstance(m.get("by_horizon"), dict) and any(str(k).lower().startswith("h") for k in m["by_horizon"])


def is_cap_metrics(m):
    return isinstance(m.get("by_horizon"), dict) and "overall" in m and not is_delay_metrics(m)


# ---------------- load everything ----------------
ensure_outputs()
cap_raw = load_parquet(str(C.CAPACITY_PREDS), C.CAPACITY_PREDS.stat().st_mtime)
fl_raw = load_parquet(str(C.FLIGHT_PREDS), C.FLIGHT_PREDS.stat().st_mtime)
sides = load_parquet(str(FLIGHT_SIDES), FLIGHT_SIDES.stat().st_mtime) if FLIGHT_SIDES.exists() else None
wx_all = R.normalise_weather(load_parquet(str(C.WEATHER_TIMELINE), C.WEATHER_TIMELINE.stat().st_mtime))
cap_all, B, hist_src = adapt_capacity(cap_raw, sides)
fl_all, sides_joined, issue_nominal = adapt_flights(fl_raw, sides)
if "sched_local" not in fl_all.columns:
    fl_all["sched_local"] = pd.NaT

mfiles = sorted(C.OUTPUTS.glob("*_metrics.json"))
metrics = load_metrics(tuple((str(p), p.stat().st_mtime) for p in mfiles))
if any(not m.get("is_synthetic") for m in metrics.values()):  # hide leftover synthetic metrics once real ones exist
    metrics = {n: m for n, m in metrics.items() if not m.get("is_synthetic")}

syn = [n for n, df in (("capacity_preds", cap_raw), ("flight_risk_preds", fl_raw), ("weather_timeline", wx_all))
       if "is_synthetic" in df.columns and bool(df["is_synthetic"].any())]
syn += [n for n, m in metrics.items() if m.get("is_synthetic")]

# ---------------- landing page ----------------
DEMO = {"ap": "ATL", "day": dt.date(2025, 1, 21), "time": dt.time(19, 0), "horizon": 12}
NO_PICK = "(pick a date below instead)"


def landing_numbers():
    """Headline numbers for the landing page, computed from the real output files (2025 test period)."""
    out = {"flights": int(sides["flight_id"].nunique()) if sides is not None else len(fl_all), "wx": len(wx_all)}
    e = fl_all[fl_all["horizon_h"] == 3].dropna(subset=["actual_delay45"]) if "actual_delay45" in fl_all.columns else fl_all.iloc[:0]
    if len(e):
        top = e.nlargest(max(1, len(e) // 10), "p_delay45")["actual_delay45"].mean()
        out["top"], out["lift"] = top, top / e["actual_delay45"].mean()
    t = cap_all[cap_all["horizon_h"] == 6].dropna(subset=["stress_rate_actual"]) if "stress_rate_actual" in cap_all.columns else cap_all.iloc[:0]
    if len(t):
        act, wat = t["stress_rate_pred"] > t["hist_action"], t["stress_rate_pred"] > t["hist_watch"]
        out["act_y"], out["norm_y"] = t.loc[act, "stress_rate_actual"].mean(), t.loc[~wat, "stress_rate_actual"].mean()
    return out


def go_to(view, preset=None):
    st.session_state["view"] = view
    if preset:
        st.session_state["preset"] = preset
    st.rerun()


def render_landing():
    theme.landing_mode_css()
    n = landing_numbers()
    theme.landing_hero([(n["flights"], 0, "", "flights analysed"), (n["wx"], 0, "", "hourly weather reports"),
                        (12, 0, " h", "of warning"),
                        (round(n.get("lift", 0), 1), 1, "×", "better at spotting late flights")])
    _, b1, b2, _ = st.columns([1, 3, 3, 1])
    if b1.button("✈  Open the live dashboard", type="primary", key="cta_open"):
        go_to("dashboard")
    if b2.button("▶  Demo: ATL winter storm", key="cta_demo"):
        go_to("dashboard", DEMO)
    st.markdown("### What it does")
    act = (f"When it says ACTION, {n['act_y']*100:.0f} of 100 flights were disrupted (normal: {n['norm_y']*100:.0f})"
           if "act_y" in n else "Alerts measured on 2025 data")
    top = f"Riskiest 10%: {n['top']:.0%} really late, {n['lift']:.1f}× the average" if "top" in n else "Ranked by risk"
    theme.feature_cards([
        ("🛬", "Airport congestion forecast", "How many flights will be disrupted in each 6-hour block at ATL or DFW, "
               "3, 6 or 12 hours ahead, against what is normal for that airport.", act),
        ("⏱", "Flights at risk", "The chance each flight is more than 45 minutes late, worked out hours before it "
              "departs or lands, with the top reasons in plain words.", top),
        ("🧭", "Advice you can check", "Four fixed rules turn the forecasts into NORMAL, WATCH or ACTION. Every alert "
              "lists the timestamped weather reports and values behind it.", "No AI-written numbers · the human decides"),
    ])
    st.markdown("### How it works")
    theme.how_it_works([
        ("🌦", "Weather + flights", f"{n['wx']:,} hourly airport weather reports and {n['flights']:,} sampled flights, 2023–2025."),
        ("🔒", "No peeking ahead", "Each forecast only uses data from before it was made. Time-stamp errors in the data were found and fixed."),
        ("🤖", "Machine learning", "LightGBM models forecast block congestion and each flight's >45-minute delay risk."),
        ("🚦", "Plain advice", "Simple rules turn forecasts into NORMAL / WATCH / ACTION with the evidence attached."),
    ])
    _, b3, _ = st.columns([2, 2, 2])
    if b3.button("Start exploring  →", type="primary", key="cta_bottom"):
        go_to("dashboard")
    st.markdown("<div class='landing-foot'>Built by Ry · Emz · Rach · Deph for Autonomous SG 2026. Data: BTS on-time sample, "
                "organiser hourly weather, IEM METAR archive. Results are on Jan–Jul 2025, which the models never saw in "
                "training.</div>", unsafe_allow_html=True)


if "view" not in st.session_state:
    st.session_state["view"] = "dashboard" if st.query_params.get("view") == "dashboard" else "landing"
if st.session_state["view"] == "landing":
    render_landing()
    st.stop()

# ---------------- sidebar ----------------
if st.sidebar.button("← Home", key="back_home"):
    go_to("landing")
preset = st.session_state.pop("preset", None)
if preset:  # demo button on the landing page: pre-fill the sidebar
    st.session_state.update({"w_ap": preset["ap"], "w_h": preset["horizon"], "w_pick": NO_PICK,
                             f"day_{preset['ap']}_{NO_PICK}": preset["day"], f"tm_{preset['ap']}": preset["time"]})
st.sidebar.title("✈ Capacity & Delay-Risk")
st.sidebar.caption("For an airline ops duty manager or ATC flow coordinator. Pick an airport and a moment in "
                   "2025: the page shows only what was known at that moment, like a live shift.")
avail = sorted(set(cap_all["airport"]) | set(fl_all["airport"]) | set(wx_all["airport"]))
ap = st.sidebar.selectbox("Airport", avail, key="w_ap",
                          **({} if "w_ap" in st.session_state else {"index": avail.index("ATL") if "ATL" in avail else 0}))
tz = C.AIRPORT_TZ.get(ap, "UTC")
if ap not in C.AIRPORT_TZ:
    st.sidebar.warning("No time zone configured for this airport — times shown in UTC.")
cap = cap_all[cap_all["airport"] == ap]
fl = fl_all[fl_all["airport"] == ap]
wx = wx_all[wx_all["airport"] == ap]
horizon = st.sidebar.radio("How far ahead to look (hours)", C.HORIZONS, horizontal=True, key="w_h",
                           **({} if "w_h" in st.session_state else {"index": min(1, len(C.HORIZONS) - 1)}),
                           help="Longer warning = more time to act, but less accurate.")

times = cap["block_start_utc"] if not cap.empty else (fl["sched_utc"] if not fl.empty else wx["obs_utc"])
lt = times.dt.tz_convert(tz)
dmin, dmax = lt.min().date(), lt.max().date()
picks = {}
if not cap.empty:
    c0 = cap[cap["horizon_h"] == cap["horizon_h"].min()]
    top = (c0.assign(day=c0["block_start_utc"].dt.tz_convert(tz).dt.date)
           .groupby("day")["stress_rate_pred"].max().nlargest(8))
    picks = {f"{d:%a %d %b %Y} · up to {v*100:.0f} of 100 flights disrupted": d for d, v in top.items()}
pick = st.sidebar.selectbox("Jump to a high-risk day", [NO_PICK] + list(picks), key="w_pick",
                            help="Days in the 2025 test period where the forecast was highest.")
default_day = picks.get(pick, next(iter(picks.values()), dmin))
default_day = min(max(default_day, dmin), dmax)
k_day, k_tm = f"day_{ap}_{pick}", f"tm_{ap}"
day = st.sidebar.date_input("Date (local)", min_value=dmin, max_value=dmax, key=k_day,
                            **({} if k_day in st.session_state else {"value": default_day}))
tm = st.sidebar.time_input("Time now (local)", step=3600, key=k_tm,
                           **({} if k_tm in st.session_state else {"value": dt.time(6, 0)}))
now_local = pd.Timestamp(dt.datetime.combine(day, tm)).tz_localize(tz, ambiguous=False, nonexistent="shift_forward")
now = now_local.tz_convert("UTC")
show_hind = st.sidebar.toggle("Show what actually happened", value=False,
                              help="Hindsight only: these outcomes were NOT available to the forecasts or rules.")
st.sidebar.caption(f"= {now:%Y-%m-%d %H:%M} UTC. Everything on the page uses only data from before this moment.")
if sides is None:
    st.sidebar.warning("data/processed/flight_sides.parquet not found — carrier/counterpart unavailable.")

if syn:
    st.warning("⚠ SYNTHETIC DEMO DATA in: " + ", ".join(syn) +
               ". Replace with real pipeline outputs (same file names) — no code change needed.")

recs = R.evaluate_all(cap, wx, fl, now, horizon, tz=tz)
overall = R.overall_level([r for _, r in recs])
bw = R.window_blocks(cap, now, horizon)
fw = R.window_flights(fl, now, horizon)
thr = C.RULES["p_delay_threshold"]
p50 = p80 = p90 = np.nan
if not cap.empty:
    p50, p80, p90 = (float(cap[c].iloc[0]) for c in HIST_COLS)
peak = float(bw["stress_rate_pred"].max()) if bw is not None and not bw.empty else None
theme.hero(ap=AIRPORT_NAMES.get(ap, ap), when=f"{now_local:%a %d %b %Y · %H:%M} local", horizon=horizon, level=overall,
           meaning=LEVEL_MEANING[overall], peak=peak, normal=p80)

tab_fc, tab_fl, tab_ev, tab_lim = st.tabs(["① Forecast & advice", "② Flights at risk", "③ How accurate is it?", "④ Data & limits"])

# ---------------- Forecast ----------------
with tab_fc:
    st.subheader(f"{AIRPORT_NAMES.get(ap, ap)} · {now_local:%a %d %b %Y, %H:%M} local · next {horizon} hours")
    st.caption("The coloured box tells you whether to act. Below it: the forecast, the four checks behind the advice, "
               "and the weather reports they are based on.")

    # headline: the single most important thing, in plain words
    worst = sorted(recs, key=lambda x: -R._RANK[x[1].level])
    alerts = [(n, r) for n, r in worst if r.level != "NORMAL"]
    if alerts:
        head = alerts[0][1].action
        why = " ".join(f"<li><b>{n}:</b> {r.reason}</li>" for n, r in alerts)
        why = f"<ul style='margin:6px 0 0 0'>{why}</ul>"
    else:
        head = f"No action needed. Nothing unusual is expected at {ap} in the next {horizon} hours."
        why = "<div style='margin-top:4px'>All four checks below are normal.</div>"
    col = LEVEL_COLOR[overall]
    st.markdown(f"<div class='status-card' style='border-left:10px solid {col};background:{col}1f;border-radius:8px;padding:14px 18px;margin:4px 0 14px 0'>"
                f"<div style='font-size:0.85rem;letter-spacing:.05em;font-weight:700;color:{col}'>{LEVEL_ICON[overall]} {overall}"
                f" · {LEVEL_MEANING[overall]}</div>"
                f"<div style='font-size:1.35rem;font-weight:700;margin-top:2px'>{head}</div>{why}</div>",
                unsafe_allow_html=True)

    c2, c3, c4 = st.columns(3)
    c2.metric(f"Expected disruption (worst {B:g}h block ahead)",
              f"{bw['stress_rate_pred'].max()*100:.0f} of 100 flights" if bw is not None and not bw.empty else "no forecast",
              help="Share of flights expected to have an ATC/airport-capacity or weather delay ≥15 min, or be cancelled.")
    c3.metric(f"Normal for {ap}", f"up to {p80*100:.0f} of 100" if not np.isnan(p80) else "n/a",
              help=f"WATCH above {p80*100:.0f}, ACTION above {p90*100:.0f} of 100. Based on {hist_src}." if not np.isnan(p80) else hist_src)
    nflag = int((fw["p_delay45"] >= thr).sum()) if fw is not None and not fw.empty else 0
    c4.metric(f"Flights likely >45 min late (next {horizon}h)", f"{nflag} of {0 if fw is None else len(fw)}",
              help=f"Flights with at least a {thr:.0%} chance of a >45-minute delay. The data is a sample (~6 ATL arrivals/day), not the full schedule.")

    left = st.container()
    with left:
        st.markdown(f"**Expected disruption per {B:g}-hour block** · bars = forecast, coloured by alert level; "
                    f"green band = normal for {ap}")
        fig = go.Figure()
        xs = []
        past = cap.drop_duplicates("block_start_utc")
        if "stress_rate_actual" in past.columns:
            past = past[(past["block_end_utc"] <= now) & (past["block_start_utc"] >= now - pd.Timedelta(hours=24))] \
                .dropna(subset=["stress_rate_actual"])
            if not past.empty:
                t = local(past["block_start_utc"], tz) + pd.Timedelta(hours=B / 2)
                xs += list(t)
                fig.add_trace(go.Bar(x=t, y=past["stress_rate_actual"] * 100, width=B * 3600 * 1000 * 0.85,
                                     name="what happened (last 24h)", marker=dict(color="rgba(148,163,184,.45)",
                                     line=dict(color="rgba(148,163,184,.9)", width=1)),
                                     hovertemplate="%{y:.0f} of 100 flights disrupted<extra>what happened</extra>"))
        if bw is not None and not bw.empty:
            t = local(bw["block_start_utc"], tz) + pd.Timedelta(hours=B / 2)
            xs += list(t)
            col = np.where(bw["stress_rate_pred"] > p90, LEVEL_COLOR["ACTION"],
                           np.where(bw["stress_rate_pred"] > p80, LEVEL_COLOR["WATCH"], "#6366f1"))
            cd = np.stack([bw["horizon_h"].astype(int).astype(str), bw["issue_time_utc"].dt.strftime("%m-%d %H:%MZ")], axis=1)
            fig.add_trace(go.Bar(x=t, y=bw["stress_rate_pred"] * 100, width=B * 3600 * 1000 * 0.85, marker_color=col,
                                 name="forecast", customdata=cd,
                                 hovertemplate="%{y:.0f} of 100 flights · forecast made %{customdata[0]}h ahead (%{customdata[1]})<extra>forecast</extra>"))
            if "baseline_hist" in bw.columns:
                fig.add_trace(go.Scatter(x=t, y=bw["baseline_hist"] * 100, mode="markers", name="typical for this block & month",
                                         marker=dict(symbol="line-ew-open", size=28, color="#10b981", line=dict(width=3))))
            if "baseline_persist" in bw.columns:
                fig.add_trace(go.Scatter(x=t, y=bw["baseline_persist"] * 100, mode="markers", name="naive guess: same as last block",
                                         marker=dict(symbol="circle-open", size=10, color="#555")))
            if show_hind and "stress_rate_actual" in bw.columns:
                fig.add_trace(go.Scatter(x=t, y=bw["stress_rate_actual"] * 100, mode="markers", name="what actually happened",
                                         marker=dict(symbol="diamond", size=12, color="black")))
        if xs and not np.isnan(p50):
            x0, x1 = min(xs) - pd.Timedelta(hours=B / 2), max(xs) + pd.Timedelta(hours=B / 2)
            fig.add_trace(go.Scatter(x=[x0, x1], y=[p50 * 100] * 2, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[x0, x1], y=[p80 * 100] * 2, mode="lines", fill="tonexty", line=dict(width=1, color="#10b981"),
                                     fillcolor="rgba(16,185,129,0.16)", name="normal range (WATCH above)"))
            fig.add_trace(go.Scatter(x=[x0, x1], y=[p90 * 100] * 2, mode="lines", line=dict(dash="dash", color=LEVEL_COLOR["ACTION"]),
                                     name="ACTION line"))
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=60, b=10),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                          yaxis_title="disrupted flights per 100", xaxis_title=f"local time ({tz})")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Disrupted = ATC/airport-capacity or weather delay ≥15 min, or cancelled. Alert lines: {hist_src}.")

    with st.container(border=True):
        st.markdown(f"**The four checks behind the advice** {badge(overall)}", unsafe_allow_html=True)
        st.caption("Simple, fixed rules applied to the forecasts and the latest weather reports, not AI. "
                   "Open a check to see the exact values and times it used.")
        grid = st.columns(2)
        for i, (name, rec) in enumerate(sorted(recs, key=lambda x: -R._RANK[x[1].level])):
            with grid[i % 2].expander(f"{LEVEL_ICON[rec.level]} {rec.level} · {name}", expanded=rec.level != "NORMAL"):
                st.markdown(f"**{rec.action}**")
                st.write(rec.reason)
                d = evidence_df(rec.evidence, tz)
                if not d.empty:
                    st.dataframe(d, hide_index=True, width="stretch")

    st.markdown("**Weather reports for the last 24 hours** · dashed line = now; nothing to the right of it is used")
    end = now + pd.Timedelta(hours=horizon) if show_hind else now
    w = wx[(wx["obs_utc"] > now - pd.Timedelta(hours=24)) & (wx["obs_utc"] <= end)].copy()
    if w.empty:
        st.info("No observations in this window.")
    else:
        w["t"] = local(w["obs_utc"], tz)
        w["obs"] = w["obs_utc"].dt.strftime("%Y-%m-%d %H:%MZ")
        k, u = w[w["wind_dir_known"]], w[~w["wind_dir_known"]]
        rh = int(w["runway_heading"].iloc[0]) if "runway_heading" in w.columns and pd.notna(w["runway_heading"].iloc[0]) \
            else C.RUNWAY_HEADING.get(ap, "?")
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.35, 0.25, 0.2, 0.2],
                            subplot_titles=(f"Crosswind across the main runway ({rh}°), knots", "Wind and gusts, knots",
                                            "Rain, mm per hour", "Weather type"))
        hov = "%{y:.0f} kt · obs %{customdata}<extra></extra>"
        fig.add_trace(go.Scatter(x=w["t"], y=w["crosswind_kt"], mode="lines", name="crosswind", line=dict(color="#06b6d4", width=2.5),
                                 customdata=w["obs"], hovertemplate=hov), 1, 1)
        fig.add_trace(go.Scatter(x=k["t"], y=k["gust_crosswind_kt"], mode="markers", name="gust crosswind",
                                 marker=dict(color="#ec4899", size=7), customdata=k["obs"], hovertemplate=hov), 1, 1)
        fig.add_trace(go.Scatter(x=u["t"], y=u["gust_crosswind_kt"], mode="markers", name="gust crosswind, at most (wind direction unknown)",
                                 marker=dict(color="#ec4899", size=10, symbol="triangle-up-open"), customdata=u["obs"],
                                 hovertemplate="≤%{y:.0f} kt UPPER BOUND · obs %{customdata}<extra></extra>"), 1, 1)
        fig.add_trace(go.Scatter(x=[w["t"].min(), w["t"].max()], y=[C.RULES["crosswind_action_kt"]] * 2, mode="lines",
                                 line=dict(dash="dot", color="grey"), name=f"{C.RULES['crosswind_action_kt']:.0f} kt rule"), 1, 1)
        if "wspd_kt" in w.columns:
            fig.add_trace(go.Scatter(x=w["t"], y=w["wspd_kt"], mode="lines", name="wind speed", line=dict(color="#94a3b8"),
                                     customdata=w["obs"], hovertemplate=hov), 2, 1)
        if "wpgt_kt" in w.columns and w["wpgt_kt"].notna().any():
            fig.add_trace(go.Scatter(x=w["t"], y=w["wpgt_kt"], mode="markers", name="gusts (when reported)",
                                     marker=dict(color="#f59e0b", size=8, symbol="diamond"),
                                     customdata=w["obs"], hovertemplate=hov), 2, 1)
        fig.add_trace(go.Bar(x=w["t"], y=w["prcp_mm"], name="precip", marker_color="#38bdf8", customdata=w["obs"],
                             hovertemplate="%{y:.1f} mm · obs %{customdata}<extra></extra>"), 3, 1)
        fig.add_trace(go.Scatter(x=w["t"], y=w["wx_group"], mode="markers", name="wx group", marker=dict(color="#a78bfa", size=9),
                                 customdata=w["obs"], hovertemplate="%{y} · obs %{customdata}<extra></extra>"), 4, 1)
        fig.update_yaxes(categoryorder="array", categoryarray=C.WX_GROUP_ORDER + ["unknown"], row=4, col=1)
        nl = str(now.tz_convert(tz).tz_localize(None))
        fig.add_vline(x=nl, line_dash="dash", line_color="black", row="all", col="all")
        if show_hind:
            fig.add_vrect(x0=nl, x1=str(end.tz_convert(tz).tz_localize(None)), fillcolor="grey", opacity=0.12,
                          line_width=0, row="all", col="all")
        fig.update_layout(height=620, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.08))
        st.plotly_chart(fig, width="stretch")
        if show_hind:
            st.caption("Grey area = hindsight observations after decision time (not used by forecasts or rules).")
        last = w[w["obs_utc"] <= now].tail(6).copy()
        if not last.empty:
            last["local"] = local(last["obs_utc"], tz).dt.strftime("%m-%d %H:%M")
            last["obs_utc"] = last["obs_utc"].dt.strftime("%Y-%m-%d %H:%MZ")
            fmt = lambda r, c: "" if pd.isna(r[c]) else (f"≤{r[c]:.0f} (upper bound)" if not r["wind_dir_known"] else f"{r[c]:.0f}")
            last["crosswind (kt)"] = last.apply(lambda r: fmt(r, "crosswind_kt"), axis=1)
            last["gust crosswind (kt)"] = last.apply(lambda r: fmt(r, "gust_crosswind_kt"), axis=1)
            cols = [c for c in ["local", "obs_utc", "wdir_deg", "wspd_kt", "wpgt_kt", "crosswind (kt)", "gust crosswind (kt)",
                                "tailwind_kt", "prcp_mm", "wx_group", "wind_source"] if c in last.columns]
            with st.expander("Latest weather reports as a table"):
                st.dataframe(last[cols].iloc[::-1].rename(columns=WX_COLS), hide_index=True, width="stretch")

    with st.form("decision"):
        st.markdown("**Your decision** — the tool recommends; the duty manager decides.")
        dcol, ncol = st.columns([1, 3])
        decision = dcol.radio("Decision", ["Accept", "Modify", "Reject"])
        note = ncol.text_input("Note / reason (logged)")
        if st.form_submit_button("Log decision"):
            row = {"logged_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%SZ"), "airport": ap,
                   "decision_time_utc": f"{now:%Y-%m-%d %H:%MZ}", "horizon_h": horizon, "tool_level": overall,
                   "tool_actions": " | ".join(r.action for _, r in recs if r.level != "NORMAL"),
                   "decision": decision, "note": note}
            C.OUTPUTS.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([row]).to_csv(C.DECISION_LOG, mode="a", header=not C.DECISION_LOG.exists(), index=False)
            st.success(f"Logged to {C.DECISION_LOG.relative_to(ROOT)}")
    if C.DECISION_LOG.exists():
        with st.expander("Decision log (latest first)"):
            log = pd.read_csv(C.DECISION_LOG).iloc[::-1].head(10)
            st.dataframe(log, hide_index=True, width="stretch")

# ---------------- At-risk flights ----------------
with tab_fl:
    st.subheader(f"Flights at risk of a >45-minute delay · {ap}")
    st.caption("Each flight gets a chance of being more than 45 minutes late, worked out a few hours before it "
               "departs or lands. ↑ means the factor pushes the risk up, ↓ pushes it down.")
    a, b2, c = st.columns(3)
    span = a.radio("Show flights in the next", [f"{horizon} hours", f"{max(C.HORIZONS)} hours"], horizontal=True)
    span_h = horizon if span == f"{horizon} hours" else max(C.HORIZONS)
    thr_sel = b2.slider("Flag flights with a chance of at least", 0.0, 1.0, float(C.RULES["p_delay_threshold"]), 0.05,
                        format="%.2f", help="0.35 = 35% chance of a >45-minute delay. The average flight is about 13%.")
    sides_sel = c.multiselect("Departures / arrivals", ["DEP", "ARR"], default=["DEP", "ARR"])
    f = R.window_flights(fl, now, span_h)
    if f is None or f.empty:
        st.info("No flights from the sample in this window. Try a longer window or another time.")
    else:
        f = f[f["side"].isin(sides_sel)].copy()
        flagged = f.sort_values("p_delay45", ascending=False)  # all flights, riskiest first; ⚠ marks the flagged ones
        n_flag = int((flagged["p_delay45"] >= thr_sel).sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Flights in this window (sample)", len(f))
        m2.metric("Flagged ⚠ (at or above your line)", n_flag)
        m3.metric("Average chance of >45 min delay", f"{f['p_delay45'].mean():.0%}")
        sl = pd.to_datetime(flagged["sched_local"], errors="coerce")
        if sl.isna().any():
            sl = local(flagged["sched_utc"], tz)
        show = pd.DataFrame({"": np.where(flagged["p_delay45"] >= thr_sel, "⚠", ""),
                             "scheduled (local)": sl.dt.strftime("%a %d %b %H:%M"),
                             "dep/arr": flagged["side"].map({"DEP": "departure", "ARR": "arrival"}).fillna(flagged["side"]),
                             "flight": flagged["flight_label"],
                             "to / from": flagged["counterpart"].astype(str),
                             "chance of >45 min delay": flagged["p_delay45"],
                             "why (top 3 factors)": flagged["top_factors"].map(explain_factors) if "top_factors" in flagged.columns else "",
                             "forecast made": flagged["horizon_h"].astype(int).astype(str) + "h before"})
        if show_hind and "actual_delay45" in flagged.columns:
            show["actually >45 min late?"] = flagged["actual_delay45"].map({1.0: "yes", 0.0: "no"})
        st.dataframe(show, hide_index=True, width="stretch",
                     column_config={"chance of >45 min delay": st.column_config.ProgressColumn(
                         "chance of >45 min delay", min_value=0.0, max_value=1.0, format="percent")})
        st.caption("The data is a sample (~6 ATL arrivals a day), not the full schedule. "
                   + ("Flight details come from flight_sides.parquet, matched on (flight_id, side). " if sides_joined else "")
                   + ("Forecast time = scheduled time minus the horizon. " if issue_nominal else ""))

# ---------------- Evaluation ----------------
with tab_ev:
    st.subheader("How accurate is it?")
    st.caption("Everything here is measured on January–July 2025, which the models never saw while learning "
               "(they trained on 2023 – mid-2024). Each model is compared with simple guesses it has to beat.")
    cm = next((m for m in metrics.values() if is_cap_metrics(m)), None)
    dm = next((m for m in metrics.values() if is_delay_metrics(m)), None)
    k1, k2, k3 = st.columns(3)
    if cm and cm.get("overall"):
        o = cm["overall"]
        k1.metric("Congestion forecast: typical error", f"{o['model_rmse']*100:.1f} per 100 flights",
                  f"{(o['model_rmse'] / o['hist_rmse'] - 1) * 100:+.1f}% vs historical average", delta_color="inverse",
                  help=f"Root-mean-square error of the disrupted-flights rate per {B:g}h block (lower is better). "
                       f"Historical average: {o['hist_rmse']*100:.1f}. Naive 'same as last block': {o['persist_rmse']*100:.1f}.")
    ev3 = fl_all[(fl_all["horizon_h"] == 3)].dropna(subset=["actual_delay45"]) if "actual_delay45" in fl_all.columns else pd.DataFrame()
    if not ev3.empty:
        top = ev3.nlargest(max(1, len(ev3) // 10), "p_delay45")["actual_delay45"].mean()
        base = ev3["actual_delay45"].mean()
        k2.metric("Riskiest 10% of flights: really >45 min late", f"{top:.0%}",
                  f"{top / base:.1f}× the average of {base:.0%}",
                  help="Flights the model ranked in its top 10% for delay risk, three hours before departure/arrival.")
    if dm:
        t3 = (dm.get("by_horizon", {}).get("h3", {}) or {}).get("test", {})
        if "model" in t3 and "calendar_only" in t3:
            lift = t3["model"]["pr_auc"] / t3["calendar_only"]["pr_auc"] - 1
            k3.metric("Does the weather help? (3h ahead)", f"{lift:+.0%}",
                      "vs timetable-only model",
                      help=f"PR-AUC at 3h: {t3['model']['pr_auc']:.3f} with weather vs {t3['calendar_only']['pr_auc']:.3f} "
                           "without. Most >45-min delays are airline-driven (late aircraft, crew), which caps what weather can explain.")
    st.markdown("**In plain words**")
    st.markdown(
        "- The congestion forecast clearly beats the naive guess and only just beats the historical average. "
        "It helps most on unusual days (storms, winter weather), which is when it matters.\n"
        "- The flight-delay model's riskiest-ranked flights run late about twice as often as the average flight, "
        "and the weather adds real value on top of the timetable.\n"
        "- Warnings get less accurate the further ahead you look (3h → 12h). Every number below shows this trade-off.\n"
        "- It does fail: see the failure cases at the bottom, each with a one-line reason.")
    st.markdown("**Warning time vs accuracy** · the earlier the warning, the harder the job")
    g1, g2 = st.columns(2)
    if cm and cm.get("by_horizon"):
        rows = [{"horizon": f"{hnum(k)}h", "h": hnum(k), "who": lbl, "err": v[f"{p}_rmse"] * 100}
                for k, v in cm["by_horizon"].items()
                for p, lbl in (("model", "Our forecast"), ("hist", "Historical average"), ("persist", "Same as last block"))]
        H = pd.DataFrame(rows).sort_values("h")
        fig = px.line(H, x="horizon", y="err", color="who", markers=True, height=300,
                      color_discrete_map={"Our forecast": "#6366f1", "Historical average": "#10b981", "Same as last block": "#94a3b8"},
                      title="Congestion forecast error (lower is better)")
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), yaxis_title="error, disrupted flights per 100",
                          xaxis_title="warning time", legend_title="")
        g1.plotly_chart(fig, width="stretch")
    if dm and dm.get("by_horizon"):
        rows = [{"horizon": f"{hnum(k)}h", "h": hnum(k), "who": lbl, "pr": v["test"][p]["pr_auc"]}
                for k, v in dm["by_horizon"].items()
                for p, lbl in (("model", "Our model"), ("calendar_only", "Timetable only"), ("base_rate", "Guess the average"))
                if p in v.get("test", {})]
        D = pd.DataFrame(rows).sort_values("h")
        fig = px.line(D, x="horizon", y="pr", color="who", markers=True, height=300,
                      color_discrete_map={"Our model": "#ec4899", "Timetable only": "#06b6d4", "Guess the average": "#94a3b8"},
                      title="Finding >45-min delays: PR-AUC (higher is better)")
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), yaxis_title="PR-AUC", xaxis_title="warning time", legend_title="")
        g2.plotly_chart(fig, width="stretch")

    st.divider()
    st.markdown("#### Full details")
    st.caption("Every metric, baseline and chart behind the summary above. Open a section to see it.")
    for name, m in metrics.items():
        title = ("Congestion forecast: all metrics" if is_cap_metrics(m) else
                 "Flight delay model: all metrics" if is_delay_metrics(m) else name)
        with st.expander(title):
            if "error" in m:
                st.error(f"Could not read {name}: {m['error']}")
            elif is_delay_metrics(m):
                render_delay(name, m, horizon)
            elif is_cap_metrics(m):
                render_capacity(name, m, horizon)
            else:
                render_legacy(name, m)
    recompute = st.expander(f"Double-check: delay metrics recomputed from the prediction file ({horizon}h)")
    ev = fl_all[fl_all["horizon_h"] == horizon] if "horizon_h" in fl_all.columns else fl_all
    if "actual_delay45" in ev.columns:
        ev = ev.dropna(subset=["actual_delay45"])
    if "actual_delay45" not in ev.columns or ev.empty:
        recompute.info("No y_true in flight_risk_preds — cannot recompute.")
    else:
        with recompute:
            thr = C.RULES["p_delay_threshold"]
            rows = []
            for a_, g in ev.groupby("airport"):
                y, p = g["actual_delay45"].astype(int).to_numpy(), g["p_delay45"].to_numpy()
                top = g.nlargest(max(1, int(len(g) * 0.1)), "p_delay45")["actual_delay45"].mean()
                rows.append({"airport": a_, "flights": len(g), "actual >45 min per 100": round(100 * y.mean(), 1),
                             "mean predicted per 100": round(100 * p.mean(), 1), f"flagged per 100 (p≥{thr})": round(100 * (p >= thr).mean(), 1),
                             "AUC": round(safe_auc(y, p), 3), "PR-AUC": round(safe_ap(y, p), 3), "Brier": round(brier_score_loss(y, p), 4),
                             "precision @ top 10%": round(top, 3), "lift @ top 10%": round(top / max(y.mean(), 1e-9), 2)})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("Airports compared only as rates per 100 flights (sample volumes differ).")
            cb = ev.assign(bin=ev.groupby("airport")["p_delay45"].transform(lambda s: pd.qcut(s, 10, labels=False, duplicates="drop")))
            cb = cb.groupby(["airport", "bin"]).agg(pred=("p_delay45", "mean"), obs=("actual_delay45", "mean"), n=("p_delay45", "size")).reset_index()
            fig = go.Figure()
            mx = float(max(cb["pred"].max(), cb["obs"].max())) * 1.05
            fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(dash="dash", color="grey"), name="perfect calibration"))
            for a_, g in cb.groupby("airport"):
                fig.add_trace(go.Scatter(x=g["pred"], y=g["obs"], mode="lines+markers", name=a_, customdata=g["n"],
                                         hovertemplate="pred %{x:.2f} · obs %{y:.2f} · n=%{customdata}<extra></extra>"))
            fig.update_layout(height=360, xaxis_title="mean predicted p(>45 min)", yaxis_title="observed rate",
                              margin=dict(l=10, r=10, t=30, b=10), title="Reliability (deciles, recomputed)")
            st.plotly_chart(fig, width="stretch")

    st.markdown("### Where it went wrong")
    st.caption("The biggest misses on the 2025 test data, each with a one-line reason worked out by rules from the "
               "timestamped weather reports. Showing these is part of an honest evaluation.")
    if CAP_FAIL.exists():
        cf = load_csv(str(CAP_FAIL), CAP_FAIL.stat().st_mtime)
        rows = []
        for _, r in cf.iterrows():
            start = local_to_utc(pd.Series([r["block_start_local"]]), [r["airport"]]).iloc[0]
            blk = pd.Series({"block_start_utc": start, "block_end_utc": start + pd.Timedelta(hours=B),
                             "issue_time_utc": _to_utc(pd.Series([r["issued_utc"]])).iloc[0],
                             "horizon_h": r["horizon_h"], "stress_rate_actual": r["y_true"], "stress_rate_pred": r["y_pred"]})
            why = r.get("why", r.get("explanation"))
            if why is None or pd.isna(why):
                why = R.explain_miss(blk, wx_all[wx_all["airport"] == r["airport"]]) if pd.notna(start) else "n/a"
            rows.append({"airport": r["airport"], "6h block (local)": str(r["block_start_local"])[:16],
                         "warning time": f"{int(r['horizon_h'])}h", "forecast /100": round(100 * r["y_pred"]),
                         "actual /100": round(100 * r["y_true"]), "why it missed": why})
        st.markdown("**Congestion forecast** (`capacity_failures.csv`)")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        with st.expander("Weather columns in capacity_failures.csv"):
            st.dataframe(_safe(cf), hide_index=True, width="stretch")
    if DELAY_FAIL.exists():
        df_ = load_csv(str(DELAY_FAIL), DELAY_FAIL.stat().st_mtime)
        rows = []
        for _, r in df_.iterrows():
            su = local_to_utc(pd.Series([r["sched_local"]]), [r["airport"]]).iloc[0]
            row = pd.Series({"sched_utc": su, "issue_time_utc": _to_utc(pd.Series([r["issue_utc"]])).iloc[0],
                             "horizon_h": r["horizon_h"], "actual_delay45": r["y_true"]})
            why = r.get("why", r.get("explanation"))
            if why is None or pd.isna(why):
                why = R.explain_flight_miss(row, wx_all[wx_all["airport"] == r["airport"]]) if pd.notna(su) else "n/a"
            rows.append({"type": str(r["case"]).replace("_", " "), "airport": r["airport"],
                         "dep/arr": {"DEP": "departure", "ARR": "arrival"}.get(str(r["side"]).upper(), r["side"]),
                         "scheduled (local)": str(r["sched_local"])[:16], "warning time": f"{int(r['horizon_h'])}h",
                         "chance given": f"{float(r['p_delay45']):.0%}", "really >45 min late?": "yes" if int(r["y_true"]) else "no",
                         "model's reasons": explain_factors(r.get("top_factors", "")), "why it missed": why})
        st.markdown("**Flight delay model** (`delay_failures.csv`)")
        st.dataframe(pd.DataFrame(rows).drop_duplicates(), hide_index=True, width="stretch")
        with st.expander("Weather columns in delay_failures.csv"):
            st.dataframe(_safe(df_), hide_index=True, width="stretch")

    with st.expander(f"Auto-detected worst capacity misses · {ap} · horizon {horizon}h"):
        if "stress_rate_actual" in cap.columns:
            ch = cap[cap["horizon_h"] == horizon].dropna(subset=["stress_rate_actual"])
            ch = ch.assign(err=ch["stress_rate_actual"] - ch["stress_rate_pred"])
            sel = pd.concat([ch.nlargest(3, "err"), ch.nsmallest(2, "err")])
            rows = [{"block (local)": f"{r['block_start_utc'].tz_convert(tz):%Y-%m-%d %H:%M}",
                     "type": "missed stress" if r["err"] > 0 else "false alarm",
                     "pred /100": round(100 * r["stress_rate_pred"]), "actual /100": round(100 * r["stress_rate_actual"]),
                     "explanation": R.explain_miss(r, wx)} for _, r in sel.iterrows()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# ---------------- Data & limits ----------------
with tab_lim:
    st.subheader("Data & limits")
    st.caption("What the data can and can't tell us, the problems we found and fixed, and how we kept the test honest.")
    st.markdown("##### What the data is")
    theme.info_cards([
        ("✈️", "A sample, not the full schedule",
         f"The flight file is a sample (~6 ATL arrivals a day), so the true arrival rate (AAR) can't be counted. We forecast a "
         f"<b>disruption proxy</b>: the share of flights per {B:g}h block with an ATC/airport or weather delay ≥15 min, or "
         "cancelled. Airports are only compared as <b>rates per 100 flights</b>."),
        ("🌦", "Hourly weather + wind direction",
         "ATL/DFW weather comes from the organisers' hourly file (24 readings a day, with wind direction). Gaps in direction "
         "and all gusts come from public METARs (Iowa Environmental Mesonet)."),
        ("🌍", "Works for any airport",
         "No airport is hard-coded. <code>python src/pipeline.py --airports KORD KDEN</code> looks up the time zone and "
         "runway heading from public data and downloads the wind automatically. Tested on ORD, DEN and SEA."),
    ])
    st.markdown("##### Problems we found and fixed")
    theme.info_cards([
        ("🕓", "Weather stamped 4–6 hours wrong",
         "In the flight file, the weather for a flight at <i>local</i> hour H is really the reading at <i>UTC</i> hour H "
         "(checked against raw METARs: 0.03 °C error). We relabel every reading to its true time."),
        ("⏱", "Hourly file stamped an hour early",
         "A row stamped hour T holds weather observed <i>during</i> hour T (it matches the METAR at ~T:52). We treat it as "
         "known only from T+1h, otherwise forecasts would see up to an hour into the future."),
        ("🧭", "Runway direction, not raw wind",
         "Crosswind and tailwind are computed against the runway (ATL 80°, DFW 180°). When wind direction is unknown, "
         "crosswind is shown as an <b>upper bound</b> (▵) and the rule says <i>verify direction</i>."),
    ])
    st.markdown("##### Keeping it honest")
    theme.info_cards([
        ("🔒", "No peeking ahead",
         "Every forecast and rule uses only weather observed, and flights finished, before it was made. Trained on 2023 – "
         "mid-2024; all results are on Jan–Jul 2025, which the models never saw. Automated leakage tests check this."),
        ("📏", f"Alert lines from real history",
         f"WATCH / ACTION fire when the forecast is worse than {C.RULES['stress_watch_q']:.0%} / "
         f"{C.RULES['stress_action_q']:.0%} of this airport's real {B:g}h blocks in training. On 2025 data, ACTION fires on "
         "~2% of blocks, which really were ~4× more disrupted than normal."),
        ("🧑‍✈️", "The human decides",
         "Numbers come from statistics and machine learning; advice comes from four fixed rules (thresholds in "
         "<code>src/dash_config.py</code>). Every alert shows its evidence, and decisions are logged with a note."),
    ])
    st.markdown("##### Limits and next steps")
    theme.info_cards([
        ("🌩", "Observations, not forecasts",
         "Inputs are weather reports up to the moment of the forecast, so a storm that hasn't started yet is invisible. "
         "This is the main failure mode (see ③). Next: add TAF and convective weather forecasts."),
        ("🧳", "Most long delays aren't weather",
         "Of >45-minute delay minutes, 45% are late aircraft and 34% airline causes, vs 8% weather and 12% ATC/airport. "
         "That caps how much weather can explain."),
        ("🔗", "Joins done carefully",
         "A flight between ATL and DFW appears once per airport; everything is joined on (flight_id, side), never "
         "flight_id alone." + (" Flight forecast times are scheduled time − horizon." if issue_nominal else "")),
    ])
    with st.expander("Rule thresholds in use"):
        st.json(C.RULES)