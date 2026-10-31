"""Predictive Airport Capacity & Delay-Risk — decision support for an airline ops duty manager.
Numbers come from model files + pure rules (src/recommend.py). The human decides."""
import datetime as dt
import json
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

st.set_page_config(page_title="Airport Capacity & Delay-Risk", page_icon="✈", layout="wide")
LEVEL_COLOR = {"NORMAL": "#2e7d32", "WATCH": "#ef6c00", "ACTION": "#c62828"}
LEVEL_ICON = {"NORMAL": "🟢", "WATCH": "🟠", "ACTION": "🔴"}
B = C.BLOCK_HOURS


# ---------------- loading ----------------
def ensure_outputs():
    missing = [p for p in (C.CAPACITY_PREDS, C.FLIGHT_PREDS, C.WEATHER_TIMELINE) if not p.exists()]
    if missing or not list(C.OUTPUTS.glob("*_metrics.json")):
        from src import make_fake_outputs
        make_fake_outputs.main([])


def _to_utc(s):
    s = pd.to_datetime(s)
    return s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")


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
def load_metrics(paths_mtimes):
    out = {}
    for p, _ in paths_mtimes:
        try:
            out[Path(p).name] = json.loads(Path(p).read_text())
        except Exception as e:  # show, don't crash
            out[Path(p).name] = {"task": Path(p).name, "error": str(e)}
    return out


def local(s, tz):
    return s.dt.tz_convert(tz).dt.tz_localize(None)


def badge(level):
    return (f"<span style='background:{LEVEL_COLOR[level]};color:white;padding:3px 12px;"
            f"border-radius:12px;font-weight:700'>{level}</span>")


def evidence_df(ev, tz):
    if not ev:
        return pd.DataFrame()
    d = pd.DataFrame(ev)
    t = pd.to_datetime(d["obs_or_valid_time"].str.replace("Z", "", regex=False), errors="coerce")
    d.insert(d.columns.get_loc("obs_or_valid_time") + 1, "local",
             t.dt.tz_localize("UTC").dt.tz_convert(tz).dt.strftime("%m-%d %H:%M").fillna(""))
    return d.astype(str).replace({"None": "", "nan": "", "NaT": ""})


def safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def safe_ap(y, p):
    return average_precision_score(y, p) if len(np.unique(y)) == 2 else np.nan


ensure_outputs()
cap_all = load_parquet(str(C.CAPACITY_PREDS), C.CAPACITY_PREDS.stat().st_mtime)
fl_all = load_parquet(str(C.FLIGHT_PREDS), C.FLIGHT_PREDS.stat().st_mtime)
wx_all = R.normalise_weather(load_parquet(str(C.WEATHER_TIMELINE), C.WEATHER_TIMELINE.stat().st_mtime))
mfiles = sorted(C.OUTPUTS.glob("*_metrics.json"))
metrics = load_metrics(tuple((str(p), p.stat().st_mtime) for p in mfiles))

hist_estimated = False
for q, c in ((0.5, "hist_p50"), (0.8, "hist_p80"), (0.9, "hist_p90")):
    if c not in cap_all.columns:
        cap_all[c] = cap_all.groupby("airport")["stress_rate_pred"].transform(lambda s, q=q: s.quantile(q))
        hist_estimated = True
if "sched_local" not in fl_all.columns:
    fl_all["sched_local"] = pd.NaT

syn = [n for n, df in (("capacity_preds", cap_all), ("flight_risk_preds", fl_all), ("weather_timeline", wx_all))
       if "is_synthetic" in df.columns and bool(df["is_synthetic"].any())]
syn += [n for n, m in metrics.items() if m.get("is_synthetic")]

# ---------------- sidebar ----------------
st.sidebar.title("✈ Capacity & Delay-Risk")
st.sidebar.caption("For: airline ops duty manager / ATC flow coordinator")
avail = sorted(set(cap_all["airport"]) | set(fl_all["airport"]) | set(wx_all["airport"]))
ap = st.sidebar.text_input("Airport code (IATA)", value="ATL").strip().upper()
st.sidebar.caption("In data: " + ", ".join(avail))
if ap not in avail:
    st.error(f"Airport '{ap}' not found in the output files. Available: {', '.join(avail)}")
    st.stop()
tz = C.AIRPORT_TZ.get(ap, "UTC")
if ap not in C.AIRPORT_TZ:
    st.sidebar.warning("No time zone configured for this airport — times shown in UTC.")
cap = cap_all[cap_all["airport"] == ap]
fl = fl_all[fl_all["airport"] == ap]
wx = wx_all[wx_all["airport"] == ap]
horizon = st.sidebar.radio("Look-ahead horizon (hours)", C.HORIZONS, index=min(1, len(C.HORIZONS) - 1), horizontal=True)

times = cap["block_start_utc"] if not cap.empty else (fl["sched_utc"] if not fl.empty else wx["obs_utc"])
lt = times.dt.tz_convert(tz)
dmin, dmax = lt.min().date(), lt.max().date()
picks = []
if not cap.empty:
    c0 = cap[cap["horizon_h"] == cap["horizon_h"].min()] if "horizon_h" in cap.columns else cap
    picks = list(c0.assign(day=c0["block_start_utc"].dt.tz_convert(tz).dt.date)
                 .groupby("day")["stress_rate_pred"].max().nlargest(8).index)
pick = st.sidebar.selectbox("Quick pick: highest predicted-stress days", ["(choose date below)"] + [str(d) for d in picks])
default_day = pd.Timestamp(pick).date() if pick != "(choose date below)" else (picks[0] if picks else dmin)
default_day = min(max(default_day, dmin), dmax)
day = st.sidebar.date_input("Decision date (local)", value=default_day, min_value=dmin, max_value=dmax, key=f"day_{ap}_{pick}")
tm = st.sidebar.time_input("Decision time (local)", value=dt.time(10, 0), step=3600, key=f"tm_{ap}")
now_local = pd.Timestamp(dt.datetime.combine(day, tm)).tz_localize(tz, ambiguous=False, nonexistent="shift_forward")
now = now_local.tz_convert("UTC")
show_hind = st.sidebar.toggle("Show hindsight (actual outcomes)", value=False,
                              help="Actuals after decision time were NOT available to the model or rules.")
st.sidebar.caption(f"Decision time = {now:%Y-%m-%d %H:%MZ}. Rules and forecasts use only data at or before this time.")

if syn:
    st.warning("⚠ SYNTHETIC DEMO DATA in: " + ", ".join(syn) +
               ". Replace with real pipeline outputs (same file names) — no code change needed.")

tab_fc, tab_fl, tab_ev, tab_lim = st.tabs(["Forecast", "At-risk flights", "Evaluation", "Data & limits"])

# ---------------- Forecast ----------------
with tab_fc:
    st.subheader(f"{ap} · {now_local:%a %d %b %Y %H:%M} local ({now:%H:%MZ}) · next {horizon}h")
    recs = R.evaluate_all(cap, wx, fl, now, horizon)
    overall = R.overall_level([r for _, r in recs])
    bw = R.window_blocks(cap, now, horizon)
    fw = R.window_flights(fl, now, horizon)
    thr = C.RULES["p_delay_threshold"]
    p50 = p80 = p90 = np.nan
    if not cap.empty:
        p50, p80, p90 = (float(cap[c].iloc[0]) for c in ("hist_p50", "hist_p80", "hist_p90"))

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"**Overall recommendation**<br>{badge(overall)}", unsafe_allow_html=True)
    c2.metric("Peak predicted stress", f"{bw['stress_rate_pred'].max()*100:.0f} /100 flights" if bw is not None and not bw.empty else "n/a")
    c3.metric("Normal band (hist p50–p80)", f"{p50*100:.0f}–{p80*100:.0f} /100" if not np.isnan(p50) else "n/a")
    nflag = int((fw["p_delay45"] >= thr).sum()) if fw is not None and not fw.empty else 0
    c4.metric(f"Flights flagged (p≥{thr:.2f})", f"{nflag} of {0 if fw is None else len(fw)}", help="Sampled flights only")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Predicted capacity stress vs historical normal band**")
        fig = go.Figure()
        past = cap.drop_duplicates("block_start_utc")
        past = past[(past["block_start_utc"] + pd.Timedelta(hours=B) <= now) &
                    (past["block_start_utc"] >= now - pd.Timedelta(hours=24))].dropna(subset=["stress_rate_actual"]) \
            if "stress_rate_actual" in past.columns else past.iloc[0:0]
        xs = []
        if not past.empty:
            t = local(past["block_start_utc"], tz) + pd.Timedelta(hours=B / 2)
            xs += list(t)
            fig.add_trace(go.Scatter(x=t, y=past["stress_rate_actual"] * 100, mode="markers+lines",
                                     name="observed, last 24h (sample)", marker=dict(color="grey"), line=dict(color="lightgrey"),
                                     customdata=past.get("n_flights", pd.Series(np.nan, index=past.index)),
                                     hovertemplate="%{y:.0f}/100 · n=%{customdata} flights<extra>observed</extra>"))
        if bw is not None and not bw.empty:
            t = local(bw["block_start_utc"], tz) + pd.Timedelta(hours=B / 2)
            xs += list(t)
            col = np.where(bw["stress_rate_pred"] > p90, LEVEL_COLOR["ACTION"],
                           np.where(bw["stress_rate_pred"] > p80, LEVEL_COLOR["WATCH"], "#5b8db8"))
            cd = np.stack([bw.get("horizon_h", pd.Series(np.nan, index=bw.index)).astype(str),
                           bw["issue_time_utc"].dt.strftime("%m-%d %H:%MZ") if "issue_time_utc" in bw.columns else [""] * len(bw)], axis=1)
            fig.add_trace(go.Bar(x=t, y=bw["stress_rate_pred"] * 100, width=B * 3600 * 1000 * 0.85, marker_color=col,
                                 name="predicted (bar colour = rule level)", customdata=cd,
                                 hovertemplate="%{y:.0f}/100 · horizon %{customdata[0]}h · issued %{customdata[1]}<extra>forecast</extra>"))
            if show_hind and "stress_rate_actual" in bw.columns:
                fig.add_trace(go.Scatter(x=t, y=bw["stress_rate_actual"] * 100, mode="markers", name="actual (hindsight)",
                                         marker=dict(symbol="diamond", size=12, color="black")))
        if xs and not np.isnan(p50):
            x0, x1 = min(xs) - pd.Timedelta(hours=B / 2), max(xs) + pd.Timedelta(hours=B / 2)
            fig.add_trace(go.Scatter(x=[x0, x1], y=[p50 * 100] * 2, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[x0, x1], y=[p80 * 100] * 2, mode="lines", fill="tonexty", line=dict(width=1, color="#2e7d32"),
                                     fillcolor="rgba(46,125,50,0.15)", name="normal band p50–p80 (WATCH above)"))
            fig.add_trace(go.Scatter(x=[x0, x1], y=[p90 * 100] * 2, mode="lines", line=dict(dash="dash", color=LEVEL_COLOR["ACTION"]),
                                     name="hist p90 (ACTION above)"))
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25),
                          yaxis_title="flights >45 min / cxl per 100", xaxis_title=f"local time ({tz})")
        st.plotly_chart(fig, use_container_width=True)
        if hist_estimated:
            st.caption("⚠ Historical percentiles missing from file — estimated from predictions.")

    with right:
        with st.container(border=True):
            st.markdown(f"**Recommendation** {badge(overall)}", unsafe_allow_html=True)
            st.caption("Pure rules on model outputs + observations. Thresholds in src/dash_config.py.")
            for name, rec in sorted(recs, key=lambda x: -R._RANK[x[1].level]):
                with st.expander(f"{LEVEL_ICON[rec.level]} {rec.level} · {name} — {rec.action}", expanded=rec.level != "NORMAL"):
                    st.write(rec.reason)
                    d = evidence_df(rec.evidence, tz)
                    if not d.empty:
                        st.dataframe(d, hide_index=True, use_container_width=True)

    st.markdown("**Weather timeline (observations; vertical line = decision time)**")
    end = now + pd.Timedelta(hours=horizon) if show_hind else now
    w = wx[(wx["obs_utc"] > now - pd.Timedelta(hours=24)) & (wx["obs_utc"] <= end)].copy()
    if w.empty:
        st.info("No observations in this window.")
    else:
        w["t"] = local(w["obs_utc"], tz)
        w["obs"] = w["obs_utc"].dt.strftime("%Y-%m-%d %H:%MZ")
        k, u = w[w["wind_dir_known"]], w[~w["wind_dir_known"]]
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.35, 0.25, 0.2, 0.2],
                            subplot_titles=(f"Crosswind vs runway {int(w['runway_heading'].iloc[0]) if 'runway_heading' in w.columns else '?'}° (kt)",
                                            "Wind / gust (kt)", "Precipitation (mm/h)", "Weather group"))
        hov = "%{y:.0f} kt · obs %{customdata}<extra></extra>"
        fig.add_trace(go.Scatter(x=w["t"], y=w["crosswind_kt"], mode="lines", name="crosswind", line=dict(color="#1f77b4"),
                                 customdata=w["obs"], hovertemplate=hov), 1, 1)
        fig.add_trace(go.Scatter(x=k["t"], y=k["gust_crosswind_kt"], mode="markers", name="gust crosswind (direction known)",
                                 marker=dict(color="#d62728", size=7), customdata=k["obs"], hovertemplate=hov), 1, 1)
        fig.add_trace(go.Scatter(x=u["t"], y=u["gust_crosswind_kt"], mode="markers", name="gust crosswind UPPER BOUND (direction unknown)",
                                 marker=dict(color="#d62728", size=10, symbol="triangle-up-open"), customdata=u["obs"],
                                 hovertemplate="≤%{y:.0f} kt UPPER BOUND · obs %{customdata}<extra></extra>"), 1, 1)
        fig.add_trace(go.Scatter(x=[w["t"].min(), w["t"].max()], y=[C.RULES["crosswind_action_kt"]] * 2, mode="lines",
                                 line=dict(dash="dot", color="grey"), name=f"{C.RULES['crosswind_action_kt']:.0f} kt rule"), 1, 1)
        if "wspd_kt" in w.columns:
            fig.add_trace(go.Scatter(x=w["t"], y=w["wspd_kt"], mode="lines", name="wind speed", line=dict(color="#7f7f7f"),
                                     customdata=w["obs"], hovertemplate=hov), 2, 1)
        if "gust_kt" in w.columns:
            fig.add_trace(go.Scatter(x=w["t"], y=w["gust_kt"], mode="lines", name="gust", line=dict(color="#ff7f0e", dash="dot"),
                                     customdata=w["obs"], hovertemplate=hov), 2, 1)
        fig.add_trace(go.Bar(x=w["t"], y=w["prcp_mm"], name="precip", marker_color="#17becf", customdata=w["obs"],
                             hovertemplate="%{y:.1f} mm · obs %{customdata}<extra></extra>"), 3, 1)
        fig.add_trace(go.Scatter(x=w["t"], y=w["wx_group"], mode="markers", name="wx group", marker=dict(color="#9467bd", size=8),
                                 customdata=w["obs"], hovertemplate="%{y} · obs %{customdata}<extra></extra>"), 4, 1)
        fig.update_yaxes(categoryorder="array", categoryarray=C.WX_GROUP_ORDER + ["unknown"], row=4, col=1)
        nl = str(now.tz_convert(tz).tz_localize(None))
        fig.add_vline(x=nl, line_dash="dash", line_color="black", row="all", col="all")
        if show_hind:
            fig.add_vrect(x0=nl, x1=str(end.tz_convert(tz).tz_localize(None)), fillcolor="grey", opacity=0.12,
                          line_width=0, row="all", col="all")
        fig.update_layout(height=620, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.08))
        st.plotly_chart(fig, use_container_width=True)
        if show_hind:
            st.caption("Grey area = hindsight observations after decision time (not used by forecasts or rules).")
        last = w[w["obs_utc"] <= now].tail(6).copy()
        if not last.empty:
            last["local"] = local(last["obs_utc"], tz).dt.strftime("%m-%d %H:%M")
            last["obs_utc"] = last["obs_utc"].dt.strftime("%Y-%m-%d %H:%MZ")
            fmt = lambda r, c: f"≤{r[c]:.0f} (upper bound)" if not r["wind_dir_known"] else f"{r[c]:.0f}"
            last["crosswind (kt)"] = last.apply(lambda r: fmt(r, "crosswind_kt"), axis=1)
            last["gust crosswind (kt)"] = last.apply(lambda r: fmt(r, "gust_crosswind_kt"), axis=1)
            cols = [c for c in ["obs_utc", "local", "drct", "wspd_kt", "gust_kt", "crosswind (kt)", "gust crosswind (kt)",
                                "tailwind_kt", "prcp_mm", "wx_group", "wind_source"] if c in last.columns]
            st.dataframe(last[cols].iloc[::-1], hide_index=True, use_container_width=True)

    with st.form("decision"):
        st.markdown("**Your decision** — the tool recommends; the duty manager decides.")
        dcol, ncol = st.columns([1, 3])
        decision = dcol.radio("Decision", ["Accept", "Modify", "Reject"], horizontal=False)
        note = ncol.text_input("Note / reason (logged)")
        if st.form_submit_button("Log decision"):
            row = {"logged_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%SZ"), "airport": ap,
                   "decision_time_utc": f"{now:%Y-%m-%d %H:%MZ}", "horizon_h": horizon, "tool_level": overall,
                   "tool_actions": " | ".join(r.action for _, r in recs if r.level != "NORMAL"),
                   "decision": decision, "note": note}
            C.OUTPUTS.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([row]).to_csv(C.DECISION_LOG, mode="a", header=not C.DECISION_LOG.exists(), index=False)
            st.success(f"Logged to {C.DECISION_LOG.relative_to(ROOT)}")

# ---------------- At-risk flights ----------------
with tab_fl:
    st.subheader(f"At-risk flights · {ap}")
    a, b2, c = st.columns(3)
    span = a.radio("Window", [f"now → +{horizon}h", f"now → +{max(C.HORIZONS)}h"], horizontal=True)
    span_h = horizon if span.startswith(f"now → +{horizon}h") else max(C.HORIZONS)
    thr_sel = b2.slider("p(>45 min) threshold", 0.0, 1.0, float(C.RULES["p_delay_threshold"]), 0.05)
    sides = c.multiselect("Side", ["DEP", "ARR"], default=["DEP", "ARR"])
    f = R.window_flights(fl, now, span_h)
    if f is None or f.empty:
        st.info("No sampled flights in this window.")
    else:
        f = f[f["side"].isin(sides)].copy()
        flagged = f[f["p_delay45"] >= thr_sel].sort_values("p_delay45", ascending=False)
        m1, m2, m3 = st.columns(3)
        m1.metric("Flights in window (sample)", len(f))
        m2.metric("Flagged", len(flagged))
        m3.metric("Flagged per 100 flights", f"{100*len(flagged)/max(len(f),1):.0f}")
        sl = flagged["sched_local"] if flagged["sched_local"].notna().all() else local(flagged["sched_utc"], tz)
        show = pd.DataFrame({"sched_local": pd.to_datetime(sl).dt.strftime("%m-%d %H:%M"), "side": flagged["side"],
                             "flight": flagged.get("flight_id"), "counterpart": flagged.get("counterpart"),
                             "carrier": flagged.get("carrier"), "p_delay45": flagged["p_delay45"],
                             "horizon_h": flagged.get("horizon_h"),
                             "issued_utc": flagged["issue_time_utc"].dt.strftime("%m-%d %H:%MZ") if "issue_time_utc" in flagged.columns else "",
                             "top_factors": flagged.get("top_factors")})
        if show_hind and "actual_delay45" in flagged.columns:
            show["actual >45 (hindsight)"] = flagged["actual_delay45"]
        st.dataframe(show, hide_index=True, use_container_width=True,
                     column_config={"p_delay45": st.column_config.ProgressColumn("p_delay45", min_value=0.0, max_value=1.0, format="%.2f")})
        st.caption("Dataset is a sample (~6 ATL arrivals/day) — this is not the full schedule. "
                   "top_factors = largest positive contributions in the model's own explanation.")

# ---------------- Evaluation ----------------
with tab_ev:
    st.subheader("Honest evaluation (test period, unseen by training)")
    for fname, m in metrics.items():
        with st.container(border=True):
            st.markdown(f"### {m.get('task', fname)}" + ("  ⚠ synthetic" if m.get("is_synthetic") else ""))
            if "error" in m:
                st.error(f"Could not read {fname}: {m['error']}")
                continue
            st.caption(f"File: outputs/{fname} · test period {m.get('test_period', '?')}"
                       + (f" · base rate {m['base_rate']:.1%}" if m.get("base_rate") is not None else ""))
            dm = pd.DataFrame(m.get("models", []))
            if not dm.empty:
                st.dataframe(dm, hide_index=True, use_container_width=True)
                pm = m.get("primary_metric") or next((c for c in dm.columns if c not in ("name", "horizon_h")), None)
                if pm in dm.columns:
                    dm["horizon"] = dm["horizon_h"].astype(str) + "h" if "horizon_h" in dm.columns else "all"
                    fig = px.bar(dm, x="name", y=pm, color="horizon", barmode="group", height=300)
                    if pm == "pr_auc" and m.get("base_rate") is not None:
                        fig.add_hline(y=m["base_rate"], line_dash="dot", line_color="grey")
                    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_title="", yaxis_title=pm)
                    st.plotly_chart(fig, use_container_width=True)
                    if pm == "pr_auc":
                        st.caption("Dotted line = base rate (PR-AUC of a random model).")
            gen = pd.DataFrame(m.get("generalisation", []))
            if not gen.empty:
                st.markdown("**Generalisation (train on one airport, test on the other)**")
                st.dataframe(gen, hide_index=True, use_container_width=True)
            for fc in m.get("failure_cases", []):
                st.markdown(f"- ❌ **{fc.get('date', '')} · {fc.get('airport', '')}** — {fc.get('what', '')}  \n  _Why:_ {fc.get('why', '')}")

    st.markdown(f"### Recomputed from the prediction file (horizon {horizon}h)")
    ev = fl_all[fl_all["horizon_h"] == horizon] if "horizon_h" in fl_all.columns else fl_all
    if "actual_delay45" in ev.columns:
        ev = ev.dropna(subset=["actual_delay45"])
    if "actual_delay45" not in ev.columns or ev.empty:
        st.info("No actual_delay45 in flight_risk_preds — cannot recompute.")
    else:
        thr = C.RULES["p_delay_threshold"]
        rows = []
        for a_, g in ev.groupby("airport"):
            y, p = g["actual_delay45"].astype(int).to_numpy(), g["p_delay45"].to_numpy()
            top = g.nlargest(max(1, int(len(g) * 0.1)), "p_delay45")["actual_delay45"].mean()
            rows.append({"airport": a_, "flights": len(g), "actual >45 min per 100": round(100 * y.mean(), 1),
                         "mean predicted per 100": round(100 * p.mean(), 1), f"flagged per 100 (p≥{thr})": round(100 * (p >= thr).mean(), 1),
                         "AUC": round(safe_auc(y, p), 3), "PR-AUC": round(safe_ap(y, p), 3), "Brier": round(brier_score_loss(y, p), 4),
                         "precision @ top 10%": round(top, 3), "lift @ top 10%": round(top / max(y.mean(), 1e-9), 2)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
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
                          margin=dict(l=10, r=10, t=30, b=10), title="Reliability (deciles)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"### Worst capacity-stress misses · {ap} · horizon {horizon}h (auto-explained by rules)")
    if "stress_rate_actual" in cap.columns:
        ch = cap[cap["horizon_h"] == horizon] if "horizon_h" in cap.columns else cap
        ch = ch.dropna(subset=["stress_rate_actual"])
        if "n_flights" in ch.columns:
            ch = ch[ch["n_flights"] >= 2]
        ch = ch.assign(err=ch["stress_rate_actual"] - ch["stress_rate_pred"])
        sel = pd.concat([ch.nlargest(3, "err"), ch.nsmallest(2, "err")])
        rows = [{"block (local)": f"{r['block_start_utc'].tz_convert(tz):%Y-%m-%d %H:%M}",
                 "type": "missed stress" if r["err"] > 0 else "false alarm", "n flights": int(r.get("n_flights", 0)),
                 "pred /100": round(100 * r["stress_rate_pred"]), "actual /100": round(100 * r["stress_rate_actual"]),
                 "explanation": R.explain_miss(r, wx)} for _, r in sel.iterrows()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("No stress_rate_actual in capacity file.")

# ---------------- Data & limits ----------------
with tab_lim:
    cm = next((m for m in metrics.values() if m.get("stress_definition")), {})
    st.markdown(f"""
### Data & limits
- **Sample data, not the full schedule.** The provided BTS file is a sample (~6 ATL arrivals/day), so the true arrival
  rate (AAR) cannot be computed. **Capacity stress is a proxy:** {cm.get('stress_definition', 'share of sampled flights in a 3h block with >45-min delay / cancellation / diversion')}.
  Blocks with very few flights are noisy, so we show `n_flights` and compare airports only as **rates per 100 flights**.
- **Weather time-zone fix.** In the provided CSV, weather attached to a flight at *local* hour H is really the observation at
  *UTC* hour H (verified against raw METARs, temp error 0.03 °C). We relabel each observation to its true `obs_utc`
  (4–6 h before the flight) instead of pretending it is concurrent.
- **Wind direction.** The CSV has no direction. We take it from public METARs (Iowa Environmental Mesonet ASOS archive,
  same stations). Crosswind/tailwind are computed against the runway heading (ATL 80°, DFW 180°).
  When direction is missing (calm/variable), **crosswind = full wind speed (upper bound)**, flagged `wind_dir_known=False`,
  and shown as ▵ "upper bound". The rule then says *verify direction* instead of acting.
- **Observations, not forecasts.** Inputs are observations available at issue time. Weather that develops after issue time
  (e.g. afternoon thunderstorms at 12h horizon) cannot be seen. This is our main failure mode. Next step: add TAF/convective forecasts.
- **No leakage.** Forecasts are selected only if issued at or before the decision time; rules read only observations ≤ decision time;
  persistence baselines use the last block that *ended* before issue time; test period = 2025, never trained on.
- **Human in the loop.** Recommendations come from transparent rules (thresholds in `src/dash_config.py`). Every card lists the
  timestamped values used. Decisions are logged with the operator's note.
""")
    with st.expander("Rule thresholds in use"):
        st.json(C.RULES)