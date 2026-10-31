import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import recommend as R

NOW = pd.Timestamp("2025-05-20 14:00", tz="UTC")
H = pd.Timedelta(hours=1)


def _blocks(pred, issue=NOW):
    return pd.DataFrame({"airport": ["ATL"], "block_start_utc": [NOW + 3 * H], "block_end_utc": [NOW + 6 * H],
                         "horizon_h": [3], "issue_time_utc": [issue], "stress_rate_pred": [pred],
                         "hist_p50": [0.10], "hist_watch": [0.20], "hist_action": [0.30]})


def _wx(gxw, known, at):
    return pd.DataFrame({"airport": ["ATL"], "obs_utc": [at], "crosswind_kt": [gxw - 5.0],
                         "gust_crosswind_kt": [float(gxw)], "wind_dir_known": [known],
                         "drct": [170.0 if known else float("nan")], "runway_heading": [80],
                         "prcp_mm": [0.0], "wx_group": ["cloudy"]})


def test_stress_levels_and_evidence():
    assert R.rule_stress(_blocks(0.45), NOW).level == "ACTION"
    assert R.rule_stress(_blocks(0.25), NOW).level == "WATCH"
    assert R.rule_stress(_blocks(0.10), NOW).level == "NORMAL"
    rec = R.rule_stress(_blocks(0.45), NOW)
    assert rec.evidence[0]["value"] == 0.45 and rec.evidence[0]["obs_or_valid_time"] == "2025-05-20 17:00Z"


def test_stress_ignores_forecast_issued_after_now():
    assert R.rule_stress(_blocks(0.9, issue=NOW + H), NOW).level == "NORMAL"


def test_crosswind_unknown_direction_is_upper_bound_watch():
    rec = R.rule_crosswind(_wx(28, False, NOW - H), NOW)
    assert rec.level == "WATCH" and "upper bound" in rec.action.lower()
    assert any(e["signal"] == "wind_dir_known" and e["value"] is False for e in rec.evidence)
    assert R.rule_crosswind(_wx(28, True, NOW - H), NOW).level == "ACTION"


def test_crosswind_ignores_future_obs():
    assert R.rule_crosswind(_wx(40, True, NOW + H), NOW).level == "NORMAL"


def test_flight_share():
    fl = pd.DataFrame({"flight_id": list("ABCD"), "side": ["ARR"] * 4,
                       "sched_utc": [NOW + k * H for k in (1, 2, 3, 4)],
                       "p_delay45": [0.6, 0.5, 0.1, 0.1]})
    rec = R.rule_flight_share(fl, NOW, 6)
    assert rec.level == "ACTION" and "buffers" in rec.action
    assert R.rule_flight_share(fl.head(2), NOW, 6).level == "NORMAL"   # below min_flights

def test_storm_group_from_pipeline_triggers_watch():
    # the pipeline writes 'storm' (config.COCO_GROUP); the rule must use the same name
    w = _wx(5, True, NOW - H).assign(wx_group="storm")
    assert R.rule_weather_obs(w, NOW).level == "WATCH"
