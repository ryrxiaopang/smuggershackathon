"""Dashboard + rules configuration. Thresholds live here so they are easy to tune."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
PROCESSED = ROOT / "data" / "processed"
CAPACITY_PREDS = OUTPUTS / "capacity_preds.parquet"
FLIGHT_PREDS = OUTPUTS / "flight_risk_preds.parquet"
WEATHER_TIMELINE = PROCESSED / "weather_timeline.parquet"
DECISION_LOG = OUTPUTS / "decision_log.csv"

# Brief assumptions. Any airport works; unknown airports fall back to UTC display.
RUNWAY_HEADING = {"ATL": 80, "DFW": 180}
AIRPORT_TZ = {"ATL": "America/New_York", "DFW": "America/Chicago"}

HORIZONS = [3, 6, 12]
BLOCK_HOURS = 3
TEST_START, TEST_END = "2025-01-01", "2025-07-31"

WX_GROUP_ORDER = ["clear", "cloudy", "fog", "rain", "showers", "freezing", "snow", "thunderstorm"]


def coco_to_group(c):
    """Meteostat condition code -> coarse weather group."""
    try:
        c = int(c)
    except (TypeError, ValueError):
        return "unknown"
    if c <= 2: return "clear"
    if c <= 4: return "cloudy"
    if c <= 6: return "fog"
    if c <= 9: return "rain"
    if c <= 13: return "freezing"
    if c <= 16: return "snow"
    if c <= 18: return "showers"
    if c <= 20: return "freezing"
    if c <= 22: return "snow"
    return "thunderstorm"


RULES = {
    # capacity stress: compare prediction with the airport's historical percentiles (columns in capacity file)
    "stress_watch_col": "hist_p80",
    "stress_action_col": "hist_p90",
    # crosswind (kt), computed relative to runway heading
    "crosswind_watch_kt": 18.0,
    "crosswind_action_kt": 25.0,
    "crosswind_unknown_dir_kt": 25.0,   # upper-bound threshold when wind direction is unknown
    # observed weather
    "wx_lookback_h": 3,
    "convective_groups": ["thunderstorm"],
    "winter_groups": ["freezing", "snow"],
    "precip_watch_mm": 5.0,             # total precipitation in lookback window
    "precip_heavy_mm_h": 2.0,           # per-hour, used for failure explanations
    # at-risk flight share in the decision window
    "p_delay_threshold": 0.35,
    "share_watch": 0.25,
    "share_action": 0.40,
    "min_flights": 3,
}