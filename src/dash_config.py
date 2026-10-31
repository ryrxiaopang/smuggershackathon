"""Dashboard + rules configuration. Thresholds live here so they are easy to tune."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import config as _shared  # noqa: E402  airports, time zones, runway headings, split dates
OUTPUTS = ROOT / "outputs"
PROCESSED = ROOT / "data" / "processed"
CAPACITY_PREDS = OUTPUTS / "capacity_preds.parquet"
FLIGHT_PREDS = OUTPUTS / "flight_risk_preds.parquet"
WEATHER_TIMELINE = PROCESSED / "weather_timeline.parquet"
DECISION_LOG = OUTPUTS / "decision_log.csv"

# From the shared config: ATL 80 / DFW 180 (brief) plus any airport looked up by src/airports.py.
RUNWAY_HEADING = _shared.RUNWAY_HEADING
AIRPORT_TZ = _shared.AIRPORT_TZ

HORIZONS = [3, 6, 12]
BLOCK_HOURS = _shared.BLOCK_HOURS  # 6h blocks, as in capacity_model.py
TRAIN_END = _shared.TRAIN_END
TEST_START, TEST_END = _shared.TEST_START, _shared.TEST_END

# Weather groups exactly as the pipeline writes them (config.COCO_GROUP), mildest first.
WX_GROUP_ORDER = ["clear", "cloud", "fog", "rain", "freezing", "snow", "storm"]


def coco_to_group(c):
    """Meteostat condition code -> weather group, same mapping as the pipeline."""
    try:
        return _shared.COCO_GROUP.get(int(c), "unknown")
    except (TypeError, ValueError):
        return "unknown"


RULES = {
    # capacity stress: compare the prediction with how disrupted this airport's 6h blocks really were in
    # the TRAIN period (realised share of stressed flights, blocks with >=3 flights). On the 2025 test set
    # (h=6) this flags ACTION on 2.2% of blocks (actual 42/100 disrupted vs 10/100 when NORMAL).
    "stress_watch_col": "hist_watch",
    "stress_action_col": "hist_action",
    "stress_watch_q": 0.75,
    "stress_action_q": 0.90,
    # crosswind (kt), computed relative to runway heading
    "crosswind_watch_kt": 18.0,
    "crosswind_action_kt": 25.0,
    "crosswind_unknown_dir_kt": 25.0,   # upper-bound threshold when wind direction is unknown
    # observed weather
    "wx_lookback_h": 3,
    "convective_groups": ["storm"],  # pipeline name for thunderstorm/lightning/hail (coco 23-27)
    "winter_groups": ["freezing", "snow"],
    "precip_watch_mm": 5.0,             # total precipitation in lookback window
    "precip_heavy_mm_h": 2.0,           # per-hour, used for failure explanations
    # at-risk flight share in the decision window
    "p_delay_threshold": 0.35,
    "share_watch": 0.25,
    "share_action": 0.40,
    "min_flights": 3,
}