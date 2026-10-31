"""Shared configuration. Import from here; do not hard-code airports, paths or dates elsewhere."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- Airports (extend these dicts to support any airport code) ---
AIRPORTS = ["ATL", "DFW"]
RUNWAY_HEADING = {"ATL": 80, "DFW": 180}  # degrees, per organiser assumption
AIRPORT_TZ = {"ATL": "America/New_York", "DFW": "America/Chicago"}

# --- Paths ---
RAW_CSV = ROOT / "data" / "raw" / "Flight_Data_With_Weather_Final.csv"
WIND_CSV = ROOT / "data" / "external" / "iem_wind_ATL_DFW.csv"
PROCESSED = ROOT / "data" / "processed"
FLIGHT_SIDES = PROCESSED / "flight_sides.parquet"
WEATHER_TIMELINE = PROCESSED / "weather_timeline.parquet"
OUTPUTS = ROOT / "outputs"
CAPACITY_PREDS = OUTPUTS / "capacity_preds.parquet"
FLIGHT_RISK_PREDS = OUTPUTS / "flight_risk_preds.parquet"

# --- Time split (never shuffle across time) ---
TRAIN_END = "2024-06-30"
VAL_START, VAL_END = "2024-07-01", "2024-12-31"
TEST_START, TEST_END = "2025-01-01", "2025-07-31"

# --- Modelling constants ---
HORIZONS_H = [3, 6, 12]
BLOCK_HOURS = 6
DELAY_THRESHOLD_MIN = 45
STRESS_DELAY_MIN = 15  # nas_delay + weather_delay >= this, or cancelled
KMH_PER_KT = 1.852

# Meteostat weather condition code (coco) -> group
# 1 clear, 2 fair, 3 cloudy, 4 overcast, 5 fog, 6 freezing fog, 7-9 rain, 10-13 freezing rain/sleet,
# 14-16 snow, 17-18 rain shower, 19-20 sleet shower, 21-22 snow shower, 23 lightning, 24 hail,
# 25-26 thunderstorm, 27 storm
COCO_GROUP = {
    1: "clear", 2: "clear", 3: "cloud", 4: "cloud", 5: "fog", 6: "fog",
    7: "rain", 8: "rain", 9: "rain", 10: "freezing", 11: "freezing", 12: "freezing", 13: "freezing",
    14: "snow", 15: "snow", 16: "snow", 17: "rain", 18: "rain", 19: "freezing", 20: "freezing",
    21: "snow", 22: "snow", 23: "storm", 24: "storm", 25: "storm", 26: "storm", 27: "storm",
}
