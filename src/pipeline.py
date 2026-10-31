"""Build the shared data files from the provided CSV (+ optional public wind direction).

Usage:
    python src/pipeline.py --airports ATL DFW

Outputs (see README "shared file contract"):
    data/processed/flight_sides.parquet      one row per flight per selected-airport end
    data/processed/weather_timeline.parquet  one row per airport per observed UTC hour

Works for any airport code that has an entry in config.AIRPORT_TZ (and config.RUNWAY_HEADING for
wind components). Wind direction comes from data/external/iem_wind_*.csv if present
(run src/fetch_wind.py); otherwise crosswind is an upper bound (= full wind speed).
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import config  # noqa: E402
from wind import wind_components  # noqa: E402

FLIGHT_COLS = ["FL_DATE", "ORIGIN", "DEST", "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM", "TAIL_NUM",
               "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "DEP_DELAY", "ARR_DELAY",
               "CANCELLED", "CANCELLATION_CODE", "DIVERTED", "DISTANCE", "CARRIER_DELAY",
               "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
               "TAXI_OUT", "TAXI_IN"]
WX_VARS = ["temp", "prcp", "wspd", "wpgt", "pres", "snow", "coco"]
NS = "datetime64[ns]"


def load_raw(path, airports):
    cols = FLIGHT_COLS + [f"{v}_{s}" for v in WX_VARS for s in ("ORIGIN", "DEST")]
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    df = df[df["ORIGIN"].isin(airports) | df["DEST"].isin(airports)].reset_index(drop=True)
    df["flight_id"] = np.arange(len(df))  # stable id per flight (both ends share it)
    return df


def hhmm_to_offset(hhmm):
    hhmm = hhmm.fillna(0).astype(int)
    return pd.to_timedelta(hhmm // 100, unit="h") + pd.to_timedelta(hhmm % 100, unit="m")


def explode_sides(df, airports):
    """One row per flight per selected-airport end. A flight between two selected airports gives 2 rows."""
    date = pd.to_datetime(df["FL_DATE"])
    dep_local = date + hhmm_to_offset(df["CRS_DEP_TIME"])
    arr_local = date + hhmm_to_offset(df["CRS_ARR_TIME"])
    # overnight arrival: scheduled arrival clock time earlier than departure -> next day
    arr_local = arr_local.where(df["CRS_ARR_TIME"] >= df["CRS_DEP_TIME"], arr_local + pd.Timedelta(days=1))

    sides = []
    for side, ap_col, other_col, hhmm_col, local, sfx, delay_col in [
        ("dep", "ORIGIN", "DEST", "CRS_DEP_TIME", dep_local, "_ORIGIN", "DEP_DELAY"),
        ("arr", "DEST", "ORIGIN", "CRS_ARR_TIME", arr_local, "_DEST", "ARR_DELAY"),
    ]:
        m = df[ap_col].isin(airports)
        s = pd.DataFrame({
            "flight_id": df.loc[m, "flight_id"],
            "airport": df.loc[m, ap_col],
            "side": side,
            "other_airport": df.loc[m, other_col],
            "carrier": df.loc[m, "OP_UNIQUE_CARRIER"],
            "flight_number": df.loc[m, "OP_CARRIER_FL_NUM"],
            "tail_num": df.loc[m, "TAIL_NUM"],
            "sched_local": local[m],
            "distance": df.loc[m, "DISTANCE"],
            "cancelled": df.loc[m, "CANCELLED"].fillna(0).astype(int),
            "cancellation_code": df.loc[m, "CANCELLATION_CODE"],
            "diverted": df.loc[m, "DIVERTED"].fillna(0).astype(int),
            "delay_min": df.loc[m, delay_col],
            "nas_delay": df.loc[m, "NAS_DELAY"],
            "weather_delay": df.loc[m, "WEATHER_DELAY"],
            "carrier_delay": df.loc[m, "CARRIER_DELAY"],
            "late_aircraft_delay": df.loc[m, "LATE_AIRCRAFT_DELAY"],
            "taxi_min": df.loc[m, "TAXI_OUT" if side == "dep" else "TAXI_IN"],
        })
        # Provided-weather time-zone shift (verified vs raw METARs): the reading attached to local
        # hour H on FL_DATE is really the observation at UTC hour H on FL_DATE.
        hh = (df.loc[m, hhmm_col].fillna(0).astype(int) // 100) % 24
        s["wx_obs_utc"] = date[m] + pd.to_timedelta(hh, unit="h")
        for v in WX_VARS:
            s[f"raw_{v}"] = df.loc[m, v + sfx]
        sides.append(s)
    return pd.concat(sides, ignore_index=True)


def add_times(fs):
    fs["sched_utc"] = pd.NaT
    for ap, tz in config.AIRPORT_TZ.items():
        m = fs["airport"] == ap
        if m.any():
            fs.loc[m, "sched_utc"] = (fs.loc[m, "sched_local"]
                                      .dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
                                      .dt.tz_convert("UTC").dt.tz_localize(None))
    fs["sched_utc"] = pd.to_datetime(fs["sched_utc"])
    fs["date_local"] = fs["sched_local"].dt.normalize()
    fs["hour_local"] = fs["sched_local"].dt.hour
    fs["block_local"] = fs["hour_local"] // config.BLOCK_HOURS * config.BLOCK_HOURS
    return fs


def add_labels(fs):
    bad = (fs["cancelled"] == 1) | (fs["diverted"] == 1)
    fs["delayed_45"] = np.where(bad, np.nan, (fs["delay_min"] > config.DELAY_THRESHOLD_MIN).astype(float))
    cap_delay = fs["nas_delay"].fillna(0) + fs["weather_delay"].fillna(0)
    fs["stress_flag"] = ((fs["cancelled"] == 1) | (cap_delay >= config.STRESS_DELAY_MIN)).astype(int)
    return fs


def add_weather(fs):
    fs["temp_c"] = fs["raw_temp"]
    fs["prcp_mm"] = fs["raw_prcp"]
    fs["wspd_kt"] = (fs["raw_wspd"] / config.KMH_PER_KT).round(1)
    # 0 gust = "no gust reported" (~98% of rows), not a measured 0
    fs["wpgt_kt"] = (fs["raw_wpgt"].where(fs["raw_wpgt"] > 0) / config.KMH_PER_KT).round(1)
    fs["pres_hpa"] = fs["raw_pres"]
    fs["snow_mm"] = fs["raw_snow"]  # always 0 at ATL/DFW; kept for the contract, do not model on it
    fs["coco"] = fs["raw_coco"].where(fs["raw_coco"] > 0)  # 0 is not a valid code
    fs["wx_group"] = fs["coco"].map(config.COCO_GROUP).fillna("unknown")
    return fs.drop(columns=[c for c in fs.columns if c.startswith("raw_")])


def load_wind(airports):
    files = sorted(glob.glob(str(ROOT / "data" / "external" / "iem_wind_*.csv")))
    if not files:
        print("! No wind file in data/external - run `python src/fetch_wind.py`. "
              "Crosswind will be an upper bound for every row.")
        return None
    w = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    w = w[w["station"].isin(airports)]
    w["obs_utc"] = pd.to_datetime(w["obs_utc"]).astype(NS)
    w = w.drop_duplicates(["station", "obs_utc"], keep="last")
    # METAR direction 0 is only used for calm wind -> direction unknown
    w["wdir_deg"] = w["drct"].where(w["drct"] > 0)
    return w.rename(columns={"station": "airport", "obs_utc": "wx_obs_utc"})[["airport", "wx_obs_utc", "wdir_deg"]]


def add_wind(fs, wind):
    if wind is not None:
        fs = fs.merge(wind, on=["airport", "wx_obs_utc"], how="left")
    else:
        fs["wdir_deg"] = np.nan
    heading = fs["airport"].map(config.RUNWAY_HEADING)
    if heading.isna().any():
        print(f"! No runway heading for {sorted(fs.loc[heading.isna(), 'airport'].unique())}; "
              "add them to config.RUNWAY_HEADING. Using crosswind upper bound for those rows.")
    w = wind_components(fs["wdir_deg"].where(heading.notna()), fs["wspd_kt"], heading.fillna(0))
    g = wind_components(fs["wdir_deg"].where(heading.notna()), fs["wpgt_kt"], heading.fillna(0))
    fs["wind_dir_known"] = w["wind_dir_known"]
    fs["headwind_kt"] = w["headwind"]
    fs["tailwind_kt"] = w["tailwind"]
    fs["crosswind_kt"] = w["crosswind"]
    fs["gust_crosswind_kt"] = g["crosswind"]  # NaN when no gust reported
    return fs


TIMELINE_COLS = ["temp_c", "prcp_mm", "wspd_kt", "wpgt_kt", "pres_hpa", "snow_mm", "coco", "wx_group",
                 "wdir_deg", "wind_dir_known", "headwind_kt", "tailwind_kt", "crosswind_kt", "gust_crosswind_kt"]


def build_timeline(fs):
    tl = (fs.sort_values(["airport", "wx_obs_utc"])
            .drop_duplicates(["airport", "wx_obs_utc"])[["airport", "wx_obs_utc"] + TIMELINE_COLS]
            .rename(columns={"wx_obs_utc": "obs_utc"}).reset_index(drop=True))
    return tl


def report(fs, tl):
    print("\n=== Data report ===")
    print(f"flight_sides: {len(fs):,} rows, {fs['flight_id'].nunique():,} flights")
    print(fs.groupby(["airport", "side"]).size().rename("rows").to_string())
    print(f"date range (local): {fs['date_local'].min().date()} -> {fs['date_local'].max().date()}")
    print(f"sched_utc missing (DST gap/ambiguous): {fs['sched_utc'].isna().sum()}")
    print(f"weather_timeline: {len(tl):,} airport-hours; per airport-day mean "
          f"{tl.groupby(['airport', tl['obs_utc'].dt.date]).size().mean():.1f} hours")
    print(f"wind direction known: {fs['wind_dir_known'].mean() * 100:.1f}% of flight rows")
    print(f"gust reported: {fs['wpgt_kt'].notna().mean() * 100:.1f}% of flight rows")
    rates = fs.groupby("airport").agg(stress_per_100=("stress_flag", "mean"), delayed45_per_100=("delayed_45", "mean")) * 100
    print("per 100 flights:\n" + rates.round(1).to_string())
    lag = (fs["sched_utc"] - fs["wx_obs_utc"]).dt.total_seconds() / 3600
    print(f"provided weather is observed {lag.median():.1f} h (median) before the scheduled time")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--airports", nargs="+", default=config.AIRPORTS)
    ap.add_argument("--csv", default=str(config.RAW_CSV))
    a = ap.parse_args()
    airports = [x.upper() for x in a.airports]
    missing_tz = [x for x in airports if x not in config.AIRPORT_TZ]
    if missing_tz:
        sys.exit(f"Add a time zone for {missing_tz} to config.AIRPORT_TZ first.")

    df = load_raw(a.csv, airports)
    fs = explode_sides(df, airports)
    fs = add_times(fs)
    fs = add_labels(fs)
    fs = add_weather(fs)
    fs = add_wind(fs, load_wind(airports))
    for c in ["sched_local", "sched_utc", "date_local", "wx_obs_utc"]:
        fs[c] = pd.to_datetime(fs[c]).astype(NS)
    fs = fs.sort_values(["sched_utc", "flight_id", "side"]).reset_index(drop=True)
    tl = build_timeline(fs)

    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    fs.to_parquet(config.FLIGHT_SIDES, index=False)
    tl.to_parquet(config.WEATHER_TIMELINE, index=False)
    report(fs, tl)
    print(f"\nSaved {config.FLIGHT_SIDES.relative_to(ROOT)} and {config.WEATHER_TIMELINE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
