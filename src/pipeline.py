"""Build the shared data files from the provided CSV (+ optional public wind direction).

Usage:
    python src/pipeline.py --airports ATL DFW
    python src/pipeline.py --airports KORD KDEN --out-dir data/processed/ORD_DEN   # any airport

Outputs (see README "shared file contract"):
    data/processed/flight_sides.parquet      one row per flight per selected-airport end
    data/processed/weather_timeline.parquet  one row per airport per observed UTC hour

Works for any airport in the CSV, given as FAA (ORD) or ICAO (KORD) code, with no manual steps:
  - time zone + runway heading: config.py, else looked up from public data (src/airports.py)
  - wind direction: data/external/iem_wind_*.csv; airports not covered are downloaded from IEM
    (src/fetch_wind.py). Without direction, crosswind is an upper bound (= full wind speed).
Use --offline to skip all downloads.
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
from airports import ensure as ensure_airports, iem_id  # noqa: E402
from fetch_wind import build_url, download, parse  # noqa: E402
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


def wind_files():
    return sorted(glob.glob(str(ROOT / "data" / "external" / "iem_wind_*.csv")))


def ensure_wind(airports, start, end, offline=False):
    """Download IEM wind for airports that no data/external/iem_wind_*.csv covers yet."""
    have = set()
    for f in wind_files():
        have |= set(pd.read_csv(f, usecols=["station"])["station"].unique())
    missing = [a for a in airports if a not in have]
    if not missing:
        return
    if offline:
        print(f"! --offline: no wind direction for {missing}; crosswind is an upper bound there.")
        return
    ids = {iem_id(a): a for a in missing}
    try:
        w = parse(download(build_url(list(ids), start, end)))
    except Exception as exc:
        print(f"! Wind download failed ({exc}); crosswind is an upper bound for {missing}.")
        return
    w["station"] = w["station"].map(ids)  # IEM station ids -> CSV airport codes
    out = ROOT / "data" / "external" / f"iem_wind_{'_'.join(missing)}.csv"
    w.to_csv(out, index=False)
    print(f"Downloaded wind for {missing}: {len(w):,} station-hours -> {out.relative_to(ROOT)}")


def load_wind(airports):
    files = wind_files()
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
    w = w.rename(columns={"station": "airport", "obs_utc": "wx_obs_utc", "sknt": "iem_sknt"})
    return w[["airport", "wx_obs_utc", "wdir_deg", "iem_sknt"]]


def add_wind(fs, wind):
    if wind is not None:
        fs = fs.merge(wind[["airport", "wx_obs_utc", "wdir_deg"]], on=["airport", "wx_obs_utc"], how="left")
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


def check_alignment(tl, wind):
    """Re-checks the weather time-zone rule for every airport, every run: CSV wind speed vs the
    independent IEM wind speed at the relabelled hour and 1 h either side. If the rule holds,
    the error is smallest (near 0) at the relabelled hour."""
    if wind is None:
        return
    w = wind.set_index(["airport", "wx_obs_utc"])["iem_sknt"]
    print("time-zone rule check, mean |CSV wind - IEM wind| kt (relabelled hour / -1 h / +1 h):")
    for ap, g in tl.groupby("airport"):
        err = {}
        for sh in (0, -1, 1):
            idx = pd.MultiIndex.from_arrays([g["airport"], g["obs_utc"] + pd.Timedelta(hours=sh)])
            err[sh] = np.nanmean(np.abs(g["wspd_kt"].to_numpy() - w.reindex(idx).to_numpy()))
        if np.isnan(err[0]):
            print(f"  {ap}: no IEM wind to compare")
            continue
        ok = err[0] < min(err[-1], err[1])
        print(f"  {ap}: {err[0]:.2f} / {err[-1]:.2f} / {err[1]:.2f}  "
              + ("OK" if ok else "CHECK: weather here may not follow the UTC-hour rule"))


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
    ap.add_argument("--airports", nargs="+", default=config.AIRPORTS, help="FAA or ICAO codes, e.g. ATL KORD")
    ap.add_argument("--csv", default=str(config.RAW_CSV))
    ap.add_argument("--out-dir", default=str(config.PROCESSED),
                    help="where to write the parquets (default: the shared data/processed)")
    ap.add_argument("--offline", action="store_true", help="no downloads; use cached files only")
    a = ap.parse_args()
    airports = ensure_airports(a.airports, offline=a.offline)
    missing_tz = [x for x in airports if x not in config.AIRPORT_TZ]
    if missing_tz:
        sys.exit(f"No time zone for {missing_tz}: run without --offline, or add it to config.AIRPORT_TZ.")

    df = load_raw(a.csv, airports)
    absent = [x for x in airports if not (df["ORIGIN"].eq(x) | df["DEST"].eq(x)).any()]
    if absent:
        sys.exit(f"No flights for {absent} in {a.csv}.")
    dates = pd.to_datetime(df["FL_DATE"])
    ensure_wind(airports, dates.min() - pd.Timedelta(days=1), dates.max() + pd.Timedelta(days=2), a.offline)
    wind = load_wind(airports)

    fs = explode_sides(df, airports)
    fs = add_times(fs)
    fs = add_labels(fs)
    fs = add_weather(fs)
    fs = add_wind(fs, wind)
    for c in ["sched_local", "sched_utc", "date_local", "wx_obs_utc"]:
        fs[c] = pd.to_datetime(fs[c]).astype(NS)
    fs = fs.sort_values(["sched_utc", "flight_id", "side"]).reset_index(drop=True)
    tl = build_timeline(fs)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fs.to_parquet(out / config.FLIGHT_SIDES.name, index=False)
    tl.to_parquet(out / config.WEATHER_TIMELINE.name, index=False)
    report(fs, tl)
    check_alignment(tl, wind)
    print(f"\nSaved {config.FLIGHT_SIDES.name} and {config.WEATHER_TIMELINE.name} to {out}")


if __name__ == "__main__":
    main()
