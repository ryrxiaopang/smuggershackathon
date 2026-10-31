"""
src/make_fake_data.py — FAKE flight_sides and weather_timeline parquets
that satisfy the shared contract column-for-column.

Not real data. Weather is a Markov chain over wx_group with pressure
random walk and seasonal temperature; stress_flag is driven by crosswind,
precipitation, snow and storm/fog state so capacity_model.py has a signal
to learn. Overwrite data/processed/*.parquet with ry's real output once
his pipeline is ready — capacity_model.py reads the same paths.

Run: python src/make_fake_data.py
"""
from __future__ import annotations
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
START = pd.Timestamp("2023-01-01")
END   = pd.Timestamp("2025-07-31")

AIRPORTS   = ("ATL", "DFW")
RUNWAY     = {"ATL": 80, "DFW": 180}
AIRPORT_TZ = {"ATL": "America/New_York", "DFW": "America/Chicago"}
OTHER_POOL = ["LAX","ORD","JFK","SFO","DEN","SEA","BOS","MCO","LAS","MIA"]
CARRIERS   = ["DL","AA","WN","UA","B6"]

OUT_DIR = Path("data/processed")

WX_GROUPS = ["clear","cloud","fog","rain","freezing","snow","storm"]
WX_COCO   = {  # Meteostat codes
    "clear":    [1, 2],
    "cloud":    [3, 4],
    "fog":      [5, 6],
    "rain":     [7, 8, 9, 17, 18, 19],
    "freezing": [10, 11, 12, 13],
    "snow":     [14, 15, 16, 20, 21, 22],
    "storm":    [23, 24, 25, 26, 27],
}


# ---------------------------------------------------------------------- weather
def _markov_wx(airport: str, hours: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Hour-by-hour wx_group with ~0.9 persistence; climatology varies by season."""
    n = len(hours)
    doy = hours.dayofyear.values
    summer = ((doy > 150) & (doy < 260)).astype(float)
    winter = ((doy < 70)  | (doy > 320)).astype(float)
    flip = rng.random(n) >= 0.9
    u = rng.random(n)
    out = np.empty(n, dtype=object)
    base = np.array([0.40, 0.28, 0.04, 0.10, 0.02, 0.03, 0.03])  # clear,cloud,fog,rain,fz,snow,storm
    cur = "clear"
    for i in range(n):
        if i == 0 or flip[i]:
            p = base.copy()
            p[6] += 0.05 * summer[i] + (0.03 if airport == "DFW" else 0.0)
            if winter[i]:
                p[4] += 0.03
                p[5] += 0.03
            p = p / p.sum()
            cur = WX_GROUPS[np.searchsorted(np.cumsum(p), u[i])]
        out[i] = cur
    return out


def gen_weather(airport: str) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash(airport)) % (2**32))
    hours = pd.date_range(START, END + pd.Timedelta(days=1), freq="h", inclusive="left")
    n = len(hours)
    doy = hours.dayofyear.values
    hod = hours.hour.values

    temp = (15
            + 15*np.sin(2*np.pi*(doy-80)/365)
            + 5*np.sin(2*np.pi*(hod-14)/24)
            + rng.normal(0, 2, n))
    if airport == "DFW":
        temp = temp + 3.0

    # Pressure as a random walk, lightly mean-reverted
    noise = rng.normal(0, 0.25, n)
    pres  = 1013 + np.cumsum(noise)
    pres  = pres - 0.001 * np.cumsum(pres - 1013)  # mean reversion

    wspd_kt = np.clip(rng.gamma(2.2, 3.5, n), 0, None)
    event   = rng.random(n) < 0.01
    wspd_kt[event] += rng.gamma(2, 7, event.sum())
    wpgt_kt = wspd_kt * (1 + rng.gamma(1.2, 0.3, n))

    wx_group = _markov_wx(airport, hours, rng)

    # Storm bursts bump gusts
    is_storm = wx_group == "storm"
    wpgt_kt[is_storm] += rng.gamma(2, 4, is_storm.sum())

    coco = np.empty(n, dtype=np.int16)
    for g in WX_GROUPS:
        mask = wx_group == g
        if mask.any():
            coco[mask] = rng.choice(WX_COCO[g], mask.sum())

    prcp = np.zeros(n)
    is_wet = np.isin(wx_group, ["rain", "storm", "freezing"])
    prcp[is_wet] = rng.gamma(1.5, 1.2, is_wet.sum())
    prcp[is_storm] += rng.gamma(2.0, 2.0, is_storm.sum())

    snow = np.zeros(n)
    cold = temp < 2
    is_snow_hr = (wx_group == "snow") | ((wx_group == "freezing") & cold)
    snow[is_snow_hr] = rng.gamma(1.5, 2.0, is_snow_hr.sum())

    wdir_deg = rng.uniform(0, 360, n)
    wind_dir_known = rng.random(n) >= 0.1

    heading = RUNWAY[airport]
    theta = np.radians(wdir_deg - heading)
    headwind  = wspd_kt * np.cos(theta)
    tailwind  = np.maximum(0, -headwind)
    crosswind = np.abs(wspd_kt * np.sin(theta))
    gust_crosswind = np.abs(wpgt_kt * np.sin(theta))

    # Unknown wind direction → crosswind falls back to full wind speed (upper bound)
    unk = ~wind_dir_known
    crosswind[unk]      = wspd_kt[unk]
    gust_crosswind[unk] = wpgt_kt[unk]
    headwind = headwind.astype(float);  headwind[unk]  = np.nan
    tailwind = tailwind.astype(float);  tailwind[unk]  = np.nan
    wdir_vals = wdir_deg.copy();         wdir_vals[unk] = np.nan

    return pd.DataFrame({
        "airport": airport,
        "obs_utc": hours,
        "temp_c":  temp.round(2),
        "prcp_mm": prcp.round(2),
        "wspd_kt": wspd_kt.round(2),
        "wpgt_kt": wpgt_kt.round(2),
        "pres_hpa": pres.round(2),
        "snow_mm": snow.round(2),
        "coco":    coco,
        "wx_group": wx_group.astype(str),
        "wdir_deg": wdir_vals,
        "wind_dir_known": wind_dir_known,
        "headwind_kt":  headwind.round(2),
        "tailwind_kt":  tailwind.round(2),
        "crosswind_kt": crosswind.round(2),
        "gust_crosswind_kt": gust_crosswind.round(2),
    })


# ---------------------------------------------------------------------- flights
def _localise(naive: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    """Attach local tz, converting to naive UTC; NaT on DST gaps/duplicates."""
    local = naive.tz_localize(ZoneInfo(tz), nonexistent="NaT", ambiguous="NaT")
    return local.tz_convert("UTC").tz_localize(None)


def gen_flights(weather: pd.DataFrame) -> pd.DataFrame:
    frames = []
    days = pd.date_range(START, END, freq="D")
    for airport in AIRPORTS:
        for side in ("dep", "arr"):
            n_per_day = RNG.poisson(12, len(days))
            day_arr = np.repeat(np.array(days, dtype="datetime64[ns]"), n_per_day)
            m = n_per_day.sum()
            hours = RNG.integers(5, 23, m).astype("int16")
            mins  = RNG.integers(0, 60, m).astype("int16")

            local_naive = (pd.DatetimeIndex(day_arr)
                           + pd.to_timedelta(hours.astype("int64"), unit="h")
                           + pd.to_timedelta(mins.astype("int64"), unit="m"))
            sched_utc = _localise(local_naive, AIRPORT_TZ[airport])
            keep = ~sched_utc.isna()
            if not keep.all():
                local_naive = local_naive[keep]
                sched_utc   = sched_utc[keep]
                hours       = hours[keep]
                day_arr     = day_arr[keep]
                m           = int(keep.sum())

            frames.append(pd.DataFrame({
                "airport":       airport,
                "side":          side,
                "other_airport": RNG.choice(OTHER_POOL, m),
                "carrier":       RNG.choice(CARRIERS, m),
                "sched_local":   local_naive,
                "sched_utc":     sched_utc,
                "date_local":    pd.DatetimeIndex(day_arr).normalize(),
                "hour_local":    hours,
                "block_local":   ((hours // 6) * 6).astype("int16"),
                "distance":      RNG.integers(400, 2500, m).astype("int32"),
                "wx_obs_utc":    sched_utc.floor("h"),
            }))
    flights = pd.concat(frames, ignore_index=True)
    flights.insert(0, "flight_id",
                   "F" + flights.index.astype(str).str.zfill(7))

    wx = weather.rename(columns={"obs_utc": "wx_obs_utc"})
    flights = flights.merge(wx, on=["airport", "wx_obs_utc"], how="left",
                            validate="many_to_one")
    return _add_outcomes(flights)


def _add_outcomes(f: pd.DataFrame) -> pd.DataFrame:
    """Stress probability tied to the weather at wx_obs_utc so the model can learn."""
    n = len(f)
    crosswind = f["crosswind_kt"].fillna(0).to_numpy()
    prcp      = f["prcp_mm"].fillna(0).to_numpy()
    snow      = f["snow_mm"].fillna(0).to_numpy()
    wx        = f["wx_group"].fillna("clear").to_numpy()

    p = (0.06
         + 0.012 * np.clip(crosswind, 0, 40)
         + 0.03 * (prcp > 0.5)
         + 0.05 * (prcp > 5.0)
         + 0.04 * (snow > 0)
         + 0.12 * (wx == "storm")
         + 0.04 * (wx == "fog")
         + 0.03 * (wx == "freezing"))
    p = np.clip(p, 0.0, 0.85)

    stress    = RNG.random(n) < p
    cancelled = stress & (RNG.random(n) < 0.20)
    diverted  = (~cancelled) & (RNG.random(n) < 0.005)

    normal_delay = np.clip(RNG.normal(5, 15, n), 0, None)
    bad_delay    = np.clip(RNG.normal(50, 25, n), 15, 240)
    delay_min = np.where(stress & ~cancelled, bad_delay, normal_delay)

    nas = np.zeros(n); wea = np.zeros(n)
    nc = stress & ~cancelled
    share = RNG.uniform(0.3, 0.9, n)
    nas[nc] = delay_min[nc] * share[nc]
    wea[nc] = delay_min[nc] - nas[nc]

    delay_out = delay_min.astype(float)
    delay_out[cancelled | diverted] = np.nan

    d45 = (delay_min > 45).astype(float)
    d45[cancelled | diverted] = np.nan

    f = f.copy()
    f["cancelled"]     = cancelled
    f["diverted"]      = diverted
    f["delay_min"]     = delay_out.round(1)
    f["delayed_45"]    = d45
    f["nas_delay"]     = nas.round(1)
    f["weather_delay"] = wea.round(1)
    f["stress_flag"]   = cancelled | ((nas + wea) >= 15)
    return f


# ---------------------------------------------------------------------- main
CONTRACT_COLS = [
    "flight_id","airport","side","other_airport","carrier","sched_local","sched_utc",
    "date_local","hour_local","block_local","distance","cancelled","diverted",
    "delay_min","delayed_45","nas_delay","weather_delay","stress_flag",
    "wx_obs_utc","temp_c","prcp_mm","wspd_kt","wpgt_kt","pres_hpa","snow_mm",
    "coco","wx_group","wdir_deg","wind_dir_known",
    "headwind_kt","tailwind_kt","crosswind_kt","gust_crosswind_kt",
]
WEATHER_COLS = [
    "airport","obs_utc","temp_c","prcp_mm","wspd_kt","wpgt_kt","pres_hpa","snow_mm",
    "coco","wx_group","wdir_deg","wind_dir_known",
    "headwind_kt","tailwind_kt","crosswind_kt","gust_crosswind_kt",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weather = pd.concat([gen_weather(a) for a in AIRPORTS], ignore_index=True)
    weather = weather[WEATHER_COLS]
    weather.to_parquet(OUT_DIR / "weather_timeline.parquet", index=False)

    flights = gen_flights(weather)
    flights = flights[CONTRACT_COLS]
    flights.to_parquet(OUT_DIR / "flight_sides.parquet", index=False)

    print(f"weather_timeline: {len(weather):,} rows  "
          f"({weather['airport'].value_counts().to_dict()})")
    print(f"flight_sides:    {len(flights):,} rows  "
          f"stress_rate={flights['stress_flag'].mean():.3f}  "
          f"cancel_rate={flights['cancelled'].mean():.3f}  "
          f"divert_rate={flights['diverted'].mean():.3f}")


if __name__ == "__main__":
    main()
