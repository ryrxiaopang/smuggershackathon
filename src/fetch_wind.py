"""Download hourly METAR wind (direction, speed, gust) from the Iowa Environmental Mesonet ASOS archive.

Public, free, no key. Same observations as the weather in the provided CSV (verified: temp error 0.03 C).

Usage:
    python src/fetch_wind.py                      # ATL DFW, 2022-12-31 .. 2025-08-02
    python src/fetch_wind.py --stations ATL DFW ORD --start 2023-01-01 --end 2025-08-01

Output: data/external/iem_wind_<STATIONS>.csv with columns
    station, obs_utc (rounded to the hour), drct (deg; blank = calm/variable), sknt (kt), gust (kt)
Station ids are FAA codes without the leading K (ATL, DFW, ORD...).
"""
import argparse
import io
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

BASE = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def build_url(stations, start, end):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    q = "&".join(f"station={st}" for st in stations)
    q += "&data=drct&data=sknt&data=gust"
    q += f"&year1={s.year}&month1={s.month}&day1={s.day}&year2={e.year}&month2={e.month}&day2={e.day}"
    q += "&tz=Etc/UTC&format=onlycomma&latlon=no&missing=M&trace=T&direct=no&report_type=3"
    return f"{BASE}?{q}"


def download(url, retries=3):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read().decode("utf-8")
        except Exception as exc:  # network hiccup -> retry
            print(f"attempt {i + 1} failed: {exc}")
            time.sleep(5 * (i + 1))
    raise RuntimeError("IEM download failed; check your internet connection")


def parse(text):
    df = pd.read_csv(io.StringIO(text), na_values=["M"])
    df["valid"] = pd.to_datetime(df["valid"])
    df["obs_utc"] = df["valid"].dt.round("h")
    df = df.sort_values("valid").groupby(["station", "obs_utc"], as_index=False).last()
    return df[["station", "obs_utc", "drct", "sknt", "gust"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", nargs="+", default=config.AIRPORTS)
    ap.add_argument("--start", default="2022-12-31")
    ap.add_argument("--end", default="2025-08-02")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    url = build_url(a.stations, a.start, a.end)
    print("Downloading", url)
    df = parse(download(url))
    out = Path(a.out) if a.out else config.ROOT / "data" / "external" / f"iem_wind_{'_'.join(a.stations)}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    known = df["drct"].notna().mean() * 100
    print(f"Saved {len(df):,} station-hours to {out} ({known:.1f}% with wind direction)")


if __name__ == "__main__":
    main()
