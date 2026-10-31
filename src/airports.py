"""Time zone and runway heading for any airport code, looked up from public data.

    python src/airports.py KORD DEN       # look up, cache and print
    python src/airports.py --self-test    # offline check of the code-handling logic

Accepts FAA/IATA codes as used in the CSV (ORD) or ICAO codes (KORD). Sources, downloaded only
when an airport is not cached yet:
  - OurAirports (https://ourairports.com/data/): ICAO <-> FAA code, state, runways
  - Iowa Environmental Mesonet ASOS station list: time zone, station id for the wind download
Results are cached in data/external/airport_meta.csv (committed), which config.py reads.
Values in config.RUNWAY_HEADING / config.AIRPORT_TZ always win (ATL 80 and DFW 180 are the
organiser's assumption; OurAirports says ATL 90).

Runway heading = the dominant runway orientation: open runways grouped by direction (to 10 deg),
the group with the most total runway length wins, heading = median true heading of its low-numbered
end (e.g. ORD 09/27 -> 90). Headwind/tailwind are relative to that direction, as for ATL 80 / DFW 180.
"""
import argparse
import csv
import io
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

OURAIRPORTS = "https://davidmegginson.github.io/ourairports-data/{}.csv"
IEM_NETWORK = "https://mesonet.agron.iastate.edu/geojson/network/{}_ASOS.geojson"
META_COLS = ["code", "icao", "name", "tz", "runway_heading", "heading_source", "iem_id"]


def _get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read().decode("utf-8")


def load_cache():
    if not config.AIRPORT_META.exists():
        return pd.DataFrame(columns=META_COLS)
    return pd.read_csv(config.AIRPORT_META, dtype={"code": str, "icao": str, "iem_id": str})


def known_codes():
    return set(config.AIRPORT_TZ) | set(load_cache()["code"])


def normalize(code, known=None, lookup=None):
    """'kord' -> 'ORD'. Offline for cached airports and K-prefixed ICAO codes of known airports;
    otherwise `lookup` (OurAirports airports table) resolves ICAO codes such as PHNL -> HNL."""
    code = code.strip().upper()
    known = known_codes() if known is None else known
    if code in known:
        return code
    cache = load_cache()
    hit = cache.loc[cache["icao"] == code, "code"]
    if len(hit):
        return hit.iloc[0]
    if len(code) == 4 and code[0] == "K" and code[1:] in known:
        return code[1:]
    if lookup is not None:
        for col in ["iata_code", "ident", "icao_code", "gps_code", "local_code"]:
            hit = lookup[lookup[col] == code]
            if len(hit):
                r = hit.iloc[0]
                return r["iata_code"] if isinstance(r["iata_code"], str) else r["local_code"]
    return code


def dominant_heading(runways):
    rw = runways[(runways["closed"] == 0) & runways["length_ft"].notna()].copy()
    from_ident = pd.to_numeric(rw["le_ident"].str.extract(r"^(\d{1,2})")[0], errors="coerce") * 10
    rw["hdg"] = rw["le_heading_degT"].fillna(from_ident)
    rw = rw.dropna(subset=["hdg"])
    if rw.empty:
        return None
    rw["orient"] = (rw["hdg"] % 180 / 10).round() % 18
    best = rw.groupby("orient")["length_ft"].sum().idxmax()
    return int(round(rw.loc[rw["orient"] == best, "hdg"].median())) % 360


def _iem_station(state, ids):
    """(tz, iem_id) from the IEM ASOS station list for that state/territory."""
    try:
        feats = json.loads(_get(IEM_NETWORK.format(state)))["features"]
    except Exception as exc:  # network down / unknown network
        print(f"! IEM station list for {state} unavailable ({exc})")
        return None, None
    by_id = {f["id"]: f["properties"] for f in feats}
    for i in ids:
        if isinstance(i, str) and i in by_id:
            return by_id[i].get("tzname"), i
    return None, None


def _ourairports_table():
    ap = pd.read_csv(io.StringIO(_get(OURAIRPORTS.format("airports"))), low_memory=False,
                     usecols=["ident", "name", "iso_country", "iso_region", "icao_code",
                              "iata_code", "gps_code", "local_code", "type"])
    return ap[ap["type"] != "closed"]


def lookup(codes, ap):
    """Metadata for codes not cached yet, from the OurAirports table `ap` + runways + IEM."""
    print(f"Looking up {codes} from OurAirports + IEM ...")
    rw = pd.read_csv(io.StringIO(_get(OURAIRPORTS.format("runways"))), low_memory=False,
                     usecols=["airport_ident", "length_ft", "closed", "le_ident", "le_heading_degT"])
    rows = []
    for code in codes:
        hit = ap[ap["iata_code"] == code]
        if hit.empty:
            hit = ap[ap["local_code"] == code]
        if hit.empty:
            print(f"! {code} not found in OurAirports")
            continue
        a = hit.sort_values("type").iloc[0]  # large_airport < medium_airport < small_airport
        state = a["iso_region"].split("-")[1] if a["iso_country"] == "US" else a["iso_country"]
        tz, iem_id = _iem_station(state, [code, a["local_code"], a["ident"], a["ident"][1:]])
        rows.append({"code": code, "icao": a["ident"], "name": a["name"], "tz": tz,
                     "runway_heading": dominant_heading(rw[rw["airport_ident"] == a["ident"]]),
                     "heading_source": "ourairports", "iem_id": iem_id or code})
    return pd.DataFrame(rows, columns=META_COLS)


def ensure(codes, offline=False):
    """Normalise codes and make sure each has a time zone (and, if possible, a runway heading).
    Updates the cache file and config.AIRPORT_TZ / config.RUNWAY_HEADING in place."""
    known = known_codes()
    out = [normalize(c, known) for c in codes]
    missing = [c for c in out if c not in config.AIRPORT_TZ]
    if missing and not offline:
        table = _ourairports_table()
        out = [normalize(c, known, table) for c in codes]
        missing = [c for c in out if c not in config.AIRPORT_TZ]
    if missing and not offline:
        new = lookup(missing, table)
        cache = load_cache()
        cache = pd.concat([cache[~cache["code"].isin(new["code"])], new], ignore_index=True)
        config.AIRPORT_META.parent.mkdir(parents=True, exist_ok=True)
        cache.sort_values("code").to_csv(config.AIRPORT_META, index=False)
        _apply(new)
    return list(dict.fromkeys(out))  # dedupe, keep order


def _apply(rows):
    for r in rows.itertuples():
        if isinstance(r.tz, str):
            config.AIRPORT_TZ.setdefault(r.code, r.tz)
        if pd.notna(r.runway_heading):
            config.RUNWAY_HEADING.setdefault(r.code, int(r.runway_heading))


def iem_id(code):
    cache = load_cache()
    hit = cache.loc[cache["code"] == code, "iem_id"]
    return hit.iloc[0] if len(hit) and isinstance(hit.iloc[0], str) else code


def _self_test():
    known = {"ATL", "DFW"}
    assert normalize(" katl", known) == "ATL" and normalize("DFW", known) == "DFW"
    fake = pd.DataFrame({"iata_code": ["HNL"], "ident": ["PHNL"], "icao_code": ["PHNL"],
                         "gps_code": ["PHNL"], "local_code": ["HNL"]})
    assert normalize("PHNL", known, fake) == "HNL"
    rw = pd.DataFrame({"length_ft": [13000, 9000, 9000, 4000], "closed": [0, 0, 1, 0],
                       "le_ident": ["17C", "13L", "09", "H1"], "le_heading_degT": [180.3, 135.0, 90.0, None]})
    assert dominant_heading(rw) == 180  # closed 09 and the helipad are ignored
    rw2 = pd.DataFrame({"length_ft": [10000], "closed": [0], "le_ident": ["08L"], "le_heading_degT": [None]})
    assert dominant_heading(rw2) == 80  # falls back to the runway number
    print("airports.py self-test passed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("codes", nargs="*")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test or not a.codes:
        return _self_test()
    for c in ensure(a.codes):
        src = "organiser" if c in config.ORGANISER_HEADING else "OurAirports"
        print(f"{c}: tz={config.AIRPORT_TZ.get(c)} runway_heading={config.RUNWAY_HEADING.get(c)} ({src}) "
              f"iem_id={iem_id(c)}")


if __name__ == "__main__":
    main()
