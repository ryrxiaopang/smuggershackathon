"""Runway-relative wind components.

wind_components(wdir_deg, speed_kt, heading_deg) -> dict of numpy arrays:
    headwind (negative = tailwind), tailwind (>=0), crosswind (>=0), wind_dir_known (bool)
If direction is unknown (NaN: calm/variable/missing), crosswind is an UPPER BOUND equal to
the full speed, and headwind/tailwind are NaN.
"""
import numpy as np


def wind_components(wdir_deg, speed_kt, heading_deg):
    wdir = np.asarray(wdir_deg, dtype=float)
    spd = np.asarray(speed_kt, dtype=float)
    hdg = np.asarray(heading_deg, dtype=float)
    known = ~np.isnan(wdir)
    theta = np.radians(wdir - hdg)
    head = np.where(known, spd * np.cos(theta), np.nan)
    cross = np.where(known, np.abs(spd * np.sin(theta)), spd)  # upper bound when unknown
    tail = np.where(known, np.clip(-head, 0, None), np.nan)
    return {
        "headwind": np.round(head, 2),
        "tailwind": np.round(tail, 2),
        "crosswind": np.round(cross, 2),
        "wind_dir_known": known,
    }


def _self_test():
    r = wind_components([80, 170, 260, np.nan], [10, 10, 10, 10], [80, 80, 80, 80])
    assert abs(r["headwind"][0] - 10) < 1e-6 and abs(r["crosswind"][0]) < 1e-6  # straight down the runway
    assert abs(r["crosswind"][1] - 10) < 1e-6  # 90 deg off = pure crosswind
    assert abs(r["tailwind"][2] - 10) < 1e-6  # reciprocal = pure tailwind
    assert r["crosswind"][3] == 10 and not r["wind_dir_known"][3]  # unknown -> upper bound
    print("wind.py self-test passed")


if __name__ == "__main__":
    _self_test()
