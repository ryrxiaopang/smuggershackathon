# Refinements (tested on the provided CSV, ATL/DFW subset)

All numbers below are leakage-safe: weather only up to the issue time T, test = 2025.

## What the data says

| Finding | Evidence | What to do |
|---|---|---|
| Weather predicts **capacity stress** much better than generic **>45-min delays** | Stress flag at 3h: PR-AUC 0.155 → 0.270 with weather (1.7×). >45 min at 3h: PR-AUC 0.220 → 0.261. | Make capacity stress the headline of the pitch; delay risk is the second view. |
| Most >45-min delay minutes are airline-driven | Late aircraft 45%, carrier 34%, NAS 12%, weather 8% of delay minutes | Say this in the pitch: it explains why weather alone can't predict all delays (honest evaluation). |
| Lift decays with horizon | Stress AUC: 0.713 (3h) → 0.685 (6h) → 0.663 (12h) | Show a "warning horizon vs accuracy" chart. It's honest and judges like it. |
| A 6-hour block rate is too noisy as a direct target | Median 4 flights per block; correlation with the previous block is only 0.26 | **emz:** predict P(stress) per flight, then average to block/day (below). |
| Flight-level model averaged to blocks beats both baselines | 6h blocks, h=3: RMSE **0.184** vs historical avg 0.200 vs persistence 0.241. Daily: **0.125** vs 0.148 vs 0.182. h=12 daily: 0.141 vs 0.148. | Use these as the RMSE baselines in the evaluation tab. |
| Time-zone rule also holds for arrivals | 2,986 airport-hours with both sides: 100% identical weather | **ry:** step 5 is done; skip it. |
| `wpgt` (gust) is 0 in 97.6% of rows | 0 means "no gust reported" | Treat 0 as missing; gust features use max(gust, wind). Handled in `src/features.py`. |
| `snow` (depth) is always 0 at ATL/DFW | Useless column | Use the snow weather codes (wx_group) instead. |
| `coco` = 0 in 8 rows | Not a valid code | Map to `unknown`. |
| pandas datetime resolution mismatch | `merge_asof` fails: datetime64[ns] vs [us] | Cast with `.astype("datetime64[ns]")` (see `_ns()` in `src/features.py`). |
| 2025 normalised rates | Stress per 100 flights: ATL 10.0, DFW 13.7. >45 min per 100: ATL 11.7, DFW 14.9 | Use per-100 comparisons in the dashboard. |

## Demo days (2025 test period)
- **9–10 Jan 2025, winter weather:** ATL 84% of flights stressed with 14 cancellations; DFW 80–88%. Snow codes are in the data. This is the best live-demo story.
- **DFW thunderstorm days:** 30 Apr (69% stressed, 7 cancellations, gusts 25 km/h), 6 May, 11 Jun 2025.

## Shared feature module: `src/features.py` (on `main`)
Both `emz` and `rach` were going to write the same "weather as of time T" code. This module does it once and is tested on the real data:
- `make_issues(flights, horizons)`: one row per flight × horizon, with `issued_utc = sched_utc − h`
- `build_features(issues, flights, timeline)`: last observed weather, observation age, 6h/12h rolling maxima for crosswind, gust-crosswind, wind and rain, weather-group counts, 3h/6h pressure change, calendar fields, and a congestion feature (stress share among same-airport flights that finished at least 3h before T)
- `feature_columns(X)`: model inputs (drops keys and all-empty columns)

Get it with `git pull origin main` (or `git merge origin/main`) on your branch.

## Follow-up prompts (paste into your existing Claude conversation)

**ry**
```
Updates: (1) the arrival-side time-zone check is already verified (100% match on 2,986 airport-hours) — skip it. (2) In the pipeline treat wpgt==0 as missing (it means no gust reported, 97.6% of rows), drop snow (always 0 at ATL/DFW), map coco 0 to 'unknown'. (3) Cast all datetime columns with .astype('datetime64[ns]') before saving. (4) The processed parquets are small (~25k rows): remove data/processed from .gitignore and commit flight_sides.parquet and weather_timeline.parquet so teammates don't need to run the pipeline. src/features.py on main reads these files, so keep the contract column names exactly.
```

**emz**
```
Change of approach (tested on the real data): a 6-hour block stress rate has a median of 4 flights per block and is too noisy as a direct target. Instead: train a LightGBM classifier for P(stress_flag) per flight using src/features.py (make_issues → build_features → feature_columns), then predict the block value as the mean of flight probabilities per airport × block (and per day). Evaluate block RMSE vs persistence and the historical average (airport × block × month from train) on blocks with ≥3 flights. Reference numbers from a quick HistGradientBoosting test at h=3: 6h-block RMSE 0.184 vs hist 0.200 vs persistence 0.241; daily 0.125 vs 0.148 vs 0.182. You should match or beat these. Keep the capacity_preds.parquet contract (y_pred = mean probability per block). Also output a small table of RMSE by horizon (3/6/12) for a "warning horizon vs accuracy" chart.
```

**rach**
```
Use src/features.py on main for all features (make_issues → build_features → feature_columns); don't write your own weather-as-of code. It already includes a congestion feature (stress share among same-airport flights that finished ≥3h before T). Reference from a quick test (2025 test, PR-AUC): 3h 0.261, 6h 0.242, 12h 0.228 vs base rate 0.132; calendar-only 3h PR-AUC ≈ 0.220. Note in your summary that >45-min delays are mostly airline-driven (late aircraft 45%, carrier 34% of delay minutes vs weather 8%, NAS 12%), which caps weather's lift. Treat gust=0 as missing (already handled in features.py).
```

**deph**
```
Updates for the dashboard and pitch: (1) Demo day: 9–10 Jan 2025 (winter weather; ATL 84% of flights stressed, 14 cancellations). Backups: DFW 30 Apr, 6 May, 11 Jun 2025 (thunderstorms). (2) Gust is missing in ~98% of rows, so base crosswind rules on max(gust crosswind, crosswind) and keep the "upper bound" label when wind direction is unknown. (3) Headline = capacity stress (weather lifts PR-AUC 1.7× at 3h); delay risk is second, and explain that most >45-min delays are airline-driven (late aircraft 45%, carrier 34%). (4) Add a "warning horizon vs accuracy" chart (3/6/12h) and an Impact panel: share of high-stress blocks flagged ≥3h ahead, plus RMSE vs the historical-average baseline. (5) 2025 per-100-flight rates: stress ATL 10.0 vs DFW 13.7. (6) The time-zone bug we found and fixed is a strong honesty point. Put it in the Data & limits tab and mention it in the pitch.
```

## Integration checklist
- 3h15: everyone merges into `main`. Order: ry → emz → rach → deph.
- Only ry edits `config.py` and the parquet contract; `src/features.py` changes go through one person (emz).
- Run end to end from a fresh clone before submitting; submit early (5% of the score).
- Pitch ≤ 3 minutes; rehearse once with a timer.
