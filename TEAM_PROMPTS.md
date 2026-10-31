# Claude prompts for each teammate

Start a new Claude Console conversation with **the shared context** followed by **your role prompt**.

To make credits last: use Sonnet by default and Opus only when stuck; never paste the CSV (paste `df.head().to_string()`, column lists or tracebacks); start a fresh conversation (shared context + role prompt) when a chat gets long.

> **Already on `main`** (reuse, don't regenerate): `config.py` (airports, runway headings, time zones, paths, split dates, coco → weather group), `src/wind.py` (wind components + self-test), `src/fetch_wind.py` (IEM wind download). Paste the relevant file into your conversation when Claude needs it.

---

## Shared context (everyone)

```
PROJECT CONTEXT — Autonomous SG 2026 hackathon, Airbus track, Challenge #5
Build a Predictive Airport Capacity & Delay-Risk tool for ATL and DFW that forecasts capacity stress and >45-min delay risk 3–12 hours ahead from weather + flight data. ~5 hours left; 4 people, each on their own git branch of https://github.com/ryrxiaopang/smuggershackathon (branches: ry, emz, rach, deph; merge to main at the 3h15 mark). Python 3.11, pandas, lightgbm, scikit-learn, streamlit, pyarrow. I am using you via the Claude Console: give me complete, runnable files (not fragments), keep answers short, and when I paste an error, fix only what's broken. If you can run code yourself, run it and check outputs.

JUDGING (weights): technical soundness 25% (right tool per job — stats/ML/rules for numbers, never an LLM doing arithmetic; works end to end on inputs beyond a demo script), problem realism 20% (named user: an airline ops duty manager / ATC flow coordinator), honest evaluation 20% (baselines, metrics, at least one visible failure case), trust & traceability 20% (every risk score shows the timestamped weather signals and rules behind it; human decides), impact 10%, time 5%.

BRIEF RULES WE MUST FOLLOW: use only the provided CSV for flights/weather, filtered to flights departing from OR arriving at ATL or DFW; assume ATL runway heading 80°, DFW runway heading 180°; no data leakage across time; compute crosswind/tailwind relative to runway heading (not raw wind speed); pipeline must accept any airport code, not hard-code one airport; normalise delays by flight volume (rates per flight, never raw delay minutes across airports); scripts process data automatically.

DATA FACTS (already verified — do not re-derive):
- data/raw/Flight_Data_With_Weather_Final.csv: 148,273 rows, 123 cols, US BTS on-time data 2023-01-01 to 2025-07-31 with weather at origin and destination. It is a SAMPLE (~6 ATL arrivals/day), so hourly arrival counts / true AAR cannot be computed. ATL/DFW subset = 25,299 rows (ATL: 8,415 deps, 5,429 arrs; DFW: 7,736 deps, 4,032 arrs).
- Key flight cols: FL_DATE, ORIGIN, DEST, OP_UNIQUE_CARRIER, CRS_DEP_TIME, CRS_ARR_TIME (local hhmm ints), DEP_DELAY, ARR_DELAY, CANCELLED, DIVERTED, DISTANCE, CARRIER_DELAY, WEATHER_DELAY, NAS_DELAY, SECURITY_DELAY, LATE_AIRCRAFT_DELAY, TAXI_OUT, TAXI_IN.
- Weather cols (suffix _ORIGIN / _DEST): temp (°C), prcp (mm), wspd (km/h), wpgt (gust, km/h), pres (hPa), snow (mm), coco (Meteostat weather condition code 1–27; mapping to groups clear/cloud/fog/rain/freezing/snow/storm is COCO_GROUP in config.py). NO wind direction column.
- TIME-ZONE BUG IN PROVIDED DATA (verified against raw METARs, temp error 0.03°C): the weather attached to a flight scheduled at LOCAL hour H on date D is actually the observation at UTC hour H on date D. So the true observation time is obs_utc = D + H hours (UTC), which is 4–6 hours BEFORE the flight. We relabel weather to its true obs_utc instead of pretending it is at flight time.
- Wind direction is filled from public METARs (Iowa Environmental Mesonet ASOS archive, same observations as the CSV): file data/external/iem_wind_ATL_DFW.csv with cols station, obs_utc (hourly, rounded), drct (deg, blank if calm/variable), sknt (kt), gust (kt). If drct is missing, crosswind falls back to an UPPER BOUND = full wind speed and the row is flagged wind_dir_known=False.
- Baseline signal check (concurrent weather, ATL/DFW subset, test = 2025): >45-min delay base rate 13.2%; calendar-only AUC 0.645 / PR-AUC 0.218; +weather AUC 0.675 / PR-AUC 0.245.

SHARED FILE CONTRACT (do not rename columns without telling the team):
data/processed/flight_sides.parquet — one row per flight per ATL/DFW end:
  flight_id, airport, side ('dep'|'arr'), other_airport, carrier, sched_local, sched_utc, date_local, hour_local, block_local (6-hour block start: 0,6,12,18), distance, cancelled, diverted, delay_min (DEP_DELAY for dep, ARR_DELAY for arr), delayed_45 (delay_min>45; NaN if cancelled/diverted), nas_delay, weather_delay, stress_flag (cancelled OR nas_delay+weather_delay>=15),
  wx_obs_utc, temp_c, prcp_mm, wspd_kt, wpgt_kt, pres_hpa, snow_mm, coco, wx_group ('clear','cloud','fog','rain','freezing','snow','storm'),
  wdir_deg, wind_dir_known, headwind_kt, tailwind_kt, crosswind_kt, gust_crosswind_kt
data/processed/weather_timeline.parquet — one row per airport per observed UTC hour (deduplicated from all rows touching that airport, both sides):
  airport, obs_utc, temp_c, prcp_mm, wspd_kt, wpgt_kt, pres_hpa, snow_mm, coco, wx_group, wdir_deg, wind_dir_known, headwind_kt, tailwind_kt, crosswind_kt, gust_crosswind_kt
outputs/capacity_preds.parquet — airport, block_start_local, horizon_h (3|6|12), issued_utc, y_true, y_pred, baseline_persist, baseline_hist
outputs/flight_risk_preds.parquet — flight_id, airport, side, sched_local, horizon_h, p_delay45, y_true, top_factors (str)
outputs/*_metrics.json — metric name → value, plus baseline values

WIND MATH: runway headings {'ATL': 80, 'DFW': 180} in config.py (dict, extendable). theta = radians(wdir - heading); headwind = spd*cos(theta) (negative = tailwind); tailwind = max(0, -headwind); crosswind = |spd*sin(theta)|; same with gust for gust_crosswind. Speeds in knots (km/h ÷ 1.852). Implemented in src/wind.py.

LEAKAGE RULE: a prediction issued at time T (UTC) for horizon h may only use weather_timeline rows with obs_utc <= T and flights already completed before T. Target time = T + h.
TIME SPLIT: train 2023-01-01→2024-06-30, validation 2024-07-01→2024-12-31, test 2025-01-01→2025-07-31. Never shuffle across time.

REPO LAYOUT: config.py, src/fetch_wind.py, src/wind.py, src/pipeline.py (ry); src/capacity_model.py (emz); src/delay_model.py (rach); app/streamlit_app.py, src/recommend.py (deph); data/raw and data/processed are gitignored; data/external is committed; requirements.txt; README.md.
```

---

## ry: Data pipeline + wind

```
MY ROLE (branch: ry): data pipeline + public wind data. Everyone else is blocked on my outputs, so ship a rough v1 of flight_sides.parquet and weather_timeline.parquet within 45 minutes, then harden.

ALREADY DONE on main: config.py, src/wind.py (wind_components + self-test), src/fetch_wind.py (IEM download → data/external/iem_wind_ATL_DFW.csv). I will paste them if you need them. First I run `python src/fetch_wind.py` and commit the CSV.

Build src/pipeline.py — CLI: python src/pipeline.py --airports ATL DFW. Steps: read CSV (only needed columns, low_memory=False); keep rows where ORIGIN or DEST in airports; explode to one row per ATL/DFW end (dep side uses _ORIGIN weather + CRS_DEP_TIME, arr side uses _DEST weather + CRS_ARR_TIME; a flight between ATL and DFW yields 2 rows); build sched_local from FL_DATE + hhmm (2400 → next day 00:00; arrival earlier than departure → next day); sched_utc via config.AIRPORT_TZ; wx_obs_utc = FL_DATE + floor(hhmm/100) hours treated AS UTC (the documented time-zone bug); convert km/h→kt; map coco→wx_group via config.COCO_GROUP; join wind drct on (airport, wx_obs_utc) from the IEM file (if the file is missing, continue with drct=NaN); compute wind components with src/wind.py for wind speed and gust; derive delayed_45, stress_flag, block_local. Then build weather_timeline by deduplicating (airport, wx_obs_utc) across both sides. Write both parquets and print a short data report (row counts, % wind_dir_known, date range, nulls).

Then verify the time-zone finding on the ARRIVAL side: for a few airport-hours, check that departures and arrivals at the same airport with the same local hour H on the same date have identical temp/wspd (they should if the rule holds). Report the result.

Deliverables: src/pipeline.py, the committed wind CSV, and a 3-line message I can paste to the team when v1 parquets are ready (they need the parquet files sent to them since data/processed is gitignored — or tell them to run the pipeline themselves).
```

---

## emz: Capacity-stress forecast

```
MY ROLE (branch: emz): capacity-stress forecast — our honest proxy for arrival acceptance rate (AAR), since the sample data cannot give true hourly arrival counts.

Until ry's parquet is ready (~45 min), build against a FAKE flight_sides/weather_timeline generated with the exact contract columns (write src/make_fake_data.py), then switch to the real files.

Build src/capacity_model.py:
1. Target: for each airport × 6-hour local block (block_local 0/6/12/18 on date_local), stress_rate = mean(stress_flag) over non-diverted flights at that airport-block (both dep and arr sides); also n_flights. Drop blocks with n_flights < 3 from training/scoring (report how many). Make block size a parameter so we can try 3h; report which is less noisy.
2. Issue times: for each block with start time S (convert to UTC), predictions are issued at T = S − h for h in {3, 6, 12}. Features may only use weather_timeline rows with obs_utc <= T: last observed values, age of last obs (hours), rolling max/mean over the 6h and 12h before T of crosswind_kt, gust_crosswind_kt, wspd_kt, prcp_mm, snow_mm, pressure change (3h/6h), counts of storm/rain/fog/snow wx_group, wind_dir_known share; plus calendar (month, day of week, block), airport one-hot, and the previous block's realised stress_rate only if that block ended before T.
3. Model: LightGBM regressor per horizon (or one model with horizon as a feature — your choice, explain it). Train/val/test per the shared split; tune lightly on val only.
4. Baselines: persistence (last completed block's stress_rate before T) and historical average (same airport × block × month in train). Report RMSE and MAE on test for model vs both baselines, per airport and per horizon, in outputs/capacity_metrics.json.
5. Generalisation check: train on ATL only, test on DFW (and reverse). Report RMSE.
6. Write outputs/capacity_preds.parquet per the contract, and save the top 10 feature importances (gain) to outputs/capacity_importance.csv.
7. Find and save 2 failure cases (largest errors on test) with the weather timeline around them to outputs/capacity_failures.csv, plus one sentence each on why the model missed.

Be honest in the summary: call the target "capacity-stress proxy (share of flights with ATC/airport-capacity or weather delay ≥15 min or cancelled)", not AAR. Keep code to one file + the fake-data script.
```

---

## rach: >45-min delay-risk classifier

```
MY ROLE (branch: rach): flight-level classifier — probability that a flight at ATL/DFW is delayed > 45 minutes, issued 3, 6 and 12 hours before its scheduled time.

Until ry's parquet is ready (~45 min), prototype directly on the raw CSV subset (ORIGIN or DEST in ATL/DFW), then switch to data/processed/flight_sides.parquet + weather_timeline.parquet.

Build src/delay_model.py:
1. Label: delayed_45 (exclude cancelled/diverted from training; report how many excluded and the base rate per airport/side).
2. For each flight and horizon h in {3,6,12}: issue time T = sched_utc − h. Weather features only from weather_timeline rows at that airport with obs_utc <= T (last values, obs age, rolling 6h/12h max of crosswind_kt, gust_crosswind_kt, wspd_kt, prcp_mm, snow_mm, storm/fog/snow counts, 3h pressure change). Calendar/schedule features: hour_local, day of week, month, side, airport, carrier (category), distance. Optional congestion feature: share of stress_flag among flights at the same airport that COMPLETED before T on the same day. Never use the flight's own delay columns or any weather after T.
3. Model: LightGBM classifier with class weighting; calibrate probabilities on the validation period (isotonic or Platt). Shared time split.
4. Metrics on 2025 test, per horizon: PR-AUC, ROC-AUC, Brier score, and precision/recall at a threshold chosen on validation for ~recall 0.6. Baselines: (a) base rate, (b) calendar-only model (no weather). Show the weather lift explicitly. Normalise: report rates per 100 flights, per airport, never raw minutes.
5. Generalisation: train ATL → test DFW and reverse.
6. Explainability for trust: for each test prediction, top_factors = top 3 SHAP contributions as a short string like "gust_crosswind_6h_max=21kt (+0.08); storm_count_12h=2 (+0.05); hour_local=18 (+0.03)". Write outputs/flight_risk_preds.parquet per contract and outputs/delay_metrics.json.
7. Save 2 failure cases: a high-confidence miss and a false alarm, with their weather timeline, to outputs/delay_failures.csv.

Report back a short honest summary: lift over baselines, where it fails, and the caveat that weather comes from observations (not forecasts) up to T.
```

---

## deph: Dashboard, recommendations, pitch

```
MY ROLE (branch: deph): Streamlit decision-support dashboard, rule-based recommendations, evaluation page and the 3-minute pitch.

Until teammates' outputs exist, generate fake files matching the contract (outputs/capacity_preds.parquet, outputs/flight_risk_preds.parquet, outputs/*_metrics.json, data/processed/weather_timeline.parquet) with src/make_fake_outputs.py, so the app works now and swaps to real files later with no code change.

Build:
1. src/recommend.py — pure rules (no LLM, no ML), every rule returns (level, action, reason, evidence) where evidence lists the exact values and timestamps used. Levels: NORMAL / WATCH / ACTION. Example rules (thresholds in config, easy to tune): predicted stress_rate above the airport's historical 80th percentile → WATCH; above 90th → ACTION "consider reducing arrival rate / ground delay program for the block"; gust_crosswind_kt upper bound ≥ 25 kt with wind_dir_known=False → WATCH "verify actual wind direction — crosswind is an upper bound"; ≥ X% of next-block flights with p_delay45 ≥ threshold → "prioritise gate/crew buffers". Unit-test 3 rules.
2. app/streamlit_app.py — pages/tabs:
   a) Forecast: airport code input (any code present in data; default ATL), horizon 3/6/12h, date-time picker within the test period. Show predicted capacity-stress for upcoming blocks vs historical normal band, the recommendation card with its evidence, and the weather timeline (crosswind, gusts, precip, wx_group) with obs timestamps. Mark crosswind values as "upper bound" where wind_dir_known is False.
   b) At-risk flights: table of flights with p_delay45, top_factors, side, sched_local; filter by threshold.
   c) Evaluation: metrics vs baselines from the JSON files, a calibration/reliability chart, ATL→DFW generalisation results, and the failure cases with a one-line explanation each. ATL vs DFW comparisons only as rates per 100 flights.
   d) Data & limits: sample data (no true AAR → proxy), weather time-zone fix, wind-direction source + upper-bound fallback, observations-not-forecasts caveat.
   Use plotly; keep it clean; the user persona is an airline ops duty manager at ATL.
3. PITCH.md — a 3-minute script: user + status quo (reactive capacity cuts, holding, taxi delays) → what the tool shows → live demo path (pick a real 2025 storm day once real outputs exist) → honest results vs baselines → one failure case → limits and next steps. Also a 5-line demo checklist.

Keep everything runnable with: pip install -r requirements.txt && streamlit run app/streamlit_app.py
```
