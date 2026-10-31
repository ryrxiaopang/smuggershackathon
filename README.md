# Predictive Airport Capacity & Delay-Risk (Autonomous SG 2026 · Airbus Challenge #5)

Forecasts **capacity stress** and **>45-min delay risk** for ATL and DFW 3, 6 and 12 hours ahead from weather and flight data. Gives an airline ops duty manager or ATC flow coordinator early warning instead of reacting when the storm arrives.

## Team and branches

| Branch | Owner | Feature | Main files |
|---|---|---|---|
| `ry` | Ry | Data pipeline + public wind data | `src/pipeline.py`, `src/fetch_wind.py`, `src/wind.py` |
| `emz` | Emz | Capacity-stress forecast (AAR proxy) | `src/capacity_model.py` |
| `rach` | Rach | >45-min delay-risk classifier | `src/delay_model.py` |
| `deph` | Deph | Dashboard, rules, evaluation page, pitch | `app/streamlit_app.py`, `src/recommend.py`, `PITCH.md` |

Work on your own branch, commit often, and **merge into `main` at the 3h15 mark**. Only `ry` changes `config.py` or the shared file contract; tell the team if you need a change. The Claude prompts for each person are in [`TEAM_PROMPTS.md`](TEAM_PROMPTS.md).

## Setup

```bash
git clone https://github.com/ryrxiaopang/smuggershackathon.git
cd smuggershackathon
git checkout <your-branch>          # ry | emz | rach | deph
pip install -r requirements.txt
# put the provided CSV at data/raw/Flight_Data_With_Weather_Final.csv (gitignored, 64 MB)
python src/wind.py                  # self-test
python src/fetch_wind.py            # downloads ATL/DFW wind direction -> data/external/ (commit this file)
```

Run order once everything exists:
```bash
python src/pipeline.py --airports ATL DFW   # -> data/processed/*.parquet
python src/capacity_model.py                # -> outputs/capacity_*
python src/delay_model.py                   # -> outputs/flight_risk_preds.parquet, delay_metrics.json
streamlit run app/streamlit_app.py
```

## Data and known issues

- **Flights + weather:** the provided `Flight_Data_With_Weather_Final.csv` (148,273 BTS flights, Jan 2023 – Jul 2025, weather at both ends). We keep flights departing from or arriving at ATL/DFW (25,299 rows).
- **Sample data → no true AAR.** The file has ~6 ATL arrivals/day (real ATL handles ~1,000 departures/day), so hourly arrival counts are impossible. We forecast a **capacity-stress proxy**: the share of flights with ATC/airport-capacity (NAS) or weather delay ≥15 min, or cancelled, per airport per 6-hour block. It is normalised by flight volume by construction.
- **Time-zone shift in the provided weather.** Checked against raw METARs: the weather on a flight scheduled at local hour H on date D is really the observation at **UTC** hour H on date D (temperature error 0.03 °C at that alignment vs ~3 °C at the true flight time). So the provided weather is 4–6 h *before* the flight. The pipeline relabels each reading to its true observation time (`wx_obs_utc`) and all features respect it.
- **No wind direction in the CSV.** Filled from the [Iowa Environmental Mesonet ASOS archive](https://mesonet.agron.iastate.edu/request/download.phtml) (same METARs as the CSV) via `src/fetch_wind.py`. Wind components use the organiser runway headings (ATL 80°, DFW 180°). If direction is missing (calm/variable), crosswind is shown as an **upper bound** (= full wind speed) and flagged `wind_dir_known = False`.
- **Units:** CSV wind speed/gust are km/h; we convert to knots (÷ 1.852).
- **Observations, not forecasts:** features use weather observed up to the prediction time, not TAF forecasts.

## Rules every module follows
- No leakage: a prediction issued at UTC time T uses only weather with `obs_utc <= T` and flights completed before T.
- Time split: train 2023-01-01 → 2024-06-30 · validation 2024-07-01 → 2024-12-31 · test 2025-01-01 → 2025-07-31.
- Airport-agnostic: airports, runway headings and time zones live in `config.py`.
- Compare airports with rates per flight, never raw delay minutes.
- Numbers come from statistics, ML and rules; no LLM does arithmetic.

## Licence
MIT
