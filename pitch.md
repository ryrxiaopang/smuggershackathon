# 3-minute pitch: Predictive Airport Capacity & Delay-Risk

*Plain text = say it. Italics = what to click. About 430 spoken words; rehearse once with a timer.*

**0:00 · The user and the problem (25s)**
Meet Dana, airline ops duty manager at Atlanta. Today she learns about a capacity squeeze when ATC cuts the arrival rate.
By then aircraft are holding, taxi queues are building and crews are timing out. Every decision is reactive.
She needs hours of warning, and a reason she can trust.

**0:25 · What the tool does (25s)**
Pick an airport, a moment and how far ahead to look: 3, 6 or 12 hours. The tool forecasts how many flights will be disrupted
in each 6-hour block, and the chance each flight is more than 45 minutes late.
Machine learning does the forecasting. Simple, fixed rules turn it into advice: normal, watch or act.
Every piece of advice shows the exact weather reports and timestamps behind it. No AI writes any number, and Dana decides.

**0:50 · Live demo: Atlanta winter storm, January 2025 (70s)**
1. *ATL · Tue 21 Jan 2025 · 19:00 · 12 hours.* It's Tuesday evening. The banner is red: **ACTION. Consider reducing the arrival rate
   for Wednesday 06:00 to 12:00.** The forecast is 49 of every 100 flights disrupted. Normal for Atlanta is up to 10.
   That is eleven hours of warning.
2. *Open the congestion check, then scroll to the weather chart.* Here is why: the forecast was issued at 23:00 UTC, which is 18:00
   in Atlanta, and the weather reports before it show sub-zero temperatures and precipitation, every reading timestamped.
3. *Toggle "Show what actually happened".* 67 of 100 flights really were disrupted.
4. *DFW · Tue 6 May 2025 · 14:00 · 6 hours.* A thunderstorm day in Dallas: red again, for the evening block.
   *Flights at risk → "12 hours" → drag the line to 0.30.* The average flight has a 13% risk; the model gives this
   afternoon's flights 31 to 42%. Seven are flagged, and six of them really were late, by 51 minutes to over three hours.
   Each one says why, in plain words.
5. *Back on Forecast: Log decision → Accept.* The tool recommends; the human decides, and it's logged.

**2:00 · Honest results (30s)**
*Tab ③.* We trained on 2023 to mid-2024 and tested on 2025, which the models never saw.
- The flight-delay model: its riskiest 10% of flights really ran late 29% of the time, more than twice the average.
  Adding weather makes it 17% better than a timetable-only model.
- The congestion forecast beats the naive "same as last block" guess clearly, but beats the historical average by only 3%.
  It earns its keep on unusual days like this storm.
- When it says ACTION, those blocks really were disrupted four times more than normal ones.
- A model trained only on Atlanta works almost as well on Dallas, so it isn't tied to one airport.

**2:30 · A failure, the limits, what's next (30s)**
Now the failure. *ATL · Tue 21 Jan · 08:00 · 6 hours:* green, no action. That afternoon, 57 of 100 flights were disrupted.
We missed it because the snow hadn't started yet: our inputs are weather *observations*, so weather that hasn't happened is invisible.
Limits:
- The data is a sample, so we forecast a disruption proxy, not the true arrival rate.
- The provided weather was time-stamped 4–6 hours wrong, and the hourly file one hour early. We found both and fixed them,
  so no forecast ever sees weather from after it was made.
- Wind direction comes from public weather reports, so crosswind is computed against the real runway.

Next: add weather forecasts (TAFs) to catch storms before they start, use the full schedule for the true arrival rate,
and pilot it with one hub ops desk. The same pipeline already runs on Chicago, Denver and Seattle with one command.

---

## Demo checklist
1. `git pull origin main`, then `streamlit run app/streamlit_app.py`. All four tabs should load; there should be no "synthetic" warning.
2. Before going on stage, set **ATL · Tue 21 Jan 2025 · 19:00 · 12h**, with "Show what actually happened" **off**.
3. Rehearse the clicks: congestion check → weather chart → hindsight toggle → DFW 6 May 14:00 6h → Flights at risk, window
   "12 hours", slider to 0.30 (hindsight on shows the "actually late?" column) → ATL 21 Jan 08:00 6h.
4. Backup: screenshots of each step in case the laptop or wifi fails. Browser zoom ~90% so the banner and chart fit on one screen.
5. One person drives the app, one speaks. Stop at 3:00 even if you're mid-sentence.

## Numbers for Q&A (all from `outputs/`, 2025 test period)
| Question | Answer |
|---|---|
| Delay model vs baselines (PR-AUC, 3h / 6h / 12h) | 0.244 / 0.255 / 0.239 vs timetable-only 0.209; base rate 13.2%; 95% CI about ±0.04 |
| ROC-AUC (3h) | 0.672 vs 0.642 timetable-only |
| Congestion forecast error (RMSE, per 100 flights) | 19.5 model vs 20.0 historical average vs 26.8 "same as last block" (3h/6h/12h: 19.5 / 19.3 / 19.8) |
| How often ACTION fires | 2.2% of 6h blocks; those averaged 42/100 disrupted vs 10/100 for normal blocks |
| Train on one airport, test on the other | Delay PR-AUC (3h): ATL-trained on DFW 0.228 vs DFW-trained 0.243. Congestion RMSE: ATL→DFW 22.4 vs 21.1 |
| Why can't weather explain all delays? | Of >45-min delay minutes: late aircraft 45%, airline 34%, ATC/airport 12%, weather 8% |
| Data leakage? | Each forecast uses only weather observed before it was issued; automated leakage tests pass |
| Runway direction? | Crosswind/tailwind relative to ATL 80° / DFW 180°; when wind direction is unknown, crosswind is shown as an upper bound |
| Airports of different size? | Always compared as rates per 100 flights, never raw delay minutes |
| Time-zone fix proof | Flight-CSV wind matches public METARs to 0.06 kt at the corrected hour vs ~2 kt one hour off, at ATL, DFW, ORD, DEN, SEA |
| Weather data | Organiser hourly file: 24 readings/day for ATL/DFW, relabelled +1h (a row for hour T matches the METAR at ~T:52); wind direction for 94.7% of flights |
| Any airport? | `python src/pipeline.py --airports KORD KDEN KSEA` looks up time zone, runway and wind automatically |
