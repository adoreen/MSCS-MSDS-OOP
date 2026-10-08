# Approach & decision log

This file records **how** each mini-project was approached: the order of work, the
assumptions made where the brief was open, and the alternatives considered. The
notebooks contain the full reasoning next to the code. This is the summary.

## Overall workflow

1. **Read the brief and rubric** and turned every bullet into a checklist item
   (one notebook per project, `src/` modules, ≥ 3 pytest tests per project, seeds,
   Findings & Limitations 150–300 words, extension, README with AI declaration).
2. **Design first.** For each project I identified the domain objects (district,
   micro-grid, fish stock, region, route) and made each one a class with a single
   responsibility. Projects 1 and 5 both needed forecasters, so I put the abstract
   `Forecaster` base class, the linear-trend model and the error metrics in
   `src/common/` instead of writing them twice.
3. **Hand calculations before code.** I worked key results out on paper first
   (det/cond of the micro-grid matrix, the feasible demand band, P* = 2,200 for the
   taxi market, MSY = 1,000 t/week). The tests and notebooks check the code against
   these values.
4. **Tests next to code.** Each module has a test file covering normal cases and
   edge cases (empty input, zero, negative, singular matrix, wrong length).
5. **Notebooks last.** They import from `src/`, explain each task in markdown,
   show a second-method check, and end with Findings & Limitations.
6. **Restart & Run All.** Every notebook is executed headlessly with
   `jupyter nbconvert --execute` before submission.

Order of work: common module → P2 (most self-contained) → P1 → P5 (reuses P1's
forecaster) → P3 → P4 (needed literature and real data).

## Design patterns used (and why)

| Pattern | Where | Why it helps |
|---|---|---|
| Abstract base class + template method | `Forecaster.fit/predict` → `_fit/_predict` | validation written once; any model plugs into the same backtest code |
| Inheritance | `HybridMicroGrid(MicroGrid)` | the 3×3 case reuses all solving/feasibility/NNLS logic |
| Composition | `FishStock` + `PriceModel` → `RiskAssessor` | each class has one job; revenue = harvest × price |
| Alternative constructor | `CropRule.from_seasonal_need` | thresholds come from sources quoted per season, not per month |
| Dataclasses | `FishStock`, `PriceModel`, `CropRule` | concise, validated parameter objects (`__post_init__`) |
| Dependency injection | `prompt_demand(input_fn=...)` | interactive input can be tested and run under Restart & Run All |
| Dunder methods | `__repr__`, `__len__` on domain classes | readable notebook output, natural `len(district)` |

## Per-project decisions

### P1: Population
* **Fibonacci-ratio model was ambiguous.** I chose ratios starting at F(3)/F(2) = 2
  (configurable via `start`). Whatever the start, the ratios tend to φ ≈ 1.618.
* **Model selection by RMSE** (penalises large misses, which are costly for planning).
* **Bootstrap:** residual bootstrap with *centred* residuals. The CAGR model
  passes exactly through the first and last points, so its raw residuals are one-sided,
  which first produced a lopsided interval. Centring fixed it.
* **Classrooms rounded up**, and a range is reported from the prediction interval.
* Added districts **Mbarara** and **Mukono**. Their figures are illustrative (my own).

### P2: Micro-grid
* Derived the feasibility band 0.75·D2 ≤ D1 ≤ 2·D2 by hand, then designed the CSV
  generator (weekday/weekend pattern) so that noise occasionally breaks it.
* Infeasible days: **NNLS + residual report** rather than clipping (clipping
  violates both constraints; the notebook shows that NNLS leaves a smaller residual).
* Third constraint for the hybrid grid: total energy x + y + z = D3. Showed the feasible D3 window.
* Sensitivity: compared norm-wise amplification with κ(A), and explained why the
  small solar component still swings by up to about 64 % (cancellation).

### P3: Fisheries
* Kept r = 0.4 *per week* from the brief but flagged it as biologically fast.
* Price floor/ceiling implemented by clipping at each step.
* **Risk rule on CV** with bands 0.10 / 0.25, classified on the *average weekly CV
  over 1,000 price paths* so one lucky path does not decide the class.
* VaR reported both as the 5 % quantile and as the shortfall below the mean.
* Closed season placed in weeks 8–15 (≈ March–April) as an assumption.

### P4: Rainfall
* Crop bands derived from FAO seasonal water needs ÷ season length (sources cited
  in the notebook). This is a simplification, discussed in the limitations.
* Peak detection: tiled the year to handle Dec→Jan wrap-around; prominence
  threshold = 25 % of each region's annual range, with a sensitivity table.
* The illustrative Kampala data came out unimodal, contradicting the known climate.
  Instead of tuning the threshold until it "worked", I tested the method on
  **real NASA POWER data (2014–2023)**, where it correctly finds two seasons.

### P5: Taxi
* Counts treated as daily passengers in both directions combined.
* Walk-forward from day 4 (the MA(3) needs three days of history). α grid 0.05–1.00.
* Flagged that α tuning and scoring use the same window (optimistic), and that the
  equilibrium model's units (per trip-hour) differ from the daily counts.
* Fleet size rounded up, with a "busy day" check.

## Data provenance

| File | Source |
|---|---|
| `data/microgrid_demand.csv` | synthetic, generated by `generate_demand_csv(seed=42)` |
| `data/nasa_power_monthly_rainfall.csv` | NASA POWER monthly API, `PRECTOTCORR_SUM`, 2014–2023, accessed 2026-10-06 |
| everything else | values given in the brief (illustrative) or simulated with fixed seeds |
