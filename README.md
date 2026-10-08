# OOP with Python: Assignment 2 (Advent 2026)

MSCS & MSDS · Five mini-projects in a Ugandan context, built as small, tested,
object-oriented Python packages with one Jupyter notebook per project.

**Student:** _[your name]_ · **Registration no.:** _[your reg. no.]_

| # | Mini-project | Notebook | Module | Tests |
|---|---|---|---|---|
| 1 | UBOS District Population Forecaster | [project1_population.ipynb](project1_population.ipynb) | [src/population.py](src/population.py) | [tests/test_population.py](tests/test_population.py) |
| 2 | Solar Micro-Grid Dispatch Planner | [project2_microgrid.ipynb](project2_microgrid.ipynb) | [src/microgrid.py](src/microgrid.py) | [tests/test_microgrid.py](tests/test_microgrid.py) |
| 3 | Lake Victoria Fish Stock & Export Risk | [project3_fisheries.ipynb](project3_fisheries.ipynb) | [src/fisheries.py](src/fisheries.py) | [tests/test_fisheries.py](tests/test_fisheries.py) |
| 4 | Rainfall Pattern & Crop Suitability | [project4_rainfall.ipynb](project4_rainfall.ipynb) | [src/rainfall.py](src/rainfall.py) | [tests/test_rainfall.py](tests/test_rainfall.py) |
| 5 | Taxi Route Revenue, Pricing & Fleet | [project5_taxi.ipynb](project5_taxi.ipynb) | [src/taxi.py](src/taxi.py) | [tests/test_taxi.py](tests/test_taxi.py) |

Shared code (used by P1 and P5): [src/common/forecasting.py](src/common/forecasting.py)
(abstract `Forecaster`, `LinearTrendForecaster`, holdout and walk-forward backtests) and
[src/common/metrics.py](src/common/metrics.py) (MAE, RMSE, MAPE).
The decision log is in [docs/approach.md](docs/approach.md).

## Setup

Requires Python 3.10+ (developed and tested on Python 3.13, Windows 11).

```bash
git clone <this-repo-url>
cd <repo>
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
# unit tests (62 tests across the 5 projects + shared code)
pytest -q

# open the notebooks interactively (run from the repository root so `src` is importable)
jupyter notebook

# or execute every notebook top-to-bottom, headlessly (equivalent to Restart & Run All)
jupyter nbconvert --to notebook --execute --inplace project*.ipynb
```

* The notebooks live in the repository root so that `from src... import ...` works with no path tricks.
* All randomness uses `np.random.default_rng(seed)` with fixed seeds. Re-running reproduces every number.
* Figures are also saved to [figures/](figures/).
* P4 uses real NASA POWER data cached in `data/`, so no internet connection is needed. Delete the CSV to re-download it.

## Repository layout

```
├── project1_population.ipynb … project5_taxi.ipynb   one notebook per mini-project
├── src/
│   ├── common/        forecasting base class + metrics (shared by P1 & P5)
│   ├── population.py  microgrid.py  fisheries.py  rainfall.py  taxi.py
├── tests/             pytest suite (test_<project>.py + test_common.py)
├── data/              generated micro-grid CSV, cached NASA POWER rainfall
├── figures/           PNGs saved by the notebooks
├── docs/approach.md   approach & decision log
├── docs/viva_guide.md likely viva questions with short answers, per project
├── requirements.txt   pyproject.toml (pytest config)
```

## Summary of findings

**P1: Population.** All five districts grow at a steady, slightly accelerating
*percentage* rate, so the CAGR (exponential) model beat the linear trend on held-out
2022–24 data in every district (test MAPE < 2.1 %). The Fibonacci-ratio model was off by
about 200 %: it implies about 62 % annual growth. Wakiso grows fastest in relative terms (≈ 6.5 %/yr).
Kampala and Gulu share the same rate (≈ 4.6 %/yr), but Kampala adds about 4× more people.
The 2029 forecasts imply about **4,900 additional primary classrooms** across the five
districts (Wakiso ≈ 2,100, Kampala ≈ 1,500), or about 5,300 at the upper 95 % bootstrap bound.

**P2: Micro-grid.** The 2×2 system is well-posed (det = −5) and well-conditioned
(κ ≈ 5.8). The vectorised solve is about 5× faster than the loop for 30 days. Under the given
coefficients the battery supplies about 90 % of the energy and 96 % of the cost (≈ UGX 0.59 m/month).
Solar is the most volatile source relative to its size (CV ≈ 0.58), because it is a small difference of
two large demands. ±5 % demand noise moves it by up to about 64 %, although the system as a whole
never amplifies error beyond κ. One infeasible day (D1 > 2·D2) was repaired with NNLS.
The diesel extension shows that a dependent third constraint makes the system singular.

**P3: Fisheries.** Logistic growth gives a clear answer that Fibonacci cannot:
**h = 0.20 achieves the maximum sustainable yield** (1,000 t/week, stock at K/2) with the
highest and most stable revenue (Low risk). h = 0.30 "mines" the stock: it out-earns h = 0.10
in year one, but it has the same 750 t/week sustainable yield and leaves the stock at a fragile 25 % of K.
A raw variance threshold is meaningless because variance is in UGX², so I used a CV-based
rule. The 5 % VaR at MSY is a shortfall of about UGX 100 bn/yr. An 8-week closed season costs
10–15 % of 5-year revenue when fishing at or below MSY, but *raises* revenue (+6 %) at h = 0.30.

**P4: Rainfall & crops.** Last year's `math.cos` method was wrong, and even correct cosine
similarity is weak for rainfall: it ignores scale (2× the rain scores 1.0). Pearson
correlation (timing) and Euclidean distance (amount) separate the regions far better.
Season detection (wrap-around + relative prominence) finds Mbarara bimodal and Gulu
unimodal (borderline). Illustrative Kampala comes out unimodal, but **real NASA POWER data
(2014–2023) show Kampala is clearly bimodal**, so the illustrative series is unrepresentative
(r ≈ 0.30 with reality). Advisory: in Mbarara, plant beans in early to mid-March and mid-September. Real data
show suitable bean rain in only about half of years, which argues for using seasonal forecasts.

**P5: Taxi.** The Ntinda fare (UGX 2,000) is 9 % **below** the UGX 2,200 equilibrium,
which creates a shortage of about 10 passengers per trip-hour. In walk-forward tests SES won on all routes,
but the grid search chose α = 1 (a naive forecast), because these short series behave like
random walks. The original 3-day MA lags turning points. Day-11 forecasts (45 / 64 / 50
passengers) need **one vehicle per route** even with a 15 % buffer. On 60 simulated days with a
weekly cycle, seasonal-naive (MAE ≈ 2.9) beats the 3-day MA (MAE ≈ 8.5) by about 3×.

## Data sources

* Values marked *illustrative* in the brief are used as given. P1's Mbarara and Mukono series
  are my own illustrative additions, **not** UBOS figures.
* `data/microgrid_demand.csv` is synthetic (seed 42).
* **NASA POWER** (Prediction Of Worldwide Energy Resources), NASA Langley Research Center,
  monthly point API, parameter `PRECTOTCORR_SUM`, 2014–2023, accessed 6 Oct 2026.
  https://power.larc.nasa.gov/
* Crop water requirements: Brouwer, C. & Heibloem, M. (1986) *Irrigation Water Management:
  Irrigation Water Needs*, FAO Training Manual No. 3; DaMatta, F.M. & Ramalho, J.D.C. (2006)
  "Impacts of drought and temperature stress on coffee physiology and production: a review",
  *Brazilian Journal of Plant Physiology* 18(1): 55–81.

## Academic integrity & AI-use declaration

I used an AI coding assistant (**Claude Code, Anthropic**) for this assignment, as the
course policy allows.

**What the AI assistant did**

* Broke the brief down into a plan and a checklist mapped to the rubric.
* Wrote the first version of the class designs, the `src/` modules, the pytest tests,
  the notebooks (including the first drafts of the Findings & Limitations sections),
  the README and `docs/approach.md`.
* Added explanatory comments to the code and wrote [docs/viva_guide.md](docs/viva_guide.md),
  which I used to study the code.
* Checked its own outputs and fixed issues, e.g. a one-sided bootstrap interval (fixed by
  centring residuals), a wrong unit-scaling example in P3, and figure-layout problems.

**What I did myself**

_[Fill this in honestly once you have done it. Delete any line that isn't true and add what you actually did. For example:]_

* _Worked through every notebook and the viva guide until I could explain each line._
* _Checked the key results by hand: …_
* _Rewrote the Findings & Limitations sections in my own words for projects …_
* _Changed these modelling choices and re-ran the analysis: …_
* _Verified the crop-water citations in P4 against the original sources._

Assumptions and judgement calls (Fibonacci ratio start, CV bands, crop thresholds,
closed-season timing, the diesel constraint) are listed in [docs/approach.md](docs/approach.md).
