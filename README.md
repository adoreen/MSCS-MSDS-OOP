# OOP with Python: Assignment 2 (Advent 2026)

**Programme:** MSCS & MSDS · **Course:** Object-Oriented Programming with Python
**Student:** _[Doreen Ainembabazi]_ · **Registration number:** _[S26M19/010]_

This repository contains my solutions to the five mini-projects in Assignment 2. Each
project models a Ugandan planning problem with its own set of classes. The reusable logic
lives in `src/`, each project has a Jupyter notebook that walks through the tasks and
interprets the results, and a pytest suite checks the code.

| # | Mini-project | Notebook | Module | Tests |
|---|---|---|---|---|
| 1 | UBOS District Population Forecaster | [project1_population.ipynb](project1_population.ipynb) | [src/population.py](src/population.py) | [tests/test_population.py](tests/test_population.py) |
| 2 | Solar Micro-Grid Dispatch Planner | [project2_microgrid.ipynb](project2_microgrid.ipynb) | [src/microgrid.py](src/microgrid.py) | [tests/test_microgrid.py](tests/test_microgrid.py) |
| 3 | Lake Victoria Fish Stock & Export Risk Model | [project3_fisheries.ipynb](project3_fisheries.ipynb) | [src/fisheries.py](src/fisheries.py) | [tests/test_fisheries.py](tests/test_fisheries.py) |
| 4 | Rainfall Pattern & Crop Suitability Analyser | [project4_rainfall.ipynb](project4_rainfall.ipynb) | [src/rainfall.py](src/rainfall.py) | [tests/test_rainfall.py](tests/test_rainfall.py) |
| 5 | Taxi Route Revenue, Pricing & Fleet Planner | [project5_taxi.ipynb](project5_taxi.ipynb) | [src/taxi.py](src/taxi.py) | [tests/test_taxi.py](tests/test_taxi.py) |

Projects 1 and 5 share a forecasting framework in [src/common/](src/common/): an abstract
`Forecaster` base class, a linear-trend model, holdout and walk-forward backtesting, and the
MAE/RMSE/MAPE metrics. This avoids writing the same logic twice.

---

## Requirements

* Python **3.10 or newer** (developed and tested on Python 3.13, Windows 11)
* The packages in [requirements.txt](requirements.txt): numpy, scipy, matplotlib, pandas, pytest and Jupyter

No internet connection is needed to run anything. The real rainfall data used in Project 4
is already saved in `data/`.

## Setup

**1. Get the code**

```bash
git clone https://github.com/bagzyea/MSCS-MSDS-OOP.git
cd MSCS-MSDS-OOP
```

**2. Create and activate a virtual environment**

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

> If PowerShell says *"running scripts is disabled on this system"*, run
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.

**3. Install the dependencies**

```bash
pip install -r requirements.txt
```

## Running the project

All commands are run from the repository root with the virtual environment active.

**Run the tests**

```bash
pytest -q
```

Expected output: `62 passed`. Every project has at least three tests, covering normal
cases plus edge cases such as empty input, zero or negative values, a singular matrix and
too little history.

**Open the notebooks**

```bash
jupyter notebook
```

Open any `project*.ipynb` file and choose **Kernel → Restart & Run All**. Each notebook
runs top to bottom in a few seconds. The notebooks are saved with their outputs, so the
results and charts can also be read without running anything.

In **VS Code**: install the *Python* and *Jupyter* extensions, open a notebook, click
**Select Kernel → Python Environments → .venv**, then **Run All**.

**Run every notebook from the command line** (equivalent to Restart & Run All):

```bash
jupyter nbconvert --to notebook --execute --inplace project*.ipynb
```

**Reproducibility.** All randomness uses `np.random.default_rng(seed)` with fixed seeds,
so every run gives identical numbers. Each notebook also saves its charts to
[figures/](figures/).

**Troubleshooting.** `ModuleNotFoundError: No module named 'src'` means Jupyter was started
outside the repository root. Close it, `cd` into the repository folder and start it again.

## Repository structure

```
├── project1_population.ipynb … project5_taxi.ipynb   one notebook per mini-project
├── src/
│   ├── common/          forecasting base class, backtesting and error metrics (P1 & P5)
│   ├── population.py    P1: DistrictPopulation, CAGR / Fibonacci forecasters, ClassroomPlanner
│   ├── microgrid.py     P2: MicroGrid, HybridMicroGrid, CostModel, input validation
│   ├── fisheries.py     P3: FishStock, PriceModel, RiskAssessor
│   ├── rainfall.py      P4: Region, CropRule, SimilarityAnalyser, NASA POWER loader
│   └── taxi.py          P5: Route, MarketEquilibrium, forecasters, FleetPlanner
├── tests/               pytest suite, one file per module
├── data/                generated micro-grid demand CSV, NASA POWER rainfall (2014–2023)
├── figures/             charts exported by the notebooks
├── docs/
│   ├── approach.md      approach, design decisions and assumptions per project
│   └── viva_guide.md    study notes on the methods and design choices
├── requirements.txt
└── pyproject.toml       pytest configuration
```

## Design overview

* **Abstract base class + template method:** `Forecaster.fit()`/`predict()` validate the input
  once, then call each subclass's `_fit()`/`_predict()`. Every model can be backtested by the same code.
* **Inheritance:** `HybridMicroGrid(MicroGrid)` reuses all of the 2×2 solving, feasibility and
  NNLS logic for the 3×3 diesel case.
* **Composition:** in P3, `RiskAssessor` combines the outputs of `FishStock` and `PriceModel`.
* **Validation & dunder methods:** every domain class rejects invalid input with a clear
  `ValueError` and implements `__repr__` (and `__len__` where it makes sense).
* All classes and public methods have docstrings and type hints.

More detail on each decision and assumption is in [docs/approach.md](docs/approach.md).

## Summary of findings

**P1: Population.** All five districts grow at a steady, slightly accelerating percentage
rate. Because of this, the CAGR (exponential) model beat the linear trend on the 2022–24 holdout in
every district (test MAPE below 2.1 %). The Fibonacci-ratio model was off by about 200 %, since it
implies about 62 % annual growth. Wakiso grows fastest in relative terms (≈ 6.5 %/yr). Kampala and Gulu
grow at the same rate (≈ 4.6 %/yr), but Kampala adds about four times as many people. The 2029
forecasts imply about **4,900 additional primary classrooms** across the five districts,
rising to about 5,300 at the upper end of the 95 % bootstrap interval.

**P2: Micro-grid.** The system is well-posed (det = −5) and well-conditioned (κ ≈ 5.8).
Solving all 30 days in one vectorised call is about 5× faster than looping. Under the given
coefficients the battery supplies about 90 % of the energy and 96 % of the cost (≈ UGX 0.59 m per month).
Solar is the most volatile source relative to its size (CV ≈ 0.58). In the Monte Carlo it moved by up
to 64 % under ±5 % demand noise, because it is a small difference of two large numbers. One
infeasible day was repaired with non-negative least squares. A linearly dependent third
constraint makes the hybrid system singular.

**P3: Fisheries.** Replacing Fibonacci growth with logistic growth shows that **h = 0.20
achieves the maximum sustainable yield** (1,000 t/week, stock at K/2), with the highest and
most stable revenue. h = 0.30 out-earns h = 0.10 in the first year only by running the stock down
to 25 % of capacity. A raw variance threshold is meaningless because variance is measured in
UGX², so risk is classified by the coefficient of variation instead. The 5 % VaR at MSY is a
shortfall of about UGX 100 bn per year. An 8-week closed season lowers 5-year revenue by
10–15 % at or below MSY, but *raises* it (+6 %) when the lake is overfished.

**P4: Rainfall & crops.** The old `math.cos` approach was incorrect, and even correct cosine
similarity is a weak measure for rainfall because it ignores scale. Pearson correlation and
Euclidean distance separate the regions far better. Season detection (with December–January
wrap-around and a relative prominence threshold) classifies Mbarara as bimodal and Gulu as
unimodal (borderline). The illustrative Kampala data comes out unimodal, but **real NASA POWER
data for 2014–2023 shows Kampala is bimodal**, which means the illustrative series is unrepresentative.
The advisory note recommends beans for Mbarara, planted in early to mid-March and mid-September.

**P5: Taxi.** The Ntinda fare of UGX 2,000 is 9 % **below** the UGX 2,200 equilibrium. That
creates a shortage of about 10 passengers per trip-hour. In walk-forward testing, simple exponential
smoothing won on every route, with the optimal α = 1 (equivalent to a naïve forecast), while
the original 3-day moving average lagged behind turning points. Day-11 forecasts need **one vehicle
per route**, even with a 15 % buffer. On 60 simulated days with a weekly cycle, a seasonal-naïve
forecaster (MAE ≈ 2.9) beats the 3-day moving average (MAE ≈ 8.5).

## Data sources

* Values marked *illustrative* in the assignment brief are used as given. The two extra
  districts in Project 1 (Mbarara, Mukono) are also illustrative and are **not** UBOS figures.
* `data/microgrid_demand.csv`: synthetic, generated by the notebook with seed 42.
* **NASA POWER** (Prediction Of Worldwide Energy Resources), NASA Langley Research Center:
  monthly point data, parameter `PRECTOTCORR_SUM`, 2014–2023, accessed 6 October 2026.
  https://power.larc.nasa.gov/
* Brouwer, C. & Heibloem, M. (1986). *Irrigation Water Management: Irrigation Water Needs.*
  FAO Training Manual No. 3. Rome: FAO.
* DaMatta, F. M. & Ramalho, J. D. C. (2006). Impacts of drought and temperature stress on
  coffee physiology and production: a review. *Brazilian Journal of Plant Physiology*, 18(1), 55–81.

## AI-use declaration

As permitted by the course policy, I used an AI coding assistant (Claude Code, Anthropic)
in this assignment. It helped plan the work against the rubric and produced first drafts
of the code, tests, notebooks and documentation, as well as the study notes in
`docs/viva_guide.md`.


