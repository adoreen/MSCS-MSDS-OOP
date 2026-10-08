# Viva preparation guide

Use this to check you can explain the work in your own words. For each question,
try to answer **out loud without looking** first, then compare with the notes.
If a note doesn't make sense to you, open the code it points to and step through
it in the notebook until it does.

A good way to practise: pick a random line in `src/`, and explain (1) what it does,
(2) why it is needed, (3) what would break without it.

---

## General / OOP questions (apply to every project)

**Q: Why put the logic in `src/` instead of the notebooks?**
So it can be imported, reused and unit-tested. Notebooks are for explaining and
showing results; the `.py` modules are the reusable "software" the brief asks for.

**Q: What is an abstract base class and why did you use one?**
`Forecaster` (in `src/common/forecasting.py`) inherits from `ABC` and has
`@abstractmethod`s `_fit` and `_predict`. You can't create a `Forecaster` directly,
only subclasses that implement those methods. It guarantees every model has the same
interface (`fit`, `predict`), so `holdout_evaluate` and `walk_forward` work for any model.

**Q: What is the template method pattern?**
The public `fit()`/`predict()` are written once in the base class and do validation,
then call the subclass hooks `_fit()`/`_predict()`. Validation is never duplicated.

**Q: Where did you use inheritance, composition, and dunder methods?**
* Inheritance: `HybridMicroGrid(MicroGrid)`; every forecaster subclasses `Forecaster`.
* Composition: P3. `RiskAssessor` uses results from `FishStock` and `PriceModel`;
  they are separate objects combined, not a class hierarchy.
* Dunders: `__repr__` and `__len__` on `DistrictPopulation`, `Route`, `Region`;
  `__post_init__` on the dataclasses for validation.

**Q: Why `copy.deepcopy(model)` in the backtests?**
Each training window needs a fresh, unfitted model. Without the copy the same object
would be refitted repeatedly and the caller's model would end up changed.

**Q: How did you make the notebooks reproducible?**
Every random step uses `np.random.default_rng(seed)` with a fixed seed. The seed is
passed in, not set globally. Running `Restart & Run All` gives identical numbers.

**Q: What do your tests check? Give an edge case.**
Normal cases against hand calculations (e.g. det = −5, P* = 2,200), plus edge cases:
empty input, negative values, zero population growth, a singular matrix, MAPE with a
zero actual, too few observations for a moving average.

**Q: What's the difference between `ddof=0` and `ddof=1`?**
NumPy divides by `n − ddof`. `ddof=0` is the population variance (÷n, NumPy default);
`ddof=1` is the sample variance (÷(n−1), what `statistics.variance` uses). We lose one
degree of freedom because the mean was estimated from the same data.

---

## P1: Population forecaster

**Q: Why does `statistics.variance` differ from `np.var`?** See `ddof` above. For n = 10,
NumPy's default is exactly 9/10 of the `statistics` value.

**Q: Write the CAGR formula. Why the exponent 1/(n−1)?**
CAGR = (last/first)^(1/(n−1)) − 1. Ten yearly values contain nine growth steps.

**Q: Kampala and Gulu have the same CAGR. Does that mean the same planning need?**
No. Both grew 1.5×, but Kampala added 600k people and Gulu 160k. Classrooms depend
on *absolute* growth.

**Q: Why did the CAGR model beat the linear trend?**
The series are slightly convex (yearly increments grow), which is what constant-%
growth looks like. A straight line under-forecasts 2022–24.

**Q: Why choose models by RMSE?**
It penalises large errors more, and a big under-forecast is the costly mistake for
classroom planning. MAE and MAPE are reported too.

**Q: How does the bootstrap prediction interval work?**
Fit → residuals (centred) → 1,000 times: fitted + resampled residuals = fake history,
refit, forecast, add a resampled residual per future year → take 2.5th/97.5th percentiles.

**Q: Why centre the residuals?**
The CAGR curve passes exactly through the first and last points, so the raw residuals
were mostly one sign. Resampling them shifted every path one way and gave a lopsided interval.

**Q: Why is the forecast's variance smaller than the actual series'?**
Variance of a trending series mostly measures the trend over the window (5 years vs 10),
and a point forecast is a smooth curve with no noise. It understates uncertainty, which
is why we use prediction intervals instead.

**Q: Is the Fibonacci model ever defensible?**
The ratios tend to φ ≈ 1.618, i.e. about 62 % growth per step. It only fits idealised breeding
(every individual reproduces each step, no deaths, no limits). Not for a district population.

**Q: How did you compute classrooms? Why round up?**
⌈(P2029 − P2024) × 1000 × 0.18 / 53⌉. A fraction of a classroom still has to be built.

---

## P2: Micro-grid

**Q: What do the determinant and condition number tell you?**
det = −5 ≠ 0, so there is a unique solution. cond ≈ 5.83 (small), so a relative error in demand is
amplified at most about 5.8× in the solution. Numerically safe.

**Q: When is a day infeasible?** From x = (2D2 − D1)/5 and y = (4D1 − 3D2)/5:
only when 0.75·D2 ≤ D1 ≤ 2·D2. Outside that, x or y is negative.

**Q: Why NNLS instead of clipping?**
Clipping one value to 0 leaves the other unchanged, so both equations are violated by an
arbitrary amount. NNLS finds the closest non-negative solution and reports the shortfall.

**Q: Why is the vectorised solve faster?**
The loop pays Python/SciPy overhead 30 times (validation, function calls, small arrays).
The vectorised call factorises A once and solves all 30 right-hand sides in compiled code.

**Q: Why pass `D.T` to `linalg.solve`?**
`solve` expects right-hand sides as columns, so D (30×2) becomes 2×30. Transpose back afterwards.

**Q: Which source is more volatile and why do you use CV?**
Solar (CV ≈ 0.58 vs 0.25). Variance is in kWh² and depends on scale; CV is unit-free.

**Q: Solar moves up to 64 % in the Monte Carlo but cond is only 5.8. Contradiction?**
No. cond bounds the error of the *whole vector* relative to its *whole size*. Solar is a
small component computed as a difference of two big numbers (cancellation), so its own
relative change can be much larger.

**Q: What happens if the third equation is linearly dependent?**
det = 0. Either infinitely many solutions (consistent RHS) or none (inconsistent). No unique
dispatch, so the class raises `LinAlgError`.

---

## P3: Fisheries

**Q: Why is Fibonacci growth unrealistic?** No carrying capacity, no deaths, no harvesting,
growth that doesn't slow when crowded, about 62 % per period forever.

**Q: Derive N\* and MSY.** Set N(t+1) = N(t): rN(1 − N/K) = hN → N\* = K(1 − h/r).
Yield Y = hN\* = hK(1 − h/r), maximised at h = r/2 → MSY = rK/4 = 1,000 t/week.

**Q: Why is "variance > 50,000" meaningless?**
Variance is in UGX². Change the unit (UGX → UGX billions) and the variance changes by
10¹⁸. The verdict flips while the risk hasn't changed. CV is unit-free.

**Q: Justify your CV risk bands.** < 0.10 low, 0.10–0.25 moderate, ≥ 0.25 high, based on
a common rule of thumb. It's a judgement call, so the bands are constructor parameters.

**Q: Why average the CV over 1,000 paths?** One price path can be lucky or unlucky.
The average is the *expected* week-to-week instability.

**Q: What does the 5 % VaR mean?** In 1 year out of 20, annual revenue falls below the
5th-percentile value. VaR = mean − that value.

**Q: Why does h = 0.30 beat h = 0.10 in year one if their sustainable yields are equal?**
It mines the existing stock: catch is high while the stock is run down from 4,000 to 2,500 t.
The windfall is temporary and the stock ends up fragile.

**Q: When does the closed season help?** Only when overfished (h = 0.30, about +6 %).
At or below MSY it just loses eight weeks of catch (−10 to −15 %).

---

## P4: Rainfall

**Q: What was wrong with `math.cos()`?** It takes an angle in radians. Cosine *similarity* is
the cosine of the angle *between two vectors*: a·b / (|a||b|).

**Q: How did you verify your cosine?** Against `1 − scipy.spatial.distance.cosine`
(SciPy returns a distance, not a similarity).

**Q: Why can cosine say a much wetter region is "similar"?** Scaling a vector doesn't change
its angle: 2× Kampala scores 1.0. Rainfall is never negative, which also inflates all scores.

**Q: How is Pearson different?** It subtracts each region's mean first, so it compares *timing*.

**Q: Why tile the array three times before `find_peaks`?** December is next to January.
Without wrap-around a season peaking at the year boundary would be missed.

**Q: What is prominence and why relative?** How far a peak rises above the higher of the
two troughs around it. 25 % of the region's own annual range keeps the rule scale-free.

**Q: Kampala came out unimodal. Is your method wrong?** The illustrative data is wet in
Jan–Feb, so there's no dry season separating the rains. With real NASA POWER data the
same code finds Kampala bimodal. The illustrative series correlates only r ≈ 0.30 with reality.

**Q: How were the crop bands derived?** FAO seasonal water need ÷ season length in months.
Limitations: crops need different amounts at different growth stages, and need ≠ rainfall.

---

## P5: Taxi

**Q: Write the equilibrium as a linear system.** Q + 0.02P = 120; Q − 0.03P = 10 →
P\* = 2,200, Q\* = 76. At UGX 2,000: Qd = 80, Qs = 70, a shortage of 10, so the fare is below equilibrium.

**Q: What is walk-forward backtesting?** To forecast day t, fit only on days 1…t−1, then move
on one day. It mimics real use: no peeking at the future.

**Q: Why start at day 4?** The 3-day moving average needs three days of history.

**Q: What does α do in SES? What does α = 1 mean?** α weights the latest value; older values
decay by (1 − α) each day. α = 1 means only today counts: the naive forecast.

**Q: Why might α = 1 win here?** The short series move like a random walk, so the latest day is
the best guide and averaging adds lag.

**Q: Is the SES MAE fair?** Slightly optimistic: α was tuned on the same days it was scored on.
A nested backtest would fix that with more data.

**Q: How many vehicles and why?** ⌈forecast × 1.15 / 112⌉ = 1 per route (112 = 8 trips × 14 seats).

**Q: Why does seasonal-naive beat the moving average on the weekly data?** The MA mixes different
weekdays and lags the Friday/Sunday cycle. Seasonal-naive compares like with like.

---

## Things to be ready to admit (limitations)

Examiners like honesty. Know at least one limitation per project:
P1 illustrative data and short test window · P2 real dispatch is an optimisation problem with
battery state-of-charge · P3 r = 0.4/week is unrealistically fast, and price ignores supply · P4 crop
bands spread evenly over months, coarse satellite data · P5 only 7 forecast errors per model,
and the equilibrium units (per trip-hour) don't match the daily counts.
