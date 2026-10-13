# Assignment 1 Report

- **Name**:
- **Student ID**:
- **Email**:
- **Group**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A | BDBA 3A]

## Dataset

UCI Bike Sharing Dataset, hourly file (`data/hour.csv`), from the UCI Machine
Learning Repository (Fanaee-T & Gama, 2013). One row is one hour of the Capital
Bikeshare system in Washington D.C. between 1 Jan 2011 and 31 Dec 2012:
17,379 rows, 17 columns (date, hour, calendar flags, weather, and the number
of bikes rented in that hour, `cnt`). There are no missing values, but 165
hours are absent from the file altogether.

I picked it because the target is easy to explain (bikes rented per hour) while
the relationship with the raw columns is clearly not linear: demand has two
rush-hour peaks on working days and a single midday hump on weekends, so a
linear model only works if the features are engineered.

## Business / real-life framing

**Scenario.** Every evening the operations team of the bike-share company
plans the next day: how many bikes the rebalancing trucks must move and how
many staff to schedule per shift. They need a forecast of rentals for each
hour of tomorrow, using the calendar and tomorrow's weather forecast.

What this implies for the pipeline:

- **Target.** Total rentals per hour (`cnt`). `casual` and `registered` are
  dropped because they add up to `cnt`: they are the answer, not inputs. The
  model predicts `log(1 + cnt)` because counts are right-skewed (median 142,
  max 977), a log target makes effects proportional (rain removes a share of
  demand, not a fixed number of bikes) and predictions can never be negative.
  All errors are converted back and reported in bikes.
- **Split.** Chronological by day: first 70% of days for training
  (1 Jan 2011 - 25 May 2012, 12,139 rows), next 15% for validation
  (26 May - 12 Sep 2012, 2,640 rows), last 15% for test (13 Sep - 31 Dec 2012,
  2,600 rows). A random split would put hours from the same day in both train
  and test, which the real planner never has: they only know the past.
- **Features allowed.** Only things known the evening before: calendar and
  (forecast) weather. No "rentals in the previous hour".
- **Metric.** It is a regression, so there is no decision threshold. The main
  metric is MAE in bikes per hour (directly "how many bikes off am I"),
  with the bias (mean of prediction - actual) next to it, because for
  operations under-forecasting (empty docks, lost rides) is worse than
  over-forecasting. The regularisation strength was chosen on validation MAE.

## Data preparation & feature engineering

Feature steps, each fitted with scikit-learn on the training set (alpha = 1)
so the effect of every step is visible. MAE and bias in bikes per hour:

| Feature set | Features | Val MAE | Val bias | Test MAE | Test bias |
|---|---|---|---|---|---|
| 1. raw columns (hour as a number) | 7 | 148.2 | -76.4 | 156.1 | -132.1 |
| 2. + one-hot hour | 29 | 103.2 | -64.6 | 121.8 | -110.9 |
| 3. + hour x working-day | 52 | 73.8 | -50.8 | 109.6 | -102.5 |
| 4. + temperature curve (temp²) | 53 | 89.5 | -85.6 | 99.5 | -91.7 |
| 5. + growth trend (final) | 54 | 60.1 | 35.9 | 49.3 | 13.7 |

- **Hour x working-day dummies (47 columns).** Hour as a number forces demand
  to go up or down in a straight line through the day, which is wrong. One-hot
  hours fix the shape, but working days (peaks at 8h and 17-18h) and
  weekends/holidays (one hump around 13h) have different shapes, so each
  (hour, day type) pair gets its own dummy. One cell (00h, non-working day) is
  left out as the reference.
- **Temperature curve.** `temp` plus `temp²`: demand rises with temperature
  but flattens and drops in the hottest hours. `atemp` (feels-like) is dropped
  because it is 0.99 correlated with `temp`.
- **Weather.** One-hot with "clear" as reference. Category 4 (heavy storm)
  has only 3 rows in two years, so it is merged into category 3.
- **Growth trend.** Years since 1 Jan 2011. Average hourly rentals are 144 in
  2011 and 235 in 2012; with a chronological split the model trains mostly on
  the smaller 2011 system and is tested on late 2012.
- **Scaling.** Continuous columns are standardised with the training mean and
  standard deviation only (validation/test never influence them). This matters
  for gradient descent, not for scikit-learn's exact solution.

What did not work as expected:

- Without the trend, every version under-forecasts badly (test bias around
  -100 bikes/hour) and step 3 barely beats the hour x day-type baseline (109.6
  vs 116.4 MAE). The trend is the single biggest improvement.
- `temp²` made validation *worse* (73.8 -> 89.5) but test better
  (109.6 -> 99.5). Validation is the summer of 2012, the hottest and
  highest-demand period. I kept the feature because the drop in demand at
  very high temperatures is visible in the data, but the evidence for it is
  mixed.
- With the trend, the bias flips to positive on validation (+33): a straight
  line fitted to a fast-growing period over-extrapolates.
- 22 rows (10 Mar 2011) have humidity 0, which looks like a sensor fault. I
  left them in; they are 0.2% of training rows.

## Modeling: three implementations, one model

Linear regression (continuous target) with an L2 penalty (ridge). All three
minimise the same objective: mean squared error on `log(1 + cnt)` plus
`alpha / n * sum(w²)` (bias not penalised), which is scikit-learn's Ridge
objective divided by n.

- **scikit-learn**: `Ridge(alpha)`, exact solution.
- **Manual PyTorch**: weight and bias tensors with `requires_grad=True`,
  `loss.backward()`, then `w -= lr * w.grad` inside `torch.no_grad()` and
  zeroing the gradients by hand. Full-batch, 3,000 epochs.
- **Standard PyTorch**: `nn.Linear` + `torch.optim.SGD`, same loss, same
  zero initialisation, full-batch, 3,000 epochs.

Tuning: alpha in {0.1, 1, 10} on validation MAE (60.2 / 60.1 / 59.2, so 10
was chosen, although the differences are tiny); learning rate in
{0.01, 0.1, 0.5} by final training loss (0.3289 / 0.2065 / 0.2065, 0.5 chosen).

| Model | Val MAE | Val RMSE | Val bias | Test MAE | Test RMSE | Test bias |
|---|---|---|---|---|---|---|
| Baseline: training average | 226.88 | 299.83 | -199.73 | 187.15 | 263.38 | -153.17 |
| Baseline: average for that hour & day type | 146.08 | 191.14 | -143.83 | 116.41 | 166.79 | -97.10 |
| scikit-learn | 59.22 | 92.02 | 33.39 | 48.85 | 77.35 | 6.08 |
| PyTorch manual loop | 59.22 | 92.02 | 33.39 | 48.85 | 77.35 | 6.09 |
| PyTorch nn.Module + optim | 59.22 | 92.02 | 33.39 | 48.85 | 77.35 | 6.09 |

**Do they agree?** Yes. The largest coefficient difference between either
PyTorch version and scikit-learn is 0.0018, and the largest difference in any
test prediction is 0.38 bikes. The two PyTorch versions match each other
because they start from the same point and take the same full-batch steps.
The small remaining gap to scikit-learn is because gradient descent stops
after 3,000 steps (and uses 32-bit floats) while scikit-learn solves exactly.
Agreement depends on the learning rate: with lr = 0.01 the loop is still far
from the minimum after 3,000 epochs (loss 0.329 vs 0.206), so it would *not*
have matched scikit-learn.

Against the baselines, the model cuts test MAE from 116.4 (average for that
hour and day type) to 48.9 bikes per hour.

## Limitations & next steps

- **The trend is a straight line.** It over-predicts on validation (+33) and
  would keep growing forever. Next step: refit regularly on a rolling window,
  or use last week's demand at the same hour as a feature (known a day ahead).
- **Perfect weather is assumed.** The model is trained on observed weather,
  but the planner only has a forecast, so real errors would be larger. I would
  need historical forecasts to measure this.
- **Only one year of seasonality.** The test period (Sep-Dec) appears once in
  training, in the smaller 2011 system, so season and growth cannot be fully
  separated. More years of data would fix this.
- **Peaks are under-predicted.** In the weekly view the model misses the
  highest rush-hour and weekend peaks, which are the hours that matter most
  for rebalancing.
- **Final models are trained on the training period only.** After choosing
  alpha, refitting on train + validation would use more recent data.
- **City-wide totals.** Rebalancing is decided per station; this dataset has
  no station detail.

## Generative AI use disclosure

I used Claude (Anthropic). It proposed the dataset and the options for the
framing, split, target and features; I chose the hour x working-day and
temperature-curve features, and it added the growth-trend feature after the
ablation showed systematic under-forecasting. It wrote the code in
`features.py`, `train.py` and `app.py` and a first draft of this report from
the pipeline's results. *[Edit this paragraph so it describes accurately what
you did yourself: what you reviewed, changed or rewrote.]*
