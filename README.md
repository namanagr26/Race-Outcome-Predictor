# Race Outcome Predictor

Project 04 from the "AI Projects That Get You Hired in F1" series — a
gradient-boosting model that predicts F1 finishing position from qualifying
result, recent driver/team form, and circuit history, trained on real
multi-season data via [FastF1](https://docs.fastf1.dev/).

> **This does not accurately predict races.** Real races have DNFs, safety
> cars, grid penalties, weather, and strategy calls that a pre-race model
> can't see. The point of this project is building and honestly evaluating
> a prediction system — not making race calls.

## How it's structured

Unlike the dashboard-style projects earlier in this series, this one is a
proper three-stage ML pipeline, split into separate scripts because pulling
several seasons of session data is slow:

| File | Purpose |
|---|---|
| `build_dataset.py` | Pulls qualifying + race results per round via FastF1, saves to `data/raw_results.csv`. Resumable — skips races already collected. |
| `features.py` | Shared feature engineering (used by both training and prediction) — rolling driver/team form and circuit history, computed with no lookahead leakage. |
| `train_model.py` | Trains a `HistGradientBoostingRegressor` on all seasons before a held-out test season, evaluates MAE and per-race rank correlation, saves the model. |
| `app.py` | Streamlit front-end: build the dataset with a progress bar, train + inspect the model, and score any race in the dataset to compare predicted vs actual order. |

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Everything can be done from the app — dataset building, training, and
prediction are each their own tab (in that order). You can also run the
underlying scripts directly from the command line if you'd rather build a
large dataset without keeping the browser tab open:

```bash
python build_dataset.py --start-year 2019 --end-year 2024
python train_model.py --test-year 2024
```

## Features used

- **QualiPosition** — where the driver qualified
- **GridPosition** — actual starting position (differs from quali when there's a grid penalty)
- **DriverFormAvg** — the driver's average finishing position over all prior races (expanding mean, not including the race being predicted)
- **TeamFormAvg** — same idea, pooled across both cars for the team
- **DriverCircuitAvg** — the driver's historical average finish at this specific circuit

All rolling features are computed with `.shift(1)` before the expanding
average, so a race's features only ever see *earlier* results — no leakage
from the outcome being predicted.

## How DNFs are handled

Retirements, disqualifications, and non-classified results have no
meaningful finishing position to regress against, so they're excluded from
training and from accuracy metrics — but they're still shown in the
prediction table in the app, since a real report shouldn't quietly drop them.
This is exactly the kind of real-world messiness the original project brief
calls out: DNFs, safety cars, and grid penalties make this a genuinely
harder problem than a clean tutorial dataset.

## Evaluation approach

The test season is *never* randomly sampled — it's the most recent season,
held out entirely, because shuffling would let the model see future form
data when predicting the past. Two metrics are reported:

- **MAE** — average error in predicted finishing position (across all races)
- **Per-race Spearman rank correlation** — averaged across races, this asks
  "did the model get the relative order right within each race?", which
  matters more than raw position error for a genuinely useful predictor.

## Where this fits in the series

Reuses the FastF1 session-loading pattern from Projects 01–03, extended into
a full offline dataset + model-training pipeline — this is the piece of the
series that's a genuine, evaluable ML system rather than a data viewer.
