"""
train_model.py
----------------
Loads data/raw_results.csv, builds features via features.py, trains a
gradient boosting model to predict finishing position, evaluates it on a
held-out season (not randomly held out — held out chronologically, since
shuffling would leak future form into the past), and saves the model.

Usage:
    python train_model.py
    python train_model.py --test-year 2024
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

from features import build_features, FEATURE_COLUMNS, TARGET_COLUMN

DATA_PATH = Path("./data/raw_results.csv")
MODEL_PATH = Path("./models/model.pkl")
MODEL_PATH.parent.mkdir(exist_ok=True)


def main(test_year: int | None):
    if not DATA_PATH.exists():
        raise SystemExit(
            f"No dataset found at {DATA_PATH}. Run build_dataset.py first, "
            f"e.g.: python build_dataset.py --start-year 2019 --end-year 2024"
        )

    raw = pd.read_csv(DATA_PATH)
    feat = build_features(raw)

    # only train/evaluate on classified finishers — DNFs have no meaningful
    # finishing position to regress against (see README for how DNFs are handled)
    trainable = feat.dropna(subset=[TARGET_COLUMN]).copy()

    if test_year is None:
        test_year = int(trainable["Year"].max())

    train_df = trainable[trainable["Year"] < test_year]
    test_df = trainable[trainable["Year"] == test_year]

    if train_df.empty or test_df.empty:
        raise SystemExit(
            f"Not enough data to hold out {test_year} as a test season. "
            f"Collect more seasons with build_dataset.py."
        )

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_depth=4, random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    # per-race rank correlation: how well did we get the *order* right, race by race
    test_df = test_df.copy()
    test_df["Predicted"] = preds
    rank_correlations = []
    for (year, round_num), group in test_df.groupby(["Year", "Round"]):
        if len(group) > 2:
            corr, _ = spearmanr(group["Predicted"], group[TARGET_COLUMN])
            if pd.notna(corr):
                rank_correlations.append(corr)
    avg_rank_corr = sum(rank_correlations) / len(rank_correlations) if rank_correlations else float("nan")

    print(f"Trained on seasons before {test_year} ({len(train_df)} rows)")
    print(f"Held out {test_year} for evaluation ({len(test_df)} rows)")
    print(f"MAE (finishing position): {mae:.2f}")
    print(f"Average per-race Spearman rank correlation: {avg_rank_corr:.3f}")

    # HistGradientBoostingRegressor doesn't expose feature_importances_ directly,
    # so we compute permutation importance on the held-out test set instead.
    perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    importances = pd.Series(perm.importances_mean, index=FEATURE_COLUMNS)
    print("\nPermutation feature importances (higher = more predictive):")
    print(importances.sort_values(ascending=False))

    joblib.dump({
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "test_year": test_year,
        "mae": mae,
        "rank_correlation": avg_rank_corr,
    }, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the race outcome predictor.")
    parser.add_argument("--test-year", type=int, default=None, help="Season to hold out for evaluation")
    args = parser.parse_args()
    main(args.test_year)
