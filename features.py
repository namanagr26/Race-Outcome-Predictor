"""
features.py
------------
Shared feature-engineering logic used by both training and prediction.
Kept separate from the data-collection and model-training scripts so the
exact same feature computation can be reused when scoring new races.

All rolling/expanding statistics are computed with `.shift(1)` before the
expanding window so that a race's features only ever use information from
*earlier* races — no leakage from the race being predicted.
"""

import pandas as pd
import numpy as np

FEATURE_COLUMNS = [
    "QualiPosition",
    "GridPosition",
    "DriverFormAvg",
    "TeamFormAvg",
    "DriverCircuitAvg",
]

TARGET_COLUMN = "FinishPosition"


def _expanding_prior_mean(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    """Expanding mean of value_col within each group, using only prior rows
    (current row excluded) — computed on data already sorted chronologically."""
    return (
        df.groupby(group_col)[value_col]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the raw per-driver-per-race results table (one row per driver per
    race, chronologically orderable by Year/Round) and returns it with
    engineered rolling-form features added.

    Expected input columns:
        Year, Round, EventName, Circuit, Driver, Team,
        QualiPosition, GridPosition, FinishPosition, Classified
    """
    df = raw.copy()
    df = df.sort_values(["Year", "Round"]).reset_index(drop=True)

    # driver's own recent form (average finishing position in prior races)
    df["DriverFormAvg"] = _expanding_prior_mean(df, "Driver", "FinishPosition")

    # team's recent form: pool both cars, then map back per team
    team_race = (
        df.groupby(["Year", "Round", "Team"])["FinishPosition"].mean().reset_index()
    )
    team_race = team_race.sort_values(["Year", "Round"]).reset_index(drop=True)
    team_race["TeamFormAvg"] = _expanding_prior_mean(team_race, "Team", "FinishPosition")
    df = df.merge(
        team_race[["Year", "Round", "Team", "TeamFormAvg"]],
        on=["Year", "Round", "Team"], how="left",
    )

    # driver's historical average at this specific circuit
    df["DriverCircuitKey"] = df["Driver"] + "||" + df["Circuit"]
    df["DriverCircuitAvg"] = _expanding_prior_mean(df, "DriverCircuitKey", "FinishPosition")
    df = df.drop(columns=["DriverCircuitKey"])

    # fallback imputation: circuit-specific -> driver form -> field median
    field_median = df["FinishPosition"].median()
    df["DriverFormAvg"] = df["DriverFormAvg"].fillna(field_median)
    df["TeamFormAvg"] = df["TeamFormAvg"].fillna(field_median)
    df["DriverCircuitAvg"] = df["DriverCircuitAvg"].fillna(df["DriverFormAvg"])

    # quali/grid position: fill any gaps with a back-of-field placeholder
    df["QualiPosition"] = df["QualiPosition"].fillna(df["GridPosition"])
    df["QualiPosition"] = df["QualiPosition"].fillna(df["QualiPosition"].median())
    df["GridPosition"] = df["GridPosition"].fillna(df["QualiPosition"])

    return df


def latest_driver_team_form(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Returns each driver's most recent DriverFormAvg/TeamFormAvg/DriverCircuitAvg
    values, i.e. their current 'form' going into a hypothetical next race.
    Useful for scoring an upcoming race where we only know quali/grid positions."""
    latest = (
        feature_df.sort_values(["Year", "Round"])
        .groupby("Driver")
        .tail(1)[["Driver", "Team", "DriverFormAvg", "TeamFormAvg"]]
    )
    return latest.reset_index(drop=True)
