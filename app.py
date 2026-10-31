"""
AI Race Outcome Predictor
---------------------------
Streamlit front-end for the dataset-builder and trained gradient-boosting
model. Lets you: build/extend the multi-season dataset, train the model,
inspect its accuracy and feature importances, and score a specific past
race to see predicted vs actual finishing order.

Run with:  streamlit run app.py

Note: this app does NOT predict races in real time from scratch on every
click — building the multi-season dataset via FastF1 is slow (loading Q + R
for every round), so that step has its own progress bar and can be resumed.
Training, once the dataset exists, is fast.
"""

from pathlib import Path

import fastf1
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

from features import build_features, FEATURE_COLUMNS, TARGET_COLUMN
from build_dataset import collect_race, COLUMNS

st.set_page_config(page_title="Race Outcome Predictor", layout="wide", page_icon="🔮")

CACHE_DIR = Path("./fastf1_cache")
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

DATA_PATH = Path("./data/raw_results.csv")
DATA_PATH.parent.mkdir(exist_ok=True)
MODEL_PATH = Path("./models/model.pkl")
MODEL_PATH.parent.mkdir(exist_ok=True)

st.title("🔮 Race Outcome Predictor")
st.caption(
    "Predicts finishing position from qualifying, recent form, and circuit history. "
    "This is a probabilistic estimate, not a forecast — real races have DNFs, safety cars, "
    "and strategy calls no pre-race model can see."
)

step1, step2, step3 = st.tabs(["1️⃣ Build dataset", "2️⃣ Train & evaluate", "3️⃣ Predict a race"])


# ----------------------------------------------------------------------
# STEP 1 — Build / extend the dataset
# ----------------------------------------------------------------------
with step1:
    st.subheader("Build the training dataset")
    st.caption(
        "Pulls qualifying + race results for each round in the chosen season range via FastF1. "
        "This is the slow step — each race requires loading two sessions. Progress is saved "
        "incrementally, so you can stop and resume later without losing work."
    )

    existing_df = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else pd.DataFrame(columns=COLUMNS)
    if not existing_df.empty:
        seasons_covered = sorted(existing_df["Year"].unique())
        st.success(f"Existing dataset: **{len(existing_df)} rows** across seasons {seasons_covered}.")
    else:
        st.info("No dataset yet — build one below.")

    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input("Start year", min_value=2018, max_value=2026, value=2019)
    with col2:
        end_year = st.number_input("End year", min_value=2018, max_value=2026, value=2024)

    if st.button("Build / extend dataset", type="primary"):
        if start_year > end_year:
            st.error("Start year must be before end year.")
        else:
            done_races = set(zip(existing_df.get("Year", []), existing_df.get("Round", [])))
            progress = st.progress(0.0, text="Starting...")
            log = st.empty()

            years = list(range(int(start_year), int(end_year) + 1))
            total_events = 0
            event_list = []
            for year in years:
                schedule = fastf1.get_event_schedule(year)
                schedule = schedule[schedule["EventFormat"] != "testing"]
                for _, event in schedule.iterrows():
                    event_list.append((year, int(event["RoundNumber"]), event["EventName"], event["Location"]))
            total_events = len(event_list)

            collected = 0
            for i, (year, round_num, event_name, circuit) in enumerate(event_list):
                if (year, round_num) not in done_races:
                    log.text(f"Loading {year} R{round_num} — {event_name}...")
                    rows = collect_race(year, round_num, event_name, circuit)
                    if rows:
                        pd.DataFrame(rows).to_csv(
                            DATA_PATH, mode="a", header=not DATA_PATH.exists(), index=False
                        )
                        collected += len(rows)
                progress.progress((i + 1) / total_events, text=f"{i + 1}/{total_events} events checked")

            st.success(f"Done. Added {collected} new driver-race rows.")
            st.rerun()


# ----------------------------------------------------------------------
# STEP 2 — Train and evaluate
# ----------------------------------------------------------------------
with step2:
    st.subheader("Train the model")

    if not DATA_PATH.exists():
        st.warning("No dataset found yet — build one in step 1 first.")
        st.stop()

    raw = pd.read_csv(DATA_PATH)
    feat = build_features(raw)
    trainable = feat.dropna(subset=[TARGET_COLUMN]).copy()

    available_years = sorted(trainable["Year"].unique())
    if len(available_years) < 2:
        st.warning("Need at least 2 seasons of data to hold one out for evaluation. Build more data in step 1.")
        st.stop()

    test_year = st.selectbox(
        "Season to hold out for evaluation (trained only on earlier seasons)",
        options=available_years, index=len(available_years) - 1,
    )

    if st.button("Train model", type="primary"):
        train_df = trainable[trainable["Year"] < test_year]
        test_df = trainable[trainable["Year"] == test_year]

        if train_df.empty:
            st.error(f"No training data before {test_year}. Pick a later hold-out season.")
            st.stop()

        with st.spinner("Training..."):
            X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
            X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

            model = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_depth=4, random_state=42,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)

            test_df = test_df.copy()
            test_df["Predicted"] = preds
            rank_corrs = []
            for (_, _), group in test_df.groupby(["Year", "Round"]):
                if len(group) > 2:
                    corr, _ = spearmanr(group["Predicted"], group[TARGET_COLUMN])
                    if pd.notna(corr):
                        rank_corrs.append(corr)
            avg_rank_corr = float(np.mean(rank_corrs)) if rank_corrs else float("nan")

            perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
            importances = pd.Series(perm.importances_mean, index=FEATURE_COLUMNS).sort_values(ascending=False)

            joblib.dump({
                "model": model, "feature_columns": FEATURE_COLUMNS,
                "test_year": test_year, "mae": mae, "rank_correlation": avg_rank_corr,
            }, MODEL_PATH)

        col1, col2 = st.columns(2)
        col1.metric("Mean absolute error (positions)", f"{mae:.2f}")
        col2.metric("Avg. per-race rank correlation", f"{avg_rank_corr:.3f}")
        st.caption(
            f"Trained on {len(train_df)} driver-races from before {test_year}, "
            f"evaluated on {len(test_df)} driver-races from {test_year}."
        )

        fig = go.Figure(go.Bar(
            x=importances.values, y=importances.index, orientation="h",
            marker_color="#00D2BE",
        ))
        fig.update_layout(
            title="Feature importance (permutation-based)", template="plotly_dark",
            height=350, xaxis_title="Importance", yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, width="stretch")
        st.success("Model trained and saved. Head to step 3 to try it on a real race.")


# ----------------------------------------------------------------------
# STEP 3 — Predict a specific race
# ----------------------------------------------------------------------
with step3:
    st.subheader("Predict a race")

    if not MODEL_PATH.exists():
        st.warning("No trained model yet — finish step 2 first.")
        st.stop()

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]

    raw = pd.read_csv(DATA_PATH)
    feat = build_features(raw)

    race_options = (
        feat[["Year", "Round", "EventName"]]
        .drop_duplicates()
        .sort_values(["Year", "Round"], ascending=[False, False])
    )
    race_options["Label"] = race_options["Year"].astype(str) + " — " + race_options["EventName"]

    selected_label = st.selectbox("Pick a race from the dataset", options=race_options["Label"].tolist())
    selected_row = race_options[race_options["Label"] == selected_label].iloc[0]

    race_data = feat[(feat["Year"] == selected_row["Year"]) & (feat["Round"] == selected_row["Round"])].copy()

    if race_data.empty:
        st.warning("No data for this race.")
        st.stop()

    X = race_data[FEATURE_COLUMNS]
    race_data["PredictedPosition"] = model.predict(X)
    race_data["PredictedRank"] = race_data["PredictedPosition"].rank(method="first").astype(int)

    display = race_data.sort_values("PredictedRank")[
        ["PredictedRank", "Driver", "Team", "QualiPosition", "FinishPosition", "Classified"]
    ].rename(columns={
        "PredictedRank": "Predicted",
        "FinishPosition": "Actual finish",
        "QualiPosition": "Quali pos",
    })
    display["Actual finish"] = display["Actual finish"].apply(
        lambda x: int(x) if pd.notna(x) else "DNF/DSQ"
    )

    st.dataframe(display, width="stretch", hide_index=True)

    classified_only = race_data.dropna(subset=["FinishPosition"])
    if len(classified_only) > 2:
        corr, _ = spearmanr(classified_only["PredictedPosition"], classified_only["FinishPosition"])
        mae_race = mean_absolute_error(classified_only["FinishPosition"], classified_only["PredictedPosition"])
        col1, col2 = st.columns(2)
        col1.metric("This race's rank correlation", f"{corr:.3f}" if pd.notna(corr) else "—")
        col2.metric("This race's MAE (positions)", f"{mae_race:.2f}")

    st.caption(
        "Predicted rank is derived from the model's raw position estimate, re-ranked within this race. "
        "DNFs/DSQs are shown as actual results but excluded from accuracy metrics, since the model "
        "predicts finishing order among classified runners, not reliability."
    )

st.divider()
st.caption("Data via FastF1 · Project 04 of the 'AI Projects That Get You Hired in F1' series.")
