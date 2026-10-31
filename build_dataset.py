"""
build_dataset.py
------------------
Pulls qualifying + race results across a range of seasons via FastF1 and
writes a raw per-driver-per-race results table to data/raw_results.csv.

This is a separate, resumable script (not run inline in the Streamlit app)
because pulling several seasons of session data over the network is slow —
FastF1 has to load Q + R for every round. Progress is saved incrementally,
so re-running the script skips races already collected.

Usage:
    python build_dataset.py --start-year 2019 --end-year 2024
"""

import argparse
import os
from pathlib import Path

import fastf1
import pandas as pd

CACHE_DIR = Path("./fastf1_cache")
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

DATA_PATH = Path("./data/raw_results.csv")
DATA_PATH.parent.mkdir(exist_ok=True)

COLUMNS = [
    "Year", "Round", "EventName", "Circuit", "Driver", "Team",
    "QualiPosition", "GridPosition", "FinishPosition", "Classified", "Status",
]


def classified_to_numeric(classified_position: str):
    """FastF1's ClassifiedPosition is a string: an integer, or one of
    R/D/E/W/F/N for retired/disqualified/etc. Returns (numeric_or_None, is_classified)."""
    try:
        return int(classified_position), True
    except (ValueError, TypeError):
        return None, False


def collect_race(year: int, round_number: int, event_name: str, circuit: str) -> list[dict]:
    rows = []
    try:
        quali = fastf1.get_session(year, round_number, "Q")
        quali.load(laps=False, telemetry=False, weather=False, messages=False)
        quali_results = quali.results[["Abbreviation", "Position"]].rename(
            columns={"Position": "QualiPosition"}
        )
    except Exception as e:
        print(f"  ! Could not load qualifying for {year} R{round_number}: {e}")
        quali_results = pd.DataFrame(columns=["Abbreviation", "QualiPosition"])

    try:
        race = fastf1.get_session(year, round_number, "R")
        race.load(laps=False, telemetry=False, weather=False, messages=False)
    except Exception as e:
        print(f"  ! Could not load race for {year} R{round_number}: {e}")
        return rows

    race_results = race.results
    for _, r in race_results.iterrows():
        driver = r["Abbreviation"]
        team = r.get("TeamName", None)
        grid = r.get("GridPosition", None)
        finish_numeric, is_classified = classified_to_numeric(r.get("ClassifiedPosition"))

        quali_row = quali_results[quali_results["Abbreviation"] == driver]
        quali_pos = quali_row["QualiPosition"].iloc[0] if not quali_row.empty else None

        rows.append({
            "Year": year, "Round": round_number, "EventName": event_name, "Circuit": circuit,
            "Driver": driver, "Team": team,
            "QualiPosition": quali_pos, "GridPosition": grid,
            "FinishPosition": finish_numeric, "Classified": is_classified,
            "Status": r.get("Status", None),
        })
    return rows


def main(start_year: int, end_year: int):
    existing = pd.DataFrame(columns=COLUMNS)
    if DATA_PATH.exists():
        existing = pd.read_csv(DATA_PATH)
        print(f"Found existing dataset with {len(existing)} rows — will skip already-collected races.")

    done_races = set(zip(existing.get("Year", []), existing.get("Round", [])))
    all_new_rows = []

    for year in range(start_year, end_year + 1):
        schedule = fastf1.get_event_schedule(year)
        schedule = schedule[schedule["EventFormat"] != "testing"]

        for _, event in schedule.iterrows():
            round_number = int(event["RoundNumber"])
            if (year, round_number) in done_races:
                continue

            event_name = event["EventName"]
            circuit = event["Location"]
            print(f"[{year} R{round_number}] {event_name} ...")
            rows = collect_race(year, round_number, event_name, circuit)
            if rows:
                all_new_rows.extend(rows)
                # write incrementally so a crash doesn't lose all progress
                pd.DataFrame(all_new_rows).to_csv(
                    DATA_PATH, mode="a", header=not DATA_PATH.exists(), index=False
                )
                all_new_rows = []

    print(f"Done. Dataset saved at {DATA_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the F1 race outcome dataset via FastF1.")
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2024)
    args = parser.parse_args()
    main(args.start_year, args.end_year)
