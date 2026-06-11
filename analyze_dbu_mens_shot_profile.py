#!/usr/bin/env python3
"""Create the shot vs. non-shot feature profiles for the DBU analysis.

The benchmark script identifies all hybrid-detected counterattacks. This script
adds a more interpretable outcome point to each counterattack and then compares:

- counterattacks that led to a shot,
- counterattacks that did not lead to a shot,
- Denmark's counterattacks.

The output CSV files are used directly in the DBU analysis notebook.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd

from build_regain_candidates import (
    TRACKING_GOAL_LEFT,
    TRACKING_GOAL_RIGHT,
    build_ball_tracking_lookup,
    build_tracking_label_index,
    forward_progress_m,
    goal_distance_tracking,
    load_events,
    nearest_tracking_ball_position,
)


# Paths
ROOT = Path(__file__).resolve().parent
EVENTS_CSV = ROOT / "Data" / "events.csv"
ANALYSIS_DIR = ROOT / "Data" / "derived" / "dbu_mens_hybrid_analysis"
OUTPUT_DIR = ANALYSIS_DIR / "shot_profile"

# Dataset setup
DATASET_KEYS = ["men", "women", "u21"]
DENMARK_TEAM_NAME = {
    "men": "Denmark",
    "women": "Denmark",
    "u21": "Denmark Under 21",
}

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
# The hybrid model is based on features from the first 20 seconds after regain,
# so the DBU outcome profile uses the same maximum window.
MAX_OUTCOME_WINDOW_S = 20.0

# Events that clearly stop or restart the attacking sequence.
STOP_RESTART_TYPES = {"throw_in", "goal_kick", "corner"}
# Event types that indicate the opponent has taken control of the ball.
ON_BALL_TYPES = {
    "pass",
    "touch",
    "duel",
    "interception",
    "shot",
    "clearance",
    "free_kick",
    "throw_in",
    "corner",
    "goal_kick",
    "goalkeeper_exit",
    "penalty",
    "postmatch_penalty",
    "postmatch_penalty_faced",
    "own_goal",
}

# The profile table keeps both original model features and the outcome features
# created in this script. The direction column is only descriptive metadata for
# the output CSV.
FEATURES = [
    ("long_progression_goal_m", "Progression toward goal (m)", "higher"),
    ("outcome_progress_m", "Outcome progress (m)", "higher"),
    ("outcome_goal_distance_m", "Outcome distance to goal (m)", "lower"),
    ("outcome_goal_progression_speed_mps", "Outcome goal progression speed (m/s)", "higher"),
    ("outcome_time_s", "Duration to outcome (s)", "higher"),
    ("long_directness_goal", "Directness to goal", "higher"),
    ("first_action_time_s", "Time to first action (s)", "lower"),
    ("event_count_pass_20s", "Pass count within 20s", "higher"),
    ("event_count_touch_20s", "Touch count within 20s", "higher"),
    ("long_start_progress_m", "Start progression (m)", "higher"),
    ("long_start_goal_distance_m", "Start goal distance (m)", "lower"),
]


def parse_timestamp_to_seconds(value):
    """Convert event timestamps from hh:mm:ss format to seconds."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    hours, minutes, seconds = text.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds.replace(",", "."))


def duel_won_event(row):
    """Return true when the event data indicates the opponent won the duel."""
    return (
        str(row.get("aerialWon", "")).strip() == "1"
        or str(row.get("groundDuelRecoveredPossession", "")).strip() == "1"
        or str(row.get("groundDuelKeptPossession", "")).strip() == "1"
    )


def opponent_on_ball_ends_event(row, possessing_team_id):
    """Detect opponent actions that should terminate the attacking sequence."""
    event_type = str(row.get("_type", "")).strip()
    team_id = str(row.get("_team_id", "")).strip()
    if not team_id or team_id == str(possessing_team_id) or event_type not in ON_BALL_TYPES:
        return False
    if event_type == "duel":
        return duel_won_event(row)
    return True


def event_outcome_position(row, candidate_team_id):
    """Convert event x/y coordinates into attacking progress and goal distance."""
    x = pd.to_numeric(row.get("x"), errors="coerce")
    y = pd.to_numeric(row.get("y"), errors="coerce")
    if pd.isna(x) or pd.isna(y):
        return np.nan, np.nan

    progress_m = float(x) * PITCH_LENGTH_M / 100.0
    event_team_id = str(row.get("_team_id", "")).strip()
    # If the outcome event belongs to the opponent, flip progress so it is still
    # measured from the original attacking team's perspective.
    if event_team_id and event_team_id != str(candidate_team_id):
        progress_m = PITCH_LENGTH_M - progress_m

    progress_m = float(np.clip(progress_m, 0.0, PITCH_LENGTH_M))
    lateral_offset_m = (float(y) - 50.0) * PITCH_WIDTH_M / 100.0
    goal_distance_m = math.hypot(PITCH_LENGTH_M - progress_m, lateral_offset_m)
    return progress_m, goal_distance_m


def tracking_endpoint_position(candidate, ball_tracking_by_match):
    """Fallback position when no decisive event is found inside the window."""
    endpoint_ts = float(candidate.start_ts) + float(candidate.long_duration_s)
    ball = nearest_tracking_ball_position(
        ball_tracking_by_match,
        str(candidate.match_id),
        endpoint_ts,
        max_dt_s=0.5,
    )
    if ball is None:
        # If tracking is unavailable at the endpoint, keep the existing model
        # feature as a fallback rather than dropping the candidate.
        progress_m = pd.to_numeric(getattr(candidate, "long_end_progress_m"), errors="coerce")
        return progress_m, np.nan

    attacking_left = bool(int(candidate.attacking_left))
    goal_xy = TRACKING_GOAL_LEFT if attacking_left else TRACKING_GOAL_RIGHT
    progress_m = forward_progress_m(ball["x"], attacking_left)
    goal_distance_m = goal_distance_tracking(ball["x"], ball["y"], goal_xy)
    return progress_m, goal_distance_m


def load_events_by_match():
    """Load the event columns needed to identify the first decisive outcome."""
    event_cols = [
        "matchId",
        "eventId",
        "teamId",
        "typePrimary",
        "matchTimestamp",
        "x",
        "y",
        "aerialWon",
        "groundDuelRecoveredPossession",
        "groundDuelKeptPossession",
    ]
    events = pd.read_csv(EVENTS_CSV, usecols=event_cols, low_memory=False)
    events["_match_id"] = events["matchId"].astype(str)
    events["_team_id"] = events["teamId"].astype(str)
    events["_event_id"] = events["eventId"].astype(str)
    events["_type"] = events["typePrimary"].fillna("").astype(str)
    events["_ts"] = events["matchTimestamp"].map(parse_timestamp_to_seconds)
    events = events.dropna(subset=["_ts"]).sort_values(["_match_id", "_ts", "_event_id"])
    return {match_id: group.reset_index(drop=True) for match_id, group in events.groupby("_match_id", sort=False)}


def first_decisive_outcome_event(window, team_id):
    """Find the first event that gives the counterattack a clear outcome."""
    for _, event in window.iterrows():
        event_type = event["_type"]
        event_team_id = str(event["_team_id"])
        if event_team_id == str(team_id) and event_type == "shot":
            return event, "shot"
        if event_type == "game_interruption":
            return event, "game_interruption"
        if event_type in STOP_RESTART_TYPES:
            return event, event_type
        if opponent_on_ball_ends_event(event, team_id):
            return event, "opponent_on_ball"
    # No decisive event was found inside the 20-second window.
    return None, "window_end_20s"


def add_outcome_features(df, events_by_match, ball_tracking_by_match):
    """Add outcome position, duration and speed features to each counterattack."""
    records = []
    for candidate in df.itertuples(index=False):
        match_id = str(candidate.match_id)
        team_id = str(candidate.team_id)
        start_ts = float(candidate.start_ts)
        end_ts = start_ts + MAX_OUTCOME_WINDOW_S
        match_events = events_by_match.get(match_id)

        if match_events is None:
            window = pd.DataFrame()
        else:
            window = match_events[(match_events["_ts"] > start_ts) & (match_events["_ts"] <= end_ts)]

        # Priority: shot, stoppage/restart, opponent possession, otherwise the
        # end of the 20-second window.
        outcome_event, outcome_type = first_decisive_outcome_event(window, team_id)
        start_progress_m = pd.to_numeric(getattr(candidate, "long_start_progress_m"), errors="coerce")
        start_goal_distance_m = pd.to_numeric(
            getattr(candidate, "long_start_goal_distance_m"), errors="coerce"
        )

        if outcome_event is None:
            # No clear event outcome: use tracking at the end of the candidate
            # window so non-shot attacks still have a comparable endpoint.
            outcome_progress_m, outcome_goal_distance_m = tracking_endpoint_position(
                candidate,
                ball_tracking_by_match,
            )
            outcome_event_id = np.nan
            outcome_time_s = min(MAX_OUTCOME_WINDOW_S, float(getattr(candidate, "long_duration_s")))
            outcome_is_window_endpoint = 1
        else:
            # Event outcome: use the event location itself, e.g. shot location or
            # where possession is lost.
            outcome_progress_m, outcome_goal_distance_m = event_outcome_position(outcome_event, team_id)
            outcome_event_id = outcome_event["_event_id"]
            outcome_time_s = max(0.0, float(outcome_event["_ts"]) - start_ts)
            outcome_is_window_endpoint = 0

        records.append(
            {
                "outcome_type": outcome_type,
                "outcome_is_window_endpoint": outcome_is_window_endpoint,
                "outcome_event_id": outcome_event_id,
                "outcome_time_s": outcome_time_s,
                "outcome_progress_m": outcome_progress_m,
                "outcome_goal_distance_m": outcome_goal_distance_m,
                "outcome_progression_from_start_m": (
                    outcome_progress_m - start_progress_m
                    if pd.notna(outcome_progress_m) and pd.notna(start_progress_m)
                    else np.nan
                ),
                "outcome_goal_progression_m": (
                    start_goal_distance_m - outcome_goal_distance_m
                    if pd.notna(start_goal_distance_m) and pd.notna(outcome_goal_distance_m)
                    else np.nan
                ),
            }
        )

    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(records)], axis=1)
    # Speed is based on reduction in distance to goal over the same start-to-
    # outcome interval as outcome_time_s.
    out["outcome_goal_progression_speed_mps"] = np.where(
        out["outcome_time_s"].gt(0) & out["outcome_goal_progression_m"].notna(),
        out["outcome_goal_progression_m"].clip(lower=0) / out["outcome_time_s"],
        np.nan,
    )
    return out


def profile_rows(df, denmark_team_name):
    """Build one mean-comparison row per feature for the notebook tables."""
    rows = []
    shot = df[df["shot_20s"].eq(1)]
    no_shot = df[df["shot_20s"].eq(0)]
    denmark = df[df["teamName"].eq(denmark_team_name)]

    for col, label, direction in FEATURES:
        shot_values = shot[col].dropna()
        no_shot_values = no_shot[col].dropna()
        denmark_values = denmark[col].dropna()
        rows.append(
            {
                "feature": col,
                "label": label,
                "direction_expected_for_success": direction,
                "shot_n": int(shot_values.size),
                "no_shot_n": int(no_shot_values.size),
                "denmark_n": int(denmark_values.size),
                "shot_mean": float(shot_values.mean()) if shot_values.size else np.nan,
                "no_shot_mean": float(no_shot_values.mean()) if no_shot_values.size else np.nan,
                "denmark_mean": float(denmark_values.mean()) if denmark_values.size else np.nan,
                "shot_minus_no_shot_mean": (
                    float(shot_values.mean() - no_shot_values.mean())
                    if shot_values.size and no_shot_values.size
                    else np.nan
                ),
                "denmark_minus_shot_mean": (
                    float(denmark_values.mean() - shot_values.mean())
                    if denmark_values.size and shot_values.size
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Event data gives the terminal event; tracking data is only needed when no
    # terminal event is found inside the 20-second window.
    events_by_match = load_events_by_match()
    candidate_events_by_match = load_events()
    label_to_tracking_folder = build_tracking_label_index(ROOT / "Data")
    ball_tracking_by_match = build_ball_tracking_lookup(candidate_events_by_match, label_to_tracking_folder)

    for dataset_key in DATASET_KEYS:
        # These files are created by build_dbu_mens_hybrid_analysis.py.
        input_csv = ANALYSIS_DIR / f"hybrid_counterattacks_{dataset_key}.csv"
        df = pd.read_csv(input_csv, low_memory=False)
        df = add_outcome_features(df, events_by_match, ball_tracking_by_match)

        profile = profile_rows(df, DENMARK_TEAM_NAME[dataset_key])
        profile_path = OUTPUT_DIR / f"shot_vs_nonshot_feature_profile_{dataset_key}.csv"
        outcome_path = OUTPUT_DIR / f"counterattacks_with_outcome_{dataset_key}.csv"

        profile.to_csv(profile_path, index=False)
        df.to_csv(outcome_path, index=False)

        print(f"Saved: {profile_path}")
        print(f"Saved: {outcome_path}")


if __name__ == "__main__":
    main()
