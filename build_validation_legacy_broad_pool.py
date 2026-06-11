#!/usr/bin/env python3
"""Build the 48-case validation candidate pool used for manual checking.

This script is a cleaned Python version of the old
counterattack_validation_legacy_broad_pool.ipynb notebook. It creates the two
CSV files that the snapshot generator needs:

- Data/derived/validation_legacy_broad_pool/focused_candidate_pool_all.csv
- Data/derived/validation_legacy_broad_pool/focused_candidate_pool_manual_annotation.csv

The script deliberately does not depend on existing snapshots. Snapshots are
generated afterwards by make_validation_candidate_snapshots.py.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "Data" / "events.csv"
FINAL_MODEL_ATTEMPTS_PATH = PROJECT_ROOT / "Data" / "derived" / "counterattack_attempts_balanced_v2_scaled_tracking_coords.csv"
OUT_DIR = PROJECT_ROOT / "Data" / "derived" / "validation_legacy_broad_pool"

POOL_ALL_PATH = OUT_DIR / "focused_candidate_pool_all.csv"
ANNOTATION_PATH = OUT_DIR / "focused_candidate_pool_manual_annotation.csv"

# These are the three matches used for the fixed 48-case validation set.
VALIDATION_MATCH_IDS = [
    "5707002",  # Poland - Sweden
    "5544238",  # Spain - Germany
    "5723756",  # Norway - Italy
]

CANDIDATE_POOL_MODE = "union_legacy_broad_and_final"

LEGACY_BROAD_CFG = {
    "max_duration_s": 17.0,
    "min_progression_x": 20.0,
    "min_directness": 0.75,
    "max_passes": 4,
    "max_start_x": None,
}

LEGACY_BALANCED_CFG = {
    "max_duration_s": 17.0,
    "min_progression_x": 25.0,
    "min_directness": 0.75,
    "max_passes": 4,
    "max_start_x": 55.0,
}

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

STOP_RESTART_TYPES = {"throw_in", "goal_kick", "corner"}


def to_float(value):
    s = (value or "").strip()
    if not s or s == "NULL":
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def to_int(value):
    x = to_float(value)
    return int(x) if x is not None else None


def ts_to_seconds(ts):
    s = (ts or "").strip()
    if not s:
        return None
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def normalize_label_no_score(label):
    s = (label or "").strip()
    if "," in s:
        s = s.split(",", 1)[0].strip()
    return s


def duel_won(row):
    return (
        (row.get("aerialWon") or "").strip() == "1"
        or (row.get("groundDuelRecoveredPossession") or "").strip() == "1"
        or (row.get("groundDuelKeptPossession") or "").strip() == "1"
    )


def is_regain_event(row):
    event_type = (row.get("typePrimary") or "").strip()
    return event_type == "interception" or (event_type == "duel" and duel_won(row))


def load_events(path=DATA_PATH):
    by_match = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            match_id = (row.get("matchId") or "").strip()
            if not match_id:
                continue
            row["_ts"] = ts_to_seconds(row.get("matchTimestamp"))
            row["_event_id"] = to_int(row.get("eventId")) or 0
            row["_minute"] = to_int(row.get("minute"))
            row["_second"] = to_int(row.get("second"))
            row["_x"] = to_float(row.get("x"))
            row["_y"] = to_float(row.get("y"))
            by_match[match_id].append(row)

    for match_id in by_match:
        by_match[match_id].sort(
            key=lambda r: ((r["_ts"] if r["_ts"] is not None else 10**18), r["_event_id"])
        )
    return by_match


def build_regain_candidates(by_match):
    out = []
    for match_id, rows in by_match.items():
        prev_control = None
        for ev in rows:
            event_type = (ev.get("typePrimary") or "").strip()
            team_id = (ev.get("teamId") or "").strip()
            ts = ev["_ts"]

            if event_type in ON_BALL_TYPES and team_id:
                is_candidate = (
                    is_regain_event(ev)
                    and prev_control is not None
                    and team_id != prev_control["team_id"]
                    and ts is not None
                )

                if is_candidate:
                    out.append(
                        {
                            "match_id": str(match_id),
                            "match_label": normalize_label_no_score(ev.get("label")),
                            "match_period": (ev.get("matchPeriod") or "").strip(),
                            "start_event_id": str(ev["_event_id"]),
                            "start_type": event_type,
                            "start_ts": ts,
                            "minute": ev["_minute"],
                            "second": ev["_second"],
                            "team_id": team_id,
                            "team_name": (ev.get("teamName") or "").strip(),
                            "player_id": str((ev.get("playerId") or "").strip()),
                            "player_position": (ev.get("playerPosition") or "").strip(),
                            "event_x": ev["_x"],
                            "event_y": ev["_y"],
                            "previous_control_team_id": prev_control["team_id"],
                            "previous_control_team_name": prev_control["team_name"],
                            "previous_control_type": prev_control["event_type"],
                            "previous_control_event_id": prev_control["event_id"],
                        }
                    )

                if event_type != "duel" or duel_won(ev):
                    prev_control = {
                        "team_id": team_id,
                        "team_name": (ev.get("teamName") or "").strip(),
                        "event_type": event_type,
                        "event_id": str(ev["_event_id"]),
                    }

    return pd.DataFrame(out)


def extract_fixed_rule_attempts(profile_name, cfg, by_match):
    out = []
    for match_id, rows in by_match.items():
        prev_control_team = None
        for i, ev in enumerate(rows):
            event_type = (ev.get("typePrimary") or "").strip()
            team_id = (ev.get("teamId") or "").strip()
            ts = ev["_ts"]

            if event_type in ON_BALL_TYPES and team_id:
                candidate = (
                    is_regain_event(ev)
                    and prev_control_team is not None
                    and team_id != prev_control_team
                    and ts is not None
                )

                if candidate:
                    start_x = ev["_x"]
                    if start_x is None:
                        if event_type != "duel" or duel_won(ev):
                            prev_control_team = team_id
                        continue

                    if cfg.get("max_start_x") is not None and start_x > cfg["max_start_x"]:
                        if event_type != "duel" or duel_won(ev):
                            prev_control_team = team_id
                        continue

                    max_x = start_x
                    prev_x = start_x
                    sum_abs_dx = 0.0
                    pass_count = 0
                    action_count = 0
                    end_ts = ts

                    for ev2 in rows[i + 1:]:
                        t2 = (ev2.get("typePrimary") or "").strip()
                        team2 = (ev2.get("teamId") or "").strip()
                        ts2 = ev2["_ts"]

                        if ts2 is None:
                            continue
                        if ts2 - ts > cfg["max_duration_s"]:
                            break
                        if t2 == "game_interruption":
                            break
                        if t2 in STOP_RESTART_TYPES:
                            break
                        if t2 in ON_BALL_TYPES and team2 and team2 != team_id:
                            break

                        if team2 == team_id:
                            end_ts = ts2
                            action_count += 1
                            if t2 == "pass":
                                pass_count += 1

                            x2 = ev2["_x"]
                            if x2 is not None:
                                max_x = max(max_x, x2)
                                if prev_x is not None:
                                    sum_abs_dx += abs(x2 - prev_x)
                                prev_x = x2

                    duration_s = max(0.0, end_ts - ts)
                    progression_x = max_x - start_x
                    directness = progression_x / sum_abs_dx if sum_abs_dx > 0 else None

                    keep = True
                    if progression_x < cfg["min_progression_x"]:
                        keep = False
                    if pass_count > cfg["max_passes"]:
                        keep = False
                    if directness is None or directness < cfg["min_directness"]:
                        keep = False

                    if keep:
                        out.append(
                            {
                                "match_id": str(match_id),
                                "team_id": str(team_id),
                                "start_event_id": str(ev["_event_id"]),
                                f"{profile_name}_duration_s": duration_s,
                                f"{profile_name}_progression_x": progression_x,
                                f"{profile_name}_directness": directness,
                                f"{profile_name}_pass_count": pass_count,
                                f"{profile_name}_action_count": action_count,
                            }
                        )

                if event_type != "duel" or duel_won(ev):
                    prev_control_team = team_id

    return pd.DataFrame(out)


def build_pool():
    by_match_events = load_events()
    regain_df = build_regain_candidates(by_match_events)

    legacy_broad_df = extract_fixed_rule_attempts("legacy_broad", LEGACY_BROAD_CFG, by_match_events)
    legacy_balanced_df = extract_fixed_rule_attempts("legacy_balanced", LEGACY_BALANCED_CFG, by_match_events)

    final_df = pd.read_csv(
        FINAL_MODEL_ATTEMPTS_PATH,
        dtype={"match_id": str, "team_id": str, "start_event_id": str},
    )
    final_small = final_df[
        [
            "match_id",
            "team_id",
            "start_event_id",
            "duration_s",
            "pass_count",
            "start_zone",
            "progression_goal_m",
            "speed_ratio",
        ]
    ].rename(
        columns={
            "duration_s": "current_final_duration_s",
            "pass_count": "current_final_pass_count",
            "start_zone": "current_final_start_zone",
            "progression_goal_m": "current_final_progression_goal_m",
            "speed_ratio": "current_final_speed_ratio",
        }
    )

    scored_df = regain_df.copy()
    if not legacy_broad_df.empty:
        scored_df = scored_df.merge(legacy_broad_df, how="left", on=["match_id", "team_id", "start_event_id"])
    if not legacy_balanced_df.empty:
        scored_df = scored_df.merge(legacy_balanced_df, how="left", on=["match_id", "team_id", "start_event_id"])
    if not final_small.empty:
        scored_df = scored_df.merge(final_small, how="left", on=["match_id", "team_id", "start_event_id"])

    scored_df["legacy_broad_prediction"] = scored_df["legacy_broad_duration_s"].notna().astype(int)
    scored_df["legacy_balanced_prediction"] = scored_df["legacy_balanced_duration_s"].notna().astype(int)
    scored_df["current_final_prediction"] = scored_df["current_final_duration_s"].notna().astype(int)

    if CANDIDATE_POOL_MODE == "legacy_broad_only":
        pool_mask = scored_df["legacy_broad_prediction"] == 1
    elif CANDIDATE_POOL_MODE == "union_legacy_broad_and_final":
        pool_mask = (scored_df["legacy_broad_prediction"] == 1) | (scored_df["current_final_prediction"] == 1)
    else:
        raise ValueError(f"Unsupported CANDIDATE_POOL_MODE: {CANDIDATE_POOL_MODE}")

    pool_df = scored_df[pool_mask].copy()
    pool_df["pool_source"] = ""
    pool_df.loc[
        (pool_df["legacy_broad_prediction"] == 1) & (pool_df["current_final_prediction"] == 1),
        "pool_source",
    ] = "both"
    pool_df.loc[
        (pool_df["legacy_broad_prediction"] == 1) & (pool_df["current_final_prediction"] == 0),
        "pool_source",
    ] = "legacy_broad_only"
    pool_df.loc[
        (pool_df["legacy_broad_prediction"] == 0) & (pool_df["current_final_prediction"] == 1),
        "pool_source",
    ] = "current_final_only"

    pool_df = pool_df.sort_values(["match_id", "start_ts", "start_event_id"]).reset_index(drop=True)
    return regain_df, scored_df, pool_df


def build_annotation_file(pool_df):
    annotation_df = pool_df[pool_df["match_id"].isin(VALIDATION_MATCH_IDS)][
        [
            "match_id",
            "match_label",
            "match_period",
            "start_event_id",
            "start_type",
            "start_ts",
            "minute",
            "second",
            "team_id",
            "team_name",
            "player_id",
            "player_position",
            "event_x",
            "event_y",
            "previous_control_team_name",
            "previous_control_type",
        ]
    ].copy()

    annotation_df["_start_ts_num"] = pd.to_numeric(annotation_df["start_ts"], errors="coerce")
    annotation_df["_start_event_num"] = pd.to_numeric(annotation_df["start_event_id"], errors="coerce")
    annotation_df = annotation_df.sort_values(["match_id", "_start_ts_num", "_start_event_num"]).reset_index(drop=True)
    annotation_df["candidate_id"] = [f"{i + 1:03d}" for i in range(len(annotation_df))]
    annotation_df["snapshot_file"] = pd.NA
    annotation_df = annotation_df.drop(columns=["_start_ts_num", "_start_event_num"])

    annotation_df = annotation_df[
        [
            "candidate_id",
            "snapshot_file",
            "match_id",
            "match_label",
            "match_period",
            "start_event_id",
            "start_type",
            "start_ts",
            "minute",
            "second",
            "team_id",
            "team_name",
            "player_id",
            "player_position",
            "event_x",
            "event_y",
            "previous_control_team_name",
            "previous_control_type",
        ]
    ].copy()

    annotation_df["human_label"] = pd.NA
    annotation_df["human_confidence"] = pd.NA
    annotation_df["human_note"] = pd.NA

    # If this file is regenerated after manual work, keep the existing labels
    # and snapshot filenames for matching candidates.
    if ANNOTATION_PATH.exists():
        existing = pd.read_csv(ANNOTATION_PATH, dtype=str)
        merge_cols = ["match_id", "team_id", "start_event_id"]
        keep_cols = merge_cols + [
            col
            for col in ["snapshot_file", "human_label", "human_confidence", "human_note"]
            if col in existing.columns
        ]
        if len(keep_cols) > len(merge_cols):
            existing_small = existing[keep_cols].drop_duplicates(subset=merge_cols)
            annotation_df = annotation_df.merge(existing_small, how="left", on=merge_cols, suffixes=("", "_old"))
            for col in ["snapshot_file", "human_label", "human_confidence", "human_note"]:
                old_col = f"{col}_old"
                if old_col in annotation_df.columns:
                    annotation_df[col] = annotation_df[old_col].where(annotation_df[old_col].notna(), annotation_df[col])
                    annotation_df = annotation_df.drop(columns=[old_col])
    return annotation_df


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing file: {DATA_PATH}")
    if not FINAL_MODEL_ATTEMPTS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {FINAL_MODEL_ATTEMPTS_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    regain_df, scored_df, pool_df = build_pool()
    annotation_df = build_annotation_file(pool_df)

    pool_df.to_csv(POOL_ALL_PATH, index=False)
    annotation_df.to_csv(ANNOTATION_PATH, index=False)

    print(f"Saved focused candidate pool: {POOL_ALL_PATH}")
    print(f"Saved manual annotation file: {ANNOTATION_PATH}")
    print(f"Selected validation matchIds: {VALIDATION_MATCH_IDS}")
    print(f"All regain candidates: {len(regain_df)}")
    print(f"Legacy broad positives: {int(scored_df['legacy_broad_prediction'].sum())}")
    print(f"Current final positives: {int(scored_df['current_final_prediction'].sum())}")
    print(f"Focused pool size: {len(pool_df)}")
    print(f"Manual annotation rows: {len(annotation_df)}")


if __name__ == "__main__":
    main()
