#!/usr/bin/env python3
"""Infer each team's attacking direction from the tracking data.

The event data does not always provide a reliable attacking direction. This
module fixes that by looking at where the home and away players stand at the
start of each half in the tracking files:

- the team with the lower average x-position starts on the left side,
- a team starting on the left attacks to the right,
- a team starting on the right attacks to the left.

The lookup is built per match and per match period, because teams switch sides
between halves.
"""

import csv
from pathlib import Path


# Mapping from event-data period names to tracking-data half ids.
PERIOD_TO_HALF = {
    "1H": "1",
    "2H": "2",
    "1E": "3",
    "2E": "4",
}


def _to_float(value):
    s = (value or "").strip()
    if not s or s == "NULL":
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _normalize_label_no_score(label):
    s = (label or "").strip()
    if "," in s:
        s = s.split(",", 1)[0].strip()
    return s


def _normalized_name(name):
    s = " ".join((name or "").strip().split()).casefold()
    s = s.replace("under 21", "u21")
    s = s.replace("under-21", "u21")
    s = s.replace("u 21", "u21")
    s = s.replace("-", " ")
    s = " ".join(s.split())
    return s


def _mean_team_x(row, prefix):
    """Mean x-position for all available players from one team in a tracking row."""
    xs = []
    for key, value in row.items():
        if not key.startswith(prefix + "_") or not key.endswith("_x"):
            continue
        x = _to_float(value)
        if x is not None:
            xs.append(x)
    if not xs:
        return None
    return sum(xs) / len(xs)


def _first_valid_half_row(track_csv: Path, half_id: str, prefix: str):
    """Find the first row in a half where team x-positions are available."""
    with track_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("half") or "").strip() != half_id:
                continue
            mean_x = _mean_team_x(row, prefix)
            if mean_x is None:
                continue
            return row
    return None


def _resolve_team_ids(rows, label_no_score):
    """Match home/away names from the label to team IDs in the event rows."""
    if " - " in label_no_score:
        home_name, away_name = [part.strip() for part in label_no_score.split(" - ", 1)]
    elif "-" in label_no_score:
        home_name, away_name = [part.strip() for part in label_no_score.split("-", 1)]
    else:
        return None, None

    name_to_id = {}
    for row in rows:
        team_name = (row.get("teamName") or "").strip()
        team_id = (row.get("teamId") or "").strip()
        if team_name and team_id and team_name not in name_to_id:
            name_to_id[team_name] = team_id

    home_norm = _normalized_name(home_name)
    away_norm = _normalized_name(away_name)
    home_id = None
    away_id = None
    for team_name, team_id in name_to_id.items():
        team_norm = _normalized_name(team_name)
        if team_norm == home_norm:
            home_id = team_id
        elif team_norm == away_norm:
            away_id = team_id
    return home_id, away_id


def build_tracking_direction_lookup(by_match, label_to_folder):
    """Build {(match_id): {(team_id, period): attacking_left}} lookup."""
    lookup = {}
    for match_id, rows in by_match.items():
        if not rows:
            continue

        label_no_score = _normalize_label_no_score(rows[0].get("label"))
        folder = label_to_folder.get(label_no_score)
        if folder is None:
            continue

        home_id, away_id = _resolve_team_ids(rows, label_no_score)
        if not home_id or not away_id:
            continue

        home_csv = folder / "home.csv"
        away_csv = folder / "away.csv"
        if not home_csv.exists() or not away_csv.exists():
            continue

        period_map = {}
        for match_period, half_id in PERIOD_TO_HALF.items():
            # Use the first valid tracking row in each period. This captures
            # which side each team starts on after the half-time side switch.
            home_row = _first_valid_half_row(home_csv, half_id, "home")
            away_row = _first_valid_half_row(away_csv, half_id, "away")
            if home_row is None or away_row is None:
                continue

            home_mean_x = _mean_team_x(home_row, "home")
            away_mean_x = _mean_team_x(away_row, "away")
            if home_mean_x is None or away_mean_x is None:
                continue

            home_starts_left = home_mean_x < away_mean_x
            # attacking_left means "this team attacks toward negative x".
            # Therefore a team that starts left attacks right, so False.
            period_map[(home_id, match_period)] = not home_starts_left
            period_map[(away_id, match_period)] = home_starts_left

        if period_map:
            lookup[str(match_id)] = period_map
    return lookup


def lookup_attacking_left(direction_lookup, match_id, team_id, match_period):
    """Return attacking_left for one team/period, or None if no lookup exists."""
    return (
        direction_lookup.get(str(match_id), {})
        .get((str(team_id), (match_period or "").strip()))
    )
