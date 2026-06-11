"""Build the model-ready regain candidate table for counterattack detection.

This script is the preprocessing and feature-engineering step for the XGBoost
models. It converts raw event/tracking data into one row per possession-regain
candidate, assigns rule-based labels, and calculates event/tracking features
over several post-regain time windows.

Main inputs:
- Data/events.csv
- Data/derived/counterattack_attempts_balanced_v2_scaled_tracking_coords.csv
- Tracking CSV files in Data/H_EURO2024, Data/Q_EURO2025 and Data/U21_EURO2025

Main output:
- Data/derived/ai_counterattack_detection/regain_candidates_labeled.csv
"""

import csv
import math
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from tracking_direction import build_tracking_direction_lookup, lookup_attacking_left


# Paths and output locations. The script can now be run from the project folder
# or by giving Python the file path directly.
PROJECT_ROOT = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT / "Data"
DATA_PATH = ROOT / "events.csv"
ATTEMPTS_PATH = ROOT / "derived" / "counterattack_attempts_balanced_v2_scaled_tracking_coords.csv"
DERIVED_DIR = ROOT / "derived" / "ai_counterattack_detection"
DERIVED_DIR.mkdir(parents=True, exist_ok=True)
CANDIDATES_PATH = DERIVED_DIR / "regain_candidates_labeled.csv"

# This prevents stale cached feature tables from being reused silently.
CANDIDATE_CACHE_VERSION = 6

# Event and tracking constants.
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
TRACKING_GOAL_LEFT = (-52.5, 0.0)
TRACKING_GOAL_RIGHT = (52.5, 0.0)
WINDOW_SPECS = [
    ("short", 2.0),
    ("mid", 5.0),
    ("ten", 10.0),
    ("fifteen", 15.0),
    ("long", 20.0),
]
SHORT_WINDOW_S = dict(WINDOW_SPECS)["short"]
MID_WINDOW_S = dict(WINDOW_SPECS)["mid"]
LONG_WINDOW_S = dict(WINDOW_SPECS)["long"]


# Basic parsing helpers.
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


def duel_won(row):
    return (
        (row.get("aerialWon") or "").strip() == "1"
        or (row.get("groundDuelRecoveredPossession") or "").strip() == "1"
        or (row.get("groundDuelKeptPossession") or "").strip() == "1"
    )


def is_regain_event(row):
    event_type = (row.get("typePrimary") or "").strip()
    return event_type == "interception" or (event_type == "duel" and duel_won(row))


# The sequence should only stop when the opponent actually controls an on-ball action.
# For duels, this means the opponent must win or keep possession.
def opponent_on_ball_ends_sequence(row, possessing_team_id):
    event_type = (row.get("typePrimary") or "").strip()
    team_id = (row.get("teamId") or "").strip()
    if not team_id or team_id == possessing_team_id or event_type not in ON_BALL_TYPES:
        return False
    if event_type == "duel":
        return duel_won(row)
    return True


def normalize_label_no_score(label):
    s = (label or "").strip()
    if "," in s:
        s = s.split(",", 1)[0].strip()
    return s


# Tracking file lookup and ball-state helpers.
def build_tracking_label_index(root=ROOT):
    label_to_folder = {}
    for comp in ["H_EURO2024", "Q_EURO2025", "U21_EURO2025"]:
        comp_path = root / comp
        if not comp_path.exists():
            continue
        for folder in sorted(comp_path.iterdir()):
            if not folder.is_dir():
                continue
            home_csv = folder / "home.csv"
            if not home_csv.exists():
                continue
            with home_csv.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                first = next(reader, None)
                if not first:
                    continue
                label = normalize_label_no_score(first.get("label"))
                if label:
                    label_to_folder[label] = folder
    return label_to_folder


def build_ball_in_play_intervals(by_match, label_to_folder, pitch_x_max=52.5, pitch_y_max=34.0):
    out = {}
    for match_id, rows in by_match.items():
        if not rows:
            continue
        label = normalize_label_no_score(rows[0].get("label"))
        folder = label_to_folder.get(label)
        if folder is None:
            continue
        home_csv = folder / "home.csv"
        if not home_csv.exists():
            continue

        intervals = []
        state = None
        start_t = None
        end_t = None
        with home_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = to_float(row.get("total_time_passed"))
                if t is None:
                    continue
                bx = to_float(row.get("ball_x"))
                by = to_float(row.get("ball_y"))
                in_play = bx is not None and by is not None and abs(bx) <= pitch_x_max and abs(by) <= pitch_y_max
                if state is None:
                    state = in_play
                    start_t = t
                    end_t = t
                elif in_play == state:
                    end_t = t
                else:
                    intervals.append((start_t, end_t, state))
                    state = in_play
                    start_t = t
                    end_t = t
        if state is not None:
            intervals.append((start_t, end_t, state))
        if intervals:
            out[match_id] = {"intervals": intervals}
    return out


def compute_in_play_duration_s(ball_state_by_match, match_id, start_ts, end_ts):
    if start_ts is None or end_ts is None or end_ts <= start_ts:
        return 0.0
    data = ball_state_by_match.get(match_id)
    if not data:
        return max(0.0, end_ts - start_ts)
    total = 0.0
    for start_t, end_t, state in data["intervals"]:
        if end_t <= start_ts:
            continue
        if start_t >= end_ts:
            break
        if not state:
            continue
        total += max(0.0, min(end_t, end_ts) - max(start_t, start_ts))
    return total


def build_ball_tracking_lookup(by_match, label_to_folder):
    out = {}
    for match_id, rows in by_match.items():
        if not rows:
            continue
        label = normalize_label_no_score(rows[0].get("label"))
        folder = label_to_folder.get(label)
        if folder is None:
            continue
        home_csv = folder / "home.csv"
        if not home_csv.exists():
            continue

        times, xs, ys = [], [], []
        with home_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = to_float(row.get("total_time_passed"))
                bx = to_float(row.get("ball_x"))
                by = to_float(row.get("ball_y"))
                if t is None or bx is None or by is None:
                    continue
                times.append(t)
                xs.append(bx)
                ys.append(by)
        if times:
            out[match_id] = {"times": times, "xs": xs, "ys": ys}
    return out


def nearest_tracking_ball_position(ball_tracking_by_match, match_id, ts, max_dt_s=0.25):
    data = ball_tracking_by_match.get(match_id)
    if not data or ts is None:
        return None
    times = data["times"]
    xs = data["xs"]
    ys = data["ys"]
    idx = bisect_right(times, ts)
    candidates = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(times):
        candidates.append(idx)
    if not candidates:
        return None
    best_idx = min(candidates, key=lambda i: abs(times[i] - ts))
    dt = abs(times[best_idx] - ts)
    if dt > max_dt_s:
        return None
    return {"t": times[best_idx], "x": xs[best_idx], "y": ys[best_idx], "dt": dt}


# Tracking geometry and sequence features.
def get_tracking_ball_window(ball_tracking_by_match, match_id, start_ts, end_ts):
    data = ball_tracking_by_match.get(match_id)
    if not data:
        return []
    times = data["times"]
    xs = data["xs"]
    ys = data["ys"]
    lo = bisect_left(times, start_ts)
    hi = bisect_right(times, end_ts)
    return list(zip(times[lo:hi], xs[lo:hi], ys[lo:hi]))


def goal_distance_tracking(x, y, goal_xy):
    return math.hypot(x - goal_xy[0], y - goal_xy[1])


def forward_progress_m(x, attacking_left):
    return (52.5 - x) if attacking_left else (x + 52.5)


def tracking_start_zone(progress_m):
    if progress_m is None:
        return "unknown"
    if progress_m < 35.0:
        return "defensive_third"
    if progress_m < 70.0:
        return "middle_third"
    return "attacking_third"


def infer_attacking_left(track_points):
    if len(track_points) < 2:
        return None
    left_d = [goal_distance_tracking(x, y, TRACKING_GOAL_LEFT) for _, x, y in track_points]
    right_d = [goal_distance_tracking(x, y, TRACKING_GOAL_RIGHT) for _, x, y in track_points]
    left_prog = left_d[0] - min(left_d)
    right_prog = right_d[0] - min(right_d)
    if abs(left_prog - right_prog) < 1e-6:
        dx = track_points[-1][1] - track_points[0][1]
        return dx < 0
    return left_prog > right_prog


def infer_attacking_left_from_start(ball_tracking_by_match, match_id, ts, lookahead_s=2.0):
    points = get_tracking_ball_window(ball_tracking_by_match, match_id, ts, ts + lookahead_s)
    if len(points) < 2:
        return None
    return infer_attacking_left(points)


def compute_tracking_features(track_points, attacking_left):
    if len(track_points) < 2 or attacking_left is None:
        return None
    goal_xy = TRACKING_GOAL_LEFT if attacking_left else TRACKING_GOAL_RIGHT
    start_x_raw = track_points[0][1]
    start_y_raw = track_points[0][2]
    start_progress_m = forward_progress_m(start_x_raw, attacking_left)
    progress_vals = [forward_progress_m(x, attacking_left) for _, x, _ in track_points]
    goal_dists = [goal_distance_tracking(x, y, goal_xy) for _, x, y in track_points]
    progression_goal_m = max(0.0, goal_dists[0] - min(goal_dists))
    progression_x = max(0.0, max(progress_vals) - progress_vals[0])
    directness_goal = None
    sum_abs_goal_delta = sum(abs(next_d - prev_d) for prev_d, next_d in zip(goal_dists[:-1], goal_dists[1:]))
    if sum_abs_goal_delta > 0:
        directness_goal = progression_goal_m / sum_abs_goal_delta
    duration_s = max(0.0, track_points[-1][0] - track_points[0][0])
    avg_progress_speed_mps = None
    if duration_s > 0:
        avg_progress_speed_mps = progression_goal_m / duration_s
    return {
        "start_progress_m": start_progress_m,
        "start_zone": tracking_start_zone(start_progress_m),
        "start_goal_distance_m": goal_dists[0],
        "min_goal_distance_m": min(goal_dists),
        "end_progress_m": progress_vals[-1],
        "progression_goal_m": progression_goal_m,
        "progression_x_m": progression_x,
        "directness_goal": directness_goal,
        "avg_progress_speed_mps": avg_progress_speed_mps,
        "tracking_start_x_m_raw": start_x_raw,
        "tracking_start_y_m_raw": start_y_raw,
    }


# Event loading and rule-label matching.
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
            by_match[match_id].append(row)
    for match_id in by_match:
        by_match[match_id].sort(key=lambda r: ((r["_ts"] if r["_ts"] is not None else 10**18), r["_event_id"]))
    return by_match


def load_positive_keys(path=ATTEMPTS_PATH):
    df = pd.read_csv(path)
    keys = set()
    for row in df.itertuples(index=False):
        # Positive labels are inherited from the final rule-based definition by exact candidate identity.
        keys.add((str(row.match_id), str(row.team_id), int(row.start_event_id)))
    return keys


def cached_positive_keys(candidates_df):
    if candidates_df.empty:
        return set()
    pos_df = candidates_df[pd.to_numeric(candidates_df["label_counterattack"], errors="coerce").fillna(0).astype(int) == 1]
    keys = set()
    for row in pos_df.itertuples(index=False):
        keys.add((str(row.match_id), str(row.team_id), int(row.start_event_id)))
    return keys


def candidate_keys(candidates_df):
    if candidates_df.empty:
        return set()
    keys = set()
    for row in candidates_df.itertuples(index=False):
        keys.add((str(row.match_id), str(row.team_id), int(row.start_event_id)))
    return keys


# Candidate window and feature construction.
def build_candidate_window(rows, start_idx, team_id, start_ts):
    # The model sees features from several post-regain windows.
    # A window can end earlier if the opponent wins the ball or play stops.
    end_times = {prefix: start_ts for prefix, _ in WINDOW_SPECS}
    terminal_end = start_ts + LONG_WINDOW_S
    stop_reason = "window_end"
    for ev in rows[start_idx + 1:]:
        ts = ev["_ts"]
        if ts is None:
            continue
        event_type = (ev.get("typePrimary") or "").strip()
        if ts - start_ts > LONG_WINDOW_S:
            terminal_end = start_ts + LONG_WINDOW_S
            break
        if event_type == "game_interruption":
            stop_reason = "game_interruption"
            terminal_end = ts
            break
        if event_type in STOP_RESTART_TYPES:
            stop_reason = event_type
            terminal_end = ts
            break
        if opponent_on_ball_ends_sequence(ev, team_id):
            # Main possession-loss stop condition.
            stop_reason = "opponent_on_ball"
            terminal_end = ts
            break
        terminal_end = ts
        delta = ts - start_ts
        for prefix, window_s in WINDOW_SPECS:
            if delta <= window_s:
                end_times[prefix] = ts

    for prefix, window_s in WINDOW_SPECS:
        if end_times[prefix] == start_ts:
            end_times[prefix] = min(terminal_end, start_ts + window_s)
    return end_times, stop_reason


def event_counts_in_window(rows, start_idx, end_ts, team_id):
    counts = Counter()
    first_action_type = None
    first_action_time_s = None
    for ev in rows[start_idx + 1:]:
        ts = ev["_ts"]
        if ts is None or ts > end_ts:
            break
        if (ev.get("teamId") or "").strip() != team_id:
            continue
        event_type = (ev.get("typePrimary") or "").strip()
        if event_type in ON_BALL_TYPES:
            counts[event_type] += 1
            if first_action_type is None:
                first_action_type = event_type
                first_action_time_s = max(0.0, ts - rows[start_idx]["_ts"])
    return counts, first_action_type, first_action_time_s


def build_regain_candidates(by_match_events, ball_state_by_match, ball_tracking_by_match, positive_keys, tracking_direction_by_match):
    rows_out = []
    regain_type_counter = Counter()
    label_counter = Counter()

    for match_no, (match_id, rows) in enumerate(by_match_events.items(), start=1):
        prev_control_team = None
        for i, ev in enumerate(rows):
            event_type = (ev.get("typePrimary") or "").strip()
            team_id = (ev.get("teamId") or "").strip()
            ts = ev["_ts"]
            if event_type not in ON_BALL_TYPES or not team_id:
                continue

            is_candidate = (
                is_regain_event(ev)
                and prev_control_team is not None
                and team_id != prev_control_team
                and ts is not None
            )
            if not is_candidate:
                if event_type != "duel" or duel_won(ev):
                    prev_control_team = team_id
                continue

            start_ball = nearest_tracking_ball_position(ball_tracking_by_match, match_id, ts)
            # Prefer the fixed half-level attacking direction inferred from player starting positions.
            # The short ball-trajectory fallback is only used when the lookup is unavailable.
            attacking_left = lookup_attacking_left(
                tracking_direction_by_match,
                match_id,
                team_id,
                (ev.get("matchPeriod") or "").strip(),
            )
            if attacking_left is None:
                attacking_left = infer_attacking_left_from_start(ball_tracking_by_match, match_id, ts)
            if start_ball is None or attacking_left is None:
                if event_type != "duel" or duel_won(ev):
                    prev_control_team = team_id
                continue

            end_times, stop_reason = build_candidate_window(rows, i, team_id, ts)
            window_points = {
                prefix: get_tracking_ball_window(ball_tracking_by_match, match_id, ts, end_times[prefix])
                for prefix, _ in WINDOW_SPECS
            }
            window_feats = {
                prefix: compute_tracking_features(points, attacking_left)
                for prefix, points in window_points.items()
            }
            short_feats = window_feats["short"]
            mid_feats = window_feats["mid"]
            long_feats = window_feats["long"]
            if mid_feats is None or long_feats is None:
                if event_type != "duel" or duel_won(ev):
                    prev_control_team = team_id
                continue

            counts_by_window = {}
            first_action_type = None
            first_action_time_s = None
            for prefix, _ in WINDOW_SPECS:
                counts, action_type, action_time_s = event_counts_in_window(rows, i, end_times[prefix], team_id)
                counts_by_window[prefix] = counts
                if prefix == "mid":
                    # The 5s window is used for the immediate first action after the regain.
                    first_action_type = action_type
                    first_action_time_s = action_time_s

            # Non-matching regain candidates become negative examples for the ML model.
            label = int((match_id, team_id, ev["_event_id"]) in positive_keys)
            regain_type = "duel_regain" if event_type == "duel" else event_type
            regain_type_counter[regain_type] += 1
            label_counter[label] += 1

            row_out = {
                "candidate_cache_version": CANDIDATE_CACHE_VERSION,
                "match_id": match_id,
                "team_id": team_id,
                "start_event_id": ev["_event_id"],
                "label_counterattack": label,
                "regain_type": regain_type,
                "match_period": (ev.get("matchPeriod") or "").strip(),
                "start_ts": ts,
                "attacking_left": int(attacking_left),
                "stop_reason": stop_reason,
                "first_action_type": first_action_type or "none",
                "first_action_time_s": first_action_time_s,
                "tracking_start_dt_s": start_ball["dt"],
                "source_label": normalize_label_no_score(ev.get("label")),
            }

            for prefix, window_s in WINDOW_SPECS:
                counts = counts_by_window[prefix]
                row_out.update({
                    f"{prefix}_duration_s": max(0.0, end_times[prefix] - ts),
                    f"in_play_{int(window_s)}s": compute_in_play_duration_s(ball_state_by_match, match_id, ts, end_times[prefix]),
                    f"event_count_pass_{int(window_s)}s": counts.get("pass", 0),
                    f"event_count_touch_{int(window_s)}s": counts.get("touch", 0),
                    f"event_count_duel_{int(window_s)}s": counts.get("duel", 0),
                    f"event_count_clearance_{int(window_s)}s": counts.get("clearance", 0),
                    f"event_count_shot_{int(window_s)}s": counts.get("shot", 0),
                })

            for prefix, feats in window_feats.items():
                if feats is None:
                    continue
                row_out.update({
                    f"{prefix}_start_progress_m": feats["start_progress_m"],
                    f"{prefix}_start_zone": feats["start_zone"],
                    f"{prefix}_start_goal_distance_m": feats["start_goal_distance_m"],
                    f"{prefix}_min_goal_distance_m": feats["min_goal_distance_m"],
                    f"{prefix}_end_progress_m": feats["end_progress_m"],
                    f"{prefix}_progression_goal_m": feats["progression_goal_m"],
                    f"{prefix}_progression_x_m": feats["progression_x_m"],
                    f"{prefix}_directness_goal": feats["directness_goal"],
                    f"{prefix}_avg_progress_speed_mps": feats["avg_progress_speed_mps"],
                })

            row_out["tracking_start_x_m_raw"] = long_feats["tracking_start_x_m_raw"]
            row_out["tracking_start_y_m_raw"] = long_feats["tracking_start_y_m_raw"]
            rows_out.append(row_out)

            if event_type != "duel" or duel_won(ev):
                prev_control_team = team_id

        print(f"Processed match {match_no}/{len(by_match_events)}")

    candidates_df = pd.DataFrame(rows_out)
    return candidates_df, regain_type_counter, label_counter


def summarize_candidates(candidates_df, regain_type_counter=None, label_counter=None):
    if regain_type_counter is None:
        regain_type_counter = Counter(candidates_df["regain_type"].fillna("unknown"))
    if label_counter is None:
        label_counter = Counter(pd.to_numeric(candidates_df["label_counterattack"], errors="coerce").fillna(0).astype(int))
    return {
        "candidate_rows": int(len(candidates_df)),
        "positive_counterattacks": int(label_counter[1]),
        "negative_counterattacks": int(label_counter[0]),
        "positive_rate": float(label_counter[1] / len(candidates_df)) if len(candidates_df) else 0.0,
        "regain_type_counts": dict(regain_type_counter),
        "candidate_definition_note": "Candidates are all open-play regains with possession change: interceptions and possession-winning duels.",
        "label_definition_note": "Positive labels come from the saved final rule-based counterattack definition in counterattack_attempts_balanced_v2_scaled_tracking_coords.csv.",
    }


# Candidate cache and script entrypoint.
def load_or_build_candidates(force_rebuild=False):
    print("Loading events")
    by_match_events = load_events()
    print(f"Loaded matches: {len(by_match_events)}")
    positive_keys = load_positive_keys()
    print(f"Loaded final positive counterattacks: {len(positive_keys)}")

    rebuild_candidates = force_rebuild
    candidates_df = None
    regain_type_counter = None
    label_counter = None

    if CANDIDATES_PATH.exists() and not force_rebuild:
        candidates_df = pd.read_csv(CANDIDATES_PATH)
        cache_version_ok = (
            "candidate_cache_version" in candidates_df.columns
            and pd.to_numeric(candidates_df["candidate_cache_version"], errors="coerce").fillna(-1).eq(CANDIDATE_CACHE_VERSION).all()
        )
        cached_keys = cached_positive_keys(candidates_df)
        candidate_key_set = candidate_keys(candidates_df)
        current_positive_keys_in_candidate_space = positive_keys & candidate_key_set
        # The cache is only reused if both the feature schema and current positive labels still match.
        if cache_version_ok and cached_keys == current_positive_keys_in_candidate_space:
            regain_type_counter = Counter(candidates_df["regain_type"].fillna("unknown"))
            label_counter = Counter(pd.to_numeric(candidates_df["label_counterattack"], errors="coerce").fillna(0).astype(int))
            print(f"Loaded cached candidate regain rows: {len(candidates_df)}")
        else:
            print(
                "Cached candidates are stale; rebuilding "
                f"(cache_version_ok={cache_version_ok}, cached positives={len(cached_keys)}, "
                f"current positives in candidate space={len(current_positive_keys_in_candidate_space)})"
            )
            rebuild_candidates = True
    else:
        rebuild_candidates = True

    if rebuild_candidates:
        label_to_tracking_folder = build_tracking_label_index()
        print("Building tracking lookups")
        ball_state_by_match = build_ball_in_play_intervals(by_match_events, label_to_tracking_folder)
        ball_tracking_by_match = build_ball_tracking_lookup(by_match_events, label_to_tracking_folder)
        tracking_direction_by_match = build_tracking_direction_lookup(by_match_events, label_to_tracking_folder)
        candidates_df, regain_type_counter, label_counter = build_regain_candidates(
            by_match_events,
            ball_state_by_match,
            ball_tracking_by_match,
            positive_keys,
            tracking_direction_by_match,
        )
        print(f"Built candidate regain rows: {len(candidates_df)}")
        candidates_df.to_csv(CANDIDATES_PATH, index=False)

    summary = summarize_candidates(candidates_df, regain_type_counter, label_counter)
    return candidates_df, summary


def main():
    _, summary = load_or_build_candidates()
    print("Saved:", CANDIDATES_PATH)
    print(summary)


if __name__ == "__main__":
    main()
