#!/usr/bin/env python3
"""Generate validation snapshots for manually checking counterattack candidates.

The script combines event data and tracking data to create one overview image
per validation candidate. Each image shows the pitch, ball trajectory, player
trails, pass markers and simple speed summaries. The output is used for manual
inspection and for matching the 48 validation clips back to their candidates.

This file contains the full plotting code. The model and rule logic are not trained here, the script is
only a visualization and validation helper.
"""

import csv
import math
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Input/output paths. These are anchored to the project folder so the script is
# not dependent on the current terminal directory.
PROJECT_ROOT = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT / "Data"
POOL_CSV = ROOT / "derived" / "validation_legacy_broad_pool" / "focused_candidate_pool_all.csv"
ANNOTATION_CSV = ROOT / "derived" / "validation_legacy_broad_pool" / "focused_candidate_pool_manual_annotation.csv"
EVENTS_CSV = ROOT / "events.csv"
OUT_DIR = ROOT / "derived" / "validation_legacy_broad_pool" / "candidate_snapshots"

# Visualization settings
MAX_CONTEXT_EXTRA_S = 3.0
MAX_PLAYER_SPEED_KMH = 40.0
TRACKING_GOAL_LEFT = (-52.5, 0.0)
TRACKING_GOAL_RIGHT = (52.5, 0.0)

# Event types used when reconstructing a candidate's visual window.
STOP_RESTART_TYPES = {"throw_in", "goal_kick", "corner"}
SNAPSHOT_RECONSTRUCTION_BASE = {
    "reference_goal_distance_m": 52.5,
    "reference_max_duration_s": 17.0,
    "reference_max_passes": 5,
    "min_duration_s": 3.0,
    "min_required_speed_kmh": 13.5,
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


def read_validation_candidates(annotation_path: Path, pool_path: Path):
    """Load validation candidates and merge annotation rows with pool metadata."""
    def read_rows(path: Path):
        with path.open(newline="", encoding="utf-8") as f:
            sample = f.read(4096)
            f.seek(0)
            lines = f.readlines()

        if lines and "," not in lines[0] and ";" not in lines[0]:
            lines = lines[1:]

        text = "".join(lines)
        if not text.strip():
            return []

        try:
            dialect = csv.Sniffer().sniff(sample if sample.strip() else text, delimiters=",;")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ";" if ";" in text and text.count(";") >= text.count(",") else ","

        return list(csv.DictReader(text.splitlines(), delimiter=delimiter))

    by_key = {}
    for row in read_rows(pool_path):
        key = (
            (row.get("match_id") or "").strip(),
            (row.get("team_id") or "").strip(),
            (row.get("start_event_id") or "").strip(),
        )
        by_key[key] = row

    out = []
    for row in read_rows(annotation_path):
        key = (
            (row.get("match_id") or "").strip(),
            (row.get("team_id") or "").strip(),
            (row.get("start_event_id") or "").strip(),
        )
        if key not in by_key:
            continue
        merged = dict(by_key[key])
        merged.update(row)
        out.append(merged)

    if not out:
        raise RuntimeError("No validation candidates found in annotation/pool files.")
    return out


def find_event_meta(match_id: str, event_id: str):
    with EVENTS_CSV.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("matchId") or "").strip() == match_id and (row.get("eventId") or "").strip() == event_id:
                label = (row.get("label") or "").strip()
                team_name = (row.get("teamName") or "").strip()
                if "," in label:
                    label = label.split(",", 1)[0].strip()
                return label, team_name
    raise RuntimeError("Could not find matching event metadata.")


def normalize_label_no_score(label: str):
    s = (label or "").strip()
    if "," in s:
        s = s.split(",", 1)[0].strip()
    return s


def normalize_team_name_for_side(name: str):
    s = (name or "").strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.replace("under 21", "u21")
    s = s.replace("under-21", "u21")
    s = s.replace("u 21", "u21")
    s = s.replace("turkiye", "turkey")
    s = s.replace("-", " ")
    s = " ".join(s.split())
    return s


def split_home_away(label_no_score: str):
    parts = [p.strip() for p in (label_no_score or "").split("-")]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return (label_no_score or "").strip(), "Away"


def infer_attack_is_home(label_no_score: str, team_name: str):
    """Return whether the attacking team is listed as home in the match label."""
    home_team, away_team = split_home_away(label_no_score)
    team_norm = normalize_team_name_for_side(team_name)
    home_norm = normalize_team_name_for_side(home_team)
    away_norm = normalize_team_name_for_side(away_team)
    if team_norm == home_norm:
        return True, home_team, away_team
    if team_norm == away_norm:
        return False, home_team, away_team
    # Fall back to substring matching for minor naming variations.
    if team_norm and (team_norm in home_norm or home_norm in team_norm):
        return True, home_team, away_team
    if team_norm and (team_norm in away_norm or away_norm in team_norm):
        return False, home_team, away_team
    raise RuntimeError(
        f"Could not infer home/away side from label '{label_no_score}' and team '{team_name}'"
    )


def period_to_half(match_period: str):
    """Map event periods to tracking half IDs."""
    mapping = {"1H": "1", "2H": "2", "1E": "3", "2E": "4"}
    return mapping.get((match_period or "").strip())


def period_start_global_s(match_period: str):
    mapping = {"1H": 0.0, "2H": 2700.0, "1E": 5400.0, "2E": 6300.0}
    return mapping.get((match_period or "").strip())


def to_period_local_s(global_ts: float, match_period: str):
    """Convert global event seconds to tracking seconds within the current half."""
    start = period_start_global_s(match_period)
    if global_ts is None or start is None:
        return None
    return global_ts - start


def load_events_by_match():
    """Load events once and group them by match for faster lookup."""
    by_match = {}
    with EVENTS_CSV.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            match_id = (row.get("matchId") or "").strip()
            if not match_id:
                continue
            row = dict(row)
            row["_ts"] = ts_to_seconds(row.get("matchTimestamp"))
            row["_event_id"] = int((row.get("eventId") or "0").strip() or 0)
            by_match.setdefault(match_id, []).append(row)
    for match_id in by_match:
        by_match[match_id].sort(key=lambda row: ((row["_ts"] if row["_ts"] is not None else 10**18), row["_event_id"]))
    return by_match


def goal_distance_tracking(x, y, goal_xy):
    return math.hypot(x - goal_xy[0], y - goal_xy[1])


def infer_attacking_left(track_points):
    """Infer attacking direction from early ball movement toward either goal."""
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


def infer_attacking_left_from_start(ball_track, ts, lookahead_s=2.0):
    pts = [p for p in ball_track if ts <= p[0] <= ts + lookahead_s]
    if len(pts) < 2:
        return None
    return infer_attacking_left(pts)


def scaled_candidate_requirements(start_goal_distance_m):
    """Approximate the final rule window when drawing legacy validation cases."""
    distance_factor = start_goal_distance_m / SNAPSHOT_RECONSTRUCTION_BASE["reference_goal_distance_m"]
    scaled_max_duration_s = SNAPSHOT_RECONSTRUCTION_BASE["reference_max_duration_s"] * distance_factor
    scaled_max_duration_s = max(SNAPSHOT_RECONSTRUCTION_BASE["min_duration_s"], scaled_max_duration_s)
    scaled_max_passes = max(1, int(math.ceil(SNAPSHOT_RECONSTRUCTION_BASE["reference_max_passes"] * distance_factor)))
    required_speed_mps = start_goal_distance_m / scaled_max_duration_s
    min_required_speed_mps = SNAPSHOT_RECONSTRUCTION_BASE["min_required_speed_kmh"] / 3.6
    required_speed_mps = max(required_speed_mps, min_required_speed_mps)
    return {
        "max_duration_s": scaled_max_duration_s,
        "max_passes": scaled_max_passes,
        "required_speed_mps": required_speed_mps,
    }


def reconstruct_final_candidate_window(match_id: str, start_event_id: str):
    """Reconstruct the model part of the candidate window for visualization.
    """
    rows = EVENTS_BY_MATCH.get(match_id)
    if not rows:
        return None

    start_idx = None
    start_row = None
    for i, row in enumerate(rows):
        if str(row["_event_id"]) == start_event_id:
            start_idx = i
            start_row = row
            break
    if start_row is None:
        return None

    start_ts = start_row["_ts"]
    team_id = (start_row.get("teamId") or "").strip()
    if start_ts is None or not team_id:
        return None

    label_no_score = normalize_label_no_score(start_row.get("label"))
    folder = find_tracking_folder_by_label(label_no_score)
    home_csv = folder / "home.csv"
    ball_track = read_ball_track(home_csv, start_ts, start_ts + 40.0, sample_every_n=1)
    if not ball_track:
        return None

    start_ball = min(ball_track, key=lambda p: abs(p[0] - start_ts))
    if abs(start_ball[0] - start_ts) > 0.25:
        return None

    attacking_left = infer_attacking_left_from_start(ball_track, start_ts)
    if attacking_left is None:
        return None

    goal_xy = TRACKING_GOAL_LEFT if attacking_left else TRACKING_GOAL_RIGHT
    start_goal_distance_m = goal_distance_tracking(start_ball[1], start_ball[2], goal_xy)
    scaled = scaled_candidate_requirements(start_goal_distance_m)

    pass_count = 0
    end_ts = start_ts
    for ev2 in rows[start_idx + 1:]:
        t2 = (ev2.get("typePrimary") or "").strip()
        team2 = (ev2.get("teamId") or "").strip()
        ts2 = ev2["_ts"]
        if ts2 is None:
            continue
        if ts2 - start_ts > scaled["max_duration_s"]:
            break
        if t2 == "game_interruption":
            break
        if t2 in STOP_RESTART_TYPES:
            break
        if t2 in ON_BALL_TYPES and team2 and team2 != team_id:
            break
        if team2 == team_id:
            end_ts = ts2
            if t2 == "pass":
                pass_count += 1
            if pass_count > scaled["max_passes"]:
                break

    return {
        "end_ts": end_ts,
        "duration_s": max(0.0, end_ts - start_ts),
    }


def candidate_model_end_ts(row):
    """Return model end time, falling back to stored durations if reconstruction fails."""
    match_id = (row.get("match_id") or "").strip()
    start_event_id = (row.get("start_event_id") or "").strip()
    start_ts = float((row.get("start_ts") or "0").strip())

    reconstructed = reconstruct_final_candidate_window(match_id, start_event_id)
    if reconstructed is not None:
        return reconstructed["end_ts"], reconstructed["duration_s"]

    duration_candidates = [
        to_float(row.get("legacy_broad_duration_s")),
        to_float(row.get("current_final_duration_s")),
    ]
    duration_candidates = [d for d in duration_candidates if d is not None and d > 0]
    duration_s = max(duration_candidates) if duration_candidates else 4.0
    return start_ts + duration_s, duration_s


def find_tracking_folder_by_label(label_no_score: str):
    """Find the tracking folder whose label matches the score-free match label."""
    for comp in ["H_EURO2024", "Q_EURO2025", "U21_EURO2025"]:
        comp_path = ROOT / comp
        if not comp_path.exists():
            continue
        for d in sorted(comp_path.iterdir()):
            if not d.is_dir():
                continue
            home = d / "home.csv"
            if not home.exists():
                continue
            with home.open(newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                first = next(r, None)
                if not first:
                    continue
                lbl = normalize_label_no_score(first.get("label"))
                if lbl == label_no_score:
                    return d
    raise RuntimeError(f"Could not match tracking folder for label: {label_no_score}")


def to_float(v):
    s = (v or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def ts_to_seconds(ts):
    s = (ts or "").strip()
    if not s:
        return None
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def read_pass_times(match_id, team_id, start_ts, end_ts):
    """Read pass timestamps for the attacking team inside the visual window."""
    out = []
    with EVENTS_CSV.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("matchId") or "").strip() != match_id:
                continue
            if (row.get("teamId") or "").strip() != team_id:
                continue
            if (row.get("typePrimary") or "").strip() != "pass":
                continue

            t_str = (row.get("matchTimestamp") or "").strip()
            if not t_str:
                continue
            try:
                h, m, sec = t_str.split(":")
                t = int(h) * 3600 + int(m) * 60 + float(sec)
            except ValueError:
                continue
            if t < start_ts or t > end_ts:
                continue
            out.append(t)
    return out


def find_next_on_ball_event_ts(match_id, end_ts):
    next_ts = None
    with EVENTS_CSV.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("matchId") or "").strip() != match_id:
                continue
            ttype = (row.get("typePrimary") or "").strip()
            if ttype not in ON_BALL_TYPES:
                continue
            t_str = (row.get("matchTimestamp") or "").strip()
            if not t_str:
                continue
            try:
                h, m, sec = t_str.split(":")
                t = int(h) * 3600 + int(m) * 60 + float(sec)
            except ValueError:
                continue
            if t <= end_ts:
                continue
            if next_ts is None or t < next_ts:
                next_ts = t
    return next_ts


def find_next_on_ball_event_ts_same_period(match_id, end_ts, match_period):
    """Find the next on-ball event in the same period, used as optional context."""
    next_ts = None
    with EVENTS_CSV.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("matchId") or "").strip() != match_id:
                continue
            if (row.get("matchPeriod") or "").strip() != match_period:
                continue
            ttype = (row.get("typePrimary") or "").strip()
            if ttype not in ON_BALL_TYPES:
                continue
            t = ts_to_seconds(row.get("matchTimestamp"))
            if t is None or t <= end_ts:
                continue
            if next_ts is None or t < next_ts:
                next_ts = t
    return next_ts


def nearest_rows(track_csv: Path, times):
    target = list(times)
    best = [{"dist": float("inf"), "row": None} for _ in target]
    with track_csv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            t = to_float(row.get("total_time_passed"))
            if t is None:
                continue
            for i, tt in enumerate(target):
                d = abs(t - tt)
                if d < best[i]["dist"]:
                    best[i] = {"dist": d, "row": row}
    return [b["row"] for b in best]


def nearest_rows_in_period(track_csv: Path, half_id: str, times):
    """Find tracking rows closest to selected local-half timestamps."""
    target = list(times)
    best = [{"dist": float("inf"), "row": None} for _ in target]
    with track_csv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("half") or "").strip() != half_id:
                continue
            t = to_float(row.get("time_passed_in_half"))
            if t is None:
                continue
            for i, tt in enumerate(target):
                d = abs(t - tt)
                if d < best[i]["dist"]:
                    best[i] = {"dist": d, "row": row}
    return [b["row"] for b in best]


def read_ball_track(track_csv: Path, start_ts: float, end_ts: float, sample_every_n=2):
    """Read continuous ball trajectory from tracking between start/end times."""
    points = []
    with track_csv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        idx = 0
        for row in r:
            t = to_float(row.get("total_time_passed"))
            if t is None or t < start_ts or t > end_ts:
                continue
            if idx % sample_every_n != 0:
                idx += 1
                continue
            bx = to_float(row.get("ball_x"))
            by = to_float(row.get("ball_y"))
            if bx is not None and by is not None:
                points.append((t, bx, by))
            idx += 1
    return points


def read_ball_track_in_period(track_csv: Path, half_id: str, start_ts: float, end_ts: float, sample_every_n=2):
    """Read ball trajectory from tracking using local-half timestamps."""
    points = []
    with track_csv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        idx = 0
        for row in r:
            if (row.get("half") or "").strip() != half_id:
                continue
            t = to_float(row.get("time_passed_in_half"))
            if t is None or t < start_ts or t > end_ts:
                continue
            if idx % sample_every_n != 0:
                idx += 1
                continue
            bx = to_float(row.get("ball_x"))
            by = to_float(row.get("ball_y"))
            if bx is not None and by is not None:
                points.append((t, bx, by))
            idx += 1
    return points


def split_track_by_time(ball_track, split_ts):
    if not ball_track:
        return [], []
    model = [p for p in ball_track if p[0] <= split_ts]
    context = [p for p in ball_track if p[0] > split_ts]
    if model and context and model[-1][0] != context[0][0]:
        context = [model[-1]] + context
    return model, context


def read_rows_in_window(track_csv: Path, start_ts: float, end_ts: float):
    rows = []
    with track_csv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            t = to_float(row.get("total_time_passed"))
            if t is None or t < start_ts or t > end_ts:
                continue
            row = dict(row)
            row["_t"] = t
            rows.append(row)
    return rows


def read_rows_in_period(track_csv: Path, half_id: str, start_ts: float, end_ts: float):
    rows = []
    with track_csv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("half") or "").strip() != half_id:
                continue
            t = to_float(row.get("time_passed_in_half"))
            if t is None or t < start_ts or t > end_ts:
                continue
            row = dict(row)
            row["_t"] = t
            rows.append(row)
    return rows


def parse_team_points(row, prefix):
    xs = []
    ys = []
    for k, v in row.items():
        if not k.startswith(prefix + "_"):
            continue
        if not k.endswith("_x"):
            continue
        base = k[:-2]
        x = to_float(v)
        y = to_float(row.get(base + "_y"))
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys


def parse_team_map(row, prefix):
    """Return {shirt_number: (x, y)} for one tracking row."""
    players = {}
    for k, v in row.items():
        if not k.startswith(prefix + "_") or not k.endswith("_x"):
            continue
        shirt = k[len(prefix) + 1 : -2]
        base = k[:-2]
        x = to_float(v)
        y = to_float(row.get(base + "_y"))
        if x is None or y is None:
            continue
        players[shirt] = (x, y)
    return players


def shirt_sort_key(shirt):
    s = str(shirt)
    try:
        return (0, int(s))
    except ValueError:
        return (1, s)


def compute_player_speed_stats(rows, prefix):
    """Calculate simple average and max speed per player for the visual side panels."""
    by_shirt_samples = {}
    prev_t = None
    prev_map = None
    for row in rows:
        t = row.get("_t")
        if t is None:
            continue
        curr_map = parse_team_map(row, prefix)
        if prev_t is not None and prev_map is not None:
            dt = t - prev_t
            if dt > 0:
                for shirt, (x2, y2) in curr_map.items():
                    if shirt not in prev_map:
                        continue
                    x1, y1 = prev_map[shirt]
                    v_kmh = (math.hypot(x2 - x1, y2 - y1) / dt) * 3.6
                    if v_kmh > MAX_PLAYER_SPEED_KMH:
                        continue
                    by_shirt_samples.setdefault(shirt, []).append(v_kmh)
        prev_t = t
        prev_map = curr_map

    stats = {}
    for shirt, vals in by_shirt_samples.items():
        if not vals:
            continue
        stats[shirt] = {
            "avg_kmh": sum(vals) / len(vals),
            "max_kmh": max(vals),
        }
    return stats


def compute_ball_speed_stats(rows_home, rows_away):
    """Calculate average/max ball speed and tracked in-play duration."""
    samples = []
    prev_t = None
    prev_xy = None
    tracked_t_start = None
    tracked_t_end = None
    for hr, ar in zip(rows_home, rows_away):
        t = hr.get("_t")
        if t is None:
            continue
        bx, by = get_ball_pos(hr, ar)
        if bx is None or by is None:
            continue
        if tracked_t_start is None:
            tracked_t_start = t
        tracked_t_end = t
        if prev_t is not None and prev_xy is not None:
            dt = t - prev_t
            if dt > 0:
                v_kmh = (math.hypot(bx - prev_xy[0], by - prev_xy[1]) / dt) * 3.6
                samples.append(v_kmh)
        prev_t = t
        prev_xy = (bx, by)
    tracked_duration_s = 0.0
    if tracked_t_start is not None and tracked_t_end is not None:
        tracked_duration_s = max(0.0, tracked_t_end - tracked_t_start)
    if not samples:
        return {"avg_kmh": None, "max_kmh": None, "tracked_duration_s": tracked_duration_s, "n_samples": 0}
    return {
        "avg_kmh": sum(samples) / len(samples),
        "max_kmh": max(samples),
        "tracked_duration_s": tracked_duration_s,
        "n_samples": len(samples),
    }


def make_snapshot_times(start_ts, end_ts):
    """Choose five evenly spaced timestamps across the visualized sequence."""
    duration_s = max(0.0, end_ts - start_ts)
    if duration_s <= 0.0:
        return [start_ts]
    tvals = [
        start_ts,
        start_ts + 0.25 * duration_s,
        start_ts + 0.50 * duration_s,
        start_ts + 0.75 * duration_s,
        end_ts,
    ]
    # De-duplicate near-equal times while preserving order.
    out = []
    for t in tvals:
        if not out or abs(t - out[-1]) > 1e-6:
            out.append(t)
    return out


def format_speed_panel(team_name, stats_dict):
    lines = [f"{team_name} speeds (km/h)", "shirt  avg | max"]
    for shirt in sorted(stats_dict.keys(), key=shirt_sort_key):
        st = stats_dict[shirt]
        lines.append(f"{shirt:>4}  {st['avg_kmh']:>4.1f} | {st['max_kmh']:>4.1f}")
    return "\n".join(lines)


def draw_pitch(ax):
    """Draw a 105x68m normalized football pitch."""
    x_min, x_max = -52.5, 52.5
    y_min, y_max = -34.0, 34.0

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_facecolor("#1c7c3f")

    # Outer lines
    ax.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], color="white", lw=2)
    # Halfway
    ax.plot([0, 0], [y_min, y_max], color="white", lw=2)
    # Center circle
    cc = plt.Circle((0, 0), 9.15, fill=False, color="white", lw=2)
    ax.add_patch(cc)

    # Penalty boxes
    # left
    ax.plot([x_min, x_min + 16.5, x_min + 16.5, x_min], [20.16, 20.16, -20.16, -20.16], color="white", lw=2)
    # right
    ax.plot([x_max, x_max - 16.5, x_max - 16.5, x_max], [20.16, 20.16, -20.16, -20.16], color="white", lw=2)

    ax.set_xticks([])
    ax.set_yticks([])


def get_ball_pos(hr, ar):
    bx = to_float(hr.get("ball_x"))
    by = to_float(hr.get("ball_y"))
    if bx is None or by is None:
        bx = to_float(ar.get("ball_x"))
        by = to_float(ar.get("ball_y"))
    return bx, by


def draw_single_snapshot(ax, hr, ar, attack_is_home, title):
    draw_pitch(ax)

    hx, hy = parse_team_points(hr, "home")
    ax_, ay_ = parse_team_points(ar, "away")
    bx, by = get_ball_pos(hr, ar)

    attack_color = "#5dade2"
    defend_color = "#f39c12"
    home_color = attack_color if attack_is_home else defend_color
    away_color = defend_color if attack_is_home else attack_color

    ax.scatter(hx, hy, s=30, c=home_color, edgecolors="black", linewidths=0.3, label="home")
    ax.scatter(ax_, ay_, s=30, c=away_color, edgecolors="black", linewidths=0.3, label="away")

    if bx is not None and by is not None:
        ax.scatter([bx], [by], s=75, c="white", edgecolors="black", linewidths=1.0, label="ball")

    ax.set_title(title, color="white", fontsize=10)


def draw_sequence_overview(
    rows_home,
    rows_away,
    attack_is_home,
    match_id,
    label_no_score,
    team_name,
    ball_track,
    model_end_ts,
    pass_times,
    speed_stats,
    model_duration_s,
    visual_duration_s,
    in_play_duration_s,
    current_final_prediction,
    output_path,
):
    """Draw the main overview image for a single validation candidate."""
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.subplots_adjust(left=0.18, right=0.82)
    draw_pitch(ax)

    attack_color = "#5dade2"
    defend_color = "#f39c12"
    home_color = attack_color if attack_is_home else defend_color
    away_color = defend_color if attack_is_home else attack_color

    # Continuous ball trajectory (includes dribbles and all movement)
    if ball_track:
        model_track, context_track = split_track_by_time(ball_track, model_end_ts)
        if model_track:
            xs = [p[1] for p in model_track]
            ys = [p[2] for p in model_track]
            ax.plot(xs, ys, color="white", lw=2.4, alpha=0.95, label="ball trajectory (model)")
            ax.scatter([xs[0]], [ys[0]], s=55, c="#f1c40f", edgecolors="black", linewidths=0.6, zorder=6)
            ax.scatter([xs[-1]], [ys[-1]], s=55, c="#e74c3c", edgecolors="black", linewidths=0.6, zorder=6)
        if context_track and len(context_track) >= 2:
            xs = [p[1] for p in context_track]
            ys = [p[2] for p in context_track]
            ax.plot(xs, ys, color="#d7dbdd", lw=1.6, alpha=0.5, linestyle="--", label="ball trajectory (context)")
            ax.scatter([xs[-1]], [ys[-1]], s=30, c="#d7dbdd", edgecolors="black", linewidths=0.4, zorder=5)

    # Pass direction markers aligned to tracking trajectory.
    # We anchor each pass at nearest ball point at pass time and draw a short forward segment.
    if ball_track and pass_times:
        track_t = [p[0] for p in ball_track]
        pass_times_sorted = sorted(pass_times)
        for idx, pt in enumerate(pass_times_sorted):
            i0 = min(range(len(track_t)), key=lambda i: abs(track_t[i] - pt))
            next_pass_t = pass_times_sorted[idx + 1] if idx + 1 < len(pass_times_sorted) else None
            t_target = track_t[i0] + 0.9
            if next_pass_t is not None:
                t_target = min(t_target, next_pass_t - 0.05)
            i1 = min(range(len(track_t)), key=lambda i: abs(track_t[i] - t_target))
            if i1 == i0:
                continue
            x0, y0 = ball_track[i0][1], ball_track[i0][2]
            x1, y1 = ball_track[i1][1], ball_track[i1][2]
            ax.scatter([x0], [y0], s=18, c="#ffd166", edgecolors="black", linewidths=0.25, zorder=7)
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#ffd166", lw=2.2, alpha=0.95),
            )

    # Attacking and defending team player trails (same shirt number across frames)
    attack_prefix = "home" if attack_is_home else "away"
    defend_prefix = "away" if attack_is_home else "home"
    attack_rows = rows_home if attack_is_home else rows_away
    defend_rows = rows_away if attack_is_home else rows_home
    by_shirt = {}
    by_shirt_def = {}
    for row in attack_rows:
        if row is None:
            continue
        pmap = parse_team_map(row, attack_prefix)
        for shirt, pos in pmap.items():
            by_shirt.setdefault(shirt, []).append(pos)
    for row in defend_rows:
        if row is None:
            continue
        pmap = parse_team_map(row, defend_prefix)
        for shirt, pos in pmap.items():
            by_shirt_def.setdefault(shirt, []).append(pos)

    # Keep only players present in at least 3 snapshots to reduce clutter
    for shirt, pts in by_shirt.items():
        if len(pts) < 3:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=home_color, alpha=0.35, lw=1.4)
        ax.scatter([xs[-1]], [ys[-1]], s=22, c=home_color, edgecolors="black", linewidths=0.2)

    # Defender trails
    for shirt, pts in by_shirt_def.items():
        if len(pts) < 3:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=away_color, alpha=0.25, lw=1.2, linestyle="--")
        ax.scatter([xs[-1]], [ys[-1]], s=18, c=away_color, edgecolors="black", linewidths=0.2)

    context_extra_s = max(0.0, visual_duration_s - model_duration_s)
    ax.set_title(
        f"Counterattack overview | matchId {match_id}\n"
        f"{label_no_score} | attacking team: {team_name} | current final: {current_final_prediction} | candidate duration: {model_duration_s:.2f}s | tracked window: {in_play_duration_s:.2f}s | context: +{context_extra_s:.2f}s",
        color="white",
        fontsize=11,
    )
    # Add lightweight legend handles for trail meaning
    ax.plot([], [], color=home_color, lw=1.8, alpha=0.55, label="attacking player trails")
    ax.plot([], [], color=away_color, lw=1.6, alpha=0.45, linestyle="--", label="defending player trails")
    ax.plot([], [], color="#ffd166", lw=2.2, alpha=0.95, label="pass direction markers (tracking-aligned)")
    leg = ax.legend(loc="upper right", facecolor="#1c7c3f", edgecolor="white", framealpha=0.8)
    for txt in leg.get_texts():
        txt.set_color("white")

    # Side panels: ball + each player's speed (avg and max km/h)
    atk_panel = format_speed_panel(speed_stats["attacking_team"], speed_stats["attacking_players"])
    def_panel = format_speed_panel(speed_stats["defending_team"], speed_stats["defending_players"])
    ball_avg = speed_stats["ball"]["avg_kmh"]
    ball_max = speed_stats["ball"]["max_kmh"]
    if ball_avg is None:
        ball_text = "Ball speed (km/h)\navg: n/a | max: n/a"
    else:
        ball_text = (
            "Ball speed (km/h)\n"
            f"avg: {ball_avg:.1f} | max: {ball_max:.1f}"
        )

    fig.text(
        0.015, 0.88, atk_panel,
        ha="left", va="top", color="white", fontsize=7.4,
        family="monospace",
        bbox=dict(facecolor="#145a32", edgecolor="white", alpha=0.78, pad=5),
    )
    fig.text(
        0.985, 0.88, def_panel,
        ha="right", va="top", color="white", fontsize=7.4,
        family="monospace",
        bbox=dict(facecolor="#145a32", edgecolor="white", alpha=0.78, pad=5),
    )
    fig.text(
        0.5, 0.06, ball_text,
        ha="center", va="bottom", color="white", fontsize=9,
        bbox=dict(facecolor="#145a32", edgecolor="white", alpha=0.78, pad=4),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, facecolor="#1c7c3f")
    plt.close(fig)


def draw_timeline_strip(rows_home, rows_away, times, attack_is_home, match_id, label_no_score, team_name):
    n = len(times)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.5))
    if n == 1:
        axes = [axes]

    for idx, (ax, t, hr, ar) in enumerate(zip(axes, times, rows_home, rows_away), start=1):
        if hr is None or ar is None:
            continue
        draw_single_snapshot(ax, hr, ar, attack_is_home, f"t={t:.2f}s")

    fig.suptitle(
        f"Counterattack timeline strip | matchId {match_id} | {label_no_score} | attacking: {team_name}",
        color="white",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "timeline_strip.png", dpi=170, facecolor="#1c7c3f")
    plt.close(fig)


def main():
    """Generate all validation snapshot PNGs and write a metadata file."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    attempts = read_validation_candidates(ANNOTATION_CSV, POOL_CSV)
    meta_lines = []
    saved_files = []

    for idx, attempt in enumerate(attempts, start=1):
        match_id = (attempt.get("match_id") or "").strip()
        team_id = (attempt.get("team_id") or "").strip()
        start_event_id = (attempt.get("start_event_id") or "").strip()
        start_ts = float((attempt.get("start_ts") or "0").strip())
        match_period = (attempt.get("match_period") or "").strip()
        end_ts, model_duration_s = candidate_model_end_ts(attempt)
        next_on_ball_ts = find_next_on_ball_event_ts_same_period(match_id, end_ts, match_period)
        if next_on_ball_ts is None:
            visual_end_ts = end_ts
        else:
            visual_end_ts = min(next_on_ball_ts, end_ts + MAX_CONTEXT_EXTRA_S)
            visual_end_ts = max(visual_end_ts, end_ts)

        label_no_score = (attempt.get("match_label") or "").strip()
        team_name = (attempt.get("team_name") or "").strip()
        current_final_prediction = int((attempt.get("current_final_prediction") or "0").strip() or 0)
        folder = find_tracking_folder_by_label(label_no_score)
        home_csv = folder / "home.csv"
        away_csv = folder / "away.csv"
        half_id = period_to_half(match_period)
        start_local = to_period_local_s(start_ts, match_period)
        end_local = to_period_local_s(end_ts, match_period)
        visual_end_local = to_period_local_s(visual_end_ts, match_period)
        next_on_ball_local = to_period_local_s(next_on_ball_ts, match_period) if next_on_ball_ts is not None else None

        if half_id is None or start_local is None or end_local is None or visual_end_local is None:
            continue

        # Snapshot times spread across actual sequence duration.
        times = make_snapshot_times(start_local, visual_end_local)
        home_rows = nearest_rows_in_period(home_csv, half_id, times)
        away_rows = nearest_rows_in_period(away_csv, half_id, times)

        # Continuous rows in sequence window for speed computation.
        seq_home_rows = read_rows_in_period(home_csv, half_id, start_local, visual_end_local)
        seq_away_rows = read_rows_in_period(away_csv, half_id, start_local, visual_end_local)
        pass_times = []
        for t in read_pass_times(match_id, team_id, start_ts, visual_end_ts):
            local_t = to_period_local_s(t, match_period)
            if local_t is not None:
                pass_times.append(local_t)
        ball_track = read_ball_track_in_period(home_csv, half_id, start_local, visual_end_local, sample_every_n=2)

        # Skip unusable cases where tracking cannot support a real visual judgement.
        if len(ball_track) < 2:
            continue

        attack_is_home, home_team, away_team = infer_attack_is_home(label_no_score, team_name)

        home_speed = compute_player_speed_stats(seq_home_rows, "home")
        away_speed = compute_player_speed_stats(seq_away_rows, "away")
        ball_speed = compute_ball_speed_stats(seq_home_rows, seq_away_rows)
        if attack_is_home:
            speed_stats = {
                "attacking_team": home_team,
                "defending_team": away_team,
                "attacking_players": home_speed,
                "defending_players": away_speed,
                "ball": ball_speed,
            }
        else:
            speed_stats = {
                "attacking_team": away_team,
                "defending_team": home_team,
                "attacking_players": away_speed,
                "defending_players": home_speed,
                "ball": ball_speed,
            }

        in_play_duration_s = ball_speed.get("tracked_duration_s") or max(0.0, visual_end_ts - start_ts)
        out_file = OUT_DIR / f"{idx:03d}_{match_id}_{start_event_id}_final{current_final_prediction}.png"
        draw_sequence_overview(
            home_rows,
            away_rows,
            attack_is_home,
            match_id,
            label_no_score,
            team_name,
            ball_track,
            end_local,
            pass_times,
            speed_stats,
            model_duration_s,
            max(0.0, visual_end_local - start_local),
            in_play_duration_s,
            current_final_prediction,
            out_file,
        )
        saved_files.append(out_file)
        meta_lines.extend([
            f"[candidate_{idx:03d}]",
            f"match_id={match_id}",
            f"label={label_no_score}",
            f"team_name={team_name}",
            f"current_final_prediction={current_final_prediction}",
            f"pool_source={(attempt.get('pool_source') or '').strip()}",
            f"tracking_folder={folder}",
            f"match_period={match_period}",
            f"half_id={half_id}",
            f"start_ts={start_ts}",
            f"start_local_ts={start_local}",
            f"candidate_end_ts={end_ts}",
            f"candidate_end_local_ts={end_local}",
            f"next_on_ball_ts={next_on_ball_ts}",
            f"next_on_ball_local_ts={next_on_ball_local}",
            f"visual_end_ts={visual_end_ts}",
            f"visual_end_local_ts={visual_end_local}",
            f"times={times}",
            "",
        ])

    meta = OUT_DIR / "snapshot_metadata.txt"
    meta.write_text("\n".join(meta_lines), encoding="utf-8")

    print("Saved sequence overviews:")
    for p in saved_files:
        print("-", p)


if __name__ == "__main__":
    EVENTS_BY_MATCH = load_events_by_match()
    main()
