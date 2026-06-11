#!/usr/bin/env python3
"""
Build the MP4 annotation set used for the human counter-attack labels.

The script starts from the trained XGBoost detector and the regain candidate
table. It selects candidates from six groups: borderline negatives, borderline
positives, clear negatives, random negatives, clear positives, and random
positives. For each selected candidate it renders a short tracking clip and
writes metadata used later when the human labels are joined back to the model.

Inputs:
- Data/derived/ai_counterattack_detection/regain_candidates_labeled.csv
- Data/derived/ai_counterattack_detection/counterattack_detector.pkl
- raw event/tracking files loaded through make_validation_candidate_snapshots.py

Outputs:
- MP4 clips in Data/derived/human_annotation_set_50_50_20_20_20_20_current_746_terminal_filtered/
- annotation_selection.csv and annotation_selection.txt
- skipped_candidates.csv
"""

import csv
import pickle
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import make_validation_candidate_snapshots as snap


# Paths. Anchor everything to this project folder instead of the terminal's
# current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent
DERIVED_DIR = PROJECT_ROOT / "Data" / "derived"
AI_DIR = DERIVED_DIR / "ai_counterattack_detection"
CANDIDATES_CSV = AI_DIR / "regain_candidates_labeled.csv"
MODEL_PKL = AI_DIR / "counterattack_detector.pkl"
OUT_DIR = DERIVED_DIR / "human_annotation_set_50_50_20_20_20_20_current_746_terminal_filtered"

# Clip rendering settings
FPS = 12
PRE_CONTEXT_S = 1.0
POST_CONTEXT_S = 1.0
TRAIL_LOOKBACK_S = 1.5
RANDOM_NEGATIVE_SEED = 42
RANDOM_POSITIVE_SEED = 43

# Quality filters so humans are not asked to label clips with unusable ball tracking.
QUALITY_CHECK_FPS = 8
MAX_MISSING_BALL_RATIO = 0.35
MAX_OFF_PITCH_RATIO = 0.85
MIN_ON_PITCH_BALL_FRAMES = 8
MAX_ANNOTATION_SEQUENCE_S = 40.0

# Final annotation design: 50-50-20-20 plus the two positive additions.
GROUP_SPECS = [
    ("01_negatives_just_under_threshold", 50),
    ("02_positives_just_over_threshold", 50),
    ("03_clear_negatives", 20),
    ("04_random_negatives", 20),
    ("05_clear_positives", 20),
    ("06_random_positives", 20),
]

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


def load_model_bundle():
    with MODEL_PKL.open("rb") as f:
        return pickle.load(f)


def load_candidates():
    df = pd.read_csv(CANDIDATES_CSV)
    if df.empty:
        raise RuntimeError(f"No candidate rows found in {CANDIDATES_CSV}")
    return df


def duel_won(row):
    return (
        (row.get("aerialWon") or "").strip() == "1"
        or (row.get("groundDuelRecoveredPossession") or "").strip() == "1"
        or (row.get("groundDuelKeptPossession") or "").strip() == "1"
    )


def opponent_on_ball_ends_sequence(row, possessing_team_id):
    event_type = (row.get("typePrimary") or "").strip()
    team_id = (row.get("teamId") or "").strip()
    if not team_id or team_id == possessing_team_id or event_type not in ON_BALL_TYPES:
        return False
    if event_type == "duel":
        return duel_won(row)
    return True


def annotation_end_info(row):
    # The model scores a fixed post-regain window, but humans should see the
    # attack until it naturally ends. I therefore extend the clip until a clear
    # possession loss, restart/stoppage, or at most 40 seconds.
    match_id = str(row["match_id"])
    team_id = str(row["team_id"])
    start_event_id = int(row["start_event_id"])
    start_ts = float(row["start_ts"])
    period = str(row["match_period"])
    rows = snap.EVENTS_BY_MATCH.get(match_id, [])
    start_idx = None
    for i, ev in enumerate(rows):
        if int(ev.get("_event_id") or 0) == start_event_id:
            start_idx = i
            break
    if start_idx is None:
        raise RuntimeError(f"Could not find start event {start_event_id} in match {match_id}")

    end_ts = start_ts
    stop_reason = "max_annotation_window"
    for ev in rows[start_idx + 1:]:
        ts = ev.get("_ts")
        if ts is None:
            continue
        if ts - start_ts > MAX_ANNOTATION_SEQUENCE_S:
            end_ts = start_ts + MAX_ANNOTATION_SEQUENCE_S
            stop_reason = "max_annotation_window"
            break
        event_type = (ev.get("typePrimary") or "").strip()
        if event_type == "game_interruption":
            end_ts = ts
            stop_reason = "game_interruption"
            break
        if event_type in STOP_RESTART_TYPES:
            end_ts = ts
            stop_reason = event_type
            break
        end_ts = ts
        if opponent_on_ball_ends_sequence(ev, team_id):
            stop_reason = "opponent_on_ball"
            break
    return {
        "annotation_end_ts": end_ts,
        "annotation_duration_s": max(0.0, end_ts - start_ts),
        "annotation_stop_reason": stop_reason,
    }


def score_candidates(df, model_bundle):
    # The model probability and distance to the threshold define the annotation
    # groups, especially the just-below and just-above threshold clips.
    feature_cols = model_bundle["feature_cols"]
    threshold = float(model_bundle["threshold"])
    preprocessor = model_bundle["preprocessor"]
    model = model_bundle["model"]

    X = preprocessor.transform(df[feature_cols])
    prob = model.predict_proba(X)[:, 1]

    scored = df.copy()
    scored["model_probability"] = prob
    scored["model_threshold"] = threshold
    scored["predicted_label"] = (scored["model_probability"] >= threshold).astype(int)
    scored["distance_to_threshold"] = scored["model_probability"] - threshold
    return scored


def clip_quality_metrics(row):
    # A candidate can be informative for the model but useless for annotation if
    # the ball disappears or stays outside the pitch. This check removes those
    # cases before clips are rendered.
    label_no_score = str(row["source_label"])
    folder = snap.find_tracking_folder_by_label(label_no_score)
    half_id = snap.period_to_half(str(row["match_period"]))
    start_local = snap.to_period_local_s(float(row["start_ts"]), str(row["match_period"]))
    if half_id is None or start_local is None:
        raise RuntimeError("Could not determine local clip timing.")

    terminal = annotation_end_info(row)
    end_local = start_local + float(terminal["annotation_duration_s"])
    clip_start = max(0.0, start_local - PRE_CONTEXT_S)
    clip_end = end_local + POST_CONTEXT_S
    clip_duration_s = max(0.1, clip_end - clip_start)
    n_frames = max(8, int(round(clip_duration_s * QUALITY_CHECK_FPS)))
    frame_times = np.linspace(clip_start, clip_end, num=n_frames, endpoint=True)

    label_no_score, team_name = find_event_meta(str(row["match_id"]), str(int(row["start_event_id"])))
    snap.infer_attack_is_home(label_no_score, team_name)

    home_rows = snap.nearest_rows_in_period(folder / "home.csv", half_id, frame_times)
    away_rows = snap.nearest_rows_in_period(folder / "away.csv", half_id, frame_times)
    ball_track = snap.read_ball_track_in_period(folder / "home.csv", half_id, clip_start, clip_end, sample_every_n=1)
    if len(ball_track) < 2:
        raise RuntimeError("Insufficient tracking for rendered clip.")

    total_frames = 0
    missing_ball_frames = 0
    off_pitch_frames = 0
    on_pitch_ball_frames = 0
    for home_row, away_row in zip(home_rows, away_rows):
        if home_row is None or away_row is None:
            continue
        total_frames += 1
        bx, by = snap.get_ball_pos(home_row, away_row)
        if bx is None or by is None:
            missing_ball_frames += 1
            continue
        if abs(bx) > 52.5 or abs(by) > 34.0:
            off_pitch_frames += 1
        else:
            on_pitch_ball_frames += 1

    if total_frames == 0:
        raise RuntimeError("No tracking frames available in clip window.")

    missing_ratio = missing_ball_frames / total_frames
    off_pitch_ratio = off_pitch_frames / total_frames
    return {
        "total_frames": total_frames,
        "missing_ball_frames": missing_ball_frames,
        "missing_ball_ratio": missing_ratio,
        "off_pitch_frames": off_pitch_frames,
        "off_pitch_ratio": off_pitch_ratio,
        "on_pitch_ball_frames": on_pitch_ball_frames,
        "annotation_end_ts": terminal["annotation_end_ts"],
        "annotation_duration_s": terminal["annotation_duration_s"],
        "annotation_stop_reason": terminal["annotation_stop_reason"],
    }


def passes_quality_filter(metrics):
    return (
        metrics["missing_ball_ratio"] <= MAX_MISSING_BALL_RATIO
        and metrics["off_pitch_ratio"] <= MAX_OFF_PITCH_RATIO
        and metrics["on_pitch_ball_frames"] >= MIN_ON_PITCH_BALL_FRAMES
        and metrics["annotation_stop_reason"] != "max_annotation_window"
    )


def select_annotation_groups(scored):
    # First choose the borderline examples, then add clear/random examples so
    # the annotation set is not only made of ambiguous model decisions.
    threshold = float(scored["model_threshold"].iloc[0])
    scored = scored.copy()
    scored["candidate_key"] = list(
        zip(
            scored["match_id"].astype(str),
            scored["team_id"].astype(str),
            scored["start_event_id"].astype(int),
        )
    )
    negatives = scored[scored["predicted_label"] == 0].copy()
    positives = scored[scored["predicted_label"] == 1].copy()

    quality_cache = {}

    def row_quality(row):
        # Quality checks are expensive because they read tracking files, so each
        # candidate is checked at most once.
        key = row["candidate_key"]
        if key not in quality_cache:
            quality_cache[key] = clip_quality_metrics(row)
        return quality_cache[key]

    def take_first_valid(df, n, ascending, random_state=None):
        if random_state is not None:
            ordered = df.sample(frac=1.0, random_state=random_state)
        else:
            ordered = df.sort_values("model_probability", ascending=ascending)
        picked = []
        for _, row in ordered.iterrows():
            try:
                metrics = row_quality(row)
            except Exception:
                continue
            if not passes_quality_filter(metrics):
                continue
            enriched = row.copy()
            for key, value in metrics.items():
                enriched[key] = value
            picked.append(enriched)
            if len(picked) >= n:
                break
        return pd.DataFrame(picked)

    negatives_near = take_first_valid(negatives, 50, ascending=False)
    positives_near = take_first_valid(positives, 50, ascending=True)
    if len(negatives_near) < 50 or len(positives_near) < 50:
        raise RuntimeError("Could not fill near-threshold groups after quality filtering.")

    used_pos_keys = set(positives_near["candidate_key"].tolist())
    positives_remaining = positives[~positives["candidate_key"].isin(used_pos_keys)].copy()

    used_neg_keys = set(negatives_near["candidate_key"].tolist())
    negatives_remaining = negatives[~negatives["candidate_key"].isin(used_neg_keys)].copy()

    clear_negatives = take_first_valid(negatives_remaining, 20, ascending=True)
    if len(clear_negatives) < 20:
        raise RuntimeError("Could not fill clear-negative group after quality filtering.")

    used_neg_keys.update(clear_negatives["candidate_key"].tolist())
    negatives_remaining = negatives[~negatives["candidate_key"].isin(used_neg_keys)].copy()

    random_negatives = take_first_valid(negatives_remaining, 20, ascending=True, random_state=RANDOM_NEGATIVE_SEED)
    if len(random_negatives) < 20:
        raise RuntimeError("Could not fill random-negative group after quality filtering.")

    clear_positives = take_first_valid(positives_remaining, 20, ascending=False)
    if len(clear_positives) < 20:
        raise RuntimeError("Could not fill clear-positive group after quality filtering.")

    used_pos_keys.update(clear_positives["candidate_key"].tolist())
    positives_remaining = positives[~positives["candidate_key"].isin(used_pos_keys)].copy()

    random_positives = take_first_valid(positives_remaining, 20, ascending=False, random_state=RANDOM_POSITIVE_SEED)
    if len(random_positives) < 20:
        raise RuntimeError("Could not fill random-positive group after quality filtering.")

    groups = [
        ("01_negatives_just_under_threshold", negatives_near, threshold),
        ("02_positives_just_over_threshold", positives_near, threshold),
        ("03_clear_negatives", clear_negatives, threshold),
        ("04_random_negatives", random_negatives, threshold),
        ("05_clear_positives", clear_positives, threshold),
        ("06_random_positives", random_positives, threshold),
    ]
    return groups


def find_event_meta(match_id: str, event_id: str):
    for row in snap.EVENTS_BY_MATCH.get(match_id, []):
        if str(row.get("_event_id")) == str(event_id):
            label = snap.normalize_label_no_score(row.get("label"))
            team_name = (row.get("teamName") or "").strip()
            return label, team_name
    raise RuntimeError(f"Could not find event metadata for match {match_id}, event {event_id}")


def rgba_frame(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[:, :, :3].copy()


def ball_pos_from_rows(home_row, away_row):
    return snap.get_ball_pos(home_row, away_row)


def draw_frame(
    home_row,
    away_row,
    attack_is_home,
    label_no_score,
    team_name,
    home_team,
    away_team,
    display_candidate_id,
    rel_t,
    model_duration_s,
    in_model_window,
    trail_points,
    group_name,
):
    # The colors are fixed across all clips to make the annotation task easier:
    # blue is always the attacking team and orange is always the defending team.
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    snap.draw_pitch(ax)

    attack_color = "#5dade2"
    defend_color = "#f39c12"
    home_color = attack_color if attack_is_home else defend_color
    away_color = defend_color if attack_is_home else attack_color

    hx, hy = snap.parse_team_points(home_row, "home")
    axx, ayy = snap.parse_team_points(away_row, "away")
    bx, by = ball_pos_from_rows(home_row, away_row)

    ax.scatter(hx, hy, s=42, c=home_color, edgecolors="black", linewidths=0.35)
    ax.scatter(axx, ayy, s=42, c=away_color, edgecolors="black", linewidths=0.35)

    if trail_points:
        xs = [p[1] for p in trail_points]
        ys = [p[2] for p in trail_points]
        ax.plot(xs, ys, color="white", lw=2.2, alpha=0.95)
        ax.scatter([xs[0]], [ys[0]], s=20, c="#d7dbdd", edgecolors="black", linewidths=0.25, zorder=4)

    if bx is not None and by is not None:
        ax.scatter([bx], [by], s=90, c="white", edgecolors="black", linewidths=0.8, zorder=5)

    if rel_t < 0:
        phase = "PRE-CONTEXT"
        phase_color = "#d7dbdd"
    elif in_model_window:
        phase = "MODEL WINDOW"
        phase_color = "#e74c3c"
    else:
        phase = "FOLLOW-THROUGH"
        phase_color = "#f4d03f"
    ax.set_title(
        f"{display_candidate_id} | {group_name} | {label_no_score} | attacking team: {team_name}\n"
        f"time from regain: {rel_t:+.2f}s | model duration: {model_duration_s:.2f}s",
        color="white",
        fontsize=12,
    )

    fig.text(
        0.5,
        0.03,
        phase,
        ha="center",
        va="bottom",
        color=phase_color,
        fontsize=12,
        weight="bold",
        bbox=dict(facecolor="#145a32", edgecolor="white", alpha=0.82, pad=5),
    )
    fig.text(
        0.02,
        0.95,
        f"Home: {home_team} | Away: {away_team}\nBlue = attacking team | Orange = defending team",
        ha="left",
        va="top",
        color="white",
        fontsize=10,
        bbox=dict(facecolor="#145a32", edgecolor="white", alpha=0.82, pad=4),
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    frame = rgba_frame(fig)
    plt.close(fig)
    return frame


def build_clip(row, display_candidate_id: str, group_name: str, out_path: Path):
    # The clip includes one second before the regain and one second after the
    # annotation end. The red "MODEL WINDOW" label shows the part scored by the
    # detector, while later frames give context for the human rater.
    match_id = str(row["match_id"]).strip()
    team_id = str(row["team_id"]).strip()
    start_event_id = str(row["start_event_id"]).strip()
    match_period = str(row["match_period"]).strip()
    start_ts = float(row["start_ts"])
    model_duration_s = float(row["long_duration_s"])
    annotation_duration_s = float(row["annotation_duration_s"])

    label_no_score, team_name = find_event_meta(match_id, start_event_id)
    folder = snap.find_tracking_folder_by_label(label_no_score)
    home_csv = folder / "home.csv"
    away_csv = folder / "away.csv"
    half_id = snap.period_to_half(match_period)
    if half_id is None:
        raise RuntimeError(f"Unsupported match period for {display_candidate_id}: {match_period}")

    start_local = snap.to_period_local_s(start_ts, match_period)
    if start_local is None:
        raise RuntimeError(f"Could not convert start timestamp for {display_candidate_id}")

    end_local = start_local + annotation_duration_s
    clip_start_local = max(0.0, start_local - PRE_CONTEXT_S)
    clip_end_local = end_local + POST_CONTEXT_S
    clip_duration_s = max(0.1, clip_end_local - clip_start_local)
    n_frames = max(12, int(round(clip_duration_s * FPS)))
    frame_times = np.linspace(clip_start_local, clip_end_local, num=n_frames, endpoint=True)

    home_rows = snap.nearest_rows_in_period(home_csv, half_id, frame_times)
    away_rows = snap.nearest_rows_in_period(away_csv, half_id, frame_times)
    ball_track = snap.read_ball_track_in_period(home_csv, half_id, clip_start_local, clip_end_local, sample_every_n=1)
    if len(ball_track) < 2:
        raise RuntimeError(f"Insufficient tracking for {display_candidate_id}")

    attack_is_home, home_team, away_team = snap.infer_attack_is_home(label_no_score, team_name)

    with imageio.get_writer(out_path, fps=FPS, codec="libx264", quality=7, pixelformat="yuv420p") as writer:
        for t, home_row, away_row in zip(frame_times, home_rows, away_rows):
            if home_row is None or away_row is None:
                continue
            rel_t = t - start_local
            trail_points = [p for p in ball_track if t - TRAIL_LOOKBACK_S <= p[0] <= t]
            frame = draw_frame(
                home_row=home_row,
                away_row=away_row,
                attack_is_home=attack_is_home,
                label_no_score=label_no_score,
                team_name=team_name,
                home_team=home_team,
                away_team=away_team,
                display_candidate_id=display_candidate_id,
                rel_t=rel_t,
                model_duration_s=model_duration_s,
                in_model_window=(0.0 <= rel_t <= model_duration_s),
                trail_points=trail_points,
                group_name=group_name,
            )
            writer.append_data(frame)


def prepare_output_dir():
    # Keep one subfolder per annotation group so the final PDF/order is easy to
    # inspect manually.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, _count in GROUP_SPECS:
        (OUT_DIR / name).mkdir(parents=True, exist_ok=True)


def write_metadata(rows):
    # This metadata file is the link between the rendered MP4 names and the
    # original candidate rows used later in the human-guided model.
    csv_path = OUT_DIR / "annotation_selection.csv"
    txt_path = OUT_DIR / "annotation_selection.txt"

    fieldnames = [
        "display_candidate_id",
        "group_name",
        "group_order",
        "model_probability",
        "model_threshold",
        "predicted_label",
        "rule_label_counterattack",
        "distance_to_threshold",
        "match_id",
        "team_id",
        "start_event_id",
        "match_period",
        "start_ts",
        "long_duration_s",
        "annotation_duration_s",
        "annotation_stop_reason",
        "source_label",
        "clip_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = []
    for row in rows:
        lines.extend(
            [
                f"[{row['display_candidate_id']}]",
                f"group_name={row['group_name']}",
                f"group_order={row['group_order']}",
                f"model_probability={row['model_probability']:.6f}",
                f"model_threshold={row['model_threshold']:.6f}",
                f"predicted_label={row['predicted_label']}",
                f"rule_label_counterattack={row['rule_label_counterattack']}",
                f"distance_to_threshold={row['distance_to_threshold']:.6f}",
                f"match_id={row['match_id']}",
                f"team_id={row['team_id']}",
                f"start_event_id={row['start_event_id']}",
                f"match_period={row['match_period']}",
                f"start_ts={row['start_ts']}",
                f"long_duration_s={row['long_duration_s']}",
                f"annotation_duration_s={row['annotation_duration_s']}",
                f"annotation_stop_reason={row['annotation_stop_reason']}",
                f"source_label={row['source_label']}",
                f"path={row['clip_path']}",
                "",
            ]
        )
    txt_path.write_text("\n".join(lines), encoding="utf-8")


def write_skipped(rows):
    csv_path = OUT_DIR / "skipped_candidates.csv"
    if not rows:
        csv_path.write_text("display_candidate_id,group_name,match_id,start_event_id,reason\n", encoding="utf-8")
        return
    fieldnames = ["display_candidate_id", "group_name", "match_id", "start_event_id", "reason"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    # Load the same event cache used by the snapshot code, then score all regain
    # candidates and render the selected annotation groups.
    snap.EVENTS_BY_MATCH = snap.load_events_by_match()
    model_bundle = load_model_bundle()
    candidates = load_candidates()
    scored = score_candidates(candidates, model_bundle)
    groups = select_annotation_groups(scored)
    prepare_output_dir()

    metadata_rows = []
    skipped_rows = []
    global_idx = 1

    for group_order, (group_name, group_df, threshold) in enumerate(groups, start=1):
        group_dir = OUT_DIR / group_name
        if group_name == "01_negatives_just_under_threshold":
            group_df = group_df.sort_values("model_probability", ascending=False)
        elif group_name == "02_positives_just_over_threshold":
            group_df = group_df.sort_values("model_probability", ascending=True)
        elif group_name == "03_clear_negatives":
            group_df = group_df.sort_values("model_probability", ascending=True)
        elif group_name == "05_clear_positives":
            group_df = group_df.sort_values("model_probability", ascending=False)
        else:
            group_df = group_df.sort_values(["match_id", "start_event_id"])

        for row in group_df.itertuples(index=False):
            display_candidate_id = f"candidate_{global_idx:03d}"
            clip_path = group_dir / f"{display_candidate_id}.mp4"
            try:
                build_clip(row._asdict(), display_candidate_id, group_name, clip_path)
            except Exception as exc:
                print(f"Skipped {display_candidate_id}: {exc}")
                skipped_rows.append(
                    {
                        "display_candidate_id": display_candidate_id,
                        "group_name": group_name,
                        "match_id": str(row.match_id),
                        "start_event_id": int(row.start_event_id),
                        "reason": str(exc),
                    }
                )
                global_idx += 1
                continue
            metadata_rows.append(
                {
                    "display_candidate_id": display_candidate_id,
                    "group_name": group_name,
                    "group_order": group_order,
                    "model_probability": float(row.model_probability),
                    "model_threshold": float(threshold),
                    "predicted_label": int(row.predicted_label),
                    "rule_label_counterattack": int(row.label_counterattack),
                    "distance_to_threshold": float(row.distance_to_threshold),
                    "match_id": str(row.match_id),
                    "team_id": str(row.team_id),
                    "start_event_id": int(row.start_event_id),
                    "match_period": str(row.match_period),
                    "start_ts": float(row.start_ts),
                    "long_duration_s": float(row.long_duration_s),
                    "annotation_duration_s": float(row.annotation_duration_s),
                    "annotation_stop_reason": str(row.annotation_stop_reason),
                    "source_label": str(row.source_label),
                    "clip_path": str(clip_path),
                }
            )
            print(f"Saved {clip_path}")
            global_idx += 1

    write_metadata(metadata_rows)
    write_skipped(skipped_rows)
    print(f"Saved metadata to {OUT_DIR / 'annotation_selection.csv'}")
    print(f"Saved metadata to {OUT_DIR / 'annotation_selection.txt'}")
    print(f"Saved skipped metadata to {OUT_DIR / 'skipped_candidates.csv'}")


if __name__ == "__main__":
    main()
