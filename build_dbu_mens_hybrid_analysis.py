#!/usr/bin/env python3
"""Build the CSV tables used in the DBU hybrid counterattack analysis.

The script takes the already scored regain candidates from the hybrid model and
joins them with match/team metadata. It then writes compact benchmark tables for
Denmark, the tournament average, selected top teams, and match winners.

The script builds the same tables for the men's,
women's and U21 datasets.
"""

from pathlib import Path
import re
import unicodedata

import pandas as pd


# Project paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data"
DERIVED_DIR = DATA_DIR / "derived"
OUTPUT_DIR = DERIVED_DIR / "dbu_mens_hybrid_analysis"

EVENTS_CSV = DATA_DIR / "events.csv"
SCORED_CANDIDATES_CSV = OUTPUT_DIR / "all_regain_candidates_hybrid_scored.csv"

# Each dataset is filtered differently because the raw project data is stored in
# one shared event file, while women/U21 availability is determined by tracking
# folders. The selected top teams are the semi-finalists/finalists used in the
# report comparison.
DATASET_SPECS = [
    {
        "key": "men",
        "display_name": "Men H_EURO2024",
        "match_prefixes": ["554"],
        "tracking_dir": None,
        "exclude_match_prefixes": [],
        "denmark_norm": "denmark",
        "top_team_norms": {"spain", "france", "netherlands", "england"},
        "top_team_label": "Semi-finalists/finalists (ESP/FRA/NED/ENG)",
    },
    {
        "key": "women",
        "display_name": "Women Q_EURO2025",
        "match_prefixes": None,
        "tracking_dir": DATA_DIR / "Q_EURO2025",
        "exclude_match_prefixes": ["554"],
        "denmark_norm": "denmark",
        "top_team_norms": {"england", "italy", "germany", "spain"},
        "top_team_label": "Semi-finalists/finalists (ENG/ITA/GER/ESP)",
    },
    {
        "key": "u21",
        "display_name": "U21 EURO2025",
        "match_prefixes": None,
        "tracking_dir": DATA_DIR / "U21_EURO2025",
        "exclude_match_prefixes": [],
        "denmark_norm": "denmark u21",
        "top_team_norms": {
            "germany u21",
            "france u21",
            "netherlands u21",
            "england u21",
        },
        "top_team_label": "Semi-finalists/finalists (ENG/NED/GER/FRA)",
    },
]


def normalize_team_name(value):
    """Normalize team names so event labels and tracking labels can be matched."""
    text = str(value).strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = text.replace(" under 21", " u21")
    aliases = {
        "turkey": "turkiye",
        "turkiye": "turkiye",
        "czech republic": "czechia",
        "czech republic u21": "czechia u21",
    }
    return aliases.get(text, text)


def label_key(home_team, away_team):
    """Create a score-independent match key from home/away team names."""
    return f"{normalize_team_name(home_team)} - {normalize_team_name(away_team)}"


def parse_tracking_label(label):
    """Parse labels from tracking files, which do not include the final score."""
    match = re.match(r"^(.*) - (.*)$", str(label).strip())
    if match is None:
        raise ValueError(f"Could not parse tracking label: {label!r}")
    home_team, away_team = match.groups()
    return label_key(home_team, away_team)


def parse_match_label(label):
    """Parse event labels, which include both team names and final score."""
    match = re.match(r"^(.*) - (.*),\s*(\d+) - (\d+)(?:\s*\(.*\))?$", str(label).strip())
    if match is None:
        raise ValueError(f"Could not parse match label: {label!r}")
    home_team, away_team, home_goals, away_goals = match.groups()
    return {
        "home_team": home_team.strip(),
        "away_team": away_team.strip(),
        "home_goals": int(home_goals),
        "away_goals": int(away_goals),
        "label_no_score": f"{home_team.strip()} - {away_team.strip()}",
        "label_key": label_key(home_team, away_team),
    }


def load_tracking_label_keys(tracking_dir):
    # Women and U21 are selected by checking which match folders have tracking
    # data available in their tournament-specific directories.
    keys = set()
    if tracking_dir is None:
        return keys
    for match_dir in sorted(tracking_dir.iterdir()):
        home_csv = match_dir / "home.csv"
        if not home_csv.exists():
            continue
        label = pd.read_csv(home_csv, nrows=1)["label"].iloc[0]
        keys.add(parse_tracking_label(label))
    return keys


def load_match_metadata(spec):
    # Build one match table and one team-game table for the selected dataset.
    events = pd.read_csv(
        EVENTS_CSV,
        usecols=["matchId", "label", "teamId", "teamName"],
        low_memory=False,
    )
    events["match_id"] = events["matchId"].astype(str)
    events["team_id"] = events["teamId"].astype(str)
    work_events = events.copy()

    # Men's EURO 2024 matches are identified by match-id prefix in the shared
    # event file.
    if spec["match_prefixes"] is not None:
        work_events = work_events[
            work_events["match_id"].str.startswith(tuple(spec["match_prefixes"]))
        ].copy()

    match_rows = []
    for match_id, label in work_events[["match_id", "label"]].drop_duplicates().itertuples(index=False):
        parsed = parse_match_label(label)
        home_norm = normalize_team_name(parsed["home_team"])
        away_norm = normalize_team_name(parsed["away_team"])
        if parsed["home_goals"] > parsed["away_goals"]:
            winner_norm = home_norm
        elif parsed["away_goals"] > parsed["home_goals"]:
            winner_norm = away_norm
        else:
            winner_norm = "draw"
        match_rows.append(
            {
                "match_id": match_id,
                "label": label,
                "label_no_score": parsed["label_no_score"],
                "home_team": parsed["home_team"],
                "away_team": parsed["away_team"],
                "home_goals": parsed["home_goals"],
                "away_goals": parsed["away_goals"],
                "label_key": parsed["label_key"],
                "winner_norm": winner_norm,
            }
        )
    matches = pd.DataFrame(match_rows)

    # Women/U21 are filtered to the matches where a corresponding tracking
    # folder exists.
    if spec["tracking_dir"] is not None:
        tracking_label_keys = load_tracking_label_keys(spec["tracking_dir"])
        matches = matches[matches["label_key"].isin(tracking_label_keys)].copy()

    for prefix in spec["exclude_match_prefixes"]:
        matches = matches[~matches["match_id"].str.startswith(prefix)].copy()

    team_matches = (
        events[events["match_id"].isin(matches["match_id"])][["match_id", "team_id", "teamName"]]
        .drop_duplicates()
    )
    team_matches["team_norm"] = team_matches["teamName"].map(normalize_team_name)
    team_matches = team_matches.merge(
        matches[["match_id", "label_no_score", "winner_norm", "home_goals", "away_goals"]],
        on="match_id",
        how="left",
        validate="many_to_one",
    )
    team_matches["team_won_match"] = team_matches["team_norm"].eq(team_matches["winner_norm"])
    return matches, team_matches


def add_metadata_and_outcomes(scored, team_matches):
    # Attach team/match context to the hybrid-scored candidate table and create
    # the outcome proxies used in the DBU benchmark tables.
    scored = scored.copy()
    scored["match_id"] = scored["match_id"].astype(str)
    scored["team_id"] = scored["team_id"].astype(str)

    enriched = scored.merge(
        team_matches[
            [
                "match_id",
                "team_id",
                "teamName",
                "team_norm",
                "label_no_score",
                "winner_norm",
                "home_goals",
                "away_goals",
                "team_won_match",
            ]
        ],
        on=["match_id", "team_id"],
        how="inner",
        validate="many_to_one",
    )

    enriched["shot_20s"] = enriched["event_count_shot_20s"].fillna(0).gt(0).astype(int)
    enriched["shot_15s"] = enriched["event_count_shot_15s"].fillna(0).gt(0).astype(int)
    # Box entry is approximated by the minimum goal distance reached in the
    # 20-second feature window. It is a proxy, not a manually tagged box entry.
    enriched["box_entry_proxy"] = enriched["long_min_goal_distance_m"].le(20.15).astype(int)
    enriched["final_third_entry_proxy"] = enriched["long_min_goal_distance_m"].le(35.0).astype(int)
    enriched["box_shot_proxy"] = (enriched["shot_20s"].eq(1) & enriched["box_entry_proxy"].eq(1)).astype(int)
    return enriched


def summarize_team_counterattacks(counterattacks, team_matches):
    # Team-level table: one row per team with volume and quality indicators.
    base = (
        team_matches.groupby(["team_norm", "teamName"], as_index=False)
        .agg(matches=("match_id", "nunique"))
    )

    stats = (
        counterattacks.groupby(["team_norm", "teamName"], as_index=False)
        .agg(
            hybrid_counterattacks=("hybrid_label", "size"),
            shot_rate=("shot_20s", "mean"),
            box_shot_proxy_rate=("box_shot_proxy", "mean"),
            box_entry_proxy_rate=("box_entry_proxy", "mean"),
            final_third_entry_proxy_rate=("final_third_entry_proxy", "mean"),
            avg_progression_goal_m=("long_progression_goal_m", "mean"),
            avg_progression_speed_mps=("long_avg_progress_speed_mps", "mean"),
            avg_duration_s=("long_duration_s", "mean"),
            avg_start_goal_distance_m=("long_start_goal_distance_m", "mean"),
            avg_directness=("long_directness_goal", "mean"),
            avg_probability=("hybrid_probability", "mean"),
        )
    )

    summary = base.merge(stats, on=["team_norm", "teamName"], how="left")
    summary["hybrid_counterattacks"] = summary["hybrid_counterattacks"].fillna(0).astype(int)
    summary["counterattacks_per_match"] = summary["hybrid_counterattacks"] / summary["matches"]
    cols = [
        "teamName",
        "matches",
        "hybrid_counterattacks",
        "counterattacks_per_match",
        "shot_rate",
        "box_shot_proxy_rate",
        "box_entry_proxy_rate",
        "final_third_entry_proxy_rate",
        "avg_progression_goal_m",
        "avg_progression_speed_mps",
        "avg_duration_s",
        "avg_start_goal_distance_m",
        "avg_directness",
        "avg_probability",
    ]
    return summary[cols].sort_values(["counterattacks_per_match", "hybrid_counterattacks"], ascending=False)


def benchmark_row(name, counterattacks, team_matches):
    # Common benchmark format used for Denmark, tournament average, top teams
    # and match winners.
    return {
        "benchmark": name,
        "teams": int(team_matches["team_norm"].nunique()),
        "matches": int(len(team_matches)),
        "hybrid_counterattacks": int(len(counterattacks)),
        "counterattacks_per_match": len(counterattacks) / len(team_matches) if len(team_matches) else 0.0,
        "shot_rate": float(counterattacks["shot_20s"].mean()) if len(counterattacks) else 0.0,
        "box_shot_proxy_rate": float(counterattacks["box_shot_proxy"].mean()) if len(counterattacks) else 0.0,
        "box_entry_proxy_rate": float(counterattacks["box_entry_proxy"].mean()) if len(counterattacks) else 0.0,
        "final_third_entry_proxy_rate": float(counterattacks["final_third_entry_proxy"].mean()) if len(counterattacks) else 0.0,
        "avg_progression_goal_m": float(counterattacks["long_progression_goal_m"].mean()) if len(counterattacks) else 0.0,
        "avg_progression_speed_mps": float(counterattacks["long_avg_progress_speed_mps"].mean()) if len(counterattacks) else 0.0,
        "avg_duration_s": float(counterattacks["long_duration_s"].mean()) if len(counterattacks) else 0.0,
    }


def build_benchmarks(counterattacks, team_matches, spec):
    # These are the main report-facing comparison groups.
    denmark_tm = team_matches[team_matches["team_norm"].eq(spec["denmark_norm"])]
    denmark_ca = counterattacks[counterattacks["team_norm"].eq(spec["denmark_norm"])]

    top_tm = team_matches[team_matches["team_norm"].isin(spec["top_team_norms"])]
    top_ca = counterattacks[counterattacks["team_norm"].isin(spec["top_team_norms"])]

    winner_tm = team_matches[team_matches["team_won_match"]]
    winner_ca = counterattacks[counterattacks["team_won_match"]]

    rows = [
        benchmark_row("Denmark", denmark_ca, denmark_tm),
        benchmark_row("Tournament average", counterattacks, team_matches),
        benchmark_row(spec["top_team_label"], top_ca, top_tm),
        benchmark_row("Match winners only", winner_ca, winner_tm),
    ]
    return pd.DataFrame(rows)


def build_denmark_group_stage_benchmarks(counterattacks, team_matches):
    # Men's-specific extra table used to compare Denmark's group-stage CAs with
    # the opponents from those same matches.
    group_labels = {"Slovenia - Denmark", "Denmark - England", "Denmark - Serbia"}
    group_tm = team_matches[team_matches["label_no_score"].isin(group_labels)]
    group_ca = counterattacks[counterattacks["label_no_score"].isin(group_labels)]

    denmark_tm = group_tm[group_tm["team_norm"].eq("denmark")]
    denmark_ca = group_ca[group_ca["team_norm"].eq("denmark")]
    non_denmark_tm = group_tm[~group_tm["team_norm"].eq("denmark")]
    non_denmark_ca = group_ca[~group_ca["team_norm"].eq("denmark")]

    rows = [
        benchmark_row("Denmark group stage", denmark_ca, denmark_tm),
        benchmark_row("Denmark opponents group stage", non_denmark_ca, non_denmark_tm),
    ]
    return pd.DataFrame(rows).drop(columns=["teams"])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SCORED_CANDIDATES_CSV.exists():
        raise FileNotFoundError(
            f"Missing {SCORED_CANDIDATES_CSV}. Run the hybrid scoring step before this DBU analysis build."
        )

    scored = pd.read_csv(SCORED_CANDIDATES_CSV, low_memory=False)

    # Build the same output files for each dataset specification.
    for spec in DATASET_SPECS:
        _, team_matches = load_match_metadata(spec)
        enriched = add_metadata_and_outcomes(scored, team_matches)
        counterattacks = enriched[enriched["hybrid_label"].eq(1)].copy()

        team_summary = summarize_team_counterattacks(counterattacks, team_matches)
        benchmarks = build_benchmarks(counterattacks, team_matches, spec)

        key = spec["key"]
        counterattacks.to_csv(OUTPUT_DIR / f"hybrid_counterattacks_{key}.csv", index=False)
        team_summary.to_csv(OUTPUT_DIR / f"team_hybrid_counterattack_summary_{key}.csv", index=False)
        benchmarks.to_csv(OUTPUT_DIR / f"denmark_vs_hybrid_benchmarks_{key}.csv", index=False)

        if key == "men":
            group_stage = build_denmark_group_stage_benchmarks(counterattacks, team_matches)
            group_stage.to_csv(OUTPUT_DIR / "denmark_group_stage_vs_hybrid_benchmarks_men.csv", index=False)

        print(f"Saved {len(counterattacks)} hybrid counter-attacks for {spec['display_name']}")
        print(f"Saved: {OUTPUT_DIR / f'hybrid_counterattacks_{key}.csv'}")
        print(f"Saved: {OUTPUT_DIR / f'team_hybrid_counterattack_summary_{key}.csv'}")
        print(f"Saved: {OUTPUT_DIR / f'denmark_vs_hybrid_benchmarks_{key}.csv'}")


if __name__ == "__main__":
    main()
