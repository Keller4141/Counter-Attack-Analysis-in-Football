#!/usr/bin/env python3
"""Run the split-perturbation robustness analysis for the hybrid model.

The script repeatedly changes which human-labeled cases are used for training
and which are held out for testing. For each split it trains both the baseline
XGBoost model and the human-guided hybrid model, evaluates them against the
held-out human labels, and saves iteration-level metrics plus summary tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from train_counterattack_detector import (
    BEST_XGB_PARAMS,
    EARLY_STOPPING_ROUNDS,
    PREFERRED_CAT_COLS,
)
from train_counterattack_detector_human_guided_eval import (
    evaluate_predictions,
    fit_model,
    load_docx_rating_table,
    load_new_human_annotations,
    load_validation48_candidate_lookup,
    map_validation_clips_to_candidates,
    predict_on_test,
)


ROOT = Path(__file__).resolve().parent
DERIVED_DIR = ROOT / "Data" / "derived"
ANNOTATION_DIR = ROOT / "Data" / "annotations"
AI_DIR = DERIVED_DIR / "ai_counterattack_detection"
LEGACY_DIR = DERIVED_DIR / "validation_legacy_broad_pool"
HUMAN_SET_DIR = DERIVED_DIR / "human_annotation_set_50_50_20_20_20_20_current_746_terminal_filtered"

CANDIDATES_CSV = AI_DIR / "regain_candidates_labeled.csv"
MODEL_METRICS_JSON = AI_DIR / "model_metrics.json"
SHARED_FEATURE_SUMMARY_JSON = AI_DIR / "shared_feature_old48_eval" / "shared_feature_summary.json"

NEW170_DOCX = ANNOTATION_DIR / "ANNOTATION_SET.docx"
NEW170_SELECTION_CSV = HUMAN_SET_DIR / "annotation_selection.csv"
VALIDATION48_DOCX = ANNOTATION_DIR / "Kontra vurdering (1).docx"
VALIDATION48_REFERENCE_DIR = ANNOTATION_DIR / "validation_48" / "reference_snapshots"

DEFAULT_OUTPUT_DIR = AI_DIR / "human_resampling_robustness"
DEFAULT_N_PERMUTATIONS = 100_000

HUMAN_WEIGHT_BY_AGREEMENT = {
    4: 3.0,
    5: 5.0,
    6: 8.0,
    7: 12.0,
}


# Command line options keep the script reproducible while still making it easy
# to run a shorter smoke test before launching the full 100-split analysis.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resampling robustness test for rule-trained vs human-guided XGBoost counterattack models."
    )
    parser.add_argument("--n-iterations", type=int, default=100)
    parser.add_argument("--human-test-size", type=int, default=48)
    parser.add_argument("--random-state", type=int, default=20260428)
    parser.add_argument(
        "--feature-source",
        choices=["current_model", "shared_human_eval"],
        default="current_model",
        help="current_model uses model_metrics.json selected_features; shared_human_eval uses shared_feature_summary.json.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=DEFAULT_N_PERMUTATIONS,
        help="Monte Carlo sign-flip permutations used for paired p-values in the summary table.",
    )
    return parser.parse_args()


def normalize_candidate_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["match_id"] = df["match_id"].astype(str)
    df["start_event_id"] = df["start_event_id"].astype(str)
    return df


def load_validation48_human_cases() -> pd.DataFrame:
    # The original 48-case validation set is used together with the newer
    # annotation set when creating alternative human train/test splits.
    ratings_df = load_docx_rating_table(VALIDATION48_DOCX)
    human_cols = [col for col in ratings_df.columns if col not in ["KLIP", "Pipeline"]]
    if len(human_cols) != 7:
        raise ValueError(f"Expected 7 human columns in {VALIDATION48_DOCX}, found {len(human_cols)}: {human_cols}")

    for col in human_cols:
        ratings_df[col] = pd.to_numeric(ratings_df[col], errors="raise").astype(int)

    ratings_df["positive_votes"] = ratings_df[human_cols].sum(axis=1).astype(int)
    ratings_df["agreement_votes"] = ratings_df["positive_votes"].apply(lambda x: max(int(x), 7 - int(x))).astype(int)
    ratings_df["human_label"] = (ratings_df["positive_votes"] >= 4).astype(int)
    ratings_df["human_weight"] = ratings_df["agreement_votes"].map(HUMAN_WEIGHT_BY_AGREEMENT).astype(float)
    ratings_df = ratings_df.rename(columns={"KLIP": "clip_id"})

    validation_clip_mapping_df = map_validation_clips_to_candidates(VALIDATION48_REFERENCE_DIR)
    validation48 = validation_clip_mapping_df.merge(
        ratings_df[["clip_id", "human_label", "positive_votes", "agreement_votes", "human_weight"]],
        on="clip_id",
        how="inner",
        validate="one_to_one",
    )
    # Keep the original source label in the output files to avoid changing
    # previously generated report tables.
    validation48["human_source"] = "old48"
    return normalize_candidate_keys(validation48)


def load_new170_human_cases() -> pd.DataFrame:
    # Exclude clips that were marked as "fejl" during annotation, since they
    # should not influence either training or testing.
    new_df = load_new_human_annotations(NEW170_DOCX, NEW170_SELECTION_CSV)
    new_df = new_df[(~new_df["annotation_error"]) & (new_df["votes"].notna())].copy()
    new_df = new_df.rename(columns={"display_candidate_id": "clip_id"})
    new_df["human_source"] = "new170"
    keep_cols = [
        "match_id",
        "start_event_id",
        "clip_id",
        "human_label",
        "positive_votes",
        "agreement_votes",
        "human_weight",
        "human_source",
    ]
    return normalize_candidate_keys(new_df[keep_cols])


def combine_human_cases() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Combine the two human annotation rounds into one pool for resampling.
    raw = pd.concat([load_new170_human_cases(), load_validation48_human_cases()], ignore_index=True)
    raw["human_label"] = raw["human_label"].astype(int)
    raw["agreement_votes"] = raw["agreement_votes"].astype(int)
    raw["human_weight"] = raw["human_weight"].astype(float)

    duplicate_mask = raw.duplicated(["match_id", "start_event_id"], keep=False)
    duplicates = raw.loc[duplicate_mask].sort_values(["match_id", "start_event_id", "agreement_votes"]).copy()

    # Keep one record per actual candidate. Prefer stronger agreement; if the
    # agreement is tied, keep the original validation label for consistency.
    priority = {"new170": 0, "old48": 1}
    raw["_source_priority"] = raw["human_source"].map(priority).fillna(0).astype(int)
    unique = (
        raw.sort_values(["match_id", "start_event_id", "agreement_votes", "_source_priority"], ascending=[True, True, False, False])
        .drop_duplicates(["match_id", "start_event_id"], keep="first")
        .drop(columns=["_source_priority"])
        .reset_index(drop=True)
    )
    return unique, duplicates


def load_feature_cols(feature_source: str) -> list[str]:
    # Default is the same feature set used by the current trained model.
    if feature_source == "shared_human_eval":
        payload = json.loads(SHARED_FEATURE_SUMMARY_JSON.read_text(encoding="utf-8"))
        return payload["shared_features"]

    payload = json.loads(MODEL_METRICS_JSON.read_text(encoding="utf-8"))
    return payload["selected_features"]


def prepare_candidates() -> pd.DataFrame:
    # This is the full regain-candidate pool with rule labels and engineered
    # features already created by the earlier preprocessing scripts.
    candidates = pd.read_csv(CANDIDATES_CSV, low_memory=False)
    candidates = normalize_candidate_keys(candidates)
    candidates["team_id"] = candidates["team_id"].astype(str)
    candidates["label_counterattack"] = (
        pd.to_numeric(candidates["label_counterattack"], errors="coerce").fillna(0).astype(int)
    )
    return candidates


def split_train_validation(df: pd.DataFrame, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    # The internal validation split is match-grouped so threshold tuning does
    # not see another candidate from the same match.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, val_idx = next(splitter.split(df, groups=df["match_id"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def summarize_results(iteration_df: pd.DataFrame) -> pd.DataFrame:
    # Report both central tendency and spread because the point of this script
    # is to show how sensitive the model is to the human train/test split.
    metric_cols = ["precision", "recall", "f1", "accuracy"]
    rows = []
    for model_name, group in iteration_df.groupby("model"):
        for metric in metric_cols:
            values = group[metric].astype(float)
            rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "p05": float(values.quantile(0.05)),
                    "p25": float(values.quantile(0.25)),
                    "median": float(values.quantile(0.50)),
                    "p75": float(values.quantile(0.75)),
                    "p95": float(values.quantile(0.95)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def paired_sign_flip_pvalues(
    differences: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
    batch_size: int = 10_000,
) -> dict[str, float]:
    # Paired sign-flip test: if there is no systematic model difference, the
    # sign of each split-level difference is exchangeable.
    observed_mean = float(np.mean(differences))
    one_sided_count = 0
    two_sided_count = 0
    completed = 0

    signs_values = np.array([-1.0, 1.0])
    while completed < n_permutations:
        batch_n = min(batch_size, n_permutations - completed)
        signs = rng.choice(signs_values, size=(batch_n, differences.size))
        null_means = (signs * differences).mean(axis=1)
        one_sided_count += int((null_means >= observed_mean).sum())
        two_sided_count += int((np.abs(null_means) >= abs(observed_mean)).sum())
        completed += batch_n

    return {
        "p_value_one_sided_hybrid_better": float((one_sided_count + 1) / (n_permutations + 1)),
        "p_value_two_sided": float((two_sided_count + 1) / (n_permutations + 1)),
    }


def summarize_paired_differences(
    iteration_df: pd.DataFrame,
    n_permutations: int,
    random_state: int,
) -> pd.DataFrame:
    # Paired comparisons use the same held-out human cases in each iteration,
    # so the difference is calculated within split before summarizing.
    metric_cols = ["precision", "recall", "f1", "accuracy"]
    comparisons = [
        ("hybrid_minus_baseline", "hybrid_xgboost", "xgboost_baseline"),
        ("hybrid_minus_rule_definition", "hybrid_xgboost", "rule_definition"),
    ]
    rows = []
    rng = np.random.default_rng(random_state)
    for metric in metric_cols:
        wide = iteration_df.pivot(index="iteration", columns="model", values=metric)
        for comparison_name, left_model, right_model in comparisons:
            values = (wide[left_model] - wide[right_model]).astype(float)
            p_values = paired_sign_flip_pvalues(values.to_numpy(), n_permutations, rng)
            rows.append(
                {
                    "comparison": comparison_name,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "p05": float(values.quantile(0.05)),
                    "p25": float(values.quantile(0.25)),
                    "median": float(values.quantile(0.50)),
                    "p75": float(values.quantile(0.75)),
                    "p95": float(values.quantile(0.95)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "positive_share": float((values > 0).mean()),
                    **p_values,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model inputs.
    candidates = prepare_candidates()
    feature_cols = load_feature_cols(args.feature_source)
    missing_features = [col for col in feature_cols if col not in candidates.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns in candidate table: {missing_features}")
    cat_cols = [col for col in PREFERRED_CAT_COLS if col in feature_cols]

    human_cases, duplicate_human_cases = combine_human_cases()
    candidate_keys = set(zip(candidates["match_id"], candidates["start_event_id"]))
    missing_humans = human_cases[
        ~human_cases[["match_id", "start_event_id"]].apply(tuple, axis=1).isin(candidate_keys)
    ].copy()
    if not missing_humans.empty:
        raise RuntimeError(f"{len(missing_humans)} human-labeled cases were not found in the candidate table.")

    if len(human_cases) <= args.human_test_size:
        raise ValueError(
            f"human-test-size={args.human_test_size} leaves no training cases because only {len(human_cases)} unique human cases exist."
        )

    splitter = StratifiedShuffleSplit(
        n_splits=args.n_iterations,
        test_size=args.human_test_size,
        random_state=args.random_state,
    )

    candidate_index = candidates.set_index(["match_id", "start_event_id"], drop=False)
    human_x = human_cases[["match_id", "start_event_id"]]
    human_y = human_cases["human_label"].astype(int).to_numpy()

    # Main split-perturbation loop.
    iteration_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    start = perf_counter()
    for iteration_idx, (human_train_idx, human_test_idx) in enumerate(splitter.split(human_x, human_y), start=1):
        iteration_seed = args.random_state + iteration_idx
        human_train = human_cases.iloc[human_train_idx].copy().reset_index(drop=True)
        human_test = human_cases.iloc[human_test_idx].copy().reset_index(drop=True)
        human_test_keys = set(zip(human_test["match_id"], human_test["start_event_id"]))

        # Human test cases are removed entirely from the training pool in this
        # iteration. This is the main leakage guard in the robustness analysis.
        trainval_df = candidates[
            ~candidates[["match_id", "start_event_id"]].apply(tuple, axis=1).isin(human_test_keys)
        ].copy()

        override_cols = ["match_id", "start_event_id", "human_label", "agreement_votes", "human_weight", "human_source"]
        trainval_df = trainval_df.merge(
            human_train[override_cols],
            on=["match_id", "start_event_id"],
            how="left",
            validate="one_to_one",
        )
        trainval_df["hybrid_label"] = trainval_df["human_label"].where(
            trainval_df["human_label"].notna(), trainval_df["label_counterattack"]
        ).astype(int)
        # Human-labeled rows get agreement-based weights; all rule-only rows
        # keep weight 1.
        trainval_df["sample_weight"] = trainval_df["human_weight"].where(
            trainval_df["human_weight"].notna(), 1.0
        ).astype(float)

        train_df, val_df = split_train_validation(trainval_df, random_state=iteration_seed)

        baseline = fit_model(
            train_df,
            val_df,
            feature_cols,
            cat_cols,
            target_col="label_counterattack",
            sample_weight_col=None,
            random_state=iteration_seed,
        )
        hybrid = fit_model(
            train_df,
            val_df,
            feature_cols,
            cat_cols,
            target_col="hybrid_label",
            sample_weight_col="sample_weight",
            random_state=iteration_seed,
        )

        # Evaluate all three approaches on exactly the same human-held-out cases.
        test_df = candidate_index.loc[list(human_test_keys)].copy().reset_index(drop=True)
        test_df = test_df.merge(
            human_test[["match_id", "start_event_id", "human_label"]],
            on=["match_id", "start_event_id"],
            how="inner",
            validate="one_to_one",
        )
        test_df["human_label_test"] = test_df["human_label"].astype(int)

        y_true = test_df["human_label_test"].astype(int).to_numpy()
        rule_pred = test_df["label_counterattack"].astype(int).to_numpy()
        rule_metrics = evaluate_predictions(y_true, rule_pred, rule_pred.astype(float))
        baseline_metrics, _ = predict_on_test(baseline, test_df, feature_cols, cat_cols, "human_label_test")
        hybrid_metrics, _ = predict_on_test(hybrid, test_df, feature_cols, cat_cols, "human_label_test")

        for model_name, metrics, bundle in [
            ("rule_definition", rule_metrics, None),
            ("xgboost_baseline", baseline_metrics, baseline),
            ("hybrid_xgboost", hybrid_metrics, hybrid),
        ]:
            row = {
                "iteration": iteration_idx,
                "model": model_name,
                "human_train_n": int(len(human_train)),
                "human_test_n": int(len(human_test)),
                "human_test_positive": int(human_test["human_label"].sum()),
                "human_test_negative": int((human_test["human_label"] == 0).sum()),
                "human_train_positive": int(human_train["human_label"].sum()),
                "human_train_negative": int((human_train["human_label"] == 0).sum()),
                **metrics,
            }
            if bundle is not None:
                row.update(
                    {
                        "threshold": float(bundle["threshold"]),
                        "validation_f1": float(bundle["validation_f1"]),
                        "validation_precision": float(bundle["validation_precision"]),
                        "validation_recall": float(bundle["validation_recall"]),
                        "train_positive_rate": float(bundle["train_positive_rate"]),
                        "val_positive_rate": float(bundle["val_positive_rate"]),
                    }
                )
            iteration_rows.append(row)

        if args.save_predictions:
            prediction_rows.append(
                test_df[["match_id", "start_event_id", "human_label_test", "label_counterattack"]].assign(
                    iteration=iteration_idx
                )
            )

        elapsed = perf_counter() - start
        print(
            f"iteration {iteration_idx:03d}/{args.n_iterations} "
            f"baseline_f1={baseline_metrics['f1']:.3f} hybrid_f1={hybrid_metrics['f1']:.3f} "
            f"elapsed={elapsed:.1f}s"
        )

    iteration_df = pd.DataFrame(iteration_rows)
    summary_df = summarize_results(iteration_df)
    paired_diff_df = summarize_paired_differences(
        iteration_df,
        n_permutations=args.n_permutations,
        random_state=args.random_state,
    )

    iteration_path = args.output_dir / "resampling_iteration_metrics.csv"
    summary_path = args.output_dir / "resampling_summary.csv"
    paired_diff_path = args.output_dir / "resampling_paired_differences.csv"
    metadata_path = args.output_dir / "resampling_metadata.json"
    duplicate_path = args.output_dir / "human_case_duplicates.csv"
    human_cases_path = args.output_dir / "human_cases_unique.csv"

    # Save both the final report tables and the intermediate case mappings.
    iteration_df.to_csv(iteration_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    paired_diff_df.to_csv(paired_diff_path, index=False)
    duplicate_human_cases.to_csv(duplicate_path, index=False)
    human_cases.to_csv(human_cases_path, index=False)

    if args.save_predictions and prediction_rows:
        pd.concat(prediction_rows, ignore_index=True).to_csv(args.output_dir / "resampling_test_predictions.csv", index=False)

    metadata = {
        "n_iterations": args.n_iterations,
        "human_test_size": args.human_test_size,
        "random_state": args.random_state,
        "feature_source": args.feature_source,
        "feature_count": len(feature_cols),
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "human_raw_cases": int(len(load_new170_human_cases()) + len(load_validation48_human_cases())),
        "human_unique_cases": int(len(human_cases)),
        "human_duplicate_rows": int(len(duplicate_human_cases)),
        "human_positive_unique": int(human_cases["human_label"].sum()),
        "human_negative_unique": int((human_cases["human_label"] == 0).sum()),
        "human_weight_scheme": HUMAN_WEIGHT_BY_AGREEMENT,
        "xgb_params": BEST_XGB_PARAMS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "paired_p_value_method": "Monte Carlo paired sign-flip permutation test on split-level metric differences.",
        "n_permutations": int(args.n_permutations),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nSummary")
    print(summary_df.to_string(index=False))
    print("\nPaired differences")
    print(paired_diff_df.to_string(index=False))
    print(f"\nSaved: {iteration_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {paired_diff_path}")
    print(f"Saved: {metadata_path}")
    print(f"Saved: {human_cases_path}")
    print(f"Saved: {duplicate_path}")


if __name__ == "__main__":
    main()
