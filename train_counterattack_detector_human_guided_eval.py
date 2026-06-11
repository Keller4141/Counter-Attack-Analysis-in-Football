#!/usr/bin/env python3
"""Train and evaluate the human-guided XGBoost counterattack model.

This script compares three approaches on the fixed 48-case human validation set:
- the rule-based definition,
- a baseline XGBoost model trained only on rule labels,
- a hybrid XGBoost model where the 170 human annotations override rule labels
  in the training data and are weighted by annotator agreement.
"""

import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from train_counterattack_detector import (
    BEST_XGB_PARAMS,
    EARLY_STOPPING_ROUNDS,
    PREFERRED_CAT_COLS,
    XGBClassifier,
    build_preprocessor,
    optimize_threshold,
)


ROOT = Path(__file__).resolve().parent
DERIVED_DIR = ROOT / "Data" / "derived"
ANNOTATION_DIR = ROOT / "Data" / "annotations"
AI_DIR = DERIVED_DIR / "ai_counterattack_detection"
HUMAN_SET_DIR = DERIVED_DIR / "human_annotation_set_50_50_20_20_20_20_current_746_terminal_filtered"
LEGACY_DIR = DERIVED_DIR / "validation_legacy_broad_pool"
OUTPUT_DIR = AI_DIR / "human_guided_old48_eval"
SHARED_FEATURE_OUTPUT_DIR = AI_DIR / "shared_feature_old48_eval"
DBU_ANALYSIS_DIR = DERIVED_DIR / "dbu_mens_hybrid_analysis"
DBU_SCORED_CANDIDATES_CSV = DBU_ANALYSIS_DIR / "all_regain_candidates_hybrid_scored.csv"

NEW_ANNOTATION_DOCX = ANNOTATION_DIR / "ANNOTATION_SET.docx"
VALIDATION48_ANNOTATION_DOCX = ANNOTATION_DIR / "Kontra vurdering (1).docx"
NEW_SELECTION_CSV = HUMAN_SET_DIR / "annotation_selection.csv"
REFERENCE_SNAPSHOT_DIR = ANNOTATION_DIR / "validation_48" / "reference_snapshots"
MODEL_METRICS_JSON = AI_DIR / "model_metrics.json"
CANDIDATES_CSV = AI_DIR / "regain_candidates_labeled.csv"

HUMAN_WEIGHT_BY_AGREEMENT = {
    4: 3.0,
    5: 5.0,
    6: 8.0,
    7: 12.0,
}

# Fixed top-30 feature specification used for the report-facing fair comparison.
# Keeping it here makes the shared-feature outputs reproducible instead of
# relying on a pre-existing CSV/JSON file in Data/derived.
REPORT_SHARED_FEATURES = [
    "ten_progression_goal_m",
    "fifteen_progression_goal_m",
    "long_progression_goal_m",
    "long_avg_progress_speed_mps",
    "fifteen_end_progress_m",
    "long_end_progress_m",
    "first_action_type",
    "long_min_goal_distance_m",
    "fifteen_avg_progress_speed_mps",
    "in_play_15s",
    "fifteen_duration_s",
    "mid_progression_goal_m",
    "in_play_20s",
    "event_count_pass_15s",
    "stop_reason",
    "fifteen_progression_x_m",
    "mid_progression_x_m",
    "long_duration_s",
    "long_progression_x_m",
    "ten_avg_progress_speed_mps",
    "mid_min_goal_distance_m",
    "ten_start_progress_m",
    "event_count_duel_15s",
    "event_count_pass_20s",
    "event_count_shot_10s",
    "event_count_shot_15s",
    "mid_start_progress_m",
    "event_count_duel_20s",
    "fifteen_min_goal_distance_m",
    "in_play_10s",
]

NON_FEATURE_COLS = {
    "candidate_cache_version",
    "match_id",
    "team_id",
    "start_event_id",
    "label_counterattack",
    "source_label",
}
EXCLUDED_MODEL_COLS = {
    "start_ts",
    "attacking_left",
    "tracking_start_x_m_raw",
    "tracking_start_y_m_raw",
}


def safe_div(num, den):
    return num / den if den else 0.0


def load_selected_features():
    # Feature list used by the current trained model and the DBU analysis.
    metrics = json.loads(MODEL_METRICS_JSON.read_text(encoding="utf-8"))
    feature_cols = metrics["selected_features"]
    cat_cols = [col for col in PREFERRED_CAT_COLS if col in feature_cols]
    return feature_cols, cat_cols


def load_new_human_annotations(docx_path: Path, selection_csv_path: Path):
    # The new170 annotation file is easier to parse as text lines because it was
    # generated from a video-annotation document rather than one clean table.
    with ZipFile(docx_path) as zf:
        xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
    text = re.sub(r"</w:p>", "\n", xml)
    text = re.sub(r"<.*?>", "", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    start_idx = None
    for idx, line in enumerate(lines):
        if line == "Christian":
            start_idx = idx + 1
            break
    if start_idx is None:
        raise RuntimeError("Could not locate annotation table header in ANNOTATION_SET.docx")

    parsed = []
    i = start_idx
    while i < len(lines):
        token = lines[i]
        match = re.match(r"^(\d+)(.*)$", token)
        if not match:
            i += 1
            continue

        candidate_num = int(match.group(1))
        suffix = match.group(2).strip().lower()
        is_error = "fejl" in suffix
        row = {
            "display_candidate_num": candidate_num,
            "annotation_error": is_error,
            "raw_row_token": token,
        }

        if is_error:
            # "fejl" marks clips excluded because the video/tracking was not reliable.
            row["votes"] = None
            parsed.append(row)
            i += 1
            continue

        votes = []
        for offset in range(1, 8):
            if i + offset >= len(lines):
                break
            vote = lines[i + offset]
            if vote in {"0", "1"}:
                votes.append(int(vote))
            else:
                break
        if len(votes) != 7:
            raise RuntimeError(f"Could not parse 7 votes for candidate {candidate_num} (token={token!r})")

        positive_votes = int(sum(votes))
        agreement = int(max(positive_votes, 7 - positive_votes))
        row.update(
            {
                "votes": votes,
                "positive_votes": positive_votes,
                "human_label": int(positive_votes >= 4),
                "agreement_votes": agreement,
                "human_weight": float(HUMAN_WEIGHT_BY_AGREEMENT[agreement]),
            }
        )
        parsed.append(row)
        i += 8

    parsed_df = pd.DataFrame(parsed)
    selection_df = pd.read_csv(selection_csv_path)
    selection_df["display_candidate_num"] = (
        selection_df["display_candidate_id"].str.extract(r"candidate_(\d+)").astype(int)
    )
    merged = selection_df.merge(parsed_df, on="display_candidate_num", how="left", validate="one_to_one")
    merged["match_id"] = merged["match_id"].astype(str)
    merged["start_event_id"] = merged["start_event_id"].astype(str)
    return merged


def load_docx_rating_table(path: Path) -> pd.DataFrame:
    # DOCX files are zipped XML files. The validation file is a normal Word
    # table, so it can be parsed directly from the table structure.
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with ZipFile(path) as zf:
        xml = zf.read("word/document.xml")

    root = ET.fromstring(xml)
    all_rows: list[list[str]] = []
    header = None

    for table_idx, tbl in enumerate(root.findall(".//w:tbl", ns), start=1):
        table_rows: list[list[str]] = []
        for tr in tbl.findall("w:tr", ns):
            row: list[str] = []
            for tc in tr.findall("w:tc", ns):
                texts = [t.text or "" for t in tc.findall(".//w:t", ns)]
                row.append("".join(texts).strip())
            if any(cell for cell in row):
                table_rows.append(row)

        if not table_rows:
            continue
        if table_idx == 1:
            header = table_rows[1]
            data_rows = table_rows[2:]
        else:
            data_rows = table_rows
        all_rows.extend(data_rows)

    if header is None:
        raise ValueError(f"Could not find a table header in {path}")

    df = pd.DataFrame(all_rows, columns=header)
    df = df[df["KLIP"].astype(str).str.fullmatch(r"\d+")].copy()
    df["KLIP"] = df["KLIP"].astype(int)
    return df.sort_values("KLIP").reset_index(drop=True)


def load_validation48_human_labels_from_docx(docx_path: Path):
    labels_df = load_docx_rating_table(docx_path)
    human_cols = [col for col in labels_df.columns if col not in ["KLIP", "Pipeline"]]
    if len(human_cols) != 7:
        raise ValueError(f"Expected 7 human label columns in {docx_path}, found {len(human_cols)}: {human_cols}")

    for col in human_cols:
        labels_df[col] = pd.to_numeric(labels_df[col], errors="raise").astype(int)

    labels_df["positive_votes"] = labels_df[human_cols].sum(axis=1).astype(int)
    labels_df["human_label"] = (labels_df["positive_votes"] >= 4).astype(int)
    labels_df = labels_df.rename(columns={"KLIP": "clip_id"})
    return labels_df[["clip_id", "human_label"]].copy()


def load_validation48_candidate_lookup(manual_csv_path: Path):
    with manual_csv_path.open(newline="", encoding="utf-8") as f:
        next(f)
        reader = csv.DictReader(f, delimiter=";")
        rows = list(reader)
    df = pd.DataFrame(rows)
    df["candidate_id"] = df["candidate_id"].astype(int)
    df["match_id"] = df["match_id"].astype(str)
    df["start_event_id"] = df["start_event_id"].astype(str)
    return df[["candidate_id", "snapshot_file", "match_id", "start_event_id", "match_label", "team_name"]].copy()


def map_validation_clips_to_candidates(reference_dir: Path, validation_lookup_df=None):
    # The snapshot filenames contain the clip id and original candidate identity.
    clip_files = sorted(reference_dir.glob("Clip*_*.png"), key=lambda p: int(p.stem.split("_")[0].replace("Clip", "")))
    if len(clip_files) != 48:
        raise RuntimeError(f"Expected 48 Clip*.png files, found {len(clip_files)}")

    rows = []
    for clip_path in clip_files:
        match = re.match(r"^Clip(\d+)_(\d+)_(\d+)_(\d+)_final([01])\.png$", clip_path.name)
        if not match:
            raise RuntimeError(f"Could not parse clip mapping from {clip_path.name}")
        clip_id = int(match.group(1))
        candidate_id = int(match.group(2))
        match_id = match.group(3)
        start_event_id = match.group(4)
        rec = {
            "match_id": match_id,
            "start_event_id": start_event_id,
            "clip_id": clip_id,
            "candidate_id": candidate_id,
            "current_clip_file": clip_path.name,
            "matched_reference_snapshot": clip_path.name,
        }
        if validation_lookup_df is not None:
            lookup_row = validation_lookup_df.loc[
                (validation_lookup_df["match_id"] == match_id)
                & (validation_lookup_df["start_event_id"] == start_event_id)
            ]
        else:
            lookup_row = pd.DataFrame()
        if not lookup_row.empty:
            lookup_dict = lookup_row.iloc[0].to_dict()
            rec.update({
                "match_label": lookup_dict.get("match_label"),
                "team_name": lookup_dict.get("team_name"),
                "snapshot_file": lookup_dict.get("snapshot_file"),
            })
        rows.append(rec)

    mapped_df = pd.DataFrame(rows).sort_values("clip_id").reset_index(drop=True)
    return mapped_df


def split_train_validation(df: pd.DataFrame, target_col: str):
    # Keep matches grouped so the same match does not appear in both train and validation.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=43)
    train_idx, val_idx = next(splitter.split(df, y=df[target_col], groups=df["match_id"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df


def fit_model(train_df, val_df, feature_cols, cat_cols, target_col, sample_weight_col=None, random_state=42):
    if XGBClassifier is None:
        raise RuntimeError("XGBoost is not available in the current environment.")

    num_cols = [col for col in feature_cols if col not in cat_cols]
    work_train = train_df.copy()
    work_val = val_df.copy()
    for col in num_cols + [target_col]:
        work_train[col] = pd.to_numeric(work_train[col], errors="coerce")
        work_val[col] = pd.to_numeric(work_val[col], errors="coerce")

    y_train = work_train[target_col].astype(int).to_numpy()
    y_val = work_val[target_col].astype(int).to_numpy()

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    # Positive counterattacks are rare, so XGBoost receives a class-imbalance correction.
    scale_pos_weight = float(neg / pos) if pos else 1.0

    preprocessor = build_preprocessor(feature_cols, cat_cols)
    X_train = preprocessor.fit_transform(work_train[feature_cols])
    X_val = preprocessor.transform(work_val[feature_cols])

    sample_weight = None
    if sample_weight_col is not None:
        # In the hybrid model, human-labeled cases are weighted by annotator agreement.
        sample_weight = work_train[sample_weight_col].astype(float).to_numpy()

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=4,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **BEST_XGB_PARAMS,
    )
    fit_kwargs = {"eval_set": [(X_val, y_val)], "verbose": False}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(X_train, y_train, **fit_kwargs)

    val_prob = model.predict_proba(X_val)[:, 1]
    # Select the decision threshold on validation F1, then keep it fixed for the 48-case test set.
    threshold_result = optimize_threshold(y_val, val_prob)

    return {
        "preprocessor": preprocessor,
        "model": model,
        "threshold": float(threshold_result["threshold"]),
        "validation_precision": float(threshold_result["precision"]),
        "validation_recall": float(threshold_result["recall"]),
        "validation_f1": float(threshold_result["f1"]),
        "train_positive_rate": float(y_train.mean()),
        "val_positive_rate": float(y_val.mean()),
    }


def evaluate_predictions(y_true, y_pred, y_prob):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def predict_on_test(model_bundle, test_df, feature_cols, cat_cols, test_label_col):
    num_cols = [col for col in feature_cols if col not in cat_cols]
    work_test = test_df.copy()
    for col in num_cols + [test_label_col]:
        work_test[col] = pd.to_numeric(work_test[col], errors="coerce")
    X_test = model_bundle["preprocessor"].transform(work_test[feature_cols])
    y_true = work_test[test_label_col].astype(int).to_numpy()
    y_prob = model_bundle["model"].predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= model_bundle["threshold"]).astype(int)
    metrics = evaluate_predictions(y_true, y_pred, y_prob)
    pred_df = work_test[["match_id", "team_id", "start_event_id", test_label_col, "label_counterattack"]].copy()
    pred_df["pred_probability"] = y_prob
    pred_df["pred_label"] = y_pred
    return metrics, pred_df


def format_shared_predictions(test_df, model_bundle, feature_cols):
    """Create the compact prediction files used by the final validation notebook."""
    work_test = test_df.copy()
    X_test = model_bundle["preprocessor"].transform(work_test[feature_cols])
    y_prob = model_bundle["model"].predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= model_bundle["threshold"]).astype(int)

    out = work_test[["clip_id", "match_id", "start_event_id", "human_label_test"]].copy()
    out = out.rename(columns={"human_label_test": "human_label"})
    out["pred_probability"] = y_prob
    out["pred_label"] = y_pred
    return out


def metrics_for_shared_summary(metrics, model_bundle=None):
    """Keep the shared-feature summary compact and report-facing."""
    keys = ["tp", "fp", "fn", "tn", "precision", "recall", "accuracy", "f1"]
    out = {key: metrics[key] for key in keys}
    if "roc_auc" in metrics:
        out["roc_auc"] = metrics["roc_auc"]
    if model_bundle is not None:
        out["threshold"] = float(model_bundle["threshold"])
    return out


def save_shared_feature_outputs(
    baseline_model,
    hybrid_model,
    test_df,
    feature_cols,
    cat_cols,
    rule_metrics,
    baseline_metrics,
    hybrid_metrics,
):
    """Save the fixed top-30 comparison files used by the validation notebook."""
    SHARED_FEATURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_shared = format_shared_predictions(test_df, baseline_model, feature_cols)
    hybrid_shared = format_shared_predictions(test_df, hybrid_model, feature_cols)
    baseline_shared.to_csv(SHARED_FEATURE_OUTPUT_DIR / "shared_feature_baseline_old48_predictions.csv", index=False)
    hybrid_shared.to_csv(SHARED_FEATURE_OUTPUT_DIR / "shared_feature_hybrid_old48_predictions.csv", index=False)

    summary = {
        "shared_features": feature_cols,
        "cat_cols": cat_cols,
        "definition_on_old48": metrics_for_shared_summary(rule_metrics),
        "baseline_on_old48": metrics_for_shared_summary(baseline_metrics, baseline_model),
        "hybrid_on_old48": metrics_for_shared_summary(hybrid_metrics, hybrid_model),
    }
    (SHARED_FEATURE_OUTPUT_DIR / "shared_feature_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def score_all_candidates_with_hybrid_model(candidates_df, hybrid_model, feature_cols, cat_cols):
    """Score every regain candidate with the report-facing hybrid model for DBU analysis."""
    scored = candidates_df.copy()
    num_cols = [col for col in feature_cols if col not in cat_cols]
    for col in num_cols:
        scored[col] = pd.to_numeric(scored[col], errors="coerce")
    X_all = hybrid_model["preprocessor"].transform(scored[feature_cols])
    prob = hybrid_model["model"].predict_proba(X_all)[:, 1]
    scored["hybrid_probability"] = prob
    scored["hybrid_threshold"] = float(hybrid_model["threshold"])
    scored["hybrid_label"] = (prob >= hybrid_model["threshold"]).astype(int)
    return scored


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SHARED_FEATURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DBU_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    candidates_df = pd.read_csv(CANDIDATES_CSV, low_memory=False)
    candidates_df["match_id"] = candidates_df["match_id"].astype(str)
    candidates_df["start_event_id"] = candidates_df["start_event_id"].astype(str)
    candidates_df["team_id"] = candidates_df["team_id"].astype(str)
    candidates_df["label_counterattack"] = pd.to_numeric(candidates_df["label_counterattack"], errors="coerce").fillna(0).astype(int)
    missing_features = [col for col in REPORT_SHARED_FEATURES if col not in candidates_df.columns]
    if missing_features:
        raise RuntimeError(f"Missing report feature columns in candidate table: {missing_features}")
    feature_cols = REPORT_SHARED_FEATURES
    cat_cols = [col for col in PREFERRED_CAT_COLS if col in feature_cols]

    new_human_df = load_new_human_annotations(NEW_ANNOTATION_DOCX, NEW_SELECTION_CSV)
    validation_labels_df = load_validation48_human_labels_from_docx(VALIDATION48_ANNOTATION_DOCX)
    validation_clip_mapping_df = map_validation_clips_to_candidates(REFERENCE_SNAPSHOT_DIR)
    validation_test_df = validation_clip_mapping_df.merge(validation_labels_df, on="clip_id", how="inner", validate="one_to_one")

    test_keys = set(zip(validation_test_df["match_id"], validation_test_df["start_event_id"]))

    # Build the fixed 48-case test dataframe from the current candidate table.
    candidate_lookup = candidates_df.set_index(["match_id", "start_event_id"], drop=False)
    test_rows = []
    missing_test = []
    for row in validation_test_df.to_dict("records"):
        key = (row["match_id"], row["start_event_id"])
        if key not in candidate_lookup.index:
            missing_test.append(row)
            continue
        candidate_row = candidate_lookup.loc[key]
        if isinstance(candidate_row, pd.DataFrame):
            candidate_row = candidate_row.iloc[0]
        merged = candidate_row.to_dict()
        merged.update(row)
        test_rows.append(merged)
    if missing_test:
        raise RuntimeError(f"{len(missing_test)} human-labeled test cases were not found in current candidate table.")
    test_df = pd.DataFrame(test_rows)
    test_df["human_label_test"] = test_df["human_label"].astype(int)

    # Remove fixed test cases from the training pool to avoid leakage.
    trainval_df = candidates_df[
        ~candidates_df.apply(lambda row: (row["match_id"], row["start_event_id"]) in test_keys, axis=1)
    ].copy()

    # Merge the 170 human labels into the training pool and remove any overlap with the fixed test set.
    usable_new_humans = new_human_df[(~new_human_df["annotation_error"]) & (new_human_df["votes"].notna())].copy()
    usable_new_humans = usable_new_humans[
        ~usable_new_humans.apply(lambda row: (row["match_id"], row["start_event_id"]) in test_keys, axis=1)
    ].copy()
    usable_new_humans = usable_new_humans[
        ["match_id", "start_event_id", "display_candidate_id", "human_label", "agreement_votes", "human_weight", "positive_votes", "group_name"]
    ]

    trainval_df = trainval_df.merge(
        usable_new_humans,
        on=["match_id", "start_event_id"],
        how="left",
    )
    trainval_df["hybrid_label"] = trainval_df["human_label"].fillna(trainval_df["label_counterattack"]).astype(int)
    trainval_df["sample_weight"] = trainval_df["human_weight"].fillna(1.0).astype(float)

    # Use the same train/validation split for baseline and hybrid, so the comparison is fair.
    train_df, val_df = split_train_validation(trainval_df, "hybrid_label")

    baseline_model = fit_model(train_df, val_df, feature_cols, cat_cols, target_col="label_counterattack")
    hybrid_model = fit_model(train_df, val_df, feature_cols, cat_cols, target_col="hybrid_label", sample_weight_col="sample_weight")

    baseline_metrics, baseline_preds = predict_on_test(
        baseline_model,
        test_df,
        feature_cols,
        cat_cols,
        test_label_col="human_label_test",
    )
    hybrid_metrics, hybrid_preds = predict_on_test(
        hybrid_model,
        test_df,
        feature_cols,
        cat_cols,
        test_label_col="human_label_test",
    )

    rule_metrics = evaluate_predictions(
        test_df["human_label_test"].astype(int).to_numpy(),
        test_df["label_counterattack"].astype(int).to_numpy(),
        test_df["label_counterattack"].astype(float).to_numpy(),
    )

    summary = {
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "human_weight_scheme": HUMAN_WEIGHT_BY_AGREEMENT,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "new_human_annotations_used_for_training": int(len(usable_new_humans)),
        "train_human_overrides": int(train_df["human_label"].notna().sum()),
        "validation_human_overrides": int(val_df["human_label"].notna().sum()),
        "baseline_validation_threshold": float(baseline_model["threshold"]),
        "baseline_validation_f1": float(baseline_model["validation_f1"]),
        "hybrid_validation_threshold": float(hybrid_model["threshold"]),
        "hybrid_validation_f1": float(hybrid_model["validation_f1"]),
        "rule_definition_on_old48": rule_metrics,
        "baseline_model_on_old48": baseline_metrics,
        "hybrid_model_on_old48": hybrid_metrics,
    }

    baseline_preds.to_csv(OUTPUT_DIR / "baseline_old48_predictions.csv", index=False)
    hybrid_preds.to_csv(OUTPUT_DIR / "hybrid_old48_predictions.csv", index=False)
    validation_clip_mapping_df.to_csv(OUTPUT_DIR / "old48_clip_mapping.csv", index=False)
    usable_new_humans.to_csv(OUTPUT_DIR / "new170_human_annotations_used.csv", index=False)
    (OUTPUT_DIR / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    save_shared_feature_outputs(
        baseline_model=baseline_model,
        hybrid_model=hybrid_model,
        test_df=test_df,
        feature_cols=feature_cols,
        cat_cols=cat_cols,
        rule_metrics=rule_metrics,
        baseline_metrics=baseline_metrics,
        hybrid_metrics=hybrid_metrics,
    )

    dbu_feature_cols, dbu_cat_cols = load_selected_features()
    missing_dbu_features = [col for col in dbu_feature_cols if col not in candidates_df.columns]
    if missing_dbu_features:
        raise RuntimeError(f"Missing DBU scoring feature columns in candidate table: {missing_dbu_features}")

    if dbu_feature_cols == feature_cols:
        dbu_hybrid_model = hybrid_model
    else:
        # The DBU section was built from the current model feature specification
        # saved in model_metrics.json, so this model is trained separately from
        # the shared-feature validation model above.
        dbu_hybrid_model = fit_model(
            train_df,
            val_df,
            dbu_feature_cols,
            dbu_cat_cols,
            target_col="hybrid_label",
            sample_weight_col="sample_weight",
        )
    scored_candidates = score_all_candidates_with_hybrid_model(candidates_df, dbu_hybrid_model, dbu_feature_cols, dbu_cat_cols)
    scored_candidates.to_csv(DBU_SCORED_CANDIDATES_CSV, index=False)

    print(json.dumps(summary, indent=2))
    print(f"Saved: {OUTPUT_DIR / 'evaluation_summary.json'}")
    print(f"Saved: {OUTPUT_DIR / 'old48_clip_mapping.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'new170_human_annotations_used.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'baseline_old48_predictions.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'hybrid_old48_predictions.csv'}")
    print(f"Saved: {SHARED_FEATURE_OUTPUT_DIR / 'shared_feature_summary.json'}")
    print(f"Saved: {SHARED_FEATURE_OUTPUT_DIR / 'shared_feature_baseline_old48_predictions.csv'}")
    print(f"Saved: {SHARED_FEATURE_OUTPUT_DIR / 'shared_feature_hybrid_old48_predictions.csv'}")
    print(f"Saved: {DBU_SCORED_CANDIDATES_CSV}")
    print(f"DBU scoring hybrid positives: {int(scored_candidates['hybrid_label'].sum())}")


if __name__ == "__main__":
    main()
