#!/usr/bin/env python3
"""Train the rule-labeled XGBoost baseline counterattack detector.

The model is trained on the regain candidate table produced by
build_regain_candidates.py. At this stage the target labels come only from the
final rule-based counterattack definition, so this script tests how well an
XGBoost model can reproduce that definition from engineered event/tracking
features.

Main outputs:
- Data/derived/ai_counterattack_detection/counterattack_detector.pkl
- Data/derived/ai_counterattack_detection/model_metrics.json
- Data/derived/ai_counterattack_detection/feature_importance.csv
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from build_regain_candidates import DERIVED_DIR, load_or_build_candidates

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


# Model selection configuration.
EARLY_STOPPING_ROUNDS = 50
TOP_K_CANDIDATES = [10, 15, 20, 25, 30, 35, 40, 45, 50]
THRESHOLD_GRID = [round(x, 2) for x in np.arange(0.30, 0.91, 0.02)]

# The final baseline uses the same top-30 feature setup as the later hybrid comparison.
SELECTED_TOP_K = 30

# Columns that identify candidates or labels rather than describing the football situation.
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
PREFERRED_CAT_COLS = [
    "regain_type",
    "match_period",
    "stop_reason",
    "first_action_type",
    "short_start_zone",
    "mid_start_zone",
    "ten_start_zone",
    "fifteen_start_zone",
    "long_start_zone",
]

# Fixed XGBoost hyperparameters used across baseline and later comparisons.
BEST_XGB_PARAMS = {
    "n_estimators": 2000,
    "max_depth": 4,
    "learning_rate": 0.04,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "gamma": 0.2,
    "reg_alpha": 0.5,
    "reg_lambda": 5.0,
    "max_delta_step": 0,
}


def split_groups(model_df, target_col):
    # Split by match_id so the same match cannot appear in both train and test.
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    trainval_idx, test_idx = next(outer.split(model_df, y=model_df[target_col], groups=model_df["match_id"]))
    trainval_df = model_df.iloc[trainval_idx].reset_index(drop=True)
    test_df = model_df.iloc[test_idx].reset_index(drop=True)

    inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=43)
    train_idx, val_idx = next(inner.split(trainval_df, y=trainval_df[target_col], groups=trainval_df["match_id"]))
    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df, test_df


def derive_feature_lists(candidates_df):
    feature_cols = [
        col for col in candidates_df.columns
        if col not in NON_FEATURE_COLS and col not in EXCLUDED_MODEL_COLS
    ]
    cat_cols = [col for col in PREFERRED_CAT_COLS if col in feature_cols]
    return feature_cols, cat_cols


def build_preprocessor(feature_cols, cat_cols):
    # Categorical variables are one-hot encoded; numeric variables use median imputation.
    num_cols = [col for col in feature_cols if col not in cat_cols]
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                cat_cols,
            ),
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                num_cols,
            ),
        ]
    )


def encoded_feature_names(preprocessor, feature_cols, cat_cols):
    num_cols = [col for col in feature_cols if col not in cat_cols]
    names = []
    if cat_cols:
        encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        names.extend(encoder.get_feature_names_out(cat_cols).tolist())
    names.extend(num_cols)
    return names


def aggregate_xgb_gain(preprocessor, model, feature_cols, cat_cols):
    # XGBoost reports gain on encoded columns. This aggregates one-hot encoded
    # categorical levels back to their original raw feature names.
    encoded_names = encoded_feature_names(preprocessor, feature_cols, cat_cols)
    encoded_gain = model.get_booster().get_score(importance_type="gain")
    prefix_to_raw = {f"{col}_": col for col in cat_cols}
    raw_gain = {feature: 0.0 for feature in feature_cols}

    for idx, encoded_name in enumerate(encoded_names):
        raw_name = encoded_name
        for prefix, raw_col in prefix_to_raw.items():
            if encoded_name.startswith(prefix):
                raw_name = raw_col
                break
        raw_gain[raw_name] += float(encoded_gain.get(f"f{idx}", 0.0))

    gain_df = pd.DataFrame(
        [{"feature": feature, "gain": gain} for feature, gain in raw_gain.items()]
    ).sort_values("gain", ascending=False)
    return gain_df


def optimize_threshold(y_true, y_prob):
    # The classification threshold is selected on validation F1, with precision as tie-breaker.
    best = None
    for threshold in THRESHOLD_GRID:
        y_pred = (y_prob >= threshold).astype(int)
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        candidate = {
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if best is None or candidate["f1"] > best["f1"] or (
            candidate["f1"] == best["f1"] and candidate["precision"] > best["precision"]
        ):
            best = candidate
    return best


def fit_xgb_model(train_df, val_df, test_df, feature_cols, cat_cols):
    num_cols = [col for col in feature_cols if col not in cat_cols]
    target_col = "label_counterattack"

    work_train = train_df.copy()
    work_val = val_df.copy()
    work_test = test_df.copy()
    for col in num_cols + [target_col]:
        work_train[col] = pd.to_numeric(work_train[col], errors="coerce")
        work_val[col] = pd.to_numeric(work_val[col], errors="coerce")
        work_test[col] = pd.to_numeric(work_test[col], errors="coerce")

    y_train = work_train[target_col].to_numpy()
    y_val = work_val[target_col].to_numpy()
    y_test = work_test[target_col].to_numpy()

    pos = y_train.sum()
    neg = len(y_train) - pos
    # Counterattacks are rare, so XGBoost is told to weight positive examples more strongly.
    scale_pos_weight = float(neg / pos) if pos else 1.0

    preprocessor = build_preprocessor(feature_cols, cat_cols)
    X_train = preprocessor.fit_transform(work_train[feature_cols])
    X_val = preprocessor.transform(work_val[feature_cols])
    X_test = preprocessor.transform(work_test[feature_cols])

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=4,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **BEST_XGB_PARAMS,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    # The threshold is learned on validation data and then applied unchanged to the test set.
    threshold_result = optimize_threshold(y_val, val_prob)
    threshold = threshold_result["threshold"]
    val_pred = (val_prob >= threshold).astype(int)
    test_pred = (test_prob >= threshold).astype(int)

    metrics = {
        "feature_count": len(feature_cols),
        "selected_features": feature_cols,
        "best_iteration": int(model.best_iteration),
        "prediction_threshold": float(threshold),
        "validation_precision": float(precision_score(y_val, val_pred, zero_division=0)),
        "validation_recall": float(recall_score(y_val, val_pred, zero_division=0)),
        "validation_f1": float(f1_score(y_val, val_pred, zero_division=0)),
        "train_matches": int(work_train["match_id"].nunique()),
        "validation_matches": int(work_val["match_id"].nunique()),
        "test_matches": int(work_test["match_id"].nunique()),
        "train_rows": int(len(work_train)),
        "validation_rows": int(len(work_val)),
        "test_rows": int(len(work_test)),
        "train_positive_rate": float(y_train.mean()),
        "validation_positive_rate": float(y_val.mean()),
        "test_positive_rate": float(y_test.mean()),
        "precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_prob)),
    }

    model_bundle = {
        # The bundle contains everything needed to score new candidates in the same feature space.
        "preprocessor": preprocessor,
        "model": model,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "threshold": float(threshold),
    }
    feature_importance = aggregate_xgb_gain(preprocessor, model, feature_cols, cat_cols)
    return model_bundle, metrics, feature_importance


def train_model(candidates_df):
    if XGBClassifier is None:
        raise RuntimeError("XGBoost is required for the final detector model.")

    target_col = "label_counterattack"
    model_df = candidates_df.copy()
    train_df, val_df, test_df = split_groups(model_df, target_col)
    all_feature_cols, all_cat_cols = derive_feature_lists(model_df)

    # First train a full model to rank all engineered features by gain.
    full_model_bundle, full_metrics, full_importance = fit_xgb_model(
        train_df, val_df, test_df, all_feature_cols, all_cat_cols
    )
    ranked_features = full_importance["feature"].tolist()

    search_results = []
    top_k_values = sorted({k for k in TOP_K_CANDIDATES if k < len(ranked_features)} | {len(ranked_features)})
    best_run = None
    run_lookup = {}
    for top_k in top_k_values:
        # Refit the model for each top-k feature set so the comparison is fair.
        feature_cols = ranked_features[:top_k]
        cat_cols = [col for col in all_cat_cols if col in feature_cols]
        model_bundle, metrics, feature_importance = fit_xgb_model(
            train_df, val_df, test_df, feature_cols, cat_cols
        )
        run = {
            "top_k": int(top_k),
            "validation_f1": metrics["validation_f1"],
            "validation_precision": metrics["validation_precision"],
            "validation_recall": metrics["validation_recall"],
            "test_f1": metrics["f1"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "roc_auc": metrics["roc_auc"],
            "threshold": metrics["prediction_threshold"],
        }
        search_results.append(run)
        run_lookup[top_k] = {
            "top_k": top_k,
            "validation_f1": run["validation_f1"],
            "validation_precision": run["validation_precision"],
            "model_bundle": model_bundle,
            "metrics": metrics,
            "feature_importance": feature_importance,
        }
        if best_run is None or run["validation_f1"] > best_run["validation_f1"] or (
            run["validation_f1"] == best_run["validation_f1"] and run["validation_precision"] > best_run["validation_precision"]
        ):
            best_run = {
                "top_k": top_k,
                "validation_f1": run["validation_f1"],
                "validation_precision": run["validation_precision"],
                "model_bundle": model_bundle,
                "metrics": metrics,
                "feature_importance": feature_importance,
            }

    # Keep the fixed top-k model used in the report, while still storing the search results.
    selected_run = run_lookup.get(SELECTED_TOP_K, best_run)

    metrics = dict(selected_run["metrics"])
    metrics["model_name"] = "xgboost_gain_ranked_top30_rule_baseline"
    metrics["selected_feature_count"] = int(selected_run["top_k"])
    metrics["selected_features"] = selected_run["model_bundle"]["feature_cols"]
    metrics["full_feature_count"] = len(all_feature_cols)
    metrics["all_candidate_features"] = all_feature_cols
    metrics["feature_search_results"] = search_results
    metrics["full_model_validation_f1"] = full_metrics["validation_f1"]
    metrics["full_model_test_f1"] = full_metrics["f1"]
    metrics["selection_policy"] = f"fixed_top_k={SELECTED_TOP_K} selected for the shared-feature baseline specification"
    metrics["best_validation_f1_top_k"] = int(best_run["top_k"])
    metrics["best_validation_f1"] = float(best_run["validation_f1"])

    return selected_run["model_bundle"], metrics, selected_run["feature_importance"]


def main():
    # load_or_build_candidates reuses the cached candidate table unless it is stale.
    candidates_df, candidate_summary = load_or_build_candidates()
    model_bundle, metrics, feature_importance = train_model(candidates_df)

    metrics["candidate_summary"] = candidate_summary

    model_path = DERIVED_DIR / "counterattack_detector.pkl"
    metrics_path = DERIVED_DIR / "model_metrics.json"
    importance_path = DERIVED_DIR / "feature_importance.csv"

    with model_path.open("wb") as f:
        pickle.dump(model_bundle, f)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    feature_importance.to_csv(importance_path, index=False)

    print("Saved:", DERIVED_DIR / "regain_candidates_labeled.csv")
    print("Saved:", importance_path)
    print("Saved:", metrics_path)
    print("Saved:", model_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
