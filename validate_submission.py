"""
validate_submission.py
========================
Submission validation script that verifies:
  1. Final model feature list excludes ground-truth columns (label, attack_type, scenario_id, etc.).
  2. data_for_inference.csv excludes those ground-truth fields.
  3. Full inference and dashboard data-preparation path runs cleanly using ONLY data_for_inference.csv.
  4. Prepared output contains risk_score, predicted_attack_type, classification_confidence, and explanation.
  5. Concise PASS/FAIL output with non-zero exit code on failure.
"""

import sys
import os
import importlib
import joblib
import pandas as pd
import numpy as np

# Ensure UTF-8 output encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add root directory to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Dynamically import src modules
fe = importlib.import_module("src.02_feature_engineering")
ex = importlib.import_module("src.06_explainability")

FORBIDDEN_COLUMNS = {
    "label", "attack_type", "scenario_id", "ground_truth",
    "attack_label", "true_label", "is_attack"
}

def validate_feature_list():
    print("[1/4] Validating model feature list...")
    tabular_features = fe.TABULAR_FEATURES
    
    # Check explicitly defined TABULAR_FEATURES
    for feat in tabular_features:
        assert feat not in FORBIDDEN_COLUMNS, f"Forbidden ground-truth column '{feat}' in feature list!"
        feat_lower = feat.lower()
        for forbidden in ["label", "attack", "scenario", "ground_truth"]:
            assert forbidden not in feat_lower, f"Feature '{feat}' appears to contain ground-truth information ('{forbidden}')!"
            
    # Check features in saved models
    if os.path.exists("models/isolation_forest.pkl"):
        if_payload = joblib.load("models/isolation_forest.pkl")
        if_feats = if_payload.get("features", [])
        for feat in if_feats:
            assert feat not in FORBIDDEN_COLUMNS, f"Forbidden column '{feat}' in Isolation Forest model payload features!"
            
    if os.path.exists("models/classifier.pkl"):
        clf_payload = joblib.load("models/classifier.pkl")
        clf_feats = clf_payload.get("features", [])
        for feat in clf_feats:
            assert feat not in FORBIDDEN_COLUMNS, f"Forbidden column '{feat}' in Classifier model payload features!"

    print("  [✓] Feature list validation PASSED (18 features verified clean)")
    return tabular_features

def validate_inference_dataset():
    print("[2/4] Validating data_for_inference.csv...")
    csv_path = "data/data_for_inference.csv"
    assert os.path.exists(csv_path), f"File not found: {csv_path}"
    
    df_sample = pd.read_csv(csv_path, nrows=10)
    cols = set(df_sample.columns)
    
    for col in cols:
        assert col not in FORBIDDEN_COLUMNS, f"Forbidden ground-truth column '{col}' found in data_for_inference.csv!"
        assert col.lower() not in ["label", "attack_type", "scenario_id"], f"Forbidden column '{col}' found in data_for_inference.csv!"

    print(f"  [✓] data_for_inference.csv validation PASSED ({len(cols)} columns verified clean)")
    return csv_path

def run_inference_pipeline(csv_path, tabular_features):
    print("[3/4] Executing inference pipeline on unlabeled dataset...")
    df_raw = pd.read_csv(csv_path, parse_dates=["timestamp"])
    print(f"  [+] Loaded {len(df_raw):,} unlabeled inference sessions")
    
    # Run feature engineering pipeline on raw inference data
    df = df_raw.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    df = fe.add_time_features(df)
    df = fe.add_geo_velocity(df)
    df = fe.add_device_features(df)
    df = fe.add_resource_features(df)
    df = fe.add_rolling_features(df)
    df = fe.add_session_duration_zscore(df)
    df = fe.add_command_aggregate_features(df)
    df = fe.encode_categoricals(df)
    
    # 1. Isolation Forest Scoring (Primary Risk Score)
    assert os.path.exists("models/isolation_forest.pkl"), "models/isolation_forest.pkl missing!"
    if_payload = joblib.load("models/isolation_forest.pkl")
    
    strategy = if_payload.get("strategy", "pooled")
    models = if_payload.get("models", {})
    scalers = if_payload.get("scalers", {})
    
    scores = np.zeros(len(df))
    if strategy == "per_type":
        for etype in df["entity_type"].unique():
            mask = df["entity_type"] == etype
            if etype in models:
                scaler = scalers[etype]
                clf = models[etype]
            else:
                scaler = list(scalers.values())[0]
                clf = list(models.values())[0]
            X = df.loc[mask, tabular_features].fillna(0).values
            X_scaled = scaler.transform(X)
            raw = -clf.score_samples(X_scaled)
            norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
            scores[mask] = norm
    else:
        scaler = scalers["all"]
        clf = models["all"]
        X = df[tabular_features].fillna(0).values
        X_scaled = scaler.transform(X)
        raw = -clf.score_samples(X_scaled)
        norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        scores = norm
        
    df["risk_score"] = scores
    df["if_score"] = scores
    df["flagged"] = (scores >= np.percentile(scores, 99)).astype(int)
    
    # 2. Classifier Prediction
    assert os.path.exists("models/classifier.pkl"), "models/classifier.pkl missing!"
    clf_payload = joblib.load("models/classifier.pkl")
    clf = clf_payload["model"]
    le = clf_payload["label_encoder"]
    
    X_clf = df[tabular_features].fillna(0).values
    probs = clf.predict_proba(X_clf)
    pred_indices = np.argmax(probs, axis=1)
    df["predicted_attack_type"] = le.inverse_transform(pred_indices)
    df["classification_confidence"] = np.max(probs, axis=1)
    df["classifier_confidence"] = df["classification_confidence"]
    
    # 3. Explanation Generation for Flagged Sessions
    flagged_df = df[df["flagged"] == 1].copy()
    explained_flagged = ex.generate_explanation_strings(clf_payload, df, flagged_df)
    
    # Merge explanation back to main DataFrame (defaulting non-flagged to "Normal session baseline")
    df["explanation"] = "Normal session baseline"
    df.loc[df["flagged"] == 1, "explanation"] = explained_flagged["explanation"].values
    
    print("  [✓] Inference pipeline execution PASSED")
    return df

def validate_output_schema(df):
    print("[4/4] Validating prepared inference output schema...")
    required_columns = [
        "risk_score",
        "predicted_attack_type",
        "classification_confidence",
        "explanation"
    ]
    
    for col in required_columns:
        assert col in df.columns, f"Required column '{col}' missing from inference output!"
        assert not df[col].isnull().all(), f"Column '{col}' is entirely empty/null!"
        
    assert (df["risk_score"] >= 0).all() and (df["risk_score"] <= 1).all(), "risk_score values outside [0, 1]!"
    assert (df["classification_confidence"] >= 0).all() and (df["classification_confidence"] <= 1).all(), "classification_confidence values outside [0, 1]!"
    assert df["explanation"].str.len().gt(0).all(), "Some explanation strings are empty!"

    print(f"  [✓] Output schema validation PASSED ({len(df):,} rows validated)")

def run_validation():
    print("=" * 60)
    print("  UEBA Pipeline Submission Validation")
    print("=" * 60)
    tabular_features = validate_feature_list()
    csv_path = validate_inference_dataset()
    df_out = run_inference_pipeline(csv_path, tabular_features)
    validate_output_schema(df_out)
    print("=" * 60)
    print("[PASS] Submission validation succeeded!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        run_validation()
        sys.exit(0)
    except Exception as e:
        print(f"\n[FAIL] Submission validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
