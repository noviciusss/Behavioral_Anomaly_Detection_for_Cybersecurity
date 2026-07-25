"""
Phase 3 — Baseline Profiling Model (Isolation Forest)
=======================================================
Trains one Isolation Forest per entity_type (or a pooled IF if data is
too sparse) on the engineered tabular features.

Outputs:
  models/isolation_forest.pkl   — trained model(s)
  data/if_scores.csv            — anomaly score per session
  outputs/if_feature_importance.png

Run AFTER 02_feature_engineering.py.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score,
    classification_report
)

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_PATH   = "data/features_tabular.csv"
OUT_SCORES      = "data/if_scores.csv"
OUT_MODEL       = "models/isolation_forest.pkl"
OUT_PLOT        = "outputs/if_feature_importance.png"
MIN_ROWS_PER_TYPE = 500    # threshold: use per-type IF only if >= this many rows
CONTAMINATION   = 0.02     # expected anomaly rate (~2%)
N_ESTIMATORS    = 200
RANDOM_STATE    = 42

TABULAR_FEATURES = [
    "entity_type_code",
    "hour_of_day", "day_of_week", "is_weekend", "is_off_hours",
    "geo_velocity_kmh", "impossible_travel_flag",
    "is_new_device", "is_spoofed_device",
    "is_new_resource",
    "failed_auth_rate_10", "failed_auth_rate_50",
    "resource_diversity_10", "resource_diversity_50",
    "session_duration_zscore",
    "cmd_seq_length", "cmd_rarity_score",
    "auth_method_code",
]

ATTACK_LABELS = [
    "brute_force", "impossible_travel", "credential_stuffing",
    "lateral_movement", "device_spoofing", "low_and_slow_exfiltration"
]

# ── Load ──────────────────────────────────────────────────────────────────────
def load_features():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["timestamp"])
    print(f"[+] Loaded {len(df):,} sessions, {df['entity_id'].nunique()} entities")
    return df

# ── Train/Evaluate ────────────────────────────────────────────────────────────
def train_isolation_forest(df):
    """
    Strategy:
      - If any entity_type has < MIN_ROWS_PER_TYPE rows, use a single pooled IF
        with entity_type_code as a feature.
      - Otherwise train one IF per entity_type for sharper personal baselines.
    """
    type_counts = df["entity_type"].value_counts()
    use_per_type = all(type_counts >= MIN_ROWS_PER_TYPE)

    print(f"\n[+] Entity type row counts:\n{type_counts.to_string()}")
    strategy = "per-entity-type" if use_per_type else "pooled (data too sparse for per-type)"
    print(f"[+] Strategy: {strategy}")

    # Train only on NORMAL sessions — IF is unsupervised
    normal_mask = df["label"] == "normal"
    scalers = {}
    models  = {}

    all_scores = np.zeros(len(df))

    if use_per_type:
        for etype in df["entity_type"].unique():
            mask_train = normal_mask & (df["entity_type"] == etype)
            mask_all   = df["entity_type"] == etype

            X_train = df.loc[mask_train, TABULAR_FEATURES].fillna(0).values
            X_all   = df.loc[mask_all,   TABULAR_FEATURES].fillna(0).values

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_all_scaled   = scaler.transform(X_all)

            clf = IsolationForest(
                n_estimators=N_ESTIMATORS,
                contamination=CONTAMINATION,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            clf.fit(X_train_scaled)

            # score_samples returns negative values; flip so higher = more anomalous
            raw_scores = -clf.score_samples(X_all_scaled)
            # Normalise to [0, 1] per type
            raw_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)

            all_scores[df["entity_type"] == etype] = raw_scores
            scalers[etype] = scaler
            models[etype]  = clf
            print(f"  [{etype}]  trained on {mask_train.sum():,} normal sessions")

        model_payload = {"strategy": "per_type", "models": models, "scalers": scalers,
                         "features": TABULAR_FEATURES}
    else:
        # Pooled
        X_train = df.loc[normal_mask, TABULAR_FEATURES].fillna(0).values
        X_all   = df[TABULAR_FEATURES].fillna(0).values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_all_scaled   = scaler.transform(X_all)

        clf = IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=CONTAMINATION,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        clf.fit(X_train_scaled)

        raw_scores = -clf.score_samples(X_all_scaled)
        all_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)

        model_payload = {"strategy": "pooled", "models": {"all": clf},
                         "scalers": {"all": scaler}, "features": TABULAR_FEATURES}
        print(f"  [pooled]  trained on {normal_mask.sum():,} normal sessions")

    return all_scores, model_payload

# ── Metrics ───────────────────────────────────────────────────────────────────
def evaluate(df, if_scores):
    """Evaluate IF scores treating all true attacks as positive class."""
    # Binary ground truth: 1 = true attack, 0 = normal/edge_case
    y_true_binary = df["label"].isin(ATTACK_LABELS).astype(int)

    # Threshold = 99th percentile of scores on normal sessions (top 1% alert budget)
    normal_scores = if_scores[df["label"] == "normal"]
    threshold_99  = np.percentile(normal_scores, 99)

    y_pred_binary = (if_scores >= threshold_99).astype(int)

    pr_auc = average_precision_score(y_true_binary, if_scores)
    roc_auc = roc_auc_score(y_true_binary, if_scores)

    print("\n" + "─" * 50)
    print("  Isolation Forest — Evaluation (IF-only baseline)")
    print("─" * 50)
    print(f"  Threshold (99th pct normal): {threshold_99:.4f}")
    print(f"  PR-AUC  (primary metric):    {pr_auc:.4f}")
    print(f"  ROC-AUC (reference):         {roc_auc:.4f}")
    print(f"\n{classification_report(y_true_binary, y_pred_binary, target_names=['normal/edge', 'attack'])}")

    # Per-attack-type breakdown
    print("  Per-attack-type detection rate (recall at 99th-pct threshold):")
    for attack in ATTACK_LABELS:
        mask = df["label"] == attack
        if mask.sum() == 0:
            continue
        detected = (if_scores[mask] >= threshold_99).sum()
        print(f"  {attack:35s}  {detected}/{mask.sum()} = {detected/mask.sum()*100:.1f}%")

    return {
        "threshold_99pct": threshold_99,
        "pr_auc":  pr_auc,
        "roc_auc": roc_auc,
        "precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
        "recall":    recall_score(y_true_binary, y_pred_binary, zero_division=0),
        "f1":        f1_score(y_true_binary, y_pred_binary, zero_division=0),
    }

# ── Feature importance plot ────────────────────────────────────────────────────
def plot_feature_importance(df, if_scores, model_payload):
    """
    Approximate feature importance for IF: mean anomaly score difference when
    feature is high vs low (split at median). Cheap, interpretable proxy
    without requiring SHAP at this stage.
    """
    high_score_mask = if_scores >= np.percentile(if_scores, 90)

    importances = {}
    for feat in TABULAR_FEATURES:
        col = df[feat].fillna(0)
        median_val = col.median()
        high_feat = high_score_mask & (col > median_val)
        low_feat  = high_score_mask & (col <= median_val)
        # Fraction of high-score sessions that have above-median feature value
        importances[feat] = high_feat.mean() - low_feat.mean()

    imp_series = pd.Series(importances).sort_values()

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#e05263" if v > 0 else "#5b8af5" for v in imp_series.values]
    ax.barh(imp_series.index, imp_series.values, color=colors)
    ax.axvline(0, color="white", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Contribution to high anomaly score (vs median)", color="white")
    ax.set_title("Isolation Forest — Approximate Feature Importance", color="white", pad=12)

    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#16213e")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333366")

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved feature importance plot: {OUT_PLOT}")

# ── Main ──────────────────────────────────────────────────────────────────────
def run_isolation_forest():
    print("=" * 60)
    print("  Isolation Forest — Phase 3")
    print("=" * 60)

    df = load_features()

    print("\n[+] Training Isolation Forest...")
    if_scores, model_payload = train_isolation_forest(df)

    metrics = evaluate(df, if_scores)

    # Save scores
    scores_df = df[["session_id", "entity_id", "entity_type", "timestamp", "label"]].copy()
    scores_df["if_score"] = if_scores
    scores_df.to_csv(OUT_SCORES, index=False)
    print(f"\n[✓] Saved IF scores: {OUT_SCORES}")

    # Save model
    model_payload["metrics"]    = metrics
    model_payload["if_scores"]  = if_scores      # keep for downstream fusion
    joblib.dump(model_payload, OUT_MODEL)
    print(f"[✓] Saved model: {OUT_MODEL}")

    # Plot
    plot_feature_importance(df, if_scores, model_payload)

    print("\n[✓] Phase 3 complete — Deliverable 2 ✅")
    return scores_df, model_payload


if __name__ == "__main__":
    run_isolation_forest()
