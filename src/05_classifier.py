"""
Phase 5 — Anomaly Classification Layer
========================================
Trains a RandomForestClassifier on sessions flagged by the blended score
to predict WHICH attack category a flagged session resembles.

Outputs:
  models/classifier.pkl             — trained RandomForest
  data/classified_alerts.csv        — flagged sessions with predicted attack type
  outputs/classifier_confusion.png  — per-class confusion matrix

Run AFTER 04_bilstm_autoencoder.py.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)

# ── Config ────────────────────────────────────────────────────────────────────
BLENDED_PATH   = "data/blended_scores.csv"
FEATURES_PATH  = "data/features_tabular.csv"
OUT_MODEL      = "models/classifier.pkl"
OUT_ALERTS     = "data/classified_alerts.csv"
OUT_CONFUSION  = "outputs/classifier_confusion.png"
RANDOM_STATE   = 42

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
def load_data():
    blended_df = pd.read_csv(BLENDED_PATH, parse_dates=["timestamp"])
    feats_df   = pd.read_csv(FEATURES_PATH)

    df = blended_df.merge(feats_df[["session_id"] + TABULAR_FEATURES],
                          on="session_id", how="left")

    # Training data: only true-labeled attacks (ground truth available in labeled dataset)
    train_mask = df["label"].isin(ATTACK_LABELS)
    df_train   = df[train_mask].copy()

    # Inference data: flagged sessions (from blended score threshold)
    df_flagged = df[df["flagged"] == 1].copy()

    print(f"[+] Flagged sessions: {len(df_flagged):,}")
    print(f"[+] Labeled attack sessions for classifier training: {len(df_train):,}")
    print(f"[+] Label distribution in training set:\n{df_train['label'].value_counts().to_string()}")

    return df, df_train, df_flagged

# ── Train ─────────────────────────────────────────────────────────────────────
def train_classifier(df_train):
    X = df_train[TABULAR_FEATURES].fillna(0).values
    y = df_train["label"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",   # handles rare attack types
        max_depth=12,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Cross-validation for honest per-class F1
    if len(np.unique(y_enc)) >= 2 and len(df_train) >= 10:
        n_splits = min(5, len(df_train) // len(np.unique(y_enc)))
        n_splits = max(n_splits, 2)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        y_pred_cv = cross_val_predict(clf, X, y_enc, cv=cv)

        print("\n" + "─" * 60)
        print("  RandomForest Classifier — Cross-Validated Performance")
        print("─" * 60)
        print(classification_report(y_enc, y_pred_cv, target_names=le.classes_))

        per_class_f1 = f1_score(y_enc, y_pred_cv, average=None, zero_division=0)
        print("  Per-class F1:")
        for cls, f1 in zip(le.classes_, per_class_f1):
            print(f"  {cls:35s}  F1={f1:.4f}")

    # Train final model on full training set
    clf.fit(X, y_enc)
    print(f"\n[+] Final RF trained on {len(X):,} attack sessions")

    return clf, le

# ── Predict on flagged sessions ───────────────────────────────────────────────
def classify_flagged(clf, le, df_flagged):
    if len(df_flagged) == 0:
        print("[!] No flagged sessions to classify.")
        return df_flagged

    X_flagged = df_flagged[TABULAR_FEATURES].fillna(0).values
    y_pred_enc = clf.predict(X_flagged)
    y_pred_proba = clf.predict_proba(X_flagged)

    df_flagged = df_flagged.copy()
    df_flagged["predicted_attack_type"] = le.inverse_transform(y_pred_enc)
    df_flagged["classifier_confidence"] = y_pred_proba.max(axis=1).round(4)

    print(f"\n[+] Classified {len(df_flagged):,} flagged sessions:")
    print(df_flagged["predicted_attack_type"].value_counts().to_string())

    return df_flagged

# ── Confusion matrix plot ──────────────────────────────────────────────────────
def plot_confusion(clf, le, df_train):
    X = df_train[TABULAR_FEATURES].fillna(0).values
    y_enc = le.transform(df_train["label"].values)

    n_splits = min(5, max(2, len(df_train) // len(le.classes_)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    y_pred_cv = cross_val_predict(clf, X, y_enc, cv=cv)

    cm = confusion_matrix(y_enc, y_pred_cv)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=le.classes_, yticklabels=le.classes_,
        ax=ax, linewidths=0.5,
        annot_kws={"size": 10}
    )
    ax.set_xlabel("Predicted", color="white", fontsize=12)
    ax.set_ylabel("Actual",    color="white", fontsize=12)
    ax.set_title("Attack Type Classifier — Normalised Confusion Matrix", color="white", pad=12)
    plt.xticks(rotation=30, ha="right", color="white", fontsize=9)
    plt.yticks(rotation=0, color="white", fontsize=9)
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#16213e")
    plt.tight_layout()
    plt.savefig(OUT_CONFUSION, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved confusion matrix: {OUT_CONFUSION}")

# ── Main ──────────────────────────────────────────────────────────────────────
def run_classifier():
    print("=" * 60)
    print("  Anomaly Classification — Phase 5")
    print("=" * 60)

    df, df_train, df_flagged = load_data()
    clf, le = train_classifier(df_train)
    df_classified = classify_flagged(clf, le, df_flagged)

    # Save classified alerts
    cols_out = ["session_id", "entity_id", "entity_type", "timestamp",
                "label", "blended_score", "if_score", "lstm_ae_score",
                "predicted_attack_type", "classifier_confidence"] + TABULAR_FEATURES
    cols_out = [c for c in cols_out if c in df_classified.columns]
    df_classified[cols_out].to_csv(OUT_ALERTS, index=False)
    print(f"\n[✓] Saved classified alerts: {OUT_ALERTS}")

    # Save model
    joblib.dump({"model": clf, "label_encoder": le, "features": TABULAR_FEATURES},
                OUT_MODEL)
    print(f"[✓] Saved classifier: {OUT_MODEL}")

    # Plot confusion matrix
    if len(df_train) >= 10:
        plot_confusion(clf, le, df_train)

    print("\n[✓] Phase 5 complete — Deliverable 4 ✅")
    return df_classified, clf, le


if __name__ == "__main__":
    run_classifier()
