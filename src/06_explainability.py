"""
Phase 6 — Explainability Layer (SHAP)
=======================================
Generates:
  1. Global SHAP feature importance bar chart (IF + RF)
  2. Per-alert SHAP waterfall plot for one representative alert of each type
  3. Per-feature BiLSTM-AE reconstruction error (proxy explainability)
  4. Human-readable explanation strings for every alert

Outputs:
  outputs/shap_plots/global_importance.png
  outputs/shap_plots/waterfall_<attack_type>.png
  outputs/shap_plots/lstm_recon_error.png
  data/alerts_with_explanations.csv

Run AFTER 05_classifier.py.
"""

import joblib
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap

# ── Config ────────────────────────────────────────────────────────────────────
ALERTS_PATH     = "data/classified_alerts.csv"
BLENDED_PATH    = "data/blended_scores.csv"
FEATURES_PATH   = "data/features_tabular.csv"
IF_MODEL_PATH   = "models/isolation_forest.pkl"
CLF_MODEL_PATH  = "models/classifier.pkl"
LSTM_PATH       = "data/sequences_lstm.pkl"
OUT_ALERTS_EXPL = "data/alerts_with_explanations.csv"
PLOT_DIR        = "outputs/shap_plots"

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

# Human-readable feature name mapping
FEATURE_LABELS = {
    "geo_velocity_kmh":        "Geo-velocity (km/h)",
    "impossible_travel_flag":  "Impossible travel flag",
    "is_new_device":           "New device fingerprint",
    "is_spoofed_device":       "Spoofed device",
    "is_new_resource":         "New resource accessed",
    "failed_auth_rate_10":     "Failed auth rate (last 10)",
    "failed_auth_rate_50":     "Failed auth rate (last 50)",
    "resource_diversity_10":   "Resource diversity (last 10)",
    "resource_diversity_50":   "Resource diversity (last 50)",
    "is_off_hours":            "Off-hours access",
    "session_duration_zscore": "Session duration z-score",
    "cmd_seq_length":          "Command sequence length",
    "cmd_rarity_score":        "Command rarity score",
    "hour_of_day":             "Hour of day",
    "day_of_week":             "Day of week",
    "is_weekend":              "Weekend access",
    "entity_type_code":        "Entity type",
    "auth_method_code":        "Auth method",
}

import os
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
def load_data():
    alerts_df  = pd.read_csv(ALERTS_PATH, parse_dates=["timestamp"])
    feats_df   = pd.read_csv(FEATURES_PATH)
    if_payload = joblib.load(IF_MODEL_PATH)
    clf_payload = joblib.load(CLF_MODEL_PATH)
    blended_df = pd.read_csv(BLENDED_PATH)

    # Merge feature matrix
    df = blended_df.merge(feats_df[["session_id"] + TABULAR_FEATURES], on="session_id", how="left")

    print(f"[+] Alerts: {len(alerts_df):,}")
    print(f"[+] Full dataset for background: {len(df):,}")

    return alerts_df, df, if_payload, clf_payload

# ── SHAP output helper ───────────────────────────────────────────────────────
def extract_shap_1d(shap_vals, class_idx, sample_idx=0):
    """
    Robustly extract a 1D SHAP vector (n_features,) for a single sample + class.
    Handles both:
      - Old SHAP format: list of (n_samples, n_features) arrays, one per class
      - New SHAP format: ndarray of shape (n_samples, n_features, n_classes)
    """
    arr = np.asarray(shap_vals)
    if arr.ndim == 3:                                  # (n_samples, n_features, n_classes)
        return arr[sample_idx, :, class_idx]           # → (n_features,)
    elif isinstance(shap_vals, list):
        return np.asarray(shap_vals[class_idx])[sample_idx]  # → (n_features,)
    else:                                              # 2D: (n_samples, n_features) binary
        return arr[sample_idx]                         # → (n_features,)

def mean_abs_shap(shap_vals):
    """
    Mean |SHAP| across all samples and classes → 1D (n_features,).
    Handles both list and 3D ndarray output.
    """
    arr = np.asarray(shap_vals)
    if arr.ndim == 3:                 # (n_samples, n_features, n_classes)
        return np.abs(arr).mean(axis=(0, 2))   # → (n_features,)
    elif isinstance(shap_vals, list):
        stacked = np.stack([np.abs(np.asarray(sv)) for sv in shap_vals], axis=0)
        return stacked.mean(axis=(0, 1))       # → (n_features,)
    else:
        return np.abs(arr).mean(axis=0)        # → (n_features,)

# ── 1. Global SHAP feature importance (RandomForest) ──────────────────────────
def global_shap_importance(clf_payload, df):
    clf = clf_payload["model"]
    le  = clf_payload["label_encoder"]

    # Use flagged sessions as background
    flagged = df[df["flagged"] == 1]
    if len(flagged) < 5:
        print("[!] Too few flagged sessions for SHAP analysis")
        return None

    X = flagged[TABULAR_FEATURES].fillna(0).values

    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X)    # list of arrays, one per class

    mean_abs = mean_abs_shap(shap_vals)   # always (n_features,)
    importance = pd.Series(mean_abs, index=TABULAR_FEATURES).sort_values(ascending=False)
    importance.index = [FEATURE_LABELS.get(f, f) for f in importance.index]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [f"#{max(50, 230 - int(v/importance.max()*180)):02x}52{min(255, 80 + int(v/importance.max()*170)):02x}"
              for v in importance.values]
    ax.barh(importance.index[::-1], importance.values[::-1], color=colors[::-1])
    ax.set_xlabel("Mean |SHAP Value|", color="white")
    ax.set_title("Global Feature Importance — Attack Classifier (SHAP)", color="white", pad=12)
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#16213e")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333366")
    plt.tight_layout()
    path = f"{PLOT_DIR}/global_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved: {path}")
    return importance, explainer, shap_vals, flagged

# ── 2. Per-alert waterfall plot ────────────────────────────────────────────────
def waterfall_plots(clf_payload, df, alerts_df):
    clf = clf_payload["model"]
    le  = clf_payload["label_encoder"]
    explainer = shap.TreeExplainer(clf)

    # One example per attack type
    attack_types = alerts_df["predicted_attack_type"].dropna().unique() if "predicted_attack_type" in alerts_df.columns else []

    for attack_type in attack_types:
        sample = alerts_df[alerts_df.get("predicted_attack_type", pd.Series(dtype=str)) == attack_type]
        if len(sample) == 0:
            continue

        # Get highest-confidence example
        if "classifier_confidence" in sample.columns:
            row = sample.loc[sample["classifier_confidence"].idxmax()]
        else:
            row = sample.iloc[0]

        # Merge features
        feat_row = df[df["session_id"] == row["session_id"]]
        if len(feat_row) == 0:
            continue
        X_single = feat_row[TABULAR_FEATURES].fillna(0).values

        shap_vals = explainer.shap_values(X_single)
        class_idx = list(le.classes_).index(attack_type) if attack_type in le.classes_ else 0
        sv_single = extract_shap_1d(shap_vals, class_idx, sample_idx=0)  # → (n_features,)

        sv_series = pd.Series(sv_single, index=[FEATURE_LABELS.get(f, f) for f in TABULAR_FEATURES])
        sv_sorted = sv_series.sort_values()

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#e05263" if v > 0 else "#5b8af5" for v in sv_sorted.values]
        ax.barh(sv_sorted.index, sv_sorted.values, color=colors)
        ax.axvline(0, color="white", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP Value (impact on attack classification)", color="white")
        ax.set_title(f"SHAP Waterfall — {attack_type.replace('_', ' ').title()}\n"
                     f"Session: {row['session_id']}  |  Entity: {row['entity_id']}",
                     color="white", pad=10, fontsize=10)
        ax.set_facecolor("#1a1a2e")
        fig.patch.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333366")
        plt.tight_layout()
        path = f"{PLOT_DIR}/waterfall_{attack_type}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[✓] Saved waterfall: {path}")

# ── 3. BiLSTM-AE reconstruction error per feature (proxy) ─────────────────────
def lstm_reconstruction_proxy(alerts_df, blended_df):
    """
    Plot mean LSTM-AE score by entity_type and by hour_of_day for flagged sessions.
    Per-feature reconstruction error requires storing per-dim errors from Phase 4;
    this is the practical proxy used when full SHAP on RNNs is too expensive.
    """
    flagged = blended_df[blended_df["flagged"] == 1]
    if len(flagged) == 0:
        return

    # Per entity_type
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#16213e")

    # Left: mean LSTM-AE score per entity type
    grp = flagged.groupby("entity_type")["lstm_ae_score"].mean().sort_values(ascending=False)
    axes[0].bar(grp.index, grp.values, color=["#5b8af5", "#e05263", "#f5a623"])
    axes[0].set_title("Mean BiLSTM-AE Reconstruction Error\nby Entity Type (Flagged Sessions)",
                       color="white")
    axes[0].set_ylabel("Mean Reconstruction Error", color="white")
    axes[0].set_facecolor("#1a1a2e")
    axes[0].tick_params(colors="white")
    axes[0].spines[:].set_color("#333366")

    # Right: blended score distribution by label
    import matplotlib.patches as mpatches
    label_colors = {
        "normal": "#5b8af5", "brute_force": "#e05263", "impossible_travel": "#f5a623",
        "credential_stuffing": "#a855f7", "lateral_movement": "#22c55e",
        "device_spoofing": "#f97316", "low_and_slow_exfiltration": "#ec4899",
        "edge_case": "#6b7280",
    }
    for label in blended_df["label"].unique():
        subset = blended_df[blended_df["label"] == label]["blended_score"]
        axes[1].hist(subset, bins=50, alpha=0.5,
                     color=label_colors.get(label, "#ffffff"),
                     label=label, density=True)
    axes[1].set_title("Blended Score Distribution by Label", color="white")
    axes[1].set_xlabel("Blended Risk Score", color="white")
    axes[1].set_ylabel("Density", color="white")
    axes[1].legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white")
    axes[1].set_facecolor("#1a1a2e")
    axes[1].tick_params(colors="white")
    axes[1].spines[:].set_color("#333366")

    plt.tight_layout()
    path = f"{PLOT_DIR}/lstm_recon_error.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved: {path}")

# ── 4. Human-readable explanation strings ────────────────────────────────────
def generate_explanation_strings(clf_payload, df, alerts_df):
    """
    For each flagged alert: extract top-3 positive SHAP contributors →
    format as human-readable string.
    e.g. "Flagged due to geo-velocity (3847 km/h) + new device fingerprint + off-hours access"
    """
    clf = clf_payload["model"]
    le  = clf_payload["label_encoder"]
    explainer = shap.TreeExplainer(clf)

    explanations = []
    for _, row in alerts_df.iterrows():
        sid = row["session_id"]
        feat_row = df[df["session_id"] == sid]
        if len(feat_row) == 0:
            explanations.append("Explanation unavailable")
            continue

        X_single = feat_row[TABULAR_FEATURES].fillna(0).values
        feat_vals = feat_row[TABULAR_FEATURES].fillna(0).iloc[0]

        try:
            shap_vals = explainer.shap_values(X_single)
            attack_type = row.get("predicted_attack_type", None)

            if attack_type in list(le.classes_):
                class_idx = list(le.classes_).index(attack_type)
                sv = extract_shap_1d(shap_vals, class_idx, sample_idx=0)
            else:
                sv = mean_abs_shap(shap_vals)

            # Top-3 positive contributors
            sv_series = pd.Series(sv, index=TABULAR_FEATURES)
            top3 = sv_series.nlargest(3)

            parts = []
            for feat, shap_val in top3.items():
                if shap_val <= 0:
                    continue
                label = FEATURE_LABELS.get(feat, feat)
                val   = feat_vals[feat]
                if feat == "geo_velocity_kmh":
                    parts.append(f"geo-velocity ({val:.0f} km/h)")
                elif feat == "impossible_travel_flag" and val == 1:
                    parts.append("impossible travel detected")
                elif feat == "is_new_device" and val == 1:
                    parts.append("new device fingerprint")
                elif feat == "is_spoofed_device" and val == 1:
                    parts.append("spoofed device fingerprint")
                elif feat == "is_new_resource" and val == 1:
                    parts.append("new resource accessed")
                elif feat == "failed_auth_rate_10":
                    parts.append(f"high auth failure rate ({val:.0%})")
                elif feat == "resource_diversity_10":
                    parts.append(f"resource diversity spike ({val:.0f} unique resources)")
                elif feat == "is_off_hours" and val == 1:
                    parts.append("off-hours access")
                elif feat == "session_duration_zscore":
                    parts.append(f"anomalous session duration (z={val:.1f})")
                elif feat == "cmd_rarity_score":
                    parts.append(f"rare command sequence (rarity={val:.2f})")
                else:
                    parts.append(label.lower())

            if parts:
                expl = "Flagged due to " + " + ".join(parts[:3])
            else:
                expl = "Flagged by anomaly score threshold"

        except Exception as e:
            expl = f"Explanation error: {e}"

        explanations.append(expl)

    alerts_df = alerts_df.copy()
    alerts_df["explanation"] = explanations
    return alerts_df

# ── Main ──────────────────────────────────────────────────────────────────────
def run_explainability():
    print("=" * 60)
    print("  Explainability — Phase 6")
    print("=" * 60)

    alerts_df, df, if_payload, clf_payload = load_data()

    print("\n[+] Computing global SHAP feature importance...")
    result = global_shap_importance(clf_payload, df)

    print("\n[+] Generating per-alert waterfall plots...")
    waterfall_plots(clf_payload, df, alerts_df)

    print("\n[+] Generating LSTM-AE reconstruction proxy plots...")
    blended_df = pd.read_csv(BLENDED_PATH)
    lstm_reconstruction_proxy(alerts_df, blended_df)

    print("\n[+] Generating human-readable explanation strings...")
    alerts_with_expl = generate_explanation_strings(clf_payload, df, alerts_df)

    # Save
    alerts_with_expl.to_csv(OUT_ALERTS_EXPL, index=False)
    print(f"[✓] Saved: {OUT_ALERTS_EXPL}")

    # Print sample explanations
    print("\n  Sample explanations:")
    for _, row in alerts_with_expl.head(5).iterrows():
        print(f"  [{row['session_id']}]  {row['explanation']}")

    print("\n[✓] Phase 6 complete — Deliverable 5 ✅")
    return alerts_with_expl


BLENDED_PATH = "data/blended_scores.csv"

if __name__ == "__main__":
    run_explainability()
