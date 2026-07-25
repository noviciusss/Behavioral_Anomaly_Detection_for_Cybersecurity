"""
Phase 7 — Cold-Start & Concept Drift Handling
===============================================
Implements two production-realistic mechanisms:

1. COLD-START: New entities with no history fall back to the entity_type
   population baseline (IF score from the type-level model) and are flagged
   with "low confidence" until N_COLD sessions accumulate.

2. CONCEPT DRIFT: Entity behavioral profiles are updated with exponential
   decay weighting so legitimate new patterns stop being flagged after they
   persist. A simple drift metric tracks when entity profiles shift significantly.

Outputs:
  data/cold_start_entities.csv    — entities in cold-start window + their scores
  data/drift_report.csv           — entities with detected profile drift
  outputs/cold_start_demo.png     — visualization of cold-start confidence buildup

Run AFTER 04_bilstm_autoencoder.py.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_PATH   = "data/features_tabular.csv"
BLENDED_PATH    = "data/blended_scores.csv"
IF_MODEL_PATH   = "models/isolation_forest.pkl"
OUT_COLD_START  = "data/cold_start_entities.csv"
OUT_DRIFT       = "data/drift_report.csv"
OUT_PLOT        = "outputs/cold_start_demo.png"

N_COLD          = 50      # sessions needed before personal baseline replaces population
DRIFT_WINDOW    = 30      # sessions in each window for drift detection
DRIFT_THRESHOLD = 2.0     # z-score threshold for flagging drift
DECAY_LAMBDA    = 0.02    # exponential decay: older sessions weighted by e^(-λ * age)

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

# ── Load ──────────────────────────────────────────────────────────────────────
def load_data():
    feats_df   = pd.read_csv(FEATURES_PATH, parse_dates=["timestamp"])
    blended_df = pd.read_csv(BLENDED_PATH,  parse_dates=["timestamp"])
    if_payload = joblib.load(IF_MODEL_PATH)

    df = blended_df.merge(feats_df[["session_id"] + TABULAR_FEATURES],
                          on="session_id", how="left")
    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

    return df, if_payload

# ── Cold-start detection ──────────────────────────────────────────────────────
def detect_cold_start_entities(df):
    """
    For each entity, find their first N_COLD sessions.
    These sessions are "cold-start" — scored against the population baseline.
    After N_COLD sessions, confidence transitions to "normal" (personal baseline).
    """
    session_counts = df.groupby("entity_id").cumcount() + 1  # 1-indexed session count per entity
    df = df.copy()
    df["session_count"] = session_counts

    cold_mask = df["session_count"] <= N_COLD
    df["cold_start"]   = cold_mask.astype(int)
    df["confidence"]   = np.where(
        cold_mask,
        # Confidence ramps from 0.2 to 0.9 over first N_COLD sessions
        0.2 + 0.7 * (df["session_count"] / N_COLD).clip(0, 1),
        1.0  # full confidence after N_COLD sessions
    )

    cold_start_df = df[cold_mask][["session_id", "entity_id", "entity_type",
                                    "timestamp", "session_count",
                                    "blended_score", "confidence", "label"]].copy()
    cold_start_df["note"] = "Cold-start: score vs population baseline (confidence ramping)"

    n_cold_entities = df.loc[cold_mask, "entity_id"].nunique()
    n_cold_sessions = cold_mask.sum()
    print(f"[+] Cold-start entities: {n_cold_entities}")
    print(f"[+] Cold-start sessions: {n_cold_sessions:,}")

    cold_start_df.to_csv(OUT_COLD_START, index=False)
    print(f"[✓] Saved: {OUT_COLD_START}")

    return df, cold_start_df

# ── Concept drift detection ───────────────────────────────────────────────────
def detect_concept_drift(df):
    """
    Compare early behavioral profile (first DRIFT_WINDOW sessions) to
    recent profile (last DRIFT_WINDOW sessions) for each entity.
    Drift = significant shift in mean feature values (z-score > DRIFT_THRESHOLD).
    """
    drift_features = [
        "hour_of_day", "resource_diversity_10", "failed_auth_rate_10",
        "is_off_hours", "cmd_rarity_score", "session_duration_zscore"
    ]

    drift_records = []

    for entity_id, grp in df.groupby("entity_id"):
        grp = grp.sort_values("timestamp")
        if len(grp) < DRIFT_WINDOW * 2:
            continue

        early  = grp.head(DRIFT_WINDOW)[drift_features].fillna(0)
        recent = grp.tail(DRIFT_WINDOW)[drift_features].fillna(0)

        # Pool std for denominator (avoid tiny std from sparse entity)
        pooled_std = (grp[drift_features].fillna(0).std() + 1e-9)

        drift_z = ((recent.mean() - early.mean()) / pooled_std).abs()
        drifted_features = drift_z[drift_z > DRIFT_THRESHOLD].index.tolist()

        if drifted_features:
            drift_records.append({
                "entity_id":        entity_id,
                "entity_type":      grp["entity_type"].iloc[0],
                "n_sessions":       len(grp),
                "drifted_features": ", ".join(drifted_features),
                "max_drift_z":      round(drift_z.max(), 2),
                "drift_severity":   "high" if drift_z.max() > 4 else "medium",
                "recommendation":   "Retrain entity profile" if drift_z.max() > 4
                                    else "Monitor — legitimate behaviour change suspected",
            })

    drift_df = pd.DataFrame(drift_records)
    if not drift_df.empty:
        drift_df = drift_df.sort_values("max_drift_z", ascending=False)
        print(f"\n[+] Concept drift detected in {len(drift_df)} entities")
        print(drift_df[["entity_id", "entity_type", "drifted_features", "max_drift_z", "drift_severity"]].head(10).to_string())
    else:
        print("[+] No significant concept drift detected in this dataset window")

    drift_df.to_csv(OUT_DRIFT, index=False)
    print(f"[✓] Saved: {OUT_DRIFT}")
    return drift_df

# ── Exponential decay profile updater ────────────────────────────────────────
def compute_decayed_profile(entity_sessions_df, feature_cols, lambda_=DECAY_LAMBDA):
    """
    Compute exponentially decay-weighted mean for an entity's feature profile.
    More recent sessions have higher weight: weight_i = exp(-λ * (N - i))
    This is what gets called during inference to update the rolling baseline.
    """
    n = len(entity_sessions_df)
    ages   = np.arange(n - 1, -1, -1)        # age: 0 = most recent
    weights = np.exp(-lambda_ * ages)
    weights /= weights.sum()

    X = entity_sessions_df[feature_cols].fillna(0).values
    decayed_mean = (weights[:, None] * X).sum(axis=0)
    return pd.Series(decayed_mean, index=feature_cols)

# ── Cold-start confidence visualization ───────────────────────────────────────
def plot_cold_start_demo(cold_start_df):
    """Visualize how confidence ramps up over the cold-start window."""
    session_range = np.arange(1, N_COLD + 5)
    confidence    = np.where(
        session_range <= N_COLD,
        0.2 + 0.7 * (session_range / N_COLD).clip(0, 1),
        1.0
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(session_range, confidence, color="#5b8af5", linewidth=2.5, label="Confidence")
    ax.axvline(N_COLD, color="#e05263", linestyle="--", linewidth=1.5,
               label=f"End of cold-start window (N={N_COLD})")
    ax.fill_between(session_range, confidence, alpha=0.15, color="#5b8af5")

    ax.annotate("Population baseline\n(entity_type level)",
                xy=(5, 0.3), color="#aaaacc", fontsize=9, style="italic")
    ax.annotate("Personal baseline\n(full confidence)",
                xy=(N_COLD + 1, 0.95), color="#aaaacc", fontsize=9, style="italic")

    ax.set_xlabel("Session count for entity", color="white")
    ax.set_ylabel("Alert confidence weight", color="white")
    ax.set_title("Cold-Start Confidence Ramp-Up\n"
                 "New entities scored against population baseline until N sessions accumulate",
                 color="white", pad=12)
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    ax.set_ylim(0, 1.1)
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#16213e")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333366")

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved: {OUT_PLOT}")

# ── Demo: exponential decay for one entity ────────────────────────────────────
def demo_decay(df):
    """Show how the decay-weighted profile differs from a simple mean for one entity."""
    entity = df["entity_id"].value_counts().idxmax()
    entity_df = df[df["entity_id"] == entity].sort_values("timestamp")
    if len(entity_df) < 20:
        return

    demo_feats = ["hour_of_day", "resource_diversity_10", "failed_auth_rate_10"]
    simple_mean = entity_df[demo_feats].fillna(0).mean()
    decayed     = compute_decayed_profile(entity_df, demo_feats)

    print(f"\n[+] Exponential decay demo — entity: {entity} ({len(entity_df)} sessions)")
    print(f"{'Feature':35s}  {'Simple mean':>12}  {'Decayed mean':>12}  {'Δ':>8}")
    print("─" * 72)
    for feat in demo_feats:
        delta = decayed[feat] - simple_mean[feat]
        print(f"  {feat:33s}  {simple_mean[feat]:>12.4f}  {decayed[feat]:>12.4f}  {delta:>+8.4f}")
    print("  (Positive Δ → recent sessions increased this feature → drift candidate)")

# ── Main ──────────────────────────────────────────────────────────────────────
def run_cold_start_drift():
    print("=" * 60)
    print("  Cold-Start & Concept Drift — Phase 7")
    print("=" * 60)

    df, if_payload = load_data()

    print("\n[+] Detecting cold-start entities...")
    df, cold_start_df = detect_cold_start_entities(df)

    print("\n[+] Detecting concept drift...")
    drift_df = detect_concept_drift(df)

    print("\n[+] Exponential decay profile demo...")
    demo_decay(df)

    print("\n[+] Plotting cold-start confidence ramp...")
    plot_cold_start_demo(cold_start_df)

    print("\n[✓] Phase 7 complete.")
    return cold_start_df, drift_df


if __name__ == "__main__":
    run_cold_start_drift()
