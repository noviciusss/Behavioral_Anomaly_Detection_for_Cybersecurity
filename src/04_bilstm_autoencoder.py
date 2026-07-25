"""
Phase 4 — Bidirectional LSTM Autoencoder  ⚡ GPU
=================================================
Trains a BiLSTM Autoencoder on NORMAL sessions only.
Anomaly score = reconstruction error (MSE per session).

GPU-specific settings:
  - Batch size 128–256 (fast convergence)
  - 1–2 BiLSTM layers, hidden dim 64 (lean, avoids overfitting synthetic data)
  - Early stopping on validation loss

Outputs:
  models/bilstm_autoencoder.keras   — trained model
  data/blended_scores.csv           — IF + LSTM-AE blended risk score per session
  outputs/ablation_table.csv        — IF-only vs IF+BiLSTM-AE metrics
  outputs/training_curve.png

Run AFTER 03_isolation_forest.py.
"""

import os
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF info logs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, classification_report

# ── Config ────────────────────────────────────────────────────────────────────
LSTM_PATH       = "data/sequences_lstm.pkl"
IF_SCORES_PATH  = "data/if_scores.csv"
FEATURES_PATH   = "data/features_tabular.csv"
OUT_MODEL       = "models/bilstm_autoencoder.keras"
OUT_BLENDED     = "data/blended_scores.csv"
OUT_ABLATION    = "outputs/ablation_table.csv"
OUT_CURVE       = "outputs/training_curve.png"

# Architecture
HIDDEN_DIM  = 64
BATCH_SIZE  = 256     # large batch takes advantage of GPU
EPOCHS      = 60
PATIENCE    = 8       # early stopping
THRESHOLD_PERCENTILE = 99   # top 1% alert budget
ALPHA       = 0.5    # blending weight: 0.5 * IF + 0.5 * LSTM-AE

ATTACK_LABELS = [
    "brute_force", "impossible_travel", "credential_stuffing",
    "lateral_movement", "device_spoofing", "low_and_slow_exfiltration"
]

# ── Verify GPU ────────────────────────────────────────────────────────────────
def check_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[⚡] GPU detected: {[g.name for g in gpus]}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("[!] No GPU detected — training on CPU (will be slower, consider reducing EPOCHS)")
    return len(gpus) > 0

# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    with open(LSTM_PATH, "rb") as f:
        lstm_payload = pickle.load(f)

    sequences  = lstm_payload["sequences"]     # (N, MAX_SEQ_LEN) int32
    labels     = np.array(lstm_payload["labels"])
    session_ids = np.array(lstm_payload["session_ids"])
    vocab_size = lstm_payload["vocab_size"]
    max_seq_len = lstm_payload["max_seq_len"]

    # Also load a few tabular scalars to concatenate at bottleneck
    feats_df = pd.read_csv(FEATURES_PATH)
    scalar_cols = ["geo_velocity_kmh", "is_off_hours", "failed_auth_rate_10",
                   "is_new_resource", "session_duration_zscore"]
    scalars = feats_df[scalar_cols].fillna(0).values.astype(np.float32)

    # Normalize scalars (0-mean, 1-std from normal sessions)
    normal_mask = labels == "normal"
    scalar_mean = scalars[normal_mask].mean(axis=0)
    scalar_std  = scalars[normal_mask].std(axis=0) + 1e-9
    scalars_normed = (scalars - scalar_mean) / scalar_std

    if_df = pd.read_csv(IF_SCORES_PATH)
    if_scores = if_df["if_score"].values

    print(f"[+] Sequences shape: {sequences.shape}")
    print(f"[+] Vocab size: {vocab_size}, Seq len: {max_seq_len}")
    print(f"[+] Scalars shape: {scalars_normed.shape}")

    return (sequences, scalars_normed, labels, session_ids,
            vocab_size, max_seq_len, if_scores,
            scalar_mean, scalar_std, scalar_cols)

# ── Model architecture ────────────────────────────────────────────────────────
def build_bilstm_autoencoder(vocab_size, max_seq_len, n_scalars, hidden_dim=HIDDEN_DIM):
    """
    Input:
      - command_seq  : (batch, max_seq_len)  — integer IDs
      - scalars      : (batch, n_scalars)    — contextual features

    Architecture:
      Embedding → BiLSTM encoder → concat scalars → RepeatVector
      → BiLSTM decoder → TimeDistributed Dense (reconstruction)
    """
    # ── Sequence input
    seq_input = keras.Input(shape=(max_seq_len,), name="command_seq")
    x = layers.Embedding(vocab_size, 32, mask_zero=True, name="embedding")(seq_input)

    # BiLSTM encoder (1 layer)
    x = layers.Bidirectional(
        layers.LSTM(hidden_dim, return_sequences=False, name="enc_lstm"),
        name="bilstm_encoder"
    )(x)  # → (batch, hidden_dim * 2)

    # ── Scalar input — concatenate at bottleneck
    scalar_input = keras.Input(shape=(n_scalars,), name="scalars")
    bottleneck = layers.Concatenate(name="bottleneck")([x, scalar_input])
    bottleneck = layers.Dense(hidden_dim, activation="relu", name="bottleneck_dense")(bottleneck)

    # ── Decoder
    x = layers.RepeatVector(max_seq_len, name="repeat")(bottleneck)
    x = layers.Bidirectional(
        layers.LSTM(hidden_dim, return_sequences=True, name="dec_lstm"),
        name="bilstm_decoder"
    )(x)  # → (batch, max_seq_len, hidden_dim * 2)

    # Reconstruct embedding dimension (32)
    out = layers.TimeDistributed(layers.Dense(32, activation="linear"), name="reconstruction")(x)

    model = keras.Model(inputs=[seq_input, scalar_input], outputs=out, name="BiLSTM_AE")
    return model

# ── Train ─────────────────────────────────────────────────────────────────────
def train_model(sequences, scalars, labels, vocab_size, max_seq_len, n_scalars):
    model = build_bilstm_autoencoder(vocab_size, max_seq_len, n_scalars)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="mse"
    )

    # Train ONLY on normal sessions
    normal_mask = labels == "normal"
    X_seq_normal    = sequences[normal_mask]
    X_scalar_normal = scalars[normal_mask]

    # Target: embedding of the input sequence
    # We'll use a simpler formulation: target = input sequence one-hot projections
    # Practical approach: target is the normalized integer sequence itself (as float)
    target_normal = (X_seq_normal / max(vocab_size, 1)).astype(np.float32)
    target_normal = np.expand_dims(target_normal, -1)
    target_normal = np.repeat(target_normal, 32, axis=-1)  # match output dim (32)

    # 80/20 train/val split on normal sessions
    n_train = int(len(X_seq_normal) * 0.8)
    idx = np.random.permutation(len(X_seq_normal))
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=PATIENCE // 2, min_lr=1e-5, verbose=1
        ),
    ]

    print(f"\n[+] Training on {n_train:,} normal sessions (val={len(val_idx):,})...")
    history = model.fit(
        x=[X_seq_normal[train_idx], X_scalar_normal[train_idx]],
        y=target_normal[train_idx],
        validation_data=(
            [X_seq_normal[val_idx], X_scalar_normal[val_idx]],
            target_normal[val_idx]
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    return model, history

# ── Compute reconstruction error ──────────────────────────────────────────────
def compute_recon_error(model, sequences, scalars, vocab_size, max_seq_len):
    """MSE reconstruction error per session — higher = more anomalous."""
    target = (sequences / max(vocab_size, 1)).astype(np.float32)
    target = np.expand_dims(target, -1)
    target = np.repeat(target, 32, axis=-1)

    recon = model.predict([sequences, scalars], batch_size=BATCH_SIZE, verbose=0)
    # Per-session MSE
    recon_error = np.mean((recon - target) ** 2, axis=(1, 2))
    # Normalise to [0, 1]
    recon_error = (recon_error - recon_error.min()) / (recon_error.max() - recon_error.min() + 1e-9)
    return recon_error

# ── Blended score ─────────────────────────────────────────────────────────────
def blend_scores(if_scores, lstm_scores, alpha=ALPHA):
    return alpha * if_scores + (1 - alpha) * lstm_scores

# ── Evaluate + ablation ───────────────────────────────────────────────────────
def evaluate_and_ablate(labels, if_scores, lstm_scores, blended_scores):
    y_true = pd.Series(labels).isin(ATTACK_LABELS).astype(int).values

    # Threshold from normal sessions
    normal_mask = labels == "normal"
    thresh_if      = np.percentile(if_scores[normal_mask], THRESHOLD_PERCENTILE)
    thresh_lstm    = np.percentile(lstm_scores[normal_mask], THRESHOLD_PERCENTILE)
    thresh_blended = np.percentile(blended_scores[normal_mask], THRESHOLD_PERCENTILE)

    results = {}
    for name, scores, thresh in [
        ("IF-only",       if_scores,      thresh_if),
        ("BiLSTM-AE-only", lstm_scores,   thresh_lstm),
        ("IF + BiLSTM-AE (blended)", blended_scores, thresh_blended),
    ]:
        y_pred = (scores >= thresh).astype(int)
        results[name] = {
            "PR-AUC":    round(average_precision_score(y_true, scores), 4),
            "ROC-AUC":   round(roc_auc_score(y_true, scores), 4),
            "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
            "Precision": round(float((y_pred & y_true).sum()) / max(y_pred.sum(), 1), 4),
            "Recall":    round(float((y_pred & y_true).sum()) / max(y_true.sum(), 1), 4),
            "Alerts (%)": round(y_pred.mean() * 100, 2),
        }

    ablation_df = pd.DataFrame(results).T
    print("\n" + "─" * 70)
    print("  ABLATION TABLE — IF-only vs BiLSTM-AE-only vs Blended")
    print("─" * 70)
    print(ablation_df.to_string())

    ablation_df.to_csv(OUT_ABLATION)
    print(f"\n[✓] Saved ablation table: {OUT_ABLATION}")

    # Full classification report for blended
    y_pred_blended = (blended_scores >= thresh_blended).astype(int)
    print("\n  Blended Score — Classification Report:")
    print(classification_report(y_true, y_pred_blended, target_names=["normal/edge", "attack"]))

    return ablation_df, thresh_blended

# ── Training curve plot ────────────────────────────────────────────────────────
def plot_training_curve(history):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.history["loss"],     label="Train Loss", color="#5b8af5", linewidth=2)
    ax.plot(history.history["val_loss"], label="Val Loss",   color="#e05263", linewidth=2)
    ax.set_xlabel("Epoch", color="white")
    ax.set_ylabel("MSE Loss", color="white")
    ax.set_title("BiLSTM Autoencoder — Training Curve", color="white", pad=12)
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#16213e")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333366")
    plt.tight_layout()
    plt.savefig(OUT_CURVE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved training curve: {OUT_CURVE}")

# ── Main ──────────────────────────────────────────────────────────────────────
def run_bilstm_autoencoder():
    print("=" * 60)
    print("  BiLSTM Autoencoder — Phase 4  ⚡ GPU")
    print("=" * 60)

    check_gpu()

    (sequences, scalars, labels, session_ids,
     vocab_size, max_seq_len, if_scores,
     scalar_mean, scalar_std, scalar_cols) = load_data()

    model, history = train_model(sequences, scalars, labels, vocab_size, max_seq_len, scalars.shape[1])

    # Save model
    model.save(OUT_MODEL)
    print(f"\n[✓] Saved model: {OUT_MODEL}")

    # Compute reconstruction error for all sessions
    print("[+] Computing reconstruction errors...")
    lstm_scores = compute_recon_error(model, sequences, scalars, vocab_size, max_seq_len)

    # Blended score
    blended_scores = blend_scores(if_scores, lstm_scores)

    # Ablation
    ablation_df, thresh_blended = evaluate_and_ablate(labels, if_scores, lstm_scores, blended_scores)

    # Save blended scores
    if_df = pd.read_csv(IF_SCORES_PATH)
    blended_df = if_df[["session_id", "entity_id", "entity_type", "timestamp", "label"]].copy()
    blended_df["if_score"]      = if_scores
    blended_df["lstm_ae_score"] = lstm_scores
    blended_df["blended_score"] = blended_scores
    blended_df["flagged"]       = (blended_scores >= thresh_blended).astype(int)
    blended_df.to_csv(OUT_BLENDED, index=False)
    print(f"[✓] Saved blended scores: {OUT_BLENDED}")

    # Save metadata for downstream scripts
    meta = {
        "scalar_mean":    scalar_mean,
        "scalar_std":     scalar_std,
        "scalar_cols":    scalar_cols,
        "threshold_99pct": thresh_blended,
        "vocab_size":     vocab_size,
        "max_seq_len":    max_seq_len,
        "alpha":          ALPHA,
    }
    import joblib
    joblib.dump(meta, "models/bilstm_meta.pkl")

    plot_training_curve(history)

    print("\n[✓] Phase 4 complete — Deliverable 3 ✅")
    return blended_df, ablation_df


if __name__ == "__main__":
    run_bilstm_autoencoder()
