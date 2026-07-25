"""
Phase 2 — EDA + Feature Engineering
======================================
Loads raw labeled data and engineers per-session and per-entity rolling
features. Outputs two feature matrices:
  - data/features_tabular.csv   — for Isolation Forest + RandomForest
  - data/sequences_lstm.pkl     — padded integer sequences for BiLSTM-AE

Run AFTER 01_data_generator.py.
"""

import pickle
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, asin

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_PATH   = "data/data_with_labels.csv"
OUT_TABULAR  = "data/features_tabular.csv"
OUT_LSTM     = "data/sequences_lstm.pkl"
MAX_SEQ_LEN  = 20    # pad/truncate command sequences to this length
MIN_SESSIONS = 10    # minimum history sessions before zscore is meaningful

# ── Haversine ─────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlam/2)**2
    return 2 * R * asin(sqrt(max(0, a)))

# ── Load ──────────────────────────────────────────────────────────────────────
def load_raw():
    df = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"])
    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    print(f"[+] Loaded {len(df):,} sessions, {df['entity_id'].nunique()} entities")
    return df

# ── Per-session features ──────────────────────────────────────────────────────
def add_time_features(df):
    df["hour_of_day"]  = df["timestamp"].dt.hour
    df["day_of_week"]  = df["timestamp"].dt.dayofweek   # 0=Mon
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_off_hours"] = ((df["hour_of_day"] < 7) | (df["hour_of_day"] > 20)).astype(int)
    return df

def add_geo_velocity(df):
    """Distance-per-hour between consecutive sessions for the same entity."""
    df = df.copy()
    df["prev_lat"]  = df.groupby("entity_id")["geo_lat"].shift(1)
    df["prev_lon"]  = df.groupby("entity_id")["geo_lon"].shift(1)
    df["prev_time"] = df.groupby("entity_id")["timestamp"].shift(1)

    def _vel(row):
        if pd.isna(row["prev_lat"]):
            return 0.0
        dt_hours = max((row["timestamp"] - row["prev_time"]).total_seconds() / 3600, 1/60)
        dist     = haversine_km(row["prev_lat"], row["prev_lon"],
                                row["geo_lat"],  row["geo_lon"])
        return dist / dt_hours   # km/h

    df["geo_velocity_kmh"] = df.apply(_vel, axis=1)
    # Impossible travel threshold: ~900 km/h (just under commercial flight speed)
    df["impossible_travel_flag"] = (df["geo_velocity_kmh"] > 900).astype(int)
    df.drop(columns=["prev_lat", "prev_lon", "prev_time"], inplace=True)
    return df

def add_device_features(df):
    """Detect new or mismatched device fingerprints."""
    # Track the set of known fingerprints per entity up to (but not including) current row
    df = df.copy()
    known_fps = {}
    is_new_device = []
    is_spoofed    = []

    for _, row in df.iterrows():
        eid = row["entity_id"]
        fp  = str(row["device_fingerprint"])
        seen = known_fps.get(eid, set())

        is_new_device.append(1 if fp not in seen else 0)
        is_spoofed.append(1 if "_SPOOFED_" in fp else 0)

        seen.add(fp)
        known_fps[eid] = seen

    df["is_new_device"] = is_new_device
    df["is_spoofed_device"] = is_spoofed
    return df

def add_resource_features(df):
    """Detect first-time resource access per entity."""
    df = df.copy()
    known_resources = {}
    is_new_resource = []

    for _, row in df.iterrows():
        eid = row["entity_id"]
        res = row["resource_accessed"]
        seen = known_resources.get(eid, set())
        is_new_resource.append(1 if res not in seen else 0)
        seen.add(res)
        known_resources[eid] = seen

    df["is_new_resource"] = is_new_resource
    return df

def _rolling_nunique(s, w):
    vals = s.to_numpy()
    n = len(vals)
    out = np.empty(n, dtype=int)
    for i in range(n):
        start = max(0, i - w + 1)
        out[i] = len(set(vals[start : i + 1]))
    return pd.Series(out, index=s.index)

def add_rolling_features(df, windows=[10, 50]):
    """
    Per-entity rolling window statistics:
      - failed auth rate
      - resource diversity
    """
    df = df.copy()
    df["auth_failed_int"] = (~df["auth_success"]).astype(int)

    for w in windows:
        grp = df.groupby("entity_id")
        df[f"failed_auth_rate_{w}"] = (
            grp["auth_failed_int"]
            .transform(lambda x: x.rolling(w, min_periods=1).mean())
        )
        df[f"resource_diversity_{w}"] = (
            grp["resource_accessed"]
            .transform(lambda x: _rolling_nunique(x, w))
        )

    df.drop(columns=["auth_failed_int"], inplace=True)
    return df

def add_session_duration_zscore(df):
    """Z-score of session duration relative to each entity's rolling history."""
    df = df.copy()
    grp = df.groupby("entity_id")["session_duration"]
    rolling_mean = grp.transform(lambda x: x.expanding().mean().shift(1))
    rolling_std  = grp.transform(lambda x: x.expanding().std().shift(1).fillna(1))
    df["session_duration_zscore"] = ((df["session_duration"] - rolling_mean) / rolling_std).fillna(0)
    return df

# ── Command sequence features (tabular) ───────────────────────────────────────
def add_command_aggregate_features(df):
    """For tabular models: sequence length + command rarity score."""
    df = df.copy()
    df["cmd_seq_length"] = df["command_sequence"].apply(
        lambda s: len(str(s).split("|")) if pd.notna(s) else 0
    )

    # Global command frequency → rarity = 1 - freq/total
    all_cmds = df["command_sequence"].dropna().str.split("|").explode()
    cmd_freq = all_cmds.value_counts(normalize=True).to_dict()

    def rarity_score(seq_str):
        if pd.isna(seq_str):
            return 0.0
        cmds = str(seq_str).split("|")
        freqs = [cmd_freq.get(c, 0) for c in cmds]
        return 1.0 - (sum(freqs) / max(len(freqs), 1))

    df["cmd_rarity_score"] = df["command_sequence"].apply(rarity_score)
    return df

# ── Command sequence encoding (LSTM) ─────────────────────────────────────────
def build_lstm_sequences(df):
    """
    Encode command_sequence as padded integer arrays for the BiLSTM-AE.
    Returns:
      sequences_array   — np.ndarray shape (N, MAX_SEQ_LEN)
      vocab             — dict mapping command str → int index
      session_ids       — list of session_id for alignment
    """
    # Build vocab from all commands
    all_cmds = set()
    for seq in df["command_sequence"].dropna():
        for cmd in str(seq).split("|"):
            all_cmds.add(cmd.strip())

    vocab = {cmd: i + 1 for i, cmd in enumerate(sorted(all_cmds))}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = len(vocab)

    sequences = []
    for seq in df["command_sequence"]:
        if pd.isna(seq) or seq == "":
            tokens = [0] * MAX_SEQ_LEN
        else:
            ids = [vocab.get(cmd.strip(), vocab["<UNK>"]) for cmd in str(seq).split("|")]
            # Pad or truncate to MAX_SEQ_LEN
            if len(ids) < MAX_SEQ_LEN:
                ids = ids + [0] * (MAX_SEQ_LEN - len(ids))
            else:
                ids = ids[:MAX_SEQ_LEN]
        sequences.append(ids if not pd.isna(seq) else [0]*MAX_SEQ_LEN)

    return np.array(sequences, dtype=np.int32), vocab, df["session_id"].tolist()

# ── Entity type encoding ──────────────────────────────────────────────────────
def encode_categoricals(df):
    df = df.copy()
    df["entity_type_code"] = df["entity_type"].map(
        {"user": 0, "service_account": 1, "edge_device": 2}
    ).fillna(0).astype(int)
    df["auth_method_code"] = pd.Categorical(df["auth_method"]).codes
    return df

# ── Final tabular feature selection ──────────────────────────────────────────
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

# ── Main ──────────────────────────────────────────────────────────────────────
def run_feature_engineering():
    print("=" * 60)
    print("  Feature Engineering — Phase 2")
    print("=" * 60)

    df = load_raw()

    print("[+] Adding time features...")
    df = add_time_features(df)

    print("[+] Computing geo-velocity...")
    df = add_geo_velocity(df)

    print("[+] Detecting new/spoofed devices...")
    df = add_device_features(df)

    print("[+] Detecting new resource access...")
    df = add_resource_features(df)

    print("[+] Computing rolling auth + resource features...")
    df = add_rolling_features(df)

    print("[+] Computing session duration z-score...")
    df = add_session_duration_zscore(df)

    print("[+] Computing command aggregate features...")
    df = add_command_aggregate_features(df)

    print("[+] Encoding categoricals...")
    df = encode_categoricals(df)

    # ── Save tabular features ─────────────────────────────────────────
    feature_df = df[["session_id", "entity_id", "entity_type", "timestamp",
                      "label"] + TABULAR_FEATURES].copy()
    feature_df.to_csv(OUT_TABULAR, index=False)
    print(f"\n[✓] Saved tabular features: {OUT_TABULAR}  ({len(feature_df):,} rows × {len(TABULAR_FEATURES)} features)")

    # ── Save LSTM sequences ────────────────────────────────────────────
    print("[+] Building LSTM integer sequences...")
    sequences, vocab, session_ids = build_lstm_sequences(df)
    lstm_payload = {
        "sequences":   sequences,      # (N, MAX_SEQ_LEN) int32
        "vocab":       vocab,          # str → int
        "session_ids": session_ids,    # for alignment with feature_df
        "labels":      df["label"].tolist(),
        "entity_ids":  df["entity_id"].tolist(),
        "max_seq_len": MAX_SEQ_LEN,
        "vocab_size":  len(vocab),
    }
    with open(OUT_LSTM, "wb") as f:
        pickle.dump(lstm_payload, f)
    print(f"[✓] Saved LSTM sequences: {OUT_LSTM}  (vocab size: {len(vocab)}, max_len: {MAX_SEQ_LEN})")

    # ── Feature summary ────────────────────────────────────────────────
    print("\n[+] Feature summary (normal vs anomaly means):")
    normal_mask  = feature_df["label"] == "normal"
    anomaly_mask = feature_df["label"].isin(["brute_force", "impossible_travel",
                                              "credential_stuffing", "lateral_movement",
                                              "device_spoofing", "low_and_slow_exfiltration"])
    for feat in ["geo_velocity_kmh", "failed_auth_rate_10", "is_new_resource",
                 "is_new_device", "is_off_hours", "resource_diversity_10"]:
        n_mean = feature_df.loc[normal_mask, feat].mean()
        a_mean = feature_df.loc[anomaly_mask, feat].mean()
        print(f"  {feat:35s}  normal={n_mean:.3f}  anomaly={a_mean:.3f}")

    print("\n[✓] Phase 2 complete.")
    return feature_df, lstm_payload


if __name__ == "__main__":
    run_feature_engineering()
