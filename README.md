# 🛡️ AI-Powered Behavioral Anomaly Detection

**Honeywell Hackathon 2026 · Solo entry · 32-hour window**

A full UEBA (User and Entity Behavior Analytics) pipeline that detects and explains insider threats and behavioral anomalies using a two-stage AI approach.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python run_all.py

# 3. Launch the dashboard
streamlit run dashboard/dashboard.py
```

---

## Architecture

```
Synthetic Data → Feature Engineering → Isolation Forest (tabular)
                                     + BiLSTM Autoencoder (sequences)
                                     → Blended Risk Score
                                     → RandomForest Classifier (attack type)
                                     → SHAP Explanations
                                     → Streamlit Dashboard
```

---

## Project Structure

```
honey_hack/
├── src/
│   ├── 01_data_generator.py       # Phase 1: Synthetic UEBA data + 7 attack types
│   ├── 02_feature_engineering.py  # Phase 2: 17 tabular features + LSTM sequences
│   ├── 03_isolation_forest.py     # Phase 3: Baseline anomaly scoring
│   ├── 04_bilstm_autoencoder.py   # Phase 4: Sequence-aware detection (GPU)
│   ├── 05_classifier.py           # Phase 5: Attack type classification
│   ├── 06_explainability.py       # Phase 6: SHAP + human-readable explanations
│   └── 07_cold_start_drift.py     # Phase 7: Cold-start & concept drift
├── dashboard/
│   └── dashboard.py               # Phase 8: Streamlit analyst dashboard
├── data/                          # Generated CSV files (created at runtime)
├── models/                        # Trained models (created at runtime)
├── outputs/                       # Plots and charts (created at runtime)
│   └── shap_plots/
├── docs/
│   └── data_assumptions.md        # Auto-generated: attack taxonomy + schema
├── run_all.py                     # Full pipeline runner
└── requirements.txt
```

---

## Detected Attack Types

| Attack | Label | Detection Mechanism |
|---|---|---|
| Brute Force | `brute_force` | Failed auth rate spike |
| Impossible Travel | `impossible_travel` | Geo-velocity > 900 km/h |
| Credential Stuffing | `credential_stuffing` | Population-level IP clustering |
| Lateral Movement | `lateral_movement` | Resource diversity spike |
| Device Spoofing | `device_spoofing` | Fingerprint inconsistency |
| Low-and-Slow Exfiltration | `low_and_slow_exfiltration` | Long-window off-hours trend |
| **Insider Drift** | **`edge_case`** | **Threshold calibration only — not attack** |

---

## Key Design Decisions

- **BiLSTM over plain LSTM**: Past+future context per command token → better reconstruction quality at no extra GPU cost
- **Train on normal only**: Both unsupervised models (IF + LSTM-AE) train exclusively on normal sessions — no labels needed at detection time
- **99th-percentile threshold**: Corresponds to top 1% alert budget — maps directly to analyst capacity
- **PR-AUC primary metric**: At 2% anomaly rate, ROC-AUC is misleading; PR-AUC is the honest metric
- **Insider Drift as `edge_case`**: Used to tune false-positive threshold, not treated as an attack target
- **Two dataset versions**: `data_with_labels.csv` for training; `data_for_inference.csv` simulates production (no labels)

---

## Pipeline Runner Options

```bash
python run_all.py                  # Run all phases (skips if output exists)
python run_all.py --phase 4        # Run only Phase 4
python run_all.py --from 3         # Run from Phase 3 onward
python run_all.py --force          # Re-run all phases even if output exists
```

---

## Requirements

- Python 3.10+
- GPU (optional but recommended for Phase 4 — CUDA-enabled TensorFlow)
- See `requirements.txt` for full dependency list

---

*Stack: numpy · pandas · faker · scikit-learn · tensorflow · shap · streamlit · plotly*
