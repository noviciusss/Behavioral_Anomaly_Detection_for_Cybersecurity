# 🛡️ AI-Powered Behavioral Anomaly Detection for Cybersecurity
**Honeywell Hackathon 2026 Submission**

---

## 🎯 What it does

This User and Entity Behavior Analytics (UEBA) production-oriented prototype detects insider threats across multi-entity telemetry (users, service accounts, IoT edge devices) without relying on pre-labeled attack data during anomaly detection. It employs **Isolation Forest** as its primary operational anomaly detector selected by held-out PR-AUC benchmarking, a **Random Forest Classifier** to categorize flagged alerts into 6 specific attack vectors with a 0.98 weighted F1-score, and a **Two-Tier Explainability Engine** that pairs detector policy feature evidence with Shapley attributions (TreeExplainer SHAP) for full threat transparency.

---

## 📋 Hackathon Deliverables Mapped

| Required Deliverable | Repository Evidence | Status |
|---|---|---|
| **1. Synthetic Telemetry Generator** | [`src/01_data_generator.py`](file:///d:/honey_hack/src/01_data_generator.py), [`data/sample_labeled_evaluation_data.csv`](file:///d:/honey_hack/data/sample_labeled_evaluation_data.csv) | ✅ 77,896 sessions generated |
| **2. Behavioral Baseline Profiler** | [`src/02_feature_engineering.py`](file:///d:/honey_hack/src/02_feature_engineering.py), [`src/03_isolation_forest.py`](file:///d:/honey_hack/src/03_isolation_forest.py) | ✅ 18 rolling features |
| **3. Sequential & Tabular Detection** | [`src/03_isolation_forest.py`](file:///d:/honey_hack/src/03_isolation_forest.py), [`src/04_bilstm_autoencoder.py`](file:///d:/honey_hack/src/04_bilstm_autoencoder.py), [`outputs/ablation_table.csv`](file:///d:/honey_hack/outputs/ablation_table.csv) | ✅ IF primary + BiLSTM extension |
| **4. Multi-Class Threat Classifier** | [`src/05_classifier.py`](file:///d:/honey_hack/src/05_classifier.py), [`models/classifier.pkl`](file:///d:/honey_hack/models/classifier.pkl), [`outputs/classifier_confusion.png`](file:///d:/honey_hack/outputs/classifier_confusion.png) | ✅ Weighted F1 = 0.9800 |
| **5. Two-Tier Explainable AI (XAI)** | [`src/06_explainability.py`](file:///d:/honey_hack/src/06_explainability.py), [`outputs/shap_plots/`](file:///d:/honey_hack/outputs/shap_plots/), [`data/alerts_with_explanations.csv`](file:///d:/honey_hack/data/alerts_with_explanations.csv) | ✅ Policy evidence + SHAP attributions |
| **6. Cold-Start & Concept Drift Adaptation** | [`src/07_cold_start_drift.py`](file:///d:/honey_hack/src/07_cold_start_drift.py), [`data/cold_start_entities.csv`](file:///d:/honey_hack/data/cold_start_entities.csv), [`data/drift_report.csv`](file:///d:/honey_hack/data/drift_report.csv) | ✅ Population fallback & exponential decay |
| **7. Interactive Analyst Dashboard** | [`dashboard/dashboard.py`](file:///d:/honey_hack/dashboard/dashboard.py), [`validate_submission.py`](file:///d:/honey_hack/validate_submission.py), [`tests/test_validate_submission.py`](file:///d:/honey_hack/tests/test_validate_submission.py) | ✅ Ranked queue, demo mode & zero leakage |

---

## ⚡ Quick Start

```bash
# 1. Environment setup
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Run the end-to-end pipeline
python run_all.py

# 3. Launch interactive analyst console
streamlit run dashboard/dashboard.py
```

---

## 🧪 Validation & No-Leakage Verification

```bash
# Run standalone submission & no-leakage verification
python validate_submission.py

# Run pytest test suite
pytest -q
```

---

## 📊 Key Results

- **Primary Operational Detector**: Isolation Forest (selected by held-out PR-AUC)
- **PR-AUC**: **0.3380** *(vs 0.0095 for BiLSTM-AE and 0.2811 for Blended)*
- **Headline Top 1.0% Alert Budget**: **779 queued alerts** out of 77,896 sessions
- **Precision@1%**: **34.92%**
- **Recall@1%**: **42.77%**
- **F1-Score@1%**: **0.3845**
- **Attack Classifier**: **Macro F1 = 0.94**, **Weighted F1 = 0.98** across 5-fold cross-validation

---

## ⚠️ Limitations & Scope

1. **Synthetic Data**: Telemetry is generated based on realistic entity profiles and Gaussian distribution shifts rather than live enterprise production logs.
2. **Predefined Attack Taxonomy**: Attack classification is trained on 6 specific threat categories (Brute Force, Low & Slow Exfiltration, Lateral Movement, Impossible Travel, Credential Stuffing, Device Spoofing).
3. **Batch Replay Prototype**: Scores event-by-event; batch replay format in this prototype rather than a live Apache Kafka streaming infrastructure.
4. **Rare-Class Uncertainty**: Classifications for low-frequency attack types (e.g. Device Spoofing with 9 samples) exhibit wider confidence intervals.

---

## 📁 Project Structure

```text
.
├── README.md                          # Project documentation
├── requirements.txt                   # Dependency list
├── run_all.py                         # Master pipeline runner
├── validate_submission.py             # Submission & no-leakage verification script
├── .gitignore                         # Environment & build exclusions
│
├── src/                               # Pipeline core python scripts (Phases 1-7)
│   ├── 01_data_generator.py
│   ├── 02_feature_engineering.py
│   ├── 03_isolation_forest.py
│   ├── 04_bilstm_autoencoder.py
│   ├── 05_classifier.py
│   ├── 06_explainability.py
│   └── 07_cold_start_drift.py
│
├── dashboard/                         # Streamlit analyst console
│   └── dashboard.py
│
├── tests/                             # Pytest test suite
│   └── test_validate_submission.py
│
├── docs/                              # Technical design & evidence documentation
│   ├── data_assumptions.md
│   ├── threat_taxonomy.md
│   ├── architecture.md
│   ├── testing_evidence.md
│   └── STREAMING_DESIGN.md
│
├── outputs/                           # Plots & evaluation artifacts
│   ├── ablation_table.csv
│   ├── classifier_confusion.png
│   ├── pr_curve.png
│   ├── if_feature_importance.png
│   ├── dashboard_demo.png
│   ├── cold_start_demo.png
│   ├── drift_demo.png
│   └── shap_plots/
│
├── data/                              # Telemetry datasets & CSV manifests
│   ├── sample_inference_data.csv
│   ├── sample_labeled_evaluation_data.csv
│   └── DATA_README.md
│
└── models/                            # Saved model binary artifacts (.pkl & .keras)
    ├── isolation_forest.pkl
    ├── classifier.pkl
    └── MODEL_README.md
```
