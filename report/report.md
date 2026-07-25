# 🛡️ AI-Powered Behavioral Anomaly Detection & Insider Threat Analysis
**Honeywell Hackathon 2026 — Comprehensive Technical & Implementation Report**

---

## Executive Summary

Insider threats represent one of the most critical cybersecurity challenges facing modern organizations. Unlike external attacks that must breach perimeter defenses, insider threats originate from authorized accounts, legitimate credentials, and trusted devices. Traditional rule-based Security Information and Event Management (SIEM) systems struggle to detect these threats because single events rarely violate static threshold rules. 

This project implements a **production-oriented prototype** for **User and Entity Behavior Analytics (UEBA)** designed to ingest high-volume entity interaction telemetry (users, service accounts, IoT/edge devices), establish rolling behavioral baselines, detect subtle statistical and sequence-level anomalies, classify attack vectors, generate two-tier human-interpretable explanations (detector policy evidence + classifier SHAP attribution), and handle real-world deployment challenges such as cold-start entities and concept drift.

### System Achievements & Key Highlights

1. **Isolation Forest Primary Detector**: Selected as the primary operational detection engine based on held-out **PR-AUC (0.3380)**. At the headline **Top 1.0% Alert Budget** (779 queued alerts out of 77,896 sessions), it achieves **34.92% Precision**, **42.77% Recall**, and **0.3845 F1-score**.
2. **Evaluated Sequence Extension**: Evaluated a 4-layer **Bidirectional LSTM Autoencoder** for sequence-aware command modeling. While benchmarked in the ablation matrix (PR-AUC 0.0095), it was not selected for primary operational detection due to weak signal separation on synthetic command paths.
3. **High-Precision Multi-Class Classification**: Trained a balanced **Random Forest Classifier** on flagged anomalous sessions to identify 6 distinct attack categories, achieving a **0.98 weighted F1-score** across 5-fold cross-validation.
4. **Two-Tier Explainable AI (XAI)**:
   - **Tier 1 (Detector Policy Evidence)**: Tabular feature rules (e.g., *Geo-velocity 2,454 km/h (>900 km/h threshold) + New device fingerprint*).
   - **Tier 2 (Attack-Type Attribution)**: **TreeExplainer SHAP** attributions explaining why the classifier categorized the alert as a specific attack vector.
5. **Production Resilience**: Implemented a **population-baseline fallback** for cold-start entities ($N < 50$ sessions) with a confidence ramp-up function, as well as an **exponential-decay profile updater** ($\lambda = 0.02$) to adapt to legitimate behavioral concept drift without exploding false-positive rates.
6. **Interactive Analyst Dashboard**: Built a zero-configuration, portable **Streamlit Analyst Console** (`dashboard/dashboard.py`) featuring a ranked alert queue sorted by IF risk score, two-tier alert detail inspection, live SHAP attribution previews, entity historical risk sparklines, score distribution charts, ablation comparison metrics, and a dedicated Demo Mode.

---

## 1. System Architecture & Workflow

The architecture operates on an event-by-event scoring design with batch telemetry replay in this prototype. Each phase transforms raw telemetry into enriched statistical artifacts, model scores, explanations, and visual metrics.

```mermaid
flowchart TD
    subgraph Data Generation & Feature Engineering
        A[Raw Entity Telemetry Log] --> B[Feature Engineering Engine]
        B --> C1[18 Tabular Statistical Features]
        B --> C2[Padded Integer Command Sequences]
    end

    subgraph Dual-Engine Unsupervised Anomaly Detection
        C1 --> D1["🌲 Primary Detector: Isolation Forest"]
        C2 --> D2["🧠 Evaluated Extension: BiLSTM-AE"]
        D1 --> E1["Isolation Forest Risk Score (S_IF)"]
        D2 --> E2["Reconstruction Error Score (S_LSTM)"]
    end

    subgraph Primary Operational Thresholding
        E1 --> F{"S_IF >= Top 1.0% Threshold?"}
        E2 -. Evaluated for Ablation .- F
        F -- No --> H[Normal Telemetry Archive]
        F -- Yes --> I[Operational Alert Queue]
    end

    subgraph Classification & Two-Tier Explainability
        I --> J[Random Forest Attack Classifier]
        J --> K1[Tier 1: Detector Policy Evidence]
        J --> K2[Tier 2: Attack-Type Attribution (SHAP)]
    end

    subgraph Production Resilience & Dashboard
        K1 --> L[Cold-Start & Drift Adaptation]
        K2 --> L
        L --> M[Streamlit Analyst Console]
    end

    style D1 fill:#1a3a1a,stroke:#22c55e,color:#ffffff
    style D2 fill:#3a2a1a,stroke:#f59e0b,color:#ffffff
    style M fill:#1a1a2e,stroke:#3b82f6,color:#ffffff
```

---

## 2. Telemetry Ingestion & Synthetic Data Architecture

### 2.1 Synthetic Telemetry Generator
Real enterprise insider threat logs are classified or restricted due to PII regulations. Phase 1 (`src/01_data_generator.py`) generates a realistic benchmark dataset simulating entity interaction events across **120 entity profiles** (Users, Service Accounts, IoT Edge Devices) over a 60-day period.

- **Total Sessions Generated**: 77,896 sessions.
- **Entity Breakdown**:
  - `user` (75 profiles): Working hours (8 AM–5 PM with Gaussian noise), home IP subnets, standard business applications.
  - `service_account` (25 profiles): 24/7 automated cron cycles, high-frequency API calls, fixed service endpoints.
  - `edge_device` (20 profiles): Periodic IoT sensor readings, telemetry pushes, firmware check-ins.

### 2.2 Attack Taxonomy & Injected Threat Vectors
The dataset includes **636 true attack sessions** ($0.82\%$ ground-truth anomaly rate) across 6 distinct insider threat categories, plus 100 legitimate role-shift edge cases ($0.13\%$):

| Attack Vector | Count | Injected Anomaly Behavior | Key Detection Feature |
|---|---|---|---|
| `brute_force` | 266 | Rapid authentication failures from single IP | `failed_auth_rate_10` spike |
| `low_and_slow_exfiltration` | 257 | Off-hours logins, gradual data downloads | `is_off_hours` + `session_duration_zscore` |
| `lateral_movement` | 62 | Access to uncharacteristic internal microservices | `resource_diversity_10` + `is_new_resource` |
| `impossible_travel` | 22 | Consecutive logins from distant cities $< 1$ hr apart | `geo_velocity_kmh` $> 900\text{ km/h}$ |
| `credential_stuffing` | 20 | Population-wide failure spikes across multiple accounts | `failed_auth_rate_50` population cluster |
| `device_spoofing` | 9 | Login with mismatched hardware fingerprint UUID | `is_spoofed_device` = 1 |

---

## 3. Feature Engineering Engine

Raw log strings (IPs, timestamps, resource names, command lists) cannot be consumed directly by ML algorithms. Phase 2 (`src/02_feature_engineering.py`) extracts **18 numeric tabular features** and **integer sequence vectors**:

1. **Haversine Geo-Velocity**:
   $$v_{\text{geo}} = \frac{d_{\text{haversine}}(\text{loc}_t, \text{loc}_{t-1})}{\Delta t} \quad (\text{km/h})$$
   Flags sessions where required travel speed exceeds physical flight capabilities ($v_{\text{geo}} > 900\text{ km/h}$).
2. **Rolling Behavioral Aggregations**:
   - `failed_auth_rate_10`, `failed_auth_rate_50`: Sliding window ratio of failed logins over past 10 and 50 sessions.
   - `resource_diversity_10`, `resource_diversity_50`: Count of unique resources accessed over sliding windows using a vectorized NumPy sliding-window algorithm (`_rolling_nunique`).
3. **Session Duration Z-Scores**:
   $$z_{\text{duration}} = \frac{t_{\text{duration}} - \mu_{\text{entity}}}{\sigma_{\text{entity}} + 1e-5}$$
4. **Command Sequence Tokenization & Rarity Scoring**:
   - Commands are tokenized, integer-encoded, and padded to uniform length $L=20$.
   - `cmd_rarity_score`: Inverse frequency score measuring how uncommon the command sequence is relative to historical baseline.

---

## 4. Anomaly Detection Engine & Model Selection

### 4.1 Primary Detector: Isolation Forest (Tabular Profiling)
- **Mechanism**: Isolation Forest constructs random decision trees. Anomalous instances require fewer partitions to isolate (shorter path length $h(x)$), yielding higher anomaly scores:
  $$S(x, n) = 2^{-\frac{\mathbb{E}(h(x))}{c(n)}}$$
- **Entity-Type Training**: Trained separate model instances per entity type (`user`, `service_account`, `edge_device`) exclusively on normal sessions.

### 4.2 Evaluated Extension: BiLSTM Autoencoder (Sequence Profiling)
- **Architecture**: `Embedding(16d) -> BiLSTM(32u) -> BiLSTM(16u bottleneck) -> RepeatVector(20) -> BiLSTM(16u) -> TimeDistributed(Dense(32, softmax))`.
- **Training Strategy**: Trained on normal command chains to minimize reconstruction error:
  $$S_{\text{LSTM}} = \frac{1}{L} \sum_{i=1}^{L} \| \mathbf{y}_i - \hat{\mathbf{y}}_i \|^2$$

### 4.3 Model Ablation Analysis & Headline Operating Point

Evaluating unsupervised anomaly detection models on highly imbalanced data ($0.82\%$ positive class) requires inspecting **Precision-Recall Area Under the Curve (PR-AUC)** rather than ROC-AUC, as ROC-AUC is artificially inflated by high true-negative counts.

| Model Strategy | PR-AUC | ROC-AUC | F1-Score | Precision | Recall | Alert Volume |
|---|---|---|---|---|---|---|
| **IF-Only (Headline Top 1.0% Budget)** | **0.3380** | **0.8788** | **0.3845** | **0.3492** | **0.4277** | **1.00% (779 alerts)** |
| IF-Only (Alternative @ 1.37% Threshold) | 0.3380 | 0.8788 | 0.3363 | 0.2685 | 0.4497 | 1.37% (1,068 alerts) |
| BiLSTM-AE-Only (Evaluated Extension) | 0.0095 | 0.4156 | 0.0000 | 0.0000 | 0.0000 | 0.99% (771 alerts) |
| Blended IF + BiLSTM-AE (Evaluated) | 0.2811 | 0.8740 | 0.3333 | 0.2665 | 0.4450 | 1.36% (1,059 alerts) |

#### Model Selection Decisions:
1. **Primary Detector: Isolation Forest (Selected by Held-Out PR-AUC = 0.3380)**: Tabular statistical aggregations (geo-velocity, rolling failure rates, resource diversity spikes) capture the primary structural anomalies of insider attacks with strong precision (**34.92%**) at the headline **Top 1.0% Alert Budget**.
2. **BiLSTM-AE (Evaluated Extension; Not Selected)**: Evaluated as a sequence-aware extension; not selected as primary due to weak synthetic-sequence performance (PR-AUC = 0.0095).
3. **Blended Score (Evaluated; Not Selected)**: Evaluated but not selected because blending diluted the strong IF signal (PR-AUC dropped from 0.3380 to 0.2811).

---

## 5. Attack Type Classification Engine

Once a session is flagged as anomalous by the primary detection threshold ($S_{\text{IF}} \ge \tau_{99}$), it is routed to a **Random Forest Classifier** trained to categorize the specific attack vector.

### 5.1 Training & Cross-Validation Results

The classifier was trained on 636 ground-truth attack sessions using 5-fold Stratified K-Fold cross-validation with `class_weight='balanced'`:

```text
────────────────────────────────────────────────────────────
  RandomForest Classifier — Cross-Validated Performance
────────────────────────────────────────────────────────────
                           precision    recall  f1-score   support

              brute_force       0.99      0.98      0.98       266
      credential_stuffing       0.75      0.90      0.82        20
          device_spoofing       1.00      1.00      1.00         9
        impossible_travel       0.95      0.86      0.90        22
         lateral_movement       0.95      0.98      0.97        62
low_and_slow_exfiltration       1.00      1.00      1.00       257

                 accuracy                           0.98       636
                macro avg       0.94      0.95      0.95       636
             weighted avg       0.98      0.98      0.98       636
```

---

## 6. Two-Tier Explainability & SHAP Integration

### 6.1 Two-Tier Explanation Design
Security analysts require clear distinctions between why a session was flagged as anomalous versus how it was categorized:

1. **Tier 1 (Detector Evidence)**: Tabular feature policy reasons explaining why Isolation Forest flagged the session (e.g., *Geo-velocity 2,454 km/h (>900 km/h limit) + New device fingerprint + Off-hours access*).
2. **Tier 2 (Attack-Type Attribution — Random Forest + SHAP)**: `shap.TreeExplainer` feature attributions explaining why the classifier categorized the alert into a specific attack vector.

```text
SES_0003406 (Impossible Travel Alert):
  • Tier 1 Evidence: Geo-velocity 2,454 km/h + Impossible travel flag + Rare command sequence (rarity=0.98)
  • Tier 2 SHAP Attribution: geo_velocity_kmh (+0.42) & impossible_travel_flag (+0.38) primary contributors to "impossible_travel" classification
```

---

## 7. Production Resilience: Cold-Start & Concept Drift

### 7.1 Cold-Start Handling for New Entities
New entities with $N < 50$ sessions fall back to the **entity-type population baseline**, with confidence scaling linearly:
$$C(N) = 0.20 + 0.70 \times \left(\frac{N}{50}\right) \quad (N < 50)$$
Identified **120 cold-start entities** across 6,000 sessions with appropriate confidence badges in the console.

### 7.2 Behavioral Concept Drift Adaptability
Entity baselines use exponential-decay weighting ($\lambda = 0.02$) to adapt to legitimate role shifts while flagging sudden spikes ($z > 2.0$). Detected 3 drifted entities (`SVC_0090`, etc.).

---

## 8. Interactive Analyst Console & Dashboard

Built a portable Streamlit web interface (`dashboard/dashboard.py`):
- **Headline KPI Header**: Displays Top 1.0% Alert Budget (779 alerts out of 77,896 sessions) and Isolation Forest primary model indicator.
- **Ranked Alert Queue**: Ordered by IF risk score.
- **Two-Tier Detail Inspector**: Side-by-side view of Tier 1 Detector Evidence and Tier 2 SHAP Attack Attribution.
- **Demo Mode**: One-click curated slice featuring examples of each attack category.

---

## 9. Deliverable Manifest & Submission Artifacts

- **Submission Validation Script**: [`validate_submission.py`](file:///d:/honey_hack/validate_submission.py) (Clean no-leakage verification)
- **Pytest Suite**: [`tests/test_validate_submission.py`](file:///d:/honey_hack/tests/test_validate_submission.py) (3/3 Passed)
- **Technical Report**: [`report/report.md`](file:///d:/honey_hack/report/report.md)
- **Pipeline Status & Flowchart**: [`pipeline_status.md`](file:///d:/honey_hack/pipeline_status.md)
- **Analyst Console**: [`dashboard/dashboard.py`](file:///d:/honey_hack/dashboard/dashboard.py)
- **Output Artifacts**:
  - `outputs/ablation_table.csv`
  - `outputs/classifier_confusion.png`
  - `outputs/shap_plots/waterfall_*.png`
  - `data/alerts_with_explanations.csv`
  - `models/isolation_forest.pkl`, `models/classifier.pkl`, `models/bilstm_autoencoder.keras`
