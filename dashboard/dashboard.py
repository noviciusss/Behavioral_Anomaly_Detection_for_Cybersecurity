"""
Phase 8 — Analyst Dashboard Console
====================================
Streamlit app: ranked alert queue + detail view.

PORTABILITY RULES:
  - Runs with: streamlit run dashboard/dashboard.py
  - No external DB, no environment variables, no hardcoded paths
  - All data loaded from relative paths (../data/, ../outputs/)
  - Works from a ZIP extract on any machine

Operational Design:
  - Primary Detector: Isolation Forest (Headline Top 1.0% Alert Budget = 779 queued alerts)
  - BiLSTM-AE: evaluated sequence-aware extension; not selected due to weak synthetic-sequence performance
  - Blended: evaluated but not selected because it diluted IF signal

Run from project root: streamlit run dashboard/dashboard.py
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UEBA Anomaly Detection Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0f1117;
    color: #e0e0e0;
  }

  .main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #333366;
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 24px;
  }

  .metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #333366;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
  }

  .risk-high   { color: #ef4444; font-weight: 700; }
  .risk-medium { color: #f59e0b; font-weight: 700; }
  .risk-low    { color: #22c55e; font-weight: 700; }

  .tier-box {
    background: #16213e;
    border: 1px solid #333366;
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 12px;
  }

  .cold-start-badge {
    background: #1e3a5f;
    border: 1px solid #3b82f6;
    border-radius: 6px;
    padding: 4px 10px;
    color: #93c5fd;
    font-size: 12px;
  }

  div[data-testid="stDataFrame"] { border-radius: 8px; }
  div[data-testid="stMetric"] { background: #1a1a2e; border-radius: 8px; padding: 10px; border: 1px solid #333366; }
</style>
""", unsafe_allow_html=True)

# ── Path helpers ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def path(*parts):
    return os.path.join(BASE_DIR, *parts)

SHAP_DIR = path("outputs", "shap_plots")

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_alerts():
    p = path("data", "alerts_with_explanations.csv")
    if not os.path.exists(p):
        p = path("data", "classified_alerts.csv")
    if not os.path.exists(p):
        p = path("data", "blended_scores.csv")
    df = pd.read_csv(p, parse_dates=["timestamp"])
    
    # Ensure operational primary score (if_score) is explicit
    if "if_score" in df.columns:
        df["primary_score"] = df["if_score"]
    elif "blended_score" in df.columns:
        df["primary_score"] = df["blended_score"]
    elif "risk_score" in df.columns:
        df["primary_score"] = df["risk_score"]
    else:
        df["primary_score"] = 0.0
        
    return df

@st.cache_data
def load_full_scores():
    p = path("data", "blended_scores.csv")
    if os.path.exists(p):
        df = pd.read_csv(p, parse_dates=["timestamp"])
        if "if_score" in df.columns:
            df["primary_score"] = df["if_score"]
        else:
            df["primary_score"] = df.get("blended_score", 0.0)
        return df
    return pd.DataFrame()

@st.cache_data
def load_cold_start():
    p = path("data", "cold_start_entities.csv")
    if os.path.exists(p):
        return pd.read_csv(p, parse_dates=["timestamp"])
    return pd.DataFrame()

@st.cache_data
def load_ablation():
    p = path("outputs", "ablation_table.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return pd.DataFrame()

# ── Curated demo slice ────────────────────────────────────────────────────────
@st.cache_data
def get_demo_slice(alerts_df):
    """
    One of each attack type (highest confidence), plus context rows.
    """
    demo_rows = []
    attack_col = "predicted_attack_type" if "predicted_attack_type" in alerts_df.columns else "label"

    for atype in alerts_df[attack_col].dropna().unique():
        subset = alerts_df[alerts_df[attack_col] == atype]
        if "classifier_confidence" in subset.columns:
            row = subset.loc[subset["classifier_confidence"].idxmax()]
        else:
            row = subset.iloc[0]
        demo_rows.append(row)

    if demo_rows:
        return pd.DataFrame(demo_rows).reset_index(drop=True)
    return alerts_df.head(20)

# ── Risk level classification ─────────────────────────────────────────────────
def risk_level(score):
    if score >= 0.74:
        return "🔴 HIGH"
    elif score >= 0.50:
        return "🟡 MEDIUM"
    else:
        return "🟢 LOW"

def risk_color(score):
    if score >= 0.74:
        return "#ef4444"
    elif score >= 0.50:
        return "#f59e0b"
    else:
        return "#22c55e"

# ── SHAP plot loader ──────────────────────────────────────────────────────────
def load_shap_waterfall(attack_type):
    if not attack_type:
        return None
    safe_name = str(attack_type).lower().replace(" ", "_")
    candidates = [
        os.path.join(SHAP_DIR, f"waterfall_{safe_name}.png"),
        os.path.join(SHAP_DIR, f"waterfall_{attack_type}.png"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

# ── Detector Evidence Formatting (Tier 1) ─────────────────────────────────────
def format_detector_evidence(row):
    reasons = []
    
    geo_v = float(row.get("geo_velocity_kmh", 0))
    if geo_v > 0:
        if geo_v > 900:
            reasons.append(f"• **Geo-velocity**: `{geo_v:.0f} km/h` *(exceeds 900 km/h physical limit)*")
        else:
            reasons.append(f"• **Geo-velocity**: `{geo_v:.0f} km/h` *(elevated velocity)*")
            
    if row.get("impossible_travel_flag", 0) == 1:
        reasons.append("• **Impossible Travel**: multi-region access in impossible timeframe")
        
    if row.get("is_spoofed_device", 0) == 1:
        reasons.append("• **Device Fingerprint**: `spoofed hash mismatch` detected")
    elif row.get("is_new_device", 0) == 1:
        reasons.append("• **Device Fingerprint**: first time login from unrecognised hardware ID")
        
    if row.get("is_new_resource", 0) == 1:
        reasons.append(f"• **Resource Access**: first access to `{row.get('resource_accessed', 'target resource')}`")
        
    fail_auth = float(row.get("failed_auth_rate_10", 0))
    if fail_auth > 0:
        reasons.append(f"• **Failed Auth Rate**: `{fail_auth:.0%}` rolling failure rate")
        
    div = float(row.get("resource_diversity_10", 0))
    if div >= 4:
        reasons.append(f"• **Resource Diversity**: `{int(div)}` distinct resources accessed in short window")
        
    if row.get("is_off_hours", 0) == 1:
        hr = row.get("hour_of_day", "—")
        reasons.append(f"• **Access Window**: off-hours access (`{hr}:00` local time)")
        
    dur_z = float(row.get("session_duration_zscore", 0))
    if abs(dur_z) > 2.0:
        reasons.append(f"• **Session Duration**: duration z-score `{dur_z:+.1f}σ` from baseline")
        
    rarity = float(row.get("cmd_rarity_score", 0))
    if rarity > 0.85:
        reasons.append(f"• **Command Sequence**: rare command pattern (rarity `{rarity:.2f}`)")

    if not reasons:
        reasons.append("• Statistical tabular feature anomaly detected by Isolation Forest partition depth")
        
    return "\n".join(reasons)

# ── Entity sparkline ──────────────────────────────────────────────────────────
def entity_risk_sparkline(entity_id, full_scores_df):
    if full_scores_df.empty:
        return None
    
    score_col = "if_score" if "if_score" in full_scores_df.columns else "primary_score"
    if score_col not in full_scores_df.columns:
        return None

    entity_df = full_scores_df[full_scores_df["entity_id"] == entity_id].sort_values("timestamp")
    if len(entity_df) < 3:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=entity_df["timestamp"],
        y=entity_df[score_col],
        mode="lines",
        line=dict(color="#22c55e", width=2),
        fill="tozeroy",
        fillcolor="rgba(34, 197, 94, 0.15)",
        name="Isolation Forest Score (Primary)",
    ))

    # Mark flagged sessions
    if "flagged" in entity_df.columns:
        flagged = entity_df[entity_df["flagged"] == 1]
        fig.add_trace(go.Scatter(
            x=flagged["timestamp"],
            y=flagged[score_col],
            mode="markers",
            marker=dict(color="#ef4444", size=8, symbol="circle"),
            name="Flagged Session",
        ))

    fig.update_layout(
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        margin=dict(l=0, r=0, t=0, b=0),
        height=200,
        showlegend=True,
        legend=dict(font=dict(color="white"), bgcolor="#1a1a2e"),
        xaxis=dict(color="white", gridcolor="#333366", showgrid=True),
        yaxis=dict(color="white", gridcolor="#333366", showgrid=True, range=[0, 1]),
        font=dict(color="white"),
    )
    return fig

# ── Main dashboard ────────────────────────────────────────────────────────────
def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; color:#e0e0e0; font-size:1.8rem;">
            🛡️ UEBA Behavioral Anomaly Detection Console
        </h1>
        <p style="margin:6px 0 0; color:#4ade80; font-size:0.9rem; font-weight: 600;">
            Primary Detector: <b>Isolation Forest</b> (Headline Operating Point: Top 1.0% Alert Budget = 779 Alerts)
        </p>
        <p style="margin:2px 0 0; color:#94a3b8; font-size:0.8rem;">
            • <b>Production-Oriented Prototype</b>: Event-by-event scoring design with batch replay<br>
            • <b>BiLSTM-AE</b>: evaluated sequence-aware extension; not selected due to weak synthetic-sequence performance<br>
            • <b>Blended Model</b>: evaluated but not selected because it diluted IF signal
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    alerts_df    = load_alerts()
    full_scores  = load_full_scores()
    cold_df      = load_cold_start()
    ablation_df  = load_ablation()

    score_col = "if_score" if "if_score" in alerts_df.columns else "primary_score"

    # ── Top-level metrics (KPI Source: Top 1.0% Headline Budget) ─────────────
    n_scored     = 77896
    n_flagged    = len(alerts_df)
    n_top1_pct   = 779
    n_high_risk  = (alerts_df[score_col] >= 0.74).sum()
    n_cold_start = cold_df["entity_id"].nunique() if not cold_df.empty else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🚨 Alert Budget (Top 1.0%)", f"{n_top1_pct:,} alerts", help="Headline operational budget: top 1.0% of scored sessions")
    with col2:
        st.metric("📊 Total Scored Sessions", f"{n_scored:,}")
    with col3:
        st.metric("🔴 High Risk (IF >= 0.74)", f"{n_high_risk:,}")
    with col4:
        st.metric("🌱 Cold-Start Entities", f"{n_cold_start:,}")

    st.divider()

    # ── Sidebar filters ───────────────────────────────────────────────────────
    st.sidebar.title("🔍 Operational Controls")
    st.sidebar.caption("Primary Detector: Isolation Forest")

    # Demo mode toggle
    demo_mode = st.sidebar.toggle("🎬 Demo Mode (curated slice)", value=False,
                                   help="Shows one example of each attack type for demos")
    if demo_mode:
        display_df = get_demo_slice(alerts_df)
        st.sidebar.info("Demo mode: showing curated alerts (one per attack type)")
    else:
        display_df = alerts_df.copy()

    # Entity type filter
    if "entity_type" in display_df.columns:
        entity_types = ["All"] + sorted(display_df["entity_type"].dropna().unique().tolist())
        selected_type = st.sidebar.selectbox("Entity Type", entity_types)
        if selected_type != "All":
            display_df = display_df[display_df["entity_type"] == selected_type]

    # Attack type filter
    attack_col = "predicted_attack_type" if "predicted_attack_type" in display_df.columns else "label"
    attack_types = ["All"] + sorted(display_df[attack_col].dropna().unique().tolist())
    selected_attack = st.sidebar.selectbox("Attack Category", attack_types)
    if selected_attack != "All":
        display_df = display_df[display_df[attack_col] == selected_attack]

    # Risk level filter
    risk_filter = st.sidebar.multiselect(
        "Risk Severity Level (IF Score)",
        ["🔴 HIGH", "🟡 MEDIUM", "🟢 LOW"],
        default=["🔴 HIGH", "🟡 MEDIUM"],
    )
    display_df = display_df[display_df[score_col].apply(risk_level).isin(risk_filter)]
    min_score = st.sidebar.slider("Minimum IF Risk Threshold", 0.0, 1.0, 0.0, 0.01)
    display_df = display_df[display_df[score_col] >= min_score]

    # Queue Ordering: Sort by Isolation Forest score descending
    display_df = display_df.sort_values(score_col, ascending=False).reset_index(drop=True)

    # ── Alert queue table ─────────────────────────────────────────────────────
    st.subheader(f"📋 Operational Alert Queue — Primary Detector: Isolation Forest  ({len(display_df):,} alerts)")

    display_df["IF Risk Score"] = display_df[score_col]
    
    TABLE_COLS_RAW = {
        "session_id":              "Session ID",
        "entity_id":               "Entity",
        "entity_type":             "Type",
        "timestamp":               "Timestamp",
        "resource_accessed":       "Resource",
        attack_col:                "Attack Category",
        "IF Risk Score":           "IF Risk Score",
        "classifier_confidence":   "Confidence",
        "explanation":             "Explanation",
    }
    available_cols = {k: v for k, v in TABLE_COLS_RAW.items() if k in display_df.columns}

    table_df = display_df[list(available_cols.keys())].rename(columns=available_cols)
    if "IF Risk Score" in table_df.columns:
        table_df["Severity"] = table_df["IF Risk Score"].apply(risk_level)
        table_df["IF Risk Score"] = table_df["IF Risk Score"].apply(lambda x: f"{x:.4f}")

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        height=380,
        column_config={
            "IF Risk Score": st.column_config.TextColumn(width="small"),
            "Severity":      st.column_config.TextColumn(width="small"),
            "Confidence":    st.column_config.ProgressColumn(min_value=0, max_value=1, width="small"),
            "Explanation":   st.column_config.TextColumn(width="large"),
        }
    )

    st.divider()

    # ── Two-Tier Detail View ──────────────────────────────────────────────────
    st.subheader("🔎 Alert Detail Inspector & Two-Tier Explanations")

    if len(display_df) > 0:
        alert_options = display_df["session_id"].tolist() if "session_id" in display_df.columns else []
        selected_sid  = st.selectbox("Select a session to inspect", alert_options, key="detail_select")

        if selected_sid:
            row = display_df[display_df["session_id"] == selected_sid].iloc[0]
            if_risk = float(row.get(score_col, 0))

            col_a, col_b = st.columns([1, 1])

            with col_a:
                st.markdown("### 🌲 Tier 1: Detector Evidence")
                st.caption("Why Isolation Forest flagged this session as anomalous")
                
                st.markdown(f"""
                | Field | Value |
                |---|---|
                | **Entity ID** | `{row.get('entity_id', '—')}` |
                | **Entity Type** | {row.get('entity_type', '—')} |
                | **Timestamp** | {str(row.get('timestamp', '—'))[:19]} |
                | **Target Resource** | `{row.get('resource_accessed', '—')}` |
                | **Primary IF Risk Score** | **{if_risk:.4f}** |
                | **Severity Level** | {risk_level(if_risk)} |
                """)

                # Cold-start badge
                if not cold_df.empty and selected_sid in cold_df.get("session_id", pd.Series()).values:
                    cold_row = cold_df[cold_df["session_id"] == selected_sid].iloc[0]
                    st.markdown(f"""
                    <div class="cold-start-badge">
                    🌱 Cold-start entity · session {int(cold_row['session_count'])} of 50 ·
                    confidence {float(cold_row['confidence']):.0%}
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("**Detection Feature Policy Evidence:**")
                evidence_text = format_detector_evidence(row)
                st.markdown(evidence_text)

            with col_b:
                st.markdown("### 🎯 Tier 2: Attack-Type Attribution")
                st.caption(f"Random Forest + SHAP: why this alert resembles '{row.get(attack_col, 'Attack')}'")
                
                st.markdown(f"""
                | Classification Attribute | Value |
                |---|---|
                | **Attributed Attack Category** | **{row.get(attack_col, '—')}** |
                | **Classifier Confidence** | **{float(row.get('classifier_confidence', 0)):.2%}** |
                """)

                # SHAP waterfall plot
                attack_type_for_plot = row.get(attack_col, None)
                waterfall_path = load_shap_waterfall(attack_type_for_plot)
                if waterfall_path:
                    st.markdown("**Attack-Type Attribution (Random Forest + SHAP)**")
                    st.image(waterfall_path, use_column_width=True)
                else:
                    st.markdown("*SHAP waterfall plot not available for this session.*")

            # Entity history sparkline
            entity_id = row.get("entity_id")
            if entity_id and not full_scores.empty:
                st.markdown(f"**Entity Risk Profile History — `{entity_id}`**")
                sparkline = entity_risk_sparkline(entity_id, full_scores)
                if sparkline:
                    st.plotly_chart(sparkline, use_container_width=True, key=f"spark_{selected_sid}")

    st.divider()

    # ── Model Ablation & Evaluation Comparison Section ──────────────────────────
    st.subheader("📊 Model Ablation & Evaluation Comparison")
    
    st.markdown("""
    > **Model Selection Decision Matrix**:
    > - **Primary Detector**: Isolation Forest *(Headline Top 1.0% Budget = PR-AUC 0.3380, Precision@1% 34.92%, F1@1% 0.3845)*
    > - **BiLSTM-AE**: evaluated sequence-aware extension; not selected due to weak synthetic-sequence performance (PR-AUC = 0.0095)
    > - **Blended Model**: evaluated but not selected because it diluted IF signal (PR-AUC dropped to 0.2811)
    """)

    if not ablation_df.empty:
        st.dataframe(ablation_df, use_container_width=True)
        st.caption("Primary evaluation metric: PR-AUC (Precision-Recall Area Under Curve). ROC-AUC shown for reference only — misleading under 0.82% anomaly imbalance.")

    # ── Score distribution ─────────────────────────────────────────────────────
    if not full_scores.empty:
        st.subheader("📈 Score Distributions (Primary vs Research Models)")
        col_p, col_q = st.columns(2)

        plot_score_col = "if_score" if "if_score" in full_scores.columns else "primary_score"

        with col_p:
            fig_hist = px.histogram(
                full_scores, x=plot_score_col,
                color="entity_type" if "entity_type" in full_scores.columns else None,
                nbins=80, title="Primary Isolation Forest Risk Score Distribution",
                template="plotly_dark",
                labels={plot_score_col: "Isolation Forest Risk Score"},
                color_discrete_sequence=["#22c55e", "#ef4444", "#f59e0b"]
            )
            fig_hist.update_layout(
                paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
                legend=dict(bgcolor="#1a1a2e"),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_q:
            if "label" in full_scores.columns:
                label_counts = full_scores["label"].value_counts().reset_index()
                label_counts.columns = ["Label", "Count"]
                fig_pie = px.pie(
                    label_counts, names="Label", values="Count",
                    title="Session Ground-Truth Label Distribution",
                    template="plotly_dark",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                )
                fig_pie.update_layout(paper_bgcolor="#1a1a2e")
                st.plotly_chart(fig_pie, use_container_width=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    <div style="text-align:center; color:#555577; font-size:0.8rem; padding: 12px 0;">
        AI-Powered Behavioral Anomaly Detection · Honeywell Hackathon 2026 · 
        Stack: Isolation Forest (Primary) + BiLSTM-AE (Evaluated Extension) + SHAP (Attack Attribution) + Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
