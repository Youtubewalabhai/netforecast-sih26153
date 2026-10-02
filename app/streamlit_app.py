"""Streamlit Offline Web Application for AI Network Attack Forecasting (NetForecast).

SIH26153 - Team CHECK_MATE
"""

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import yaml

from netforecast.attack.mitre_map import MitreMapper
from netforecast.data.load_csv import clean_dataframe, generate_synthetic_flows, load_cic_csv
from netforecast.data.pcap_features import create_sample_pcap, extract_features_from_pcap
from netforecast.explain.shap_explain import SequenceExplainer
from netforecast.features.windows import aggregate_flows_to_windows, build_sequential_dataset
from netforecast.models.rollout import RolloutSimulator
from netforecast.models.world_model import NetworkWorldModel

# Streamlit Page Config
st.set_page_config(
    page_title="NetForecast | AI Attack Forecasting World Model",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    .stMetric {
        background-color: #1a2234;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
    }
    .mitre-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 12px;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_config_and_model():
    """Load configuration, mapper, and trained PyTorch World Model."""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "model": {
                "hidden_dim": 128,
                "num_layers": 2,
                "seq_len": 12,
                "k_rollout": 5,
                "type": "lstm",
                "dropout": 0.2,
                "device": "cpu",
            },
            "data": {"time_window_seconds": 10.0},
        }

    mapper = MitreMapper(config)
    model_path = Path("models/world_model.pt")

    if model_path.exists():
        checkpoint = torch.load(model_path, map_location="cpu")
        input_dim = checkpoint.get("input_dim", 72)
        feature_cols = checkpoint.get("feature_cols", [])
        model = NetworkWorldModel(
            input_dim=input_dim,
            hidden_dim=config["model"].get("hidden_dim", 128),
            num_layers=config["model"].get("num_layers", 2),
            num_stages=mapper.num_stages,
            dropout=config["model"].get("dropout", 0.2),
            model_type=config["model"].get("type", "lstm"),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    else:
        # Fallback untrained model for initial UI inspection
        input_dim = 72
        feature_cols = []
        model = NetworkWorldModel(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=2,
            num_stages=mapper.num_stages,
        )
        model.eval()

    return config, mapper, model, feature_cols


def main():
    st.title("🛡️ NetForecast: Predictive Network Dynamics World Model")
    st.caption(
        "SIH26153: AI-based Network Attack Forecasting from Network Traffic Data | Team CHECK_MATE (Fully Offline)"
    )

    config, mapper, model, trained_feature_cols = load_config_and_model()

    # Sidebar Controls
    st.sidebar.header("🕹️ Telemetry Input & Control")
    mode = st.sidebar.radio(
        "Select Telemetry Source",
        ["Demo Mode (Pre-loaded / Synthetic Attack)", "Upload Flow CSV (CIC-IDS-2018)", "Upload Packet PCAP"],
    )

    k_rollout = st.sidebar.slider("Forecast Horizon (K Steps)", min_value=1, max_value=10, value=5)
    window_sec = st.sidebar.slider("State Window Size (Seconds)", min_value=5, max_value=30, value=10)
    risk_threshold = st.sidebar.slider("Infiltration Alert Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

    df_flows = None
    source_description = ""

    if mode == "Demo Mode (Pre-loaded / Synthetic Attack)":
        source_description = "Synthetic Network Telemetry Sequence (Demo Mode - Offline)"
        df_flows = generate_synthetic_flows(num_samples=400, random_state=42)
    elif mode == "Upload Flow CSV (CIC-IDS-2018)":
        uploaded_file = st.sidebar.file_uploader("Upload Network Flow CSV", type=["csv"])
        if uploaded_file is not None:
            df_flows = pd.read_csv(uploaded_file, low_memory=False)
            df_flows = clean_dataframe(df_flows)
            source_description = f"Uploaded CSV: {uploaded_file.name} ({len(df_flows)} records)"
        else:
            st.info("👆 Please upload a CIC-IDS-2018 compatible CSV or switch to Demo Mode.")
            return
    elif mode == "Upload Packet PCAP":
        uploaded_pcap = st.sidebar.file_uploader("Upload PCAP/PCAPNG capture", type=["pcap", "pcapng"])
        if uploaded_pcap is not None:
            temp_pcap_path = Path("scratch/uploaded_temp.pcap")
            temp_pcap_path.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_pcap_path, "wb") as f:
                f.write(uploaded_pcap.getbuffer())
            with st.spinner("Extracting packet-level telemetry with Scapy..."):
                pcap_features_df = extract_features_from_pcap(temp_pcap_path, window_seconds=window_sec)
            if pcap_features_df.empty:
                st.error("No valid IP packets found in PCAP.")
                return
            st.success(f"Extracted {len(pcap_features_df)} time windows from PCAP.")
            st.dataframe(pcap_features_df.head(10))
            return
        else:
            st.info("👆 Please upload a .pcap file or click below to generate a sample PCAP.")
            if st.button("Generate & Process Sample Test PCAP"):
                sample_pcap_path = Path("scratch/sample_demo.pcap")
                create_sample_pcap(sample_pcap_path, num_packets=50)
                pcap_df = extract_features_from_pcap(sample_pcap_path, window_seconds=window_sec)
                st.success("Generated and parsed Scapy test PCAP:")
                st.dataframe(pcap_df)
            return

    # Process Flow Telemetry
    state_df = aggregate_flows_to_windows(df_flows, window_seconds=window_sec, mapper=mapper)
    st.sidebar.success(f"Aggregated {len(df_flows)} flows into {len(state_df)} state windows ($S_t$).")

    # Metrics Summary Row
    total_windows = len(state_df)
    infil_windows = int(state_df["is_infiltration"].sum())
    attack_ratio = (infil_windows / max(total_windows, 1)) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Telemetry Source", mode.split(" ")[0])
    col2.metric("Total Windows (S_t)", total_windows)
    col3.metric("Observed Infiltrations", infil_windows)
    col4.metric("Attack Window %", f"{attack_ratio:.1f}%")

    # Temporal Sequence Rollout
    seq_len = 12
    if len(state_df) < (seq_len + k_rollout):
        st.warning(
            f"Dataset has {len(state_df)} windows, but at least {seq_len + k_rollout} are required for {seq_len}-step sequence rollout."
        )
        return

    # Prepare features
    meta_cols = ["window_idx", "window_time", "is_infiltration", "stage_id", "dominant_label"]
    feature_cols = [c for c in state_df.columns if c not in meta_cols]

    # Initialize model if not already dimensioned
    if model.input_dim != len(feature_cols):
        model = NetworkWorldModel(
            input_dim=len(feature_cols),
            hidden_dim=128,
            num_layers=2,
            num_stages=mapper.num_stages,
        )
        model.eval()

    seq_data = build_sequential_dataset(
        state_df, seq_len=seq_len, k_rollout=k_rollout, feature_cols=feature_cols
    )

    # Simulation & Forecasting
    simulator = RolloutSimulator(model=model, mapper=mapper, device="cpu")
    rollout_out = simulator.rollout_batch(seq_data["X"], k_steps=k_rollout)

    # Main Visualizations
    st.subheader("📈 Real-Time State Progression & K-Step Infiltration Rollout")

    # Plot Timeline
    history_infil = seq_data["Y_infil_next"].ravel()
    pred_step1 = rollout_out["infil_probabilities"][:, 0]
    pred_stepK = rollout_out["infil_probabilities"][:, -1]

    fig = go.Figure()
    time_indices = list(range(len(history_infil)))

    fig.add_trace(
        go.Scatter(
            x=time_indices,
            y=history_infil,
            mode="lines",
            name="Ground Truth Infiltration",
            line=dict(color="#10b981", width=2, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time_indices,
            y=pred_step1,
            mode="lines+markers",
            name=f"World Model Forecast (t+1, {window_sec}s ahead)",
            line=dict(color="#3b82f6", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time_indices,
            y=pred_stepK,
            mode="lines",
            name=f"Autoregressive Forecast (t+{k_rollout}, {k_rollout*window_sec}s ahead)",
            line=dict(color="#ef4444", width=2, dash="dash"),
        )
    )

    # Add threshold line
    fig.add_hline(
        y=risk_threshold,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text=f"Alert Threshold ({risk_threshold})",
    )

    fig.update_layout(
        title="Network Infiltration Risk Trajectory Over Time Windows",
        xaxis_title="Time Window Index ($S_t$)",
        yaxis_title="Predicted Infiltration Probability $P(Infiltration)$",
        template="plotly_dark",
        height=400,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # MITRE ATT&CK Stage Timeline and Inspection
    st.subheader("🎯 MITRE ATT&CK Stage Mapping & Forecast Details")
    selected_idx = st.slider(
        "Select Time Window to Inspect Multi-Step Rollout & Explainability",
        min_value=0,
        max_value=len(seq_data["X"]) - 1,
        value=min(10, len(seq_data["X"]) - 1),
    )

    sample_seq = seq_data["X"][selected_idx]
    single_rollout = simulator.rollout_single_sequence(sample_seq, k_steps=k_rollout)

    col_sim1, col_sim2 = st.columns([1, 1])

    with col_sim1:
        st.markdown(f"#### K-Step Rollout Forecast for Window `{selected_idx}`")
        rollout_table = []
        for step in range(k_rollout):
            prob = single_rollout["infil_probabilities"][step]
            stage = single_rollout["predicted_stage_names"][step]
            alert = "🚨 HIGH RISK" if prob >= risk_threshold else "✅ NORMAL"
            rollout_table.append(
                {
                    "Horizon": f"t+{step+1} (+{(step+1)*window_sec}s)",
                    "Infiltration Risk": f"{prob * 100:.1f}%",
                    "MITRE ATT&CK Stage": stage,
                    "Threat Status": alert,
                }
            )
        st.dataframe(pd.DataFrame(rollout_table), use_container_width=True)

    with col_sim2:
        st.markdown("#### Explainability: Top Driving Network Telemetry Features")
        explainer = SequenceExplainer(
            model=model,
            feature_names=feature_cols,
            background_data=seq_data["X"][:30],
        )
        explanation = explainer.explain_instance_permutation(sample_seq, num_permutations=15)
        top_f = explanation["top_features"][:7]

        feat_df = pd.DataFrame(top_f)
        fig_feat = px.bar(
            feat_df,
            x="importance",
            y="feature",
            orientation="h",
            title=f"Feature Attributions ({explanation['method']})",
            color="importance",
            color_continuous_scale="Viridis",
            template="plotly_dark",
            height=280,
        )
        fig_feat.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_feat, use_container_width=True)

    # MITRE Matrix Reference Table
    with st.expander("ℹ️ MITRE ATT&CK Enterprise Matrix Alignment Reference"):
        st.markdown(
            """
            | Dataset Attack Label | Mapped MITRE Stage | MITRE Tactic / Technique | Primary Action |
            | :--- | :--- | :--- | :--- |
            | **Benign** | Benign (Stage 0) | None | Normal operations |
            | **FTP-BruteForce / SSH-Bruteforce** | Initial Access (Stage 2) | Credential Access (T1110) | Automated authentication spray |
            | **Infiltration** | Lateral Movement (Stage 3) | Lateral Movement (T1021) / Exfiltration | Privilege escalation & discovery |
            | **Bot** | Command and Control (Stage 4) | C2 Channel (T1071) | Beaconing and remote tasks |
            | **DoS / DDoS** | Impact (Stage 5) | Denial of Service (T1498) | Resource starvation |
            """
        )


if __name__ == "__main__":
    main()
