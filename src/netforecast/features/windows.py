"""Time-window State Vector Construction and Sequence Dataset Preparation."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from netforecast.attack.mitre_map import MitreMapper

logger = logging.getLogger(__name__)


def aggregate_flows_to_windows(
    df: pd.DataFrame,
    window_seconds: float = 10.0,
    mapper: Optional[MitreMapper] = None,
) -> pd.DataFrame:
    """Aggregate per-flow network records into fixed time-window state vectors S_t.

    Args:
        df: Cleaned dataframe containing network flows with 'Timestamp' and 'Label'.
        window_seconds: Window length in seconds (default: 10s).
        mapper: MitreMapper instance for mapping labels to MITRE stages.

    Returns:
        pd.DataFrame where each row is a time-window state vector S_t.
    """
    if mapper is None:
        mapper = MitreMapper()

    df = df.copy()
    if not np.issubdtype(df["Timestamp"].dtype, np.datetime64):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    df = df.sort_values(by="Timestamp").reset_index(drop=True)
    start_time = df["Timestamp"].min()

    # Calculate window index for each flow
    time_diff_sec = (df["Timestamp"] - start_time).dt.total_seconds()
    df["window_idx"] = (time_diff_sec // window_seconds).astype(int)

    # Separate numeric features from metadata
    drop_cols = ["Timestamp", "Label", "window_idx"]
    feature_cols = [
        c
        for c in df.columns
        if c not in drop_cols and np.issubdtype(df[c].dtype, np.number)
    ]

    window_records = []
    grouped = df.groupby("window_idx")

    for widx, group in grouped:
        window_start = start_time + pd.Timedelta(seconds=float(widx * window_seconds))
        flow_count = len(group)

        # Aggregate flow statistical features
        mean_stats = group[feature_cols].mean().to_dict()
        std_stats = group[feature_cols].std().fillna(0.0).to_dict()
        max_stats = group[feature_cols].max().to_dict()

        record = {
            "window_idx": widx,
            "window_time": window_start,
            "flow_count": flow_count,
        }

        # Add statistical prefixes to distinguish aggregates
        for col in feature_cols:
            record[f"{col}_mean"] = mean_stats.get(col, 0.0)
            record[f"{col}_std"] = std_stats.get(col, 0.0)
            record[f"{col}_max"] = max_stats.get(col, 0.0)

        # Label aggregation:
        # 1. Infiltration flag: 1 if any flow is marked as infiltration
        # 2. Stage label: Highest-severity attack stage present in this window
        stage_ids = []
        infil_flags = []
        raw_labels = group["Label"].tolist() if "Label" in group.columns else ["Benign"]

        for lbl in raw_labels:
            info = mapper.get_stage_info(str(lbl))
            stage_ids.append(info["stage_id"])
            infil_flags.append(info["is_infiltration"])

        record["is_infiltration"] = int(any(f == 1 for f in infil_flags))
        record["stage_id"] = int(max(stage_ids)) if stage_ids else 0
        record["dominant_label"] = group["Label"].mode()[0] if "Label" in group.columns else "Benign"

        window_records.append(record)

    state_df = pd.DataFrame(window_records)
    state_df = state_df.sort_values(by="window_idx").reset_index(drop=True)
    return state_df


def build_sequential_dataset(
    state_df: pd.DataFrame,
    seq_len: int = 12,
    k_rollout: int = 5,
    feature_cols: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """Build temporal sliding sequence dataset for PyTorch World Model training.

    Args:
        state_df: Dataframe of time-window states S_t.
        seq_len: Number of historical windows (e.g. 12 windows).
        k_rollout: Forecast horizon K (e.g. 5 steps forward).
        feature_cols: Specific feature columns to include in state vector S_t.

    Returns:
        Dictionary containing:
            'X': Historical state sequences, shape (N, seq_len, num_features)
            'Y_next_state': Ground truth S_t+1, shape (N, num_features)
            'Y_infil_next': Ground truth infiltration probability at t+1, shape (N, 1)
            'Y_stage_next': Ground truth MITRE stage at t+1, shape (N,)
            'Y_infil_rollout': Infiltration labels across t+1..t+K, shape (N, k_rollout)
            'Y_stage_rollout': Stage labels across t+1..t+K, shape (N, k_rollout)
    """
    if feature_cols is None:
        meta_cols = ["window_idx", "window_time", "is_infiltration", "stage_id", "dominant_label"]
        feature_cols = [c for c in state_df.columns if c not in meta_cols]

    features = state_df[feature_cols].values.astype(np.float32)
    infil_labels = state_df["is_infiltration"].values.astype(np.float32)
    stage_labels = state_df["stage_id"].values.astype(np.int64)

    total_len = len(state_df)
    min_required = seq_len + k_rollout

    if total_len < min_required:
        raise ValueError(
            f"Not enough windows ({total_len}) for seq_len={seq_len} + k_rollout={k_rollout}. Need at least {min_required}."
        )

    X_list = []
    Y_next_state_list = []
    Y_infil_next_list = []
    Y_stage_next_list = []
    Y_infil_rollout_list = []
    Y_stage_rollout_list = []

    for i in range(total_len - seq_len - k_rollout + 1):
        # Sequence input: [t-seq_len+1 ... t]
        x_seq = features[i : i + seq_len]

        # Target S_t+1
        y_next_s = features[i + seq_len]

        # Target classification at t+1
        y_infil_next = infil_labels[i + seq_len]
        y_stage_next = stage_labels[i + seq_len]

        # Rollout targets across [t+1 ... t+K]
        y_infil_k = infil_labels[i + seq_len : i + seq_len + k_rollout]
        y_stage_k = stage_labels[i + seq_len : i + seq_len + k_rollout]

        X_list.append(x_seq)
        Y_next_state_list.append(y_next_s)
        Y_infil_next_list.append([y_infil_next])
        Y_stage_next_list.append(y_stage_next)
        Y_infil_rollout_list.append(y_infil_k)
        Y_stage_rollout_list.append(y_stage_k)

    return {
        "X": np.array(X_list, dtype=np.float32),
        "Y_next_state": np.array(Y_next_state_list, dtype=np.float32),
        "Y_infil_next": np.array(Y_infil_next_list, dtype=np.float32),
        "Y_stage_next": np.array(Y_stage_next_list, dtype=np.int64),
        "Y_infil_rollout": np.array(Y_infil_rollout_list, dtype=np.float32),
        "Y_stage_rollout": np.array(Y_stage_rollout_list, dtype=np.int64),
        "feature_cols": feature_cols,
    }


def split_and_scale_windows(
    state_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler, List[str]]:
    """Strict time-based split of windows into train, val, and test partitions,

    fitting StandardScaler ONLY on the train partition to prevent data leakage.
    """
    if feature_cols is None:
        meta_cols = ["window_idx", "window_time", "is_infiltration", "stage_id", "dominant_label"]
        feature_cols = [c for c in state_df.columns if c not in meta_cols]

    n = len(state_df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = state_df.iloc[:train_end].copy()
    val_df = state_df.iloc[train_end:val_end].copy()
    test_df = state_df.iloc[val_end:].copy()

    # Fit scaler ONLY on train data
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)

    train_df[feature_cols] = scaler.transform(train_df[feature_cols].values)
    val_df[feature_cols] = scaler.transform(val_df[feature_cols].values)
    test_df[feature_cols] = scaler.transform(test_df[feature_cols].values)

    return train_df, val_df, test_df, scaler, feature_cols
