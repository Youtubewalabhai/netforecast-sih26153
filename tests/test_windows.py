"""Unit tests for time-window state aggregation and sequential dataset construction."""

import numpy as np
import pandas as pd
import pytest

from netforecast.attack.mitre_map import MitreMapper
from netforecast.data.load_csv import generate_synthetic_flows
from netforecast.features.windows import (
    aggregate_flows_to_windows,
    build_sequential_dataset,
    split_and_scale_windows,
)


def test_window_aggregation_and_shapes():
    """Verify window aggregation groups flows into state vectors S_t."""
    df = generate_synthetic_flows(num_samples=300, random_state=42)
    mapper = MitreMapper()

    state_df = aggregate_flows_to_windows(df, window_seconds=5.0, mapper=mapper)

    assert len(state_df) > 0
    assert "window_idx" in state_df.columns
    assert "is_infiltration" in state_df.columns
    assert "stage_id" in state_df.columns
    assert "flow_count" in state_df.columns


def test_sequential_dataset_shapes():
    """Verify sliding window sequences match (N, seq_len, D) dimensions."""
    df = generate_synthetic_flows(num_samples=500, random_state=42)
    mapper = MitreMapper()
    state_df = aggregate_flows_to_windows(df, window_seconds=5.0, mapper=mapper)

    train_df, val_df, test_df, scaler, feat_cols = split_and_scale_windows(
        state_df, train_ratio=0.7, val_ratio=0.15
    )

    seq_len = 8
    k_rollout = 4
    seq_data = build_sequential_dataset(
        train_df, seq_len=seq_len, k_rollout=k_rollout, feature_cols=feat_cols
    )

    num_samples = len(train_df) - seq_len - k_rollout + 1
    assert seq_data["X"].shape == (num_samples, seq_len, len(feat_cols))
    assert seq_data["Y_next_state"].shape == (num_samples, len(feat_cols))
    assert seq_data["Y_infil_next"].shape == (num_samples, 1)
    assert seq_data["Y_stage_next"].shape == (num_samples,)
    assert seq_data["Y_infil_rollout"].shape == (num_samples, k_rollout)
    assert seq_data["Y_stage_rollout"].shape == (num_samples, k_rollout)
