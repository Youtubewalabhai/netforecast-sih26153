"""Unit tests for CSV loader, cleaning, and synthetic flow generation."""

import pandas as pd
import pytest

from netforecast.data.load_csv import clean_dataframe, generate_synthetic_flows


def test_clean_dataframe():
    """Verify cleaning of NaN, Inf, and timestamp parsing."""
    raw_data = {
        "Timestamp": ["14/02/2018 08:30:00", "14/02/2018 08:30:01", "invalid_time"],
        "Dst Port": [80, 443, 22],
        "Flow Duration": [100.0, float("inf"), 300.0],
        "Label": ["Benign", "Infiltration", "Benign"],
    }
    df = pd.DataFrame(raw_data)
    cleaned = clean_dataframe(df)

    # Invalid timestamp row should be dropped
    assert len(cleaned) == 2
    assert "Timestamp" in cleaned.columns
    assert pd.api.types.is_datetime64_any_dtype(cleaned["Timestamp"])
    # Infinite value should be converted to 0.0 or finite
    assert not cleaned["Flow Duration"].isin([float("inf"), float("-inf")]).any()


def test_generate_synthetic_flows():
    """Verify synthetic dataset generator produces expected schema and distribution."""
    df = generate_synthetic_flows(num_samples=200, random_state=42)

    assert len(df) == 200
    assert "Label" in df.columns
    assert "Dst Port" in df.columns
    assert "Timestamp" in df.columns
    assert set(df["Label"].unique()).issubset({"Benign", "FTP-BruteForce", "Infiltration"})
