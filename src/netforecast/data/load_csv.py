"""CIC-IDS-2018 CSV Loader, Cleaning, and Label Parsing."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean flow dataframe: standardize columns, replace inf/-inf with nan and drop/impute."""
    # Strip whitespace from columns
    df.columns = [c.strip() for c in df.columns]

    # Replace infinity with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Check for Timestamp column
    timestamp_col = None
    for candidate in ["Timestamp", "timestamp", "Time"]:
        if candidate in df.columns:
            timestamp_col = candidate
            break

    if timestamp_col is not None:
        try:
            df["Timestamp"] = pd.to_datetime(
                df[timestamp_col], format="%d/%m/%Y %H:%M:%S", errors="coerce"
            )
            # If coercion failed for some, try generic parser
            if df["Timestamp"].isna().mean() > 0.5:
                df["Timestamp"] = pd.to_datetime(df[timestamp_col], errors="coerce")
        except Exception:
            df["Timestamp"] = pd.to_datetime(df[timestamp_col], errors="coerce")
    else:
        # Generate monotonic timestamps if missing
        df["Timestamp"] = pd.date_range(
            start="2018-02-14 00:00:00", periods=len(df), freq="100ms"
        )

    if "Label" in df.columns:
        df = df[df["Label"].astype(str).str.strip() != "Label"]

    # Drop rows where Timestamp is NaT
    df = df.dropna(subset=["Timestamp"])
    df = df.sort_values(by="Timestamp").reset_index(drop=True)

    # Convert numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    return df


def load_cic_csv(
    file_path: Union[str, Path],
    benign_subsample: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load and clean a single CIC-IDS-2018 CSV file.

    Args:
        file_path: Path to the CSV file.
        benign_subsample: Fraction of Benign rows to keep (0.0 to 1.0) to speed up training.
        random_state: Random seed for sampling.

    Returns:
        Cleaned pandas DataFrame.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path}")

    logger.info("Loading CSV: %s", path.name)
    df = pd.read_csv(path, low_memory=False)
    df = clean_dataframe(df)

    if "Label" not in df.columns:
        # Look for case variations
        for c in df.columns:
            if c.lower() == "label":
                df["Label"] = df[c]
                break

    if "Label" in df.columns and benign_subsample < 1.0:
        benign_mask = df["Label"].astype(str).str.strip().str.lower() == "benign"
        benign_df = df[benign_mask]
        attack_df = df[~benign_mask]

        if len(benign_df) > 0:
            benign_sample = benign_df.sample(
                frac=benign_subsample, random_state=random_state
            )
            df = (
                pd.concat([benign_sample, attack_df])
                .sort_values(by="Timestamp")
                .reset_index(drop=True)
            )

    return df


def load_multiple_csvs(
    raw_dir: Union[str, Path],
    file_list: List[str],
    benign_subsample: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load multiple CIC-IDS-2018 CSV files and combine them."""
    raw_path = Path(raw_dir)
    dfs = []
    for fname in file_list:
        fpath = raw_path / fname
        if fpath.exists():
            df = load_cic_csv(
                fpath, benign_subsample=benign_subsample, random_state=random_state
            )
            dfs.append(df)
        else:
            logger.warning("File not found: %s", fpath)

    if not dfs:
        raise FileNotFoundError(
            f"None of the specified files were found in {raw_dir}: {file_list}"
        )

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values(by="Timestamp").reset_index(drop=True)
    return combined


def generate_synthetic_flows(
    num_samples: int = 500, random_state: int = 42
) -> pd.DataFrame:
    """Generate synthetic network flow data ONLY for unit tests and smoke tests.

    NOTE: Labelled clearly as 'synthetic, not for reporting'.
    """
    np.random.seed(random_state)
    start_time = pd.Timestamp("2026-01-01 10:00:00")
    timestamps = [
        start_time + pd.Timedelta(milliseconds=i * 200) for i in range(num_samples)
    ]

    labels = []
    # 70% Benign, 15% FTP-BruteForce, 15% Infiltration
    for i in range(num_samples):
        if i < int(num_samples * 0.70):
            labels.append("Benign")
        elif i < int(num_samples * 0.85):
            labels.append("FTP-BruteForce")
        else:
            labels.append("Infiltration")

    data = {
        "Timestamp": timestamps,
        "Dst Port": np.random.choice([80, 443, 21, 22, 8080], size=num_samples),
        "Protocol": np.random.choice([6, 17], size=num_samples),
        "Flow Duration": np.random.exponential(scale=1000000, size=num_samples),
        "Tot Fwd Pkts": np.random.poisson(lam=10, size=num_samples) + 1,
        "Tot Bwd Pkts": np.random.poisson(lam=8, size=num_samples),
        "TotLen Fwd Pkts": np.random.gamma(shape=2, scale=500, size=num_samples),
        "TotLen Bwd Pkts": np.random.gamma(shape=2, scale=1500, size=num_samples),
        "Fwd Pkt Len Max": np.random.uniform(50, 1500, size=num_samples),
        "Fwd Pkt Len Min": np.random.uniform(0, 50, size=num_samples),
        "Fwd Pkt Len Mean": np.random.uniform(20, 500, size=num_samples),
        "Fwd Pkt Len Std": np.random.uniform(0, 200, size=num_samples),
        "Bwd Pkt Len Max": np.random.uniform(50, 1500, size=num_samples),
        "Bwd Pkt Len Min": np.random.uniform(0, 50, size=num_samples),
        "Bwd Pkt Len Mean": np.random.uniform(20, 500, size=num_samples),
        "Bwd Pkt Len Std": np.random.uniform(0, 200, size=num_samples),
        "Flow Byts/s": np.random.exponential(scale=50000, size=num_samples),
        "Flow Pkts/s": np.random.exponential(scale=100, size=num_samples),
        "Flow IAT Mean": np.random.exponential(scale=5000, size=num_samples),
        "Flow IAT Std": np.random.exponential(scale=2000, size=num_samples),
        "Flow IAT Max": np.random.exponential(scale=20000, size=num_samples),
        "Flow IAT Min": np.random.exponential(scale=100, size=num_samples),
        "FIN Flag Cnt": np.random.binomial(1, 0.3, size=num_samples),
        "SYN Flag Cnt": np.random.binomial(1, 0.4, size=num_samples),
        "RST Flag Cnt": np.random.binomial(1, 0.05, size=num_samples),
        "PSH Flag Cnt": np.random.binomial(1, 0.5, size=num_samples),
        "ACK Flag Cnt": np.random.binomial(1, 0.9, size=num_samples),
        "URG Flag Cnt": np.random.binomial(1, 0.01, size=num_samples),
        "Down/Up Ratio": np.random.choice([0, 1, 2, 3], size=num_samples),
        "Pkt Size Avg": np.random.uniform(30, 800, size=num_samples),
        "Init Fwd Win Byts": np.random.choice([8192, 29200, 65535], size=num_samples),
        "Init Bwd Win Byts": np.random.choice([8192, 29200, 65535], size=num_samples),
        "Label": labels,
    }

    df = pd.DataFrame(data)
    df = clean_dataframe(df)
    return df
