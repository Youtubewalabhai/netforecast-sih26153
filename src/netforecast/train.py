"""Training Pipeline for NetForecast World Model and Baseline.

SIH26153 - Team CHECK_MATE
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, TensorDataset

from netforecast.attack.mitre_map import MitreMapper
from netforecast.data.load_csv import generate_synthetic_flows, load_multiple_csvs
from netforecast.features.windows import (
    aggregate_flows_to_windows,
    build_sequential_dataset,
    split_and_scale_windows,
)
from netforecast.models.baseline import BaselineKStepClassifier
from netforecast.models.world_model import NetworkWorldModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("train")


def set_seed(seed: int = 42):
    """Ensure reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_world_model(
    model: NetworkWorldModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: str = "cpu",
) -> NetworkWorldModel:
    """Train the multi-task PyTorch World Model."""
    model = model.to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["model"].get("learning_rate", 0.001),
        weight_decay=1e-5,
    )
    epochs = config["model"].get("epochs", 15)

    mse_loss_fn = nn.MSELoss()
    bce_loss_fn = nn.BCEWithLogitsLoss()
    ce_loss_fn = nn.CrossEntropyLoss()

    w_dyn = config["model"]["loss_weights"].get("dynamics", 1.0)
    w_inf = config["model"]["loss_weights"].get("infiltration", 2.0)
    w_stg = config["model"]["loss_weights"].get("stage", 1.5)

    best_val_loss = float("inf")
    best_weights = None

    logger.info("Starting World Model training for %d epochs on %s...", epochs, device)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_dyn_loss = 0.0
        train_inf_loss = 0.0
        train_stg_loss = 0.0

        for batch_x, batch_y_next_s, batch_y_inf, batch_y_stg in train_loader:
            batch_x = batch_x.to(device)
            batch_y_next_s = batch_y_next_s.to(device)
            batch_y_inf = batch_y_inf.to(device)
            batch_y_stg = batch_y_stg.to(device)

            optimizer.zero_grad()
            next_state_pred, infil_logits, stage_logits = model(batch_x)

            loss_dyn = mse_loss_fn(next_state_pred, batch_y_next_s)
            loss_inf = bce_loss_fn(infil_logits, batch_y_inf)
            loss_stg = ce_loss_fn(stage_logits, batch_y_stg)

            total_loss = (w_dyn * loss_dyn) + (w_inf * loss_inf) + (w_stg * loss_stg)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item() * len(batch_x)
            train_dyn_loss += loss_dyn.item() * len(batch_x)
            train_inf_loss += loss_inf.item() * len(batch_x)
            train_stg_loss += loss_stg.item() * len(batch_x)

        train_loss /= len(train_loader.dataset)
        train_dyn_loss /= len(train_loader.dataset)
        train_inf_loss /= len(train_loader.dataset)
        train_stg_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y_next_s, batch_y_inf, batch_y_stg in val_loader:
                batch_x = batch_x.to(device)
                batch_y_next_s = batch_y_next_s.to(device)
                batch_y_inf = batch_y_inf.to(device)
                batch_y_stg = batch_y_stg.to(device)

                next_s, inf_l, stg_l = model(batch_x)
                l_dyn = mse_loss_fn(next_s, batch_y_next_s)
                l_inf = bce_loss_fn(inf_l, batch_y_inf)
                l_stg = ce_loss_fn(stg_l, batch_y_stg)

                v_loss = (w_dyn * l_dyn) + (w_inf * l_inf) + (w_stg * l_stg)
                val_loss += v_loss.item() * len(batch_x)

        val_loss /= len(val_loader.dataset)

        logger.info(
            "Epoch [%02d/%02d] | Train Loss: %.4f (Dyn: %.4f, Inf: %.4f, Stg: %.4f) | Val Loss: %.4f",
            epoch,
            epochs,
            train_loss,
            train_dyn_loss,
            train_inf_loss,
            train_stg_loss,
            val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()

    if best_weights is not None:
        model.load_state_dict(best_weights)
        logger.info("Loaded best model weights with Val Loss: %.4f", best_val_loss)

    return model


def main():
    parser = argparse.ArgumentParser(description="Train NetForecast World Model and Baseline")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config.yaml"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic dataset for smoke-test mode ONLY (not for reporting)",
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config["project"].get("seed", 42)
    set_seed(seed)

    mapper = MitreMapper(config)
    window_sec = config["data"].get("time_window_seconds", 10.0)

    # 1. Load Data
    if args.synthetic:
        logger.warning("Running in SYNTHETIC smoke-test mode. Note: synthetic, not for reporting.")
        df_flows = generate_synthetic_flows(num_samples=1000, random_state=seed)
    else:
        raw_dir = config["data"]["raw_dir"]
        selected_files = config["data"]["selected_files"]
        benign_subsample = config["data"].get("benign_subsample_ratio", 0.1)

        try:
            df_flows = load_multiple_csvs(
                raw_dir,
                selected_files,
                benign_subsample=benign_subsample,
                random_state=seed,
            )
        except Exception as e:
            logger.error("Failed to load raw CSVs: %s", e)
            logger.error(
                "Please download the dataset using scripts/download_data.sh or place CSVs in %s",
                raw_dir,
            )
            return

    logger.info("Raw flow records loaded: %d", len(df_flows))

    # 2. Window Aggregation
    state_df = aggregate_flows_to_windows(
        df_flows, window_seconds=window_sec, mapper=mapper
    )
    logger.info("Constructed %d time-window state vectors S_t", len(state_df))

    # 3. Time-based Partitioning & Scaling
    train_ratio = config["training"].get("train_ratio", 0.70)
    val_ratio = config["training"].get("val_ratio", 0.15)

    train_df, val_df, test_df, scaler, feature_cols = split_and_scale_windows(
        state_df, train_ratio=train_ratio, val_ratio=val_ratio
    )
    logger.info("Train windows: %d | Val windows: %d | Test windows: %d", len(train_df), len(val_df), len(test_df))

    seq_len = config["model"].get("seq_len", 12)
    k_rollout = config["model"].get("k_rollout", 5)

    # 4. Sequential Datasets
    train_seq = build_sequential_dataset(train_df, seq_len=seq_len, k_rollout=k_rollout, feature_cols=feature_cols)
    val_seq = build_sequential_dataset(val_df, seq_len=seq_len, k_rollout=k_rollout, feature_cols=feature_cols)

    input_dim = len(feature_cols)
    config["model"]["input_dim"] = input_dim

    # 5. DataLoaders
    batch_size = config["model"].get("batch_size", 64)
    train_ds = TensorDataset(
        torch.tensor(train_seq["X"]),
        torch.tensor(train_seq["Y_next_state"]),
        torch.tensor(train_seq["Y_infil_next"]),
        torch.tensor(train_seq["Y_stage_next"]),
    )
    val_ds = TensorDataset(
        torch.tensor(val_seq["X"]),
        torch.tensor(val_seq["Y_next_state"]),
        torch.tensor(val_seq["Y_infil_next"]),
        torch.tensor(val_seq["Y_stage_next"]),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 6. Initialize and Train World Model
    device = config["model"].get("device", "cpu")
    model = NetworkWorldModel(
        input_dim=input_dim,
        hidden_dim=config["model"].get("hidden_dim", 128),
        num_layers=config["model"].get("num_layers", 2),
        num_stages=mapper.num_stages,
        dropout=config["model"].get("dropout", 0.2),
        model_type=config["model"].get("type", "lstm"),
    )

    trained_model = train_world_model(model, train_loader, val_loader, config, device=device)

    # 7. Train Baseline Logistic Regression
    logger.info("Training Logistic Regression Baseline...")
    baseline = BaselineKStepClassifier(random_state=seed)
    baseline.fit(train_seq["X"], train_seq["Y_infil_next"])

    # 8. Save Artifacts
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = config["paths"].get("model_save", "models/world_model.pt")
    scaler_path = config["paths"].get("scaler_save", "models/scaler.joblib")
    baseline_path = config["paths"].get("baseline_save", "models/baseline.joblib")

    # Save PyTorch Model
    torch.save(
        {
            "model_state_dict": trained_model.state_dict(),
            "input_dim": input_dim,
            "feature_cols": feature_cols,
            "config": config,
        },
        model_path,
    )
    logger.info("World model saved to: %s", model_path)

    # Save Scaler & Baseline
    joblib.dump({"scaler": scaler, "feature_cols": feature_cols}, scaler_path)
    baseline.save(baseline_path)
    logger.info("Scaler and baseline saved successfully.")


if __name__ == "__main__":
    main()
