"""Comprehensive Evaluation Pipeline for NetForecast and Baselines.

Generates rigorous metrics, generalization tests, confusion matrices, and plots.
Outputs to results/metrics.json and results/metrics.md.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from netforecast.attack.mitre_map import MitreMapper
from netforecast.data.load_csv import generate_synthetic_flows, load_multiple_csvs
from netforecast.features.windows import (
    aggregate_flows_to_windows,
    build_sequential_dataset,
    split_and_scale_windows,
)
from netforecast.models.baseline import BaselineKStepClassifier
from netforecast.models.rollout import RolloutSimulator
from netforecast.models.world_model import NetworkWorldModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("evaluate")


def compute_binary_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, Any]:
    """Compute detailed classification metrics for binary infiltration forecasting."""
    y_true_flat = y_true.ravel()
    y_pred = (y_prob >= threshold).astype(int)

    prec = precision_score(y_true_flat, y_pred, zero_division=0)
    rec = recall_score(y_true_flat, y_pred, zero_division=0)
    f1 = f1_score(y_true_flat, y_pred, zero_division=0)
    acc = accuracy_score(y_true_flat, y_pred)

    cm = confusion_matrix(y_true_flat, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    try:
        auc = (
            float(roc_auc_score(y_true_flat, y_prob))
            if len(np.unique(y_true_flat)) > 1
            else 0.5
        )
    except Exception:
        auc = 0.5

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "fpr": float(fpr),
        "roc_auc": float(auc),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def save_confusion_matrix_plot(
    cm: np.ndarray,
    labels: List[str],
    title: str,
    output_path: Path,
):
    """Save formatted confusion matrix heatmap plot."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_roc_curve_plot(
    y_true: np.ndarray,
    probs_dict: Dict[str, np.ndarray],
    output_path: Path,
):
    """Save overlaid ROC curve plot for comparison."""
    plt.figure(figsize=(7, 6))
    for name, probs in probs_dict.items():
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, probs)
            auc_val = roc_auc_score(y_true, probs)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random Chance")
    plt.title("ROC Curve - Infiltration Forecasting")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate NetForecast World Model vs Baseline")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config.yaml"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic dataset for smoke-test mode ONLY (not for reporting)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    mapper = MitreMapper(config)
    plots_dir = Path(config["paths"].get("plots_dir", "results/plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    if args.synthetic:
        logger.warning("Running evaluation on SYNTHETIC smoke-test dataset.")
        df_flows = generate_synthetic_flows(num_samples=1000, random_state=42)
    else:
        raw_dir = config["data"]["raw_dir"]
        selected_files = config["data"]["selected_files"]
        benign_subsample = config["data"].get("benign_subsample_ratio", 0.1)

        try:
            df_flows = load_multiple_csvs(
                raw_dir,
                selected_files,
                benign_subsample=benign_subsample,
                random_state=42,
            )
        except Exception as e:
            logger.error("Failed to load evaluation data: %s", e)
            return

    # 2. Window Aggregation & Partitioning
    window_sec = config["data"].get("time_window_seconds", 10.0)
    state_df = aggregate_flows_to_windows(
        df_flows, window_seconds=window_sec, mapper=mapper
    )

    train_ratio = config["training"].get("train_ratio", 0.70)
    val_ratio = config["training"].get("val_ratio", 0.15)
    train_df, val_df, test_df, scaler, feature_cols = split_and_scale_windows(
        state_df, train_ratio=train_ratio, val_ratio=val_ratio
    )

    seq_len = config["model"].get("seq_len", 12)
    k_rollout = config["model"].get("k_rollout", 5)

    val_seq = build_sequential_dataset(
        val_df, seq_len=seq_len, k_rollout=k_rollout, feature_cols=feature_cols
    )
    test_seq = build_sequential_dataset(
        test_df, seq_len=seq_len, k_rollout=k_rollout, feature_cols=feature_cols
    )

    # 3. Load Trained Models
    model_path = config["paths"].get("model_save", "models/world_model.pt")
    baseline_path = config["paths"].get("baseline_save", "models/baseline.joblib")

    device = config["model"].get("device", "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    model = NetworkWorldModel(
        input_dim=len(feature_cols),
        hidden_dim=config["model"].get("hidden_dim", 128),
        num_layers=config["model"].get("num_layers", 2),
        num_stages=mapper.num_stages,
        dropout=config["model"].get("dropout", 0.2),
        model_type=config["model"].get("type", "lstm"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    baseline = BaselineKStepClassifier()
    baseline.load(baseline_path)

    # 4. Tune Thresholds on Validation Split
    x_val_tensor = torch.tensor(val_seq["X"], dtype=torch.float32).to(device)
    with torch.no_grad():
        _, val_inf_logits, _ = model(x_val_tensor)
        wm_val_probs = torch.sigmoid(val_inf_logits).squeeze(-1).cpu().numpy()

    y_val_inf = val_seq["Y_infil_next"].ravel()

    # World Model threshold tuning
    best_wm_thresh = 0.5
    best_wm_f1 = -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f1_t = f1_score(y_val_inf, (wm_val_probs >= t).astype(int), zero_division=0)
        if f1_t > best_wm_f1:
            best_wm_f1 = f1_t
            best_wm_thresh = float(t)

    # Baseline threshold tuning
    best_base_thresh = baseline.tune_threshold(val_seq["X"], y_val_inf)
    logger.info("Tuned validation thresholds -> World Model: %.4f (Val F1: %.4f), Baseline: %.4f", best_wm_thresh, best_wm_f1, best_base_thresh)

    # 5. Infiltration Step-1 Predictions on Test Split
    x_test_tensor = torch.tensor(test_seq["X"], dtype=torch.float32).to(device)
    with torch.no_grad():
        next_s_pred, inf_logits, stg_logits = model(x_test_tensor)
        wm_probs = torch.sigmoid(inf_logits).squeeze(-1).cpu().numpy()
        wm_stage_probs = F.softmax(stg_logits, dim=-1).cpu().numpy()
        wm_stage_preds = np.argmax(wm_stage_probs, axis=-1)

    y_test_inf = test_seq["Y_infil_next"].ravel()
    y_test_stg = test_seq["Y_stage_next"].ravel()

    wm_metrics = compute_binary_metrics(y_test_inf, wm_probs, threshold=best_wm_thresh)
    wm_metrics["tuned_threshold"] = best_wm_thresh

    base_metrics = baseline.evaluate(test_seq["X"], y_test_inf, threshold=best_base_thresh)
    base_metrics["tuned_threshold"] = best_base_thresh

    # 6. K-Step Rollout Simulation Evaluation
    simulator = RolloutSimulator(model=model, mapper=mapper, device=device)
    rollout_results = simulator.rollout_batch(test_seq["X"], k_steps=k_rollout)

    rollout_step_metrics = {}
    for step_k in range(k_rollout):
        step_prob = rollout_results["infil_probabilities"][:, step_k]
        step_true = test_seq["Y_infil_rollout"][:, step_k]
        rollout_step_metrics[f"step_{step_k+1}"] = compute_binary_metrics(
            step_true, step_prob, threshold=best_wm_thresh
        )

    # 7. Generalization / Holdout Attack Test
    holdout_attack_name = config["data"].get("holdout_attack", "Infilteration")
    holdout_mask = (test_df["dominant_label"] == holdout_attack_name).values
    holdout_windows = test_df[holdout_mask]

    gen_metrics = {"holdout_attack": holdout_attack_name, "sample_count": len(holdout_windows)}
    if len(holdout_windows) >= (seq_len + k_rollout):
        holdout_seq = build_sequential_dataset(
            holdout_windows, seq_len=seq_len, k_rollout=k_rollout, feature_cols=feature_cols
        )
        h_x_tensor = torch.tensor(holdout_seq["X"], dtype=torch.float32).to(device)
        with torch.no_grad():
            _, h_inf_logits, _ = model(h_x_tensor)
            h_probs = torch.sigmoid(h_inf_logits).squeeze(-1).cpu().numpy()
        
        h_y_true = holdout_seq["Y_infil_next"].ravel()
        num_benign = int(np.sum(h_y_true == 0))
        num_attack = int(np.sum(h_y_true == 1))

        wm_gen_eval = compute_binary_metrics(h_y_true, h_probs, threshold=best_wm_thresh)
        base_gen_eval = baseline.evaluate(holdout_seq["X"], h_y_true, threshold=best_base_thresh)

        gen_metrics["benign_samples"] = num_benign
        gen_metrics["attack_samples"] = num_attack
        gen_metrics["world_model"] = wm_gen_eval
        gen_metrics["baseline"] = base_gen_eval
        
        if num_benign == 0:
            gen_metrics["fpr_explanation"] = (
                "FPR is 0.0000 because the held-out attack slice comprises exclusively attack windows (0 benign samples, TN=0, FP=0), making false positive count strictly 0."
            )
        else:
            gen_metrics["fpr_explanation"] = (
                f"Evaluated on {num_benign} benign windows and {num_attack} attack windows (FP={wm_gen_eval['fp']}, TN={wm_gen_eval['tn']})."
            )
    else:
        gen_metrics["note"] = f"Insufficient continuous holdout windows ({len(holdout_windows)}) for sequence rollout."

    # 8. Generate Evaluation Plots
    cm_wm = confusion_matrix(y_test_inf, (wm_probs >= best_wm_thresh).astype(int), labels=[0, 1])
    save_confusion_matrix_plot(
        cm_wm,
        labels=["Benign", "Infiltration"],
        title=f"World Model Confusion Matrix (Thresh={best_wm_thresh:.2f})",
        output_path=plots_dir / "cm_world_model.png",
    )

    cm_base = confusion_matrix(
        y_test_inf, (baseline.predict_proba(test_seq["X"]) >= best_base_thresh).astype(int), labels=[0, 1]
    )
    save_confusion_matrix_plot(
        cm_base,
        labels=["Benign", "Infiltration"],
        title=f"Baseline Confusion Matrix (Thresh={best_base_thresh:.2f})",
        output_path=plots_dir / "cm_baseline.png",
    )

    save_roc_curve_plot(
        y_test_inf,
        {"World Model": wm_probs, "Baseline (Logistic Regression)": baseline.predict_proba(test_seq["X"])},
        output_path=plots_dir / "roc_curve.png",
    )

    # 9. Save Metrics JSON & Markdown
    all_metrics = {
        "is_synthetic": bool(args.synthetic),
        "num_test_sequences": len(test_seq["X"]),
        "time_window_seconds": window_sec,
        "k_rollout": k_rollout,
        "world_model_step1": wm_metrics,
        "baseline_step1": base_metrics,
        "rollout_multi_step": rollout_step_metrics,
        "generalization_holdout_test": gen_metrics,
    }

    metrics_json_path = Path(config["paths"].get("metrics_json", "results/metrics.json"))
    metrics_md_path = Path(config["paths"].get("metrics_md", "results/metrics.md"))

    metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    # Build Markdown Summary Table
    md_content = f"""# NetForecast Evaluation Results

> **Data Source**: {"SYNTHETIC (smoke-test only, not for reporting)" if args.synthetic else "CIC-IDS-2018 Cleaned Telemetry"}  
> **Evaluation Split**: Strict Time-Based Partition ({len(test_seq['X'])} test sequences)  
> **Validation Threshold Tuning**: World Model = {best_wm_thresh:.4f}, Baseline = {best_base_thresh:.4f}  
> **Forecast Horizon**: K = {k_rollout} windows ({k_rollout * window_sec}s forward)

## Step t+1 Forecasting Performance

| Model | Decision Threshold | F1-Score | Precision | Recall | False Positive Rate (FPR) | ROC-AUC | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **World Model (Ours)** | **{best_wm_thresh:.2f}** | **{wm_metrics['f1']:.4f}** | **{wm_metrics['precision']:.4f}** | **{wm_metrics['recall']:.4f}** | **{wm_metrics['fpr']:.4f}** | **{wm_metrics['roc_auc']:.4f}** | **{wm_metrics['accuracy']:.4f}** |
| **Baseline (Logistic Reg.)** | {best_base_thresh:.2f} | {base_metrics['f1']:.4f} | {base_metrics['precision']:.4f} | {base_metrics['recall']:.4f} | {base_metrics['fpr']:.4f} | {base_metrics['roc_auc']:.4f} | {base_metrics['accuracy']:.4f} |

## Autoregressive Rollout Degradation (K-Step Ahead)

| Forecast Horizon | F1-Score | Precision | Recall | FPR | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for step_k in range(k_rollout):
        m = rollout_step_metrics[f"step_{step_k+1}"]
        step_sec = (step_k + 1) * window_sec
        md_content += f"| **t+{step_k+1} ({step_sec:.0f}s)** | {m['f1']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['fpr']:.4f} | {m['roc_auc']:.4f} |\n"

    md_content += f"""
## Zero-Shot Attack Generalization Test (Holdout: {holdout_attack_name})
- **Sample Composition**: {gen_metrics.get('attack_samples', 0)} attack windows, {gen_metrics.get('benign_samples', 0)} benign windows (Total: {gen_metrics.get('sample_count', 0)})
- **FPR Explanation**: {gen_metrics.get('fpr_explanation', 'N/A')}
"""
    if "world_model" in gen_metrics:
        gm = gen_metrics["world_model"]
        md_content += f"- **World Model Performance on Unseen Attack**: F1 = **{gm['f1']:.4f}** (Precision = {gm['precision']:.4f}, Recall = {gm['recall']:.4f}, FPR = {gm['fpr']:.4f})\n"

    with open(metrics_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Evaluation metrics successfully written to %s and %s", metrics_json_path, metrics_md_path)


if __name__ == "__main__":
    main()
