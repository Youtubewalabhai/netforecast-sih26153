"""Model Explainability Module (SHAP + Permutation Importance Fallback).

Provides local and global feature attributions for infiltration risk predictions.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from netforecast.models.world_model import NetworkWorldModel

logger = logging.getLogger(__name__)


class SequenceExplainer:
    """Explains PyTorch World Model infiltration forecasting using SHAP with permutation fallback."""

    def __init__(
        self,
        model: NetworkWorldModel,
        feature_names: List[str],
        background_data: Optional[np.ndarray] = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.feature_names = feature_names
        self.background_data = background_data
        self.device = device
        self.model.eval()

    def _model_prediction_fn(self, flat_x: np.ndarray, seq_len: int, num_feats: int) -> np.ndarray:
        """Wrapper to evaluate flattened sequences."""
        N = flat_x.shape[0]
        x_reshaped = flat_x.reshape(N, seq_len, num_feats)
        tensor_x = torch.tensor(x_reshaped, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            _, infil_logit, _ = self.model(tensor_x)
            infil_prob = torch.sigmoid(infil_logit).squeeze(-1).cpu().numpy()

        return infil_prob

    def explain_instance_shap(
        self,
        input_seq: np.ndarray,
        num_background: int = 50,
        nsamples: int = 100,
    ) -> Dict[str, Any]:
        """Explain single sequence using SHAP KernelExplainer.

        Falls back to permutation importance if SHAP fails or is too slow.
        """
        try:
            import shap
        except ImportError:
            logger.warning("SHAP library not found, falling back to Permutation Feature Importance.")
            return self.explain_instance_permutation(input_seq)

        seq_len, num_feats = input_seq.shape
        flat_input = input_seq.reshape(1, seq_len * num_feats)

        # Prepare background data
        if self.background_data is not None and len(self.background_data) > 0:
            bg_samples = self.background_data[:num_background]
            if bg_samples.ndim == 3:
                bg_flat = bg_samples.reshape(len(bg_samples), seq_len * num_feats)
            else:
                bg_flat = bg_samples
        else:
            bg_flat = np.zeros((10, seq_len * num_feats))

        try:
            predict_fn = lambda x: self._model_prediction_fn(x, seq_len, num_feats)
            explainer = shap.KernelExplainer(predict_fn, bg_flat)
            shap_values = explainer.shap_values(flat_input, nsamples=nsamples)

            if isinstance(shap_values, list):
                sv = np.array(shap_values[0]).flatten()
            else:
                sv = np.array(shap_values).flatten()

            # Aggregate feature contributions across sequence steps
            sv_reshaped = sv.reshape(seq_len, num_feats)
            # Sum absolute SHAP values across time steps to rank features
            feature_impacts = np.sum(np.abs(sv_reshaped), axis=0)

            # Sort top features
            ranked_indices = np.argsort(feature_impacts)[::-1]
            top_features = []
            for idx in ranked_indices:
                name = self.feature_names[idx] if idx < len(self.feature_names) else f"Feature_{idx}"
                top_features.append(
                    {
                        "feature": name,
                        "importance": float(feature_impacts[idx]),
                        "mean_signed_shap": float(np.mean(sv_reshaped[:, idx])),
                    }
                )

            return {
                "method": "SHAP (KernelExplainer)",
                "top_features": top_features,
                "shap_matrix": sv_reshaped,
            }
        except Exception as e:
            logger.warning(
                "SHAP explanation failed (%s). Falling back to Permutation Importance.", e
            )
            return self.explain_instance_permutation(input_seq)

    def explain_instance_permutation(
        self,
        input_seq: np.ndarray,
        num_permutations: int = 30,
    ) -> Dict[str, Any]:
        """Feature importance via localized feature noise / masking permutation."""
        seq_len, num_feats = input_seq.shape
        tensor_orig = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, base_logit, _ = self.model(tensor_orig)
            base_prob = float(torch.sigmoid(base_logit).item())

        feature_importances = []

        for feat_idx in range(num_feats):
            drops = []
            for _ in range(num_permutations):
                perturbed = input_seq.copy()
                # Zero out or inject Gaussian noise into the specific feature channel
                perturbed[:, feat_idx] = np.random.normal(0, 1, size=seq_len)
                perturbed_tensor = torch.tensor(perturbed, dtype=torch.float32).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    _, p_logit, _ = self.model(perturbed_tensor)
                    p_prob = float(torch.sigmoid(p_logit).item())

                diff = abs(base_prob - p_prob)
                drops.append(diff)

            mean_importance = float(np.mean(drops))
            feat_name = self.feature_names[feat_idx] if feat_idx < len(self.feature_names) else f"Feature_{feat_idx}"
            feature_importances.append({"feature": feat_name, "importance": mean_importance})

        feature_importances.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "method": "Permutation Feature Importance (Fallback)",
            "top_features": feature_importances,
            "base_probability": base_prob,
        }
