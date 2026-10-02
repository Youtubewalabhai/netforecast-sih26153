"""Autoregressive K-Step Forward Simulation (Roll-Out) Module."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from netforecast.attack.mitre_map import MitreMapper
from netforecast.models.world_model import NetworkWorldModel


class RolloutSimulator:
    """Simulates future network dynamics autoregressively using a trained World Model."""

    def __init__(
        self,
        model: NetworkWorldModel,
        mapper: Optional[MitreMapper] = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.mapper = mapper if mapper is not None else MitreMapper()
        self.device = device
        self.model.eval()

    def rollout_single_sequence(
        self,
        input_seq: Union[np.ndarray, torch.Tensor],
        k_steps: int = 5,
    ) -> Dict[str, Any]:
        """Roll forward a single history sequence for K steps.

        Args:
            input_seq: Array of shape (seq_len, num_features) or (1, seq_len, num_features).
            k_steps: Number of forward simulation steps K.

        Returns:
            Dictionary containing:
                - 'infil_probabilities': List of K infiltration probabilities
                - 'predicted_stage_ids': List of K predicted MITRE stage IDs
                - 'predicted_stage_names': List of K predicted MITRE stage names
                - 'stage_probabilities': Array of shape (K, num_stages)
                - 'future_states': Array of shape (K, num_features)
        """
        if isinstance(input_seq, np.ndarray):
            curr_tensor = torch.tensor(input_seq, dtype=torch.float32)
        else:
            curr_tensor = input_seq.clone().detach().to(torch.float32)

        if curr_tensor.ndim == 2:
            curr_tensor = curr_tensor.unsqueeze(0)  # shape: (1, seq_len, num_features)

        curr_tensor = curr_tensor.to(self.device)

        infil_probs = []
        stage_ids = []
        stage_names = []
        stage_probs_list = []
        future_states = []

        with torch.no_grad():
            for step in range(k_steps):
                next_state_pred, infil_logit, stage_logit = self.model(curr_tensor)

                infil_p = torch.sigmoid(infil_logit).item()
                stage_p = F.softmax(stage_logit, dim=-1).squeeze(0).cpu().numpy()
                stage_id = int(np.argmax(stage_p))
                stage_name = self.mapper.get_stage_name(stage_id)

                infil_probs.append(infil_p)
                stage_ids.append(stage_id)
                stage_names.append(stage_name)
                stage_probs_list.append(stage_p)
                future_states.append(next_state_pred.squeeze(0).cpu().numpy())

                # Autoregressive update: drop oldest window, append predicted state
                # curr_tensor: (1, seq_len, feat_dim)
                next_state_reshaped = next_state_pred.unsqueeze(1)  # (1, 1, feat_dim)
                curr_tensor = torch.cat([curr_tensor[:, 1:, :], next_state_reshaped], dim=1)

        return {
            "infil_probabilities": infil_probs,
            "predicted_stage_ids": stage_ids,
            "predicted_stage_names": stage_names,
            "stage_probabilities": np.array(stage_probs_list),
            "future_states": np.array(future_states),
        }

    def rollout_batch(
        self,
        batch_seqs: Union[np.ndarray, torch.Tensor],
        k_steps: int = 5,
    ) -> Dict[str, np.ndarray]:
        """Roll forward a batch of history sequences.

        Args:
            batch_seqs: Array of shape (N, seq_len, num_features).
            k_steps: Number of forward simulation steps K.

        Returns:
            Dictionary containing:
                - 'infil_probabilities': Array of shape (N, K)
                - 'predicted_stage_ids': Array of shape (N, K)
                - 'stage_probabilities': Array of shape (N, K, num_stages)
                - 'future_states': Array of shape (N, K, num_features)
        """
        if isinstance(batch_seqs, np.ndarray):
            curr_tensor = torch.tensor(batch_seqs, dtype=torch.float32)
        else:
            curr_tensor = batch_seqs.clone().detach().to(torch.float32)

        curr_tensor = curr_tensor.to(self.device)
        N = curr_tensor.shape[0]

        all_infil_probs = []
        all_stage_ids = []
        all_stage_probs = []
        all_future_states = []

        with torch.no_grad():
            for step in range(k_steps):
                next_state_pred, infil_logit, stage_logit = self.model(curr_tensor)

                infil_p = torch.sigmoid(infil_logit).squeeze(-1).cpu().numpy()  # shape (N,)
                stage_p = F.softmax(stage_logit, dim=-1).cpu().numpy()  # shape (N, num_stages)
                stage_pred = np.argmax(stage_p, axis=-1)  # shape (N,)

                all_infil_probs.append(infil_p)
                all_stage_ids.append(stage_pred)
                all_stage_probs.append(stage_p)
                all_future_states.append(next_state_pred.cpu().numpy())

                # Autoregressive slide
                next_state_reshaped = next_state_pred.unsqueeze(1)
                curr_tensor = torch.cat([curr_tensor[:, 1:, :], next_state_reshaped], dim=1)

        # Transpose lists from (K, N, ...) to (N, K, ...)
        res_infil = np.array(all_infil_probs).T  # (N, K)
        res_stages = np.array(all_stage_ids).T  # (N, K)
        res_stage_probs = np.transpose(np.array(all_stage_probs), (1, 0, 2))  # (N, K, num_stages)
        res_states = np.transpose(np.array(all_future_states), (1, 0, 2))  # (N, K, feat_dim)

        return {
            "infil_probabilities": res_infil,
            "predicted_stage_ids": res_stages,
            "stage_probabilities": res_stage_probs,
            "future_states": res_states,
        }
