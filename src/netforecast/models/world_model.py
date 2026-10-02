"""PyTorch Network Dynamics World Model Architecture.

Learns state-transition dynamics P(S_{t+1} | S_t, ..., S_{t-W}) and predicts
infiltration probability and MITRE ATT&CK stage progression.
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetworkWorldModel(nn.Module):
    """LSTM / Transformer World Model for Network State Dynamics."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_stages: int = 6,
        dropout: float = 0.2,
        model_type: str = "lstm",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_stages = num_stages
        self.model_type = model_type.lower()

        # Backbone temporal encoder
        if self.model_type == "lstm":
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif self.model_type == "transformer":
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.latent_norm = nn.LayerNorm(hidden_dim)

        # Head 1: Dynamics / Next-State Prediction (MSE loss)
        self.next_state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

        # Head 2: Infiltration Probability (BCE loss)
        self.infil_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Head 3: MITRE ATT&CK Stage Classification (CrossEntropy loss)
        self.stage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_stages),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through world model.

        Args:
            x: Input sequence of state vectors, shape (batch_size, seq_len, input_dim)

        Returns:
            Tuple of:
                - next_state_pred: shape (batch_size, input_dim)
                - infil_logits: shape (batch_size, 1)
                - stage_logits: shape (batch_size, num_stages)
        """
        if self.model_type == "lstm":
            out, (h_n, _) = self.encoder(x)
            # Use the final hidden state of the top layer
            latent = h_n[-1]  # shape (batch_size, hidden_dim)
        else:
            proj = self.input_proj(x)
            encoded = self.encoder(proj)
            # Mean pooling across sequence dimension
            latent = torch.mean(encoded, dim=1)

        latent = self.latent_norm(latent)

        next_state_pred = self.next_state_head(latent)
        infil_logits = self.infil_head(latent)
        stage_logits = self.stage_head(latent)

        return next_state_pred, infil_logits, stage_logits

    def predict_step(
        self, x: torch.Tensor
    ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """Run single evaluation step returning probabilities and predictions."""
        self.eval()
        with torch.no_grad():
            next_state, infil_logit, stage_logit = self.forward(x)
            infil_prob = torch.sigmoid(infil_logit)
            stage_prob = F.softmax(stage_logit, dim=-1)
            stage_pred = torch.argmax(stage_prob, dim=-1)

        return {
            "next_state": next_state,
            "infil_prob": infil_prob,
            "stage_prob": stage_prob,
            "stage_pred": stage_pred,
        }
