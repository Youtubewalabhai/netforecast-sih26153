"""Unit tests for World Model forward pass, loss heads, and autoregressive rollout."""

import numpy as np
import pytest
import torch

from netforecast.attack.mitre_map import MitreMapper
from netforecast.models.rollout import RolloutSimulator
from netforecast.models.world_model import NetworkWorldModel


def test_world_model_forward_shapes():
    """Verify multi-task output shapes for single batch."""
    batch_size = 8
    seq_len = 10
    input_dim = 15
    num_stages = 6

    model = NetworkWorldModel(
        input_dim=input_dim, hidden_dim=32, num_layers=1, num_stages=num_stages
    )
    x = torch.randn(batch_size, seq_len, input_dim)

    next_state, infil_logits, stage_logits = model(x)

    assert next_state.shape == (batch_size, input_dim)
    assert infil_logits.shape == (batch_size, 1)
    assert stage_logits.shape == (batch_size, num_stages)


def test_autoregressive_rollout_shapes():
    """Verify K-step autoregressive rollout produces matching temporal dimensions."""
    seq_len = 8
    input_dim = 12
    k_steps = 4
    num_stages = 6

    model = NetworkWorldModel(
        input_dim=input_dim, hidden_dim=32, num_layers=1, num_stages=num_stages
    )
    mapper = MitreMapper()
    simulator = RolloutSimulator(model=model, mapper=mapper, device="cpu")

    # Single sequence test
    single_seq = np.random.randn(seq_len, input_dim).astype(np.float32)
    single_res = simulator.rollout_single_sequence(single_seq, k_steps=k_steps)

    assert len(single_res["infil_probabilities"]) == k_steps
    assert len(single_res["predicted_stage_ids"]) == k_steps
    assert single_res["stage_probabilities"].shape == (k_steps, num_stages)
    assert single_res["future_states"].shape == (k_steps, input_dim)

    # Batch rollout test
    batch_seq = np.random.randn(5, seq_len, input_dim).astype(np.float32)
    batch_res = simulator.rollout_batch(batch_seq, k_steps=k_steps)

    assert batch_res["infil_probabilities"].shape == (5, k_steps)
    assert batch_res["predicted_stage_ids"].shape == (5, k_steps)
    assert batch_res["stage_probabilities"].shape == (5, k_steps, num_stages)
    assert batch_res["future_states"].shape == (5, k_steps, input_dim)
