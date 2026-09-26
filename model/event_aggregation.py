
from __future__ import annotations

import torch


def aggregate_event_pairs(
    event_features: torch.Tensor,
    event_run_index: torch.Tensor,
    event_condition: torch.Tensor,
    run_count: int,
) -> torch.Tensor:
   
    if event_features.shape[0] != event_run_index.numel():
        raise ValueError("event feature count does not match event_run_index")
    if event_condition.numel() != event_run_index.numel():
        raise ValueError("event_condition count does not match")
    if run_count <= 0:
        raise ValueError("run_count must be positive")
    if not torch.all((event_condition == 0) | (event_condition == 1)):
        raise ValueError("event_condition must be 0=flash_l or 1=flash_r")
    group = event_run_index.long() * 2 + event_condition.long()
    sums = event_features.new_zeros(run_count * 2, event_features.shape[-1])
    counts = event_features.new_zeros(run_count * 2, 1)
    sums.index_add_(0, group, event_features)
    counts.index_add_(
        0,
        group,
        torch.ones_like(group, dtype=event_features.dtype).unsqueeze(-1),
    )
    if (counts == 0).any():
        raise RuntimeError("a run is missing flash_l or flash_r")
    means = (sums / counts).reshape(run_count, 2, -1)
    return torch.cat([means[:, 0], means[:, 1]], dim=-1)
