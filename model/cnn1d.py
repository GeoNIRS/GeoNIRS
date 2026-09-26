
from __future__ import annotations

import torch
from torch import nn

from .classification_head import LinearClassificationHead
from .event_aggregation import aggregate_event_pairs


class DirectEventCNN1DClassifier(nn.Module):
   
    def __init__(
        self,
        physical_channels: int = 70,
        time_points: int = 60,
        conv_dim: int = 64,
        embedding_dim: int = 128,
        dropout: float = 0.5,
        num_classes: int = 2,
        event_pair_mode: bool = True,
    ) -> None:
        super().__init__()
        if min(physical_channels, time_points, conv_dim, embedding_dim) <= 0:
            raise ValueError("CNN dimensions must be positive")
        if conv_dim % 8 or embedding_dim % 8:
            raise ValueError("conv and embedding dimensions must be divisible by 8")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        self.physical_channels = int(physical_channels)
        self.time_points = int(time_points)
        self.conv_dim = int(conv_dim)
        self.embedding_dim = int(embedding_dim)
        self.event_feature_dim = self.embedding_dim
        self.event_pair_mode = bool(event_pair_mode)
        self.num_classes = int(num_classes)


        self.paired_feature_dim = (
            2 * self.event_feature_dim if self.event_pair_mode
            else self.event_feature_dim
        )
        self.dropout = float(dropout)
        input_channels = 2 * self.physical_channels
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, self.conv_dim, kernel_size=7, padding=3),
            nn.GroupNorm(8, self.conv_dim),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                self.conv_dim, self.embedding_dim, kernel_size=5, padding=2
            ),
            nn.GroupNorm(8, self.embedding_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(self.dropout),
        )
        self.linear = LinearClassificationHead(
            self.paired_feature_dim,
            2 if self.event_pair_mode else self.num_classes,
        )

    def event_features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        hbo = batch["hbo"].float()
        hbr = batch["hbr"].float()
        valid = batch["valid"].bool()
        if hbo.shape != hbr.shape or hbo.shape != valid.shape:
            raise ValueError("HbO, HbR, and valid masks must have equal shapes")
        if hbo.ndim != 3 or hbo.shape[-1] != self.time_points:
            raise ValueError(
                f"expected event signals [E,C,{self.time_points}], got {hbo.shape}"
            )
        channel_count = int(hbo.shape[1])
        if channel_count != self.physical_channels:
            raise ValueError(
                "Fair Input channel count must equal the declared physical_channels "
                f"({self.physical_channels}); channel padding/selection is not allowed "
                "inside the model"
            )
        hbo = hbo.masked_fill(~valid, 0.0)
        hbr = hbr.masked_fill(~valid, 0.0)
        signal = torch.stack((hbo, hbr), dim=1).flatten(start_dim=1, end_dim=2)
        return self.encoder(signal)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        events = self.event_features(batch)

        if not self.event_pair_mode:
            return self.linear(events)
        paired = aggregate_event_pairs(
            events,
            batch["event_run_index"],
            batch["event_condition"],
            int(batch["label"].numel()),
        )
        return self.linear(paired)
