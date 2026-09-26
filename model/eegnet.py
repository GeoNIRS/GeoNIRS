
from __future__ import annotations

import torch
from torch import nn

from .classification_head import LinearClassificationHead
from .event_aggregation import aggregate_event_pairs


class EEGNetAdaptedClassifier(nn.Module):
    def __init__(
        self,
        physical_channels: int = 70,
        time_points: int = 60,
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int = 16,
        temporal_kernel: int = 5,
        separable_kernel: int = 5,
        dropout: float = 0.5,
        num_classes: int = 2,
        event_pair_mode: bool = True,
    ) -> None:
        super().__init__()
        dimensions = (
            physical_channels,
            time_points,
            f1,
            depth_multiplier,
            f2,
            temporal_kernel,
            separable_kernel,
        )
        if min(dimensions) <= 0:
            raise ValueError("EEGNet dimensions must be positive")
        if temporal_kernel % 2 == 0 or separable_kernel % 2 == 0:
            raise ValueError("temporal kernels must be odd for same padding")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

        self.physical_channels = int(physical_channels)
        self.time_points = int(time_points)
        self.f1 = int(f1)
        self.depth_multiplier = int(depth_multiplier)
        self.f2 = int(f2)
        self.temporal_kernel = int(temporal_kernel)
        self.separable_kernel = int(separable_kernel)
        self.dropout = float(dropout)
        self.num_classes = int(num_classes)
        self.event_pair_mode = bool(event_pair_mode)

        spatial_filters = self.f1 * self.depth_multiplier
        self.temporal = nn.Sequential(
            nn.Conv2d(
                2,
                self.f1,
                kernel_size=(1, self.temporal_kernel),
                padding=(0, self.temporal_kernel // 2),
                bias=False,
            ),
            nn.BatchNorm2d(self.f1),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(
                self.f1,
                spatial_filters,
                kernel_size=(self.physical_channels, 1),
                groups=self.f1,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2)),
            nn.Dropout(self.dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(
                spatial_filters,
                spatial_filters,
                kernel_size=(1, self.separable_kernel),
                padding=(0, self.separable_kernel // 2),
                groups=spatial_filters,
                bias=False,
            ),
            nn.Conv2d(spatial_filters, self.f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(self.dropout),
            nn.Flatten(),
        )
        pooled_time = self.time_points // 2 // 4
        if pooled_time <= 0:
            raise ValueError("time_points are too short for EEGNet pooling")
        self.event_feature_dim = self.f2 * pooled_time


        self.paired_feature_dim = (
            2 * self.event_feature_dim if self.event_pair_mode
            else self.event_feature_dim
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
        if channel_count > self.physical_channels:
            raise ValueError(
                f"received {channel_count} channels, expected at most "
                f"{self.physical_channels}"
            )
        hbo = hbo.masked_fill(~valid, 0.0)
        hbr = hbr.masked_fill(~valid, 0.0)
        if channel_count < self.physical_channels:
            padding = self.physical_channels - channel_count
            hbo = torch.nn.functional.pad(hbo, (0, 0, 0, padding))
            hbr = torch.nn.functional.pad(hbr, (0, 0, 0, padding))
        signal = torch.stack((hbo, hbr), dim=1)
        features = self.separable(self.spatial(self.temporal(signal)))
        if features.shape[-1] != self.event_feature_dim:
            raise RuntimeError(
                f"EEGNet produced {features.shape[-1]} features, expected "
                f"{self.event_feature_dim}"
            )
        return features

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
