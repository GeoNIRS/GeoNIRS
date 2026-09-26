
from __future__ import annotations

import torch
from torch import nn

from .classification_head import HRFCropMixin, LinearClassificationHead
from .event_aggregation import aggregate_event_pairs


class DWSConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size) -> None:
        super().__init__()
        self.depth_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, groups=in_channels
        )
        self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.point_conv(self.depth_conv(value))


class FNIRSNet(nn.Module):
   
    def __init__(
        self,
        num_class: int = 2,
        time_points: int = 85,
        physical_channels: int = 70,
        num_dhr_filters: int = 4,
        num_global_filters: int = 8,
    ) -> None:
        super().__init__()
        self.input_channels = int(physical_channels)
        self.time_points = int(time_points)
        self.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=num_dhr_filters,
            kernel_size=(1, self.time_points),
        )
        self.bn1 = nn.BatchNorm2d(num_dhr_filters)
        self.conv2 = DWSConv(
            in_channels=num_dhr_filters,
            out_channels=num_global_filters,
            kernel_size=(self.input_channels, 1),
        )
        self.bn2 = nn.BatchNorm2d(num_global_filters)
        self.act = nn.Sigmoid()
        self.fc = nn.Linear(num_global_filters, num_class)

    @property
    def feature_dim(self) -> int:
        return int(self.fc.in_features)

    def forward_features(self, value: torch.Tensor) -> torch.Tensor:
        expected = (2, self.input_channels, self.time_points)
        if value.ndim != 4 or tuple(value.shape[1:]) != expected:
            raise ValueError(f"expected [B,{expected}], got {tuple(value.shape)}")
        value = self.act(self.bn1(self.conv1(value)))
        value = self.act(self.bn2(self.conv2(value)))
        return value.flatten(start_dim=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.fc(self.forward_features(value))


class FNIRSNetLinearClassifier(nn.Module, HRFCropMixin):
   
    def __init__(
        self,
        backbone: FNIRSNet,
        sampling_rate_hz: float = 5.0,
        pre_onset_seconds: float = 5.0,
        hrf_start_seconds: float = -2.0,
        hrf_end_seconds: float = 15.0,
        freeze_batch_norm_stats: bool = True,
        num_classes: int = 2,
        event_pair_mode: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.event_pair_mode = bool(event_pair_mode)
        self.num_classes = int(num_classes)
        self.freeze_batch_norm_stats = bool(freeze_batch_norm_stats)
        self._set_hrf_crop(
            sampling_rate_hz,
            pre_onset_seconds,
            hrf_start_seconds,
            hrf_end_seconds,
        )
        self.backbone.fc.requires_grad_(False)
        self.event_feature_dim = int(backbone.feature_dim)
        self.paired_feature_dim = (
            2 * self.event_feature_dim if self.event_pair_mode
            else self.event_feature_dim
        )
        self.linear = LinearClassificationHead(
            self.paired_feature_dim,
            2 if self.event_pair_mode else self.num_classes,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_batch_norm_stats:
            self.backbone.bn1.eval()
            self.backbone.bn2.eval()
        return self

    def event_features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:


        cropped = self.crop_batch(batch) if self.event_pair_mode else batch
        hbo = cropped["hbo"].float()
        hbr = cropped["hbr"].float()
        expected_channels = self.backbone.input_channels
        if hbo.shape[1] != expected_channels:
            raise ValueError(
                "Fair Input channel count must equal the declared physical_channels "
                f"({expected_channels}); channel padding/selection is not allowed "
                "inside the model"
            )
        return self.backbone.forward_features(torch.stack([hbo, hbr], dim=1))

    def feature_parameters(self):
        return [
            parameter
            for name, parameter in self.backbone.named_parameters()
            if not name.startswith("fc.") and parameter.requires_grad
        ]

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.event_features(batch)
        if not self.event_pair_mode:
            return self.linear(features)
        paired = aggregate_event_pairs(
            features,
            batch["event_run_index"],
            batch["event_condition"],
            int(batch["label"].numel()),
        )
        return self.linear(paired)
