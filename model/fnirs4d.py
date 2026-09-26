
from __future__ import annotations

import math

import torch
from torch import nn

from .classification_head import HRFCropMixin, LinearClassificationHead
from .event_aggregation import aggregate_event_pairs


class S4DKernel(nn.Module):
   

    def __init__(
        self,
        d_model: int,
        state_dim: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        learning_rate: float | None = None,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or state_dim % 2:
            raise ValueError("state_dim must be a positive even integer")
        hidden = int(d_model)
        log_dt = torch.rand(hidden) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        c_value = torch.randn(hidden, state_dim // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(c_value))
        self._register_ssm_parameter("log_dt", log_dt, learning_rate)
        self._register_ssm_parameter(
            "log_A_real",
            torch.log(0.5 * torch.ones(hidden, state_dim // 2)),
            learning_rate,
        )
        self._register_ssm_parameter(
            "A_imag",
            math.pi
            * torch.arange(state_dim // 2).unsqueeze(0).repeat(hidden, 1),
            learning_rate,
        )

    def _register_ssm_parameter(
        self, name: str, tensor: torch.Tensor, learning_rate: float | None
    ) -> None:
        if learning_rate == 0.0:
            self.register_buffer(name, tensor)
            return
        self.register_parameter(name, nn.Parameter(tensor))
        optimizer_options = {"weight_decay": 0.0}
        if learning_rate is not None:
            optimizer_options["lr"] = float(learning_rate)
        setattr(getattr(self, name), "_optim", optimizer_options)

    def forward(self, length: int) -> torch.Tensor:
        dt = torch.exp(self.log_dt)
        c_value = torch.view_as_complex(self.C)
        a_value = -torch.exp(self.log_A_real) + 1j * self.A_imag
        dt_a = a_value * dt.unsqueeze(-1)
        exponents = dt_a.unsqueeze(-1) * torch.arange(
            length, device=a_value.device
        )
        c_value = c_value * torch.expm1(dt_a) / a_value
        return 2 * torch.einsum(
            "hn,hnl->hl", c_value, torch.exp(exponents)
        ).real


class TiedDropout(nn.Module):
  

    def __init__(self, probability: float) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")
        self.probability = float(probability)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return value
        keep = 1.0 - self.probability
        mask = torch.rand(
            value.shape[:2] + (1,) * (value.ndim - 2), device=value.device
        ) < keep
        return value * mask.to(value.dtype) / keep


class S4D(nn.Module):
    

    def __init__(
        self,
        d_model: int,
        state_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.skip = nn.Parameter(torch.randn(d_model))
        self.kernel = S4DKernel(d_model, state_dim=state_dim)
        self.activation = nn.GELU()
        self.dropout = TiedDropout(dropout)
        self.output_linear = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        length = int(value.shape[-1])
        kernel = self.kernel(length)
        kernel_frequency = torch.fft.rfft(kernel, n=2 * length)
        value_frequency = torch.fft.rfft(value, n=2 * length)
        result = torch.fft.irfft(
            value_frequency * kernel_frequency, n=2 * length
        )[..., :length]
        result = result + value * self.skip.unsqueeze(-1)
        return self.output_linear(self.dropout(self.activation(result)))


class MultiscaleMeanHemoglobinBranch(nn.Module):


    def __init__(
        self,
        signal_channels: int,
        signal_length: int,
        short_average_size: int = 10,
    ) -> None:
        super().__init__()
        if signal_length < short_average_size:
            raise ValueError("signal_length is shorter than the averaging kernel")
        output_channels = signal_channels // 2
        self.channel_average = nn.AvgPool2d(
            kernel_size=(signal_channels, 1), stride=1
        )
        self.temporal_average = nn.AvgPool2d(
            kernel_size=(1, short_average_size), stride=1
        )
        self.full_scale = nn.Sequential(
            nn.Conv2d(1, output_channels, kernel_size=(1, signal_length)),
            nn.BatchNorm2d(output_channels),
            nn.ELU(),
        )
        self.short_scale = nn.Sequential(
            nn.Conv2d(
                1,
                output_channels,
                kernel_size=(1, signal_length - short_average_size + 1),
            ),
            nn.BatchNorm2d(output_channels),
            nn.ELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        channel_mean = self.channel_average(value)
        short_mean = self.temporal_average(channel_mean)
        full_features = self.full_scale(channel_mean).flatten(start_dim=1)
        short_features = self.short_scale(short_mean).flatten(start_dim=1)
        return torch.cat((full_features, short_features), dim=1)


class FNIRS4D(nn.Module):

    def __init__(
        self,
        num_classes: int = 2,
        physical_channels: int = 70,
        time_points: int = 60,
        num_layers: int = 4,
        state_dim: int = 64,
        dropout: float = 0.5,
        prenorm: bool = False,
    ) -> None:
        super().__init__()
        if physical_channels <= 0 or time_points < 10 or num_layers <= 0:
            raise ValueError("invalid fNIRS4D input dimensions")
        self.input_channels = int(physical_channels)
        self.time_points = int(time_points)
        self.signal_channels = 2 * self.input_channels
        self.prenorm = bool(prenorm)

        self.mean_hemoglobin_branch = MultiscaleMeanHemoglobinBranch(
            self.signal_channels, self.time_points
        )
        self.knowledge_embedding = nn.Sequential(
            nn.Conv2d(
                1,
                self.signal_channels,
                kernel_size=(1, self.time_points),
            ),
            nn.InstanceNorm2d(self.signal_channels),
            nn.Conv2d(
                self.signal_channels, self.signal_channels, kernel_size=(1, 1)
            ),
        )
        self.s4d_layers = nn.ModuleList(
            [
                S4D(
                    self.signal_channels,
                    state_dim=state_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(self.signal_channels) for _ in range(num_layers)]
        )
        self.dropouts = nn.ModuleList(
            [nn.Dropout1d(dropout) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(self.feature_dim, num_classes)

    @property
    def feature_dim(self) -> int:
        return 2 * self.signal_channels

    def forward_features(self, value: torch.Tensor) -> torch.Tensor:
        expected = (1, self.signal_channels, self.time_points)
        if value.ndim != 4 or tuple(value.shape[1:]) != expected:
            raise ValueError(f"expected [B,{expected}], got {tuple(value.shape)}")

        mean_features = self.mean_hemoglobin_branch(value)
        encoded = self.knowledge_embedding(value)



        encoded = encoded.flatten(start_dim=2).transpose(-1, -2)
        for layer, norm, dropout in zip(
            self.s4d_layers, self.norms, self.dropouts
        ):
            residual = encoded
            if self.prenorm:
                residual = norm(residual.transpose(-1, -2)).transpose(-1, -2)
            residual = dropout(layer(residual))
            encoded = encoded + residual
            if not self.prenorm:
                encoded = norm(encoded.transpose(-1, -2)).transpose(-1, -2)
        sequence_features = encoded.transpose(-1, -2).mean(dim=1)
        return torch.cat((sequence_features, mean_features), dim=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.fc(self.forward_features(value))



fNIRS4D = FNIRS4D


class FNIRS4DLinearClassifier(nn.Module, HRFCropMixin):

    def __init__(
        self,
        backbone: FNIRS4D,
        sampling_rate_hz: float = 5.0,
        pre_onset_seconds: float = 5.0,
        hrf_start_seconds: float = -2.0,
        hrf_end_seconds: float = 15.0,
        num_classes: int = 2,
        event_pair_mode: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.event_pair_mode = bool(event_pair_mode)
        self.num_classes = int(num_classes)
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

    def event_features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:


        cropped = self.crop_batch(batch) if self.event_pair_mode else batch
        hbo = cropped["hbo"].float()
        hbr = cropped["hbr"].float()
        expected_channels = self.backbone.input_channels
        if hbo.shape[1] != expected_channels:
            raise ValueError(
                "Fair Input channel count must equal the declared input_channels "
                f"({expected_channels}); channel padding/selection is not allowed "
                "inside the model"
            )
        hbt = hbo + hbr
        model_input = torch.cat((hbo, hbt), dim=1).unsqueeze(1)
        return self.backbone.forward_features(model_input)

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
