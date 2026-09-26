
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn

from .classification_head import HRFCropMixin, LinearClassificationHead
from .event_aggregation import aggregate_event_pairs


class Residual(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        _, _, _, heads = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda value: rearrange(value, "b n (h d) -> b h n d", h=heads),
            qkv,
        )
        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if mask is not None:
            mask_value = -torch.finfo(dots.dtype).max
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            if mask.shape[-1] != dots.shape[-1]:
                raise ValueError("mask has incorrect dimensions")
            pair_mask = rearrange(mask, "b i -> b () i ()") * rearrange(
                mask, "b j -> b () () j"
            )
            dots.masked_fill_(~pair_mask, mask_value)
        attention = dots.softmax(dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attention, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            PreNorm(
                                dim,
                                Attention(
                                    dim,
                                    heads=heads,
                                    dim_head=dim_head,
                                    dropout=dropout,
                                ),
                            )
                        ),
                        Residual(
                            PreNorm(
                                dim,
                                FeedForward(dim, mlp_dim, dropout=dropout),
                            )
                        ),
                    ]
                )
            )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for attention, feed_forward in self.layers:
            x = attention(x, mask=mask)
            x = feed_forward(x)
        return x


class FNIRST(nn.Module):

    def __init__(
        self,
        num_class: int = 2,
        physical_channels: int = 70,
        time_points: int = 60,
        dim: int = 64,
        depth: int = 6,
        heads: int = 8,
        mlp_dim: int = 64,
        pool: str = "cls",
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        print_shapes: bool = False,
    ) -> None:
        super().__init__()
        if pool not in ("cls", "mean"):
            raise ValueError("pool must be 'cls' or 'mean'")
        if physical_channels < 5:
            raise ValueError("physical_channels must be at least 5")
        temporal_bins = math.floor((int(time_points) - 30) / 4) + 1
        if temporal_bins <= 0:
            raise ValueError("time_points must be at least 30")

        self.input_channels = int(physical_channels)
        self.time_points = int(time_points)
        self.dim = int(dim)
        self.pool = pool
        self.print_shapes = bool(print_shapes)
        self.temporal_bins = temporal_bins



        self.spatial_conv = nn.Conv2d(
            in_channels=2,
            out_channels=8,
            kernel_size=(5, 30),
            stride=(1, 4),
        )
        self.spatial_rearrange = Rearrange("b c h w -> b h (c w)")
        self.spatial_embedding = nn.Sequential(
            nn.Linear(temporal_bins * 8, dim),
            nn.LayerNorm(dim),
        )




        self.channel_conv = nn.Conv2d(
            in_channels=2,
            out_channels=8,
            kernel_size=(1, 30),
            stride=(1, 4),
        )
        self.channel_rearrange = Rearrange("b c h w -> b h (c w)")
        self.channel_embedding = nn.Sequential(
            nn.Linear(temporal_bins * 8, dim),
            nn.LayerNorm(dim),
        )



        official_max_tokens = 100
        if self.input_channels > official_max_tokens:
            raise ValueError("official positional embedding supports <=100 channels")
        self.pos_embedding_spatial = nn.Parameter(
            torch.randn(1, official_max_tokens + 1, dim)
        )
        self.cls_token_spatial = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_spatial = nn.Dropout(emb_dropout)
        self.transformer_spatial = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )

        self.pos_embedding_channel = nn.Parameter(
            torch.randn(1, official_max_tokens + 1, dim)
        )
        self.cls_token_channel = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_channel = nn.Dropout(emb_dropout)
        self.transformer_channel = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )

        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, num_class),
        )

    @property
    def feature_dim(self) -> int:
        return self.dim * 2

    def _trace(self, label: str, value: torch.Tensor) -> None:
        if self.print_shapes:
            print(f"{label}: {value.shape}")

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.mean(dim=1) if self.pool == "mean" else tokens[:, 0]

    def forward_features(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        expected = (2, self.input_channels, self.time_points)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"expected [B,{expected}], got {tuple(x.shape)}")
        self._trace("input", x)

        spatial = self.spatial_conv(x)
        self._trace("spatial conv", spatial)
        spatial = self.spatial_rearrange(spatial)
        self._trace("spatial rearrange", spatial)
        spatial = self.spatial_embedding(spatial)
        self._trace("spatial transformer input (before CLS)", spatial)

        channel = self.channel_conv(x)
        self._trace("channel conv", channel)
        channel = self.channel_rearrange(channel)
        self._trace("channel rearrange", channel)
        channel = self.channel_embedding(channel)
        self._trace("channel transformer input (before CLS)", channel)

        batch_size, spatial_tokens, _ = spatial.shape
        spatial_cls = repeat(
            self.cls_token_spatial, "() n d -> b n d", b=batch_size
        )
        spatial = torch.cat((spatial_cls, spatial), dim=1)
        spatial = spatial + self.pos_embedding_spatial[:, : spatial_tokens + 1]
        spatial = self.dropout_spatial(spatial)
        self._trace("spatial transformer input", spatial)
        spatial = self.transformer_spatial(spatial, mask)
        self._trace("spatial transformer output", spatial)

        _, channel_tokens, _ = channel.shape
        channel_cls = repeat(
            self.cls_token_channel, "() n d -> b n d", b=batch_size
        )
        channel = torch.cat((channel_cls, channel), dim=1)
        channel = channel + self.pos_embedding_channel[:, : channel_tokens + 1]
        channel = self.dropout_channel(channel)
        self._trace("channel transformer input", channel)
        channel = self.transformer_channel(channel, mask)
        self._trace("channel transformer output", channel)

        spatial_feature = self.to_latent(self._pool(spatial))
        channel_feature = self.to_latent(self._pool(channel))
        features = torch.cat((spatial_feature, channel_feature), dim=1)
        self._trace("forward_features", features)
        return features

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.mlp_head(self.forward_features(x, mask))


class FNIRSTLinearClassifier(nn.Module, HRFCropMixin):

    def __init__(
        self,
        backbone: FNIRST,
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
        self.backbone.mlp_head.requires_grad_(False)
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
        return self.backbone.forward_features(torch.stack([hbo, hbr], dim=1))

    def feature_parameters(self):
        return [
            parameter
            for name, parameter in self.backbone.named_parameters()
            if not name.startswith("mlp_head.") and parameter.requires_grad
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
