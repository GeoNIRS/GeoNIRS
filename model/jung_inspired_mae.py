from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch
from torch import nn

FAMILY = "jung_inspired_vanilla_mae"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class MAEConfig:
    patch_samples: int = 10
    minimum_patch_valid_fraction: float = 0.8
    mask_ratio: float = 0.5
    encoder_dim: int = 192
    encoder_layers: int = 4
    encoder_heads: int = 8
    decoder_dim: int = 96
    decoder_layers: int = 2
    decoder_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    def __post_init__(self):
        for prefix in ("encoder", "decoder"):
            dim, heads = getattr(self, prefix + "_dim"), getattr(self, prefix + "_heads")
            if dim <= 0 or heads <= 0 or dim % heads or dim % 2:
                raise ValueError("dimensions must be positive, even and divisible by heads")
            if getattr(self, prefix + "_layers") < 1:
                raise ValueError("at least one Transformer layer is required")
        if self.patch_samples < 1 or not 0 < self.mask_ratio < 1:
            raise ValueError("invalid patch size or mask ratio")
        if not 0 < self.minimum_patch_valid_fraction <= 1 or not 0 <= self.dropout < 1:
            raise ValueError("invalid validity threshold or dropout")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")


def time_positions(count, dim, device, dtype):
    position = torch.arange(count, device=device, dtype=torch.float32)[:, None]
    frequency = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000) / dim))
    result = torch.empty(count, dim, device=device)
    result[:, 0::2], result[:, 1::2] = (position * frequency).sin(), (position * frequency).cos()
    return result.to(dtype)


def transformer(dim, heads, depth, cfg):
    layer = nn.TransformerEncoderLayer(dim, heads, int(dim * cfg.mlp_ratio),
                                       dropout=cfg.dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, depth, norm=nn.LayerNorm(dim), enable_nested_tensor=False)


class JungInspiredMAE(nn.Module):
    def __init__(self, config: MAEConfig | dict | None = None):
        super().__init__()
        self.config = config if isinstance(config, MAEConfig) else MAEConfig(**(config or {}))
        cfg = self.config
        self.patch_embed = nn.Linear(2 * cfg.patch_samples, cfg.encoder_dim)
        self.encoder = transformer(cfg.encoder_dim, cfg.encoder_heads, cfg.encoder_layers, cfg)
        self.decoder_embed = nn.Linear(cfg.encoder_dim, cfg.decoder_dim)
        self.mask_token = nn.Parameter(torch.empty(1, 1, cfg.decoder_dim))
        self.decoder = transformer(cfg.decoder_dim, cfg.decoder_heads, cfg.decoder_layers, cfg)
        self.prediction = nn.Linear(cfg.decoder_dim, 2 * cfg.patch_samples)


        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.zeros_(module.in_proj_bias)
        nn.init.normal_(self.mask_token, std=0.02)

    @staticmethod
    def is_encoder_parameter(name):
        return name.startswith(("patch_embed.", "encoder."))

    def encoder_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if self.is_encoder_parameter(k)}

    def patchify(self, batch):
        hbo, hbr = batch["hbo"], batch["hbr"]
        if hbo.ndim != 3 or hbo.shape != hbr.shape:
            raise ValueError("expected matching HbO/HbR [B,C,T]")
        b, c, t = hbo.shape
        q = self.config.patch_samples
        if min(b, c, t) < 1 or t % q:
            raise ValueError("nonempty input T must be divisible by patch_samples; no cropping/padding")
        valid = batch["valid_time_mask"].bool()
        channel_valid = batch["valid_channel_mask"].bool()
        if valid.shape != hbo.shape or channel_valid.shape != (b, c):
            raise ValueError("expected validity [B,C,T] and [B,C]")
        valid = valid & channel_valid[..., None]
        signal = torch.stack((hbo, hbr), dim=-1)
        if not torch.isfinite(signal[valid]).all():
            raise ValueError("non-finite values at valid signal positions")
        signal = torch.where(valid[..., None], signal, 0)
        patches = signal.reshape(b, c, t // q, q * 2)
        scalar_valid = valid.reshape(b, c, t // q, q)[..., None].expand(-1, -1, -1, -1, 2).reshape_as(patches)
        patch_valid = valid.reshape(b, c, t // q, q).sum(-1) >= math.ceil(q * self.config.minimum_patch_valid_fraction)
        return patches, scalar_valid, patch_valid

    def random_mask(self, valid, generator=None):
        flat = valid.flatten(0, 1)
        counts = flat.sum(-1)
        keep = (counts * (1 - self.config.mask_ratio)).floor().long().clamp_min(1)

        eligible = counts >= 2
        scores = torch.rand(flat.shape, device=flat.device, generator=generator).masked_fill(~flat, float("inf"))
        ranks = scores.argsort(dim=-1).argsort(dim=-1)
        return (flat & (ranks >= keep[:, None]) & eligible[:, None]).reshape_as(valid)

    def _encode_visible(self, patches, visible):
        n, p, _ = patches.shape
        dim = self.config.encoder_dim
        output = patches.new_zeros(n, p, dim)
        active = visible.any(-1)
        if not active.any():
            return output
        selected, visibility = patches[active], visible[active]
        counts = visibility.sum(-1)
        length = int(counts.max())

        positions = torch.arange(p, device=patches.device).expand(selected.shape[0], -1)
        indices = positions.masked_fill(~visibility, p).argsort(dim=-1)[:, :length]
        packed_valid = torch.arange(length, device=patches.device)[None, :] < counts[:, None]
        packed = selected.gather(1, indices[..., None].expand(-1, -1, selected.shape[-1]))
        tokens = self.patch_embed(packed)
        pos = time_positions(p, dim, patches.device, tokens.dtype)
        tokens = tokens + pos[indices]
        tokens = self.encoder(tokens, src_key_padding_mask=~packed_valid)
        tokens = tokens.masked_fill(~packed_valid[..., None], 0)
        full = tokens.new_zeros(selected.shape[0], p, dim)
        full.scatter_(1, indices[..., None].expand(-1, -1, dim), tokens)
        output = output.to(full.dtype)
        output[active] = full
        return output

    def encode(self, batch):
        patches, _, valid = self.patchify(batch)
        b, c, p, f = patches.shape
        encoded = self._encode_visible(patches.reshape(b * c, p, f), valid.reshape(b * c, p))
        return {"foundation": encoded.reshape(b, c, p, -1), "valid": valid}

    def forward(self, batch, *, generator=None, masked=None):
        patches, scalar_valid, valid = self.patchify(batch)
        b, c, p, f = patches.shape
        if masked is None:
            masked = self.random_mask(valid, generator)
        if masked.shape != valid.shape or masked.dtype != torch.bool or (masked & ~valid).any():
            raise ValueError("masked must be bool [B,C,P], a subset of valid patches")
        visible = valid & ~masked
        if (masked.any(-1) & ~visible.any(-1)).any():
            raise ValueError("each reconstructed channel needs at least one visible patch")
        eligible = masked.any(-1)
        encoded = self._encode_visible(patches.reshape(b * c, p, f),
                                       (visible & eligible[..., None]).reshape(b * c, p))
        active = eligible.flatten()
        prediction = patches.new_zeros(b * c, p, f)
        if active.any():
            latent = self.decoder_embed(encoded[active])
            shown = visible.reshape(b * c, p)[active]
            tokens = torch.where(shown[..., None], latent, self.mask_token.to(latent.dtype))
            tokens = tokens + time_positions(p, self.config.decoder_dim, tokens.device, tokens.dtype)
            decoded = self.decoder(tokens, src_key_padding_mask=~valid.reshape(b * c, p)[active])
            active_prediction = self.prediction(decoded)


            prediction = prediction.to(active_prediction.dtype)
            prediction[active] = active_prediction
        prediction = prediction.reshape_as(patches)
        loss_mask = scalar_valid & masked[..., None]

        errors = (prediction.float() - patches.float()).square().masked_fill(~loss_mask, 0)
        denominators = loss_mask.sum(dim=(1, 2, 3))
        per_window = errors.sum(dim=(1, 2, 3)) / denominators.clamp_min(1)
        usable = denominators > 0


        loss_sum = per_window.sum() + self.mask_token.sum() * 0
        return {"loss": loss_sum / usable.sum().clamp_min(1), "loss_sum": loss_sum,
                "valid_windows": usable.sum(), "per_window_loss": per_window,
                "prediction": prediction, "target": patches, "loss_mask": loss_mask,
                "masked": masked, "visible": visible & eligible[..., None],
                "encoded_visible": encoded.reshape(b, c, p, -1), "patch_valid": valid}


def model_payload(model):
    return {"format_version": FORMAT_VERSION, "model_family": FAMILY,
            "model_config": asdict(model.config), "model": model.state_dict(),
            "encoder": model.encoder_state_dict()}


def load_mae_checkpoint(path: str | Path, *, random_init=False):

    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("model_family") != FAMILY or saved.get("format_version") != FORMAT_VERSION:
        raise ValueError("not a supported vanilla MAE checkpoint; refusing GeoNIRS/legacy weights")
    model = JungInspiredMAE(saved["model_config"])

    if random_init:
        expected = model.state_dict()
        if set(saved["model"]) != set(expected) or any(saved["model"][k].shape != v.shape for k, v in expected.items()):
            raise ValueError("checkpoint architecture/weights mismatch")
    else:
        model.load_state_dict(saved["model"], strict=True)
    return model, saved
