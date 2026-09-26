from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from model.ours import Ours


DECODER_PREFIXES = ("hbo_to_hbr_decoder.", "hbr_to_hbo_decoder.")
FEATURE_STD_EPSILON = 1e-6
FLATTEN_VARIANTS = ("flatten_linear", "strong_dropout_linear")
TEMPORAL_BIN_VARIANTS = ("temporal_bin2x_linear", "temporal_bin2x_linear_moderate")
LINEAR_HEAD_VARIANTS = FLATTEN_VARIANTS + TEMPORAL_BIN_VARIANTS
SUPPORTED_HEAD_VARIANTS = LINEAR_HEAD_VARIANTS + ("temporal_pool_mlp", "mean_std_linear")


class LoRALinear(nn.Module):

    def __init__(self, source: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        self.source = source
        self.source.weight.requires_grad_(False)
        if source.bias is not None:
            self.source.bias.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, source.in_features))
        self.b = nn.Parameter(torch.zeros(source.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=5 ** 0.5)
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.source(x) + (x @ self.a.t() @ self.b.t()) * self.scale

    @property
    def weight(self) -> torch.Tensor:
        return self.source.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.source.bias


def install_linear_lora(module: nn.Module, rank: int, alpha: float, targets: str, prefix: str = "") -> list[str]:
    if targets != "all_nn_Linear":
        raise ValueError(f"unsupported canonical LoRA target policy: {targets}")
    installed: list[str] = []
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
            installed.append(full_name)
        else:
            installed.extend(install_linear_lora(child, rank, alpha, targets, full_name))
    return installed


def valid_temporal_mean_std(joint: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if joint.ndim != 4 or valid.shape != joint.shape[:3]:
        raise ValueError("expected joint [B,L,P,F] and valid [B,L,P]")
    mask = valid.unsqueeze(-1).to(joint.dtype)
    count = mask.sum(dim=2).clamp_min(1.0)
    mean = (joint * mask).sum(dim=2) / count
    variance = ((joint - mean.unsqueeze(2)) * mask).square().sum(dim=2) / count
    return mean, variance.clamp_min(0.0).sqrt()


def valid_aware_temporal_bin2x(foundation: torch.Tensor, valid: torch.Tensor, patch_drop_prob: float = 0.1, training: bool = True):
    if foundation.ndim != 5 or valid.ndim != 3 or foundation.shape[0] != valid.shape[0] or foundation.shape[2] != valid.shape[1] or foundation.shape[3] != valid.shape[2] or foundation.shape[3] % 2:
        raise ValueError("expected foundation [B,S,L,P,F] and valid [B,L,P] with even P")
    batch, streams, latent, patches, feature = foundation.shape
    bins = patches // 2
    valid = valid.bool()
    time_valid = valid.any(dim=1)
    dropped = torch.zeros_like(time_valid)
    if training and patch_drop_prob > 0.0:
        for b in range(batch):
            candidates = time_valid[b].nonzero(as_tuple=False).flatten()
            if candidates.numel() <= 1:
                continue
            selected = torch.rand(candidates.numel(), device=foundation.device) < patch_drop_prob
            if bool(selected.all()):
                selected[int(torch.randint(0, int(candidates.numel()), (1,), device=foundation.device).item())] = False
            dropped[b, candidates[selected]] = True
    pooled_valid = valid & ~dropped[:, None, :]
    grouped_valid = pooled_valid.reshape(batch, latent, bins, 2)
    grouped = foundation.reshape(batch, streams, latent, bins, 2, feature)
    counts = grouped_valid.sum(dim=-1, dtype=foundation.dtype)
    weights = grouped_valid.unsqueeze(1).unsqueeze(-1).to(foundation.dtype)
    pooled = (grouped * weights).sum(dim=-2) / counts.unsqueeze(1).unsqueeze(-1).clamp_min(1.0)
    pooled = torch.where((counts > 0).unsqueeze(1).unsqueeze(-1), pooled, torch.zeros_like(pooled))
    return pooled, counts > 0, dropped


class OursEventClassifier(nn.Module):

    def __init__(self, backbone: Ours, mode: str, *, head_variant: str, dropout: float, patch_dropout: float, decoder_prefixes: tuple[str, ...], num_classes: int = 2, downstream_patch_count: int | None = None) -> None:
        super().__init__()
        if mode not in ("lp", "lora", "ft"):
            raise ValueError(mode)
        if head_variant not in SUPPORTED_HEAD_VARIANTS:
            raise ValueError(f"unsupported head variant: {head_variant}")
        self.backbone, self.mode, self.num_classes = backbone, mode, int(num_classes)
        self.head_variant, self.patch_dropout, self.decoder_prefixes = head_variant, float(patch_dropout), tuple(f"{x}." for x in decoder_prefixes)
        patch_count = int(downstream_patch_count or backbone.patch_count)
        if patch_count <= 0 or patch_count > backbone.patch_count:
            raise ValueError("downstream patch count must fit pretrained encoder")
        self.foundation_shape = (2, backbone.num_latent_sources, patch_count, backbone.output_feature_dim)
        if self.head_variant in FLATTEN_VARIANTS:
            self.event_feature_dim = math.prod(self.foundation_shape)
        elif self.head_variant in TEMPORAL_BIN_VARIANTS:
            self.event_feature_dim = 2 * backbone.num_latent_sources * (patch_count // 2) * backbone.output_feature_dim
        elif self.head_variant == "temporal_pool_mlp":
            self.event_feature_dim = 2 * backbone.num_latent_sources * backbone.output_feature_dim
        else:
            self.event_feature_dim = 4 * backbone.num_latent_sources * backbone.output_feature_dim
        self.classifier_feature_dim = int(self.event_feature_dim)
        self.register_buffer("feature_mean", torch.zeros(self.classifier_feature_dim))
        self.register_buffer("feature_std", torch.ones(self.classifier_feature_dim))
        self.register_buffer("feature_standardizer_fitted", torch.tensor(False))
        if self.head_variant in LINEAR_HEAD_VARIANTS:
            self.dropout = nn.Dropout(float(dropout)); self.linear = nn.Linear(self.classifier_feature_dim, self.num_classes)
        elif self.head_variant == "temporal_pool_mlp":
            self.norm = nn.LayerNorm(self.classifier_feature_dim); self.classifier = nn.Sequential(nn.Linear(self.classifier_feature_dim, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, self.num_classes))
        else:
            self.norm = nn.LayerNorm(self.classifier_feature_dim); self.dropout = nn.Dropout(0.2); self.classifier = nn.Linear(self.classifier_feature_dim, self.num_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.mode == "lp":
            self.backbone.eval()
        for name, module in self.backbone.named_modules():
            if name.startswith(self.decoder_prefixes):
                module.eval()
        return self

    def window_foundation(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.mode == "lp":
            with torch.no_grad():
                foundation = self.backbone.encode(batch)["foundation"]
        else:
            foundation = self.backbone.encode(batch)["foundation"]
        if tuple(foundation.shape[1:]) != self.foundation_shape:
            raise RuntimeError(f"expected foundation (*,{self.foundation_shape}), got {tuple(foundation.shape)}")
        return foundation

    def event_foundation_features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.head_variant in FLATTEN_VARIANTS:
            foundation = self.window_foundation(batch)
            windows = foundation.reshape(foundation.shape[0], self.event_feature_dim)
        else:
            if self.mode == "lp":
                with torch.no_grad(): encoded = self.backbone.encode(batch)
            else:
                encoded = self.backbone.encode(batch)
            foundation = encoded["foundation"]
            if tuple(foundation.shape[1:]) != self.foundation_shape:
                raise RuntimeError(f"expected foundation (*,{self.foundation_shape}), got {tuple(foundation.shape)}")
            if self.head_variant in TEMPORAL_BIN_VARIANTS:
                pooled, _, _ = valid_aware_temporal_bin2x(foundation, encoded["valid"], self.patch_dropout, self.training)
                windows = pooled.reshape(foundation.shape[0], self.event_feature_dim)
            else:
                joint = torch.cat([foundation[:, 0], foundation[:, 1]], dim=-1)
                mean, std = valid_temporal_mean_std(joint, encoded["valid"])
                windows = mean.flatten(start_dim=1) if self.head_variant == "temporal_pool_mlp" else torch.cat([mean, std], dim=-1).flatten(start_dim=1)
        return windows

    def set_standardizer(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feature_mean.copy_(mean.to(self.feature_mean)); self.feature_std.copy_(std.to(self.feature_std)); self.feature_standardizer_fitted.fill_(True)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = (self.event_foundation_features(batch) - self.feature_mean) / self.feature_std
        if self.head_variant in LINEAR_HEAD_VARIANTS:
            return self.linear(self.dropout(features))
        if self.head_variant == "temporal_pool_mlp":
            return self.classifier(self.norm(features))
        return self.classifier(self.dropout(self.norm(features)))


def is_decoder(name: str, decoder_prefixes: tuple[str, ...] = DECODER_PREFIXES) -> bool:
    prefixes = tuple(x if x.endswith(".") else f"{x}." for x in decoder_prefixes)
    return name.startswith(prefixes)


def configure_trainability(model: OursEventClassifier, method: str, adaptation_config: dict[str, Any]) -> None:
    for parameter in model.parameters(): parameter.requires_grad_(False)
    if model.head_variant in LINEAR_HEAD_VARIANTS: model.linear.requires_grad_(True)
    else: model.norm.requires_grad_(True); model.classifier.requires_grad_(True)
    if method == "ft":
        for name, parameter in model.backbone.named_parameters(): parameter.requires_grad_(not is_decoder(name, model.decoder_prefixes))
    elif method == "lora":
        if getattr(model, "_canonical_lora_installed", False):
            raise RuntimeError("canonical all-linear LoRA was already installed")
        lora = adaptation_config["lora"]
        model._canonical_lora_linear_names = install_linear_lora(model.backbone, int(lora["rank"]), float(lora["alpha"]), str(lora["targets"]))
        model._canonical_lora_installed = True
    elif method != "lp":
        raise ValueError(method)


def audit_trainability(model: OursEventClassifier, method: str) -> dict[str, Any]:
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    backbone_names = [name for name in names if name.startswith("backbone.")]
    decoder_trainable = [name for name in names if "decoder" in name]
    if method != "lora" and decoder_trainable:
        raise RuntimeError("decoder parameter is trainable")
    if method == "lora" and any(not (name.endswith(".a") or name.endswith(".b")) for name in decoder_trainable):
        raise RuntimeError("non-adapter decoder parameter is trainable")
    if method == "lp" and backbone_names: raise RuntimeError("LP backbone is not frozen")
    if method == "lora" and any(not (name.endswith(".a") or name.endswith(".b")) for name in backbone_names): raise RuntimeError("LoRA has non-adapter trainable backbone tensor")
    if method == "ft":
        expected = {f"backbone.{name}" for name, _ in model.backbone.named_parameters() if not is_decoder(name, model.decoder_prefixes)}
        if set(backbone_names) != expected: raise RuntimeError("FT did not unfreeze complete foundation path")
    adapter_names = [name for name in names if name.startswith("backbone.") and (name.endswith(".a") or name.endswith(".b"))]
    head_names = [name for name in names if not name.startswith("backbone.")]
    frozen_original = sum(p.numel() for name, p in model.backbone.named_parameters() if not p.requires_grad and ".source." in name)
    frozen_backbone = sum(p.numel() for p in model.backbone.parameters() if not p.requires_grad)
    return {"total_parameters": sum(p.numel() for p in model.parameters()), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable_parameter_names": names, "trainable_modules": sorted({name.rsplit('.', 1)[0] for name in names}), "linear_modules_total": len(getattr(model, "_canonical_lora_linear_names", [])) if method == "lora" else None, "lora_modules_installed": len(getattr(model, "_canonical_lora_linear_names", [])) if method == "lora" else 0, "lora_module_names": list(getattr(model, "_canonical_lora_linear_names", [])) if method == "lora" else [], "lora_trainable_parameter_names": adapter_names if method == "lora" else [], "lora_trainable_parameter_count": sum(p.numel() for name, p in model.named_parameters() if name in adapter_names), "classification_head_trainable_parameter_count": sum(p.numel() for name, p in model.named_parameters() if name in head_names), "frozen_backbone_original_parameter_count": frozen_original, "frozen_backbone_parameter_count": frozen_backbone}


@torch.no_grad()
def fit_standardizer(model: OursEventClassifier, loader, device) -> dict[str, Any]:
    model.eval(); features = torch.cat([model.event_foundation_features({k: v.to(device) if torch.is_tensor(v) else v for k, v in raw.items()}).cpu() for raw in loader]).float()
    if not torch.isfinite(features).all(): raise RuntimeError("feature standardizer input has NaN/Inf")
    mean, raw_std = features.mean(0), features.std(0, correction=0); std = raw_std.clamp_min(FEATURE_STD_EPSILON); model.set_standardizer(mean, std)
    return {"train_events": int(features.shape[0]), "dimensions": int(features.shape[1]), "near_constant_dimensions": int((raw_std < FEATURE_STD_EPSILON).sum()), "global_mean": float(features.mean()), "global_std": float(features.std(correction=0))}


def decay_allowed(name: str, parameter: nn.Parameter) -> bool:
    return parameter.ndim > 1 and not name.endswith(".bias") and "norm" not in name.lower()


def build_optimizer(model: OursEventClassifier, method: str, *, head_lr: float, backbone_lr: float, lora_lr: float, weight_decay: float):
    head_lr = float(head_lr); backbone_lr = float(backbone_lr); lora_lr = float(lora_lr); wd = float(weight_decay)
    grouped: dict[tuple[str, bool], list[tuple[str, nn.Parameter]]] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            component = "head" if name.startswith(("linear.", "norm.", "classifier.")) else "backbone"; grouped.setdefault((component, decay_allowed(name, parameter)), []).append((name, parameter))
    groups, audit = [], []
    for (component, decay), named in sorted(grouped.items()):
        lr = head_lr if component == "head" else (lora_lr if method == "lora" else backbone_lr)
        group = {"params": [p for _, p in named], "lr": lr, "initial_lr": lr, "weight_decay": wd if decay else 0.0, "group_name": f"{component}_{'decay' if decay else 'no_decay'}"}; groups.append(group)
        audit.append({"name": group["group_name"], "lr": lr, "weight_decay": group["weight_decay"], "parameters": sum(p.numel() for _, p in named), "parameter_names": [n for n, _ in named]})
    return torch.optim.AdamW(groups), audit


__all__ = ["OursEventClassifier", "LoRALinear", "install_linear_lora", "configure_trainability", "audit_trainability", "fit_standardizer", "build_optimizer", "valid_aware_temporal_bin2x"]
