import torch
from torch import nn


class MAEEventClassifier(nn.Module):
    def __init__(self, backbone, mode, channels, points, num_classes, dropout=0.1):
        super().__init__()
        if mode not in ("lp", "lora", "ft") or points % backbone.config.patch_samples:
            raise ValueError("expected lp/lora/ft and an unmodified, patch-divisible window")
        self.backbone, self.mode = backbone, mode


        self.head_variant = "flatten_linear"
        self.foundation_shape = (channels, points // backbone.config.patch_samples, backbone.config.encoder_dim)
        self.classifier_feature_dim = channels * (points // backbone.config.patch_samples) * backbone.config.encoder_dim
        self.register_buffer("feature_mean", torch.zeros(self.classifier_feature_dim))
        self.register_buffer("feature_std", torch.ones(self.classifier_feature_dim))
        self.register_buffer("feature_standardizer_fitted", torch.tensor(False))
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.classifier_feature_dim, num_classes)
        for name, parameter in backbone.named_parameters():




            parameter.requires_grad_(mode == "ft" and backbone.is_encoder_parameter(name))
        self.train()

    def train(self, mode=True):
        super().train(mode)
        if self.mode == "lp":
            self.backbone.eval()

        self.backbone.decoder.eval()
        self.backbone.decoder_embed.eval()
        self.backbone.prediction.eval()
        return self

    def event_foundation_features(self, batch):
        if self.mode == "lp":
            with torch.no_grad():
                features = self.backbone.encode(batch)["foundation"]
        else:
            features = self.backbone.encode(batch)["foundation"]
        if tuple(features.shape[1:]) != self.foundation_shape:
            raise ValueError("frozen Fair Input channel/time shape changed")
        return features.flatten(1)

    @torch.no_grad()
    def set_standardizer(self, mean, std):
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("invalid train-only feature standardizer")
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std)
        self.feature_standardizer_fitted.fill_(True)

    def forward(self, batch):
        features = (self.event_foundation_features(batch) - self.feature_mean) / self.feature_std
        return self.linear(self.dropout(features))
