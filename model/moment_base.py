from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

try:
    from momentfm import MOMENTPipeline
except ImportError as exc:
    raise ImportError(
        "MOMENT is not installed. Install it with: pip install momentfm"
    ) from exc


_MOMENT_1_BASE_CONFIG = {
    "task_name": "reconstruction",
    "model_name": "MOMENT",
    "transformer_type": "encoder_only",
    "d_model": None,
    "seq_len": 512,
    "patch_len": 8,
    "patch_stride_len": 8,
    "device": "cpu",
    "transformer_backbone": "google/flan-t5-base",
    "model_kwargs": {},
    "t5_config": {
        "architectures": ["T5ForConditionalGeneration"],
        "d_ff": 2048,
        "d_kv": 64,
        "d_model": 768,
        "decoder_start_token_id": 0,
        "dropout_rate": 0.1,
        "eos_token_id": 1,
        "feed_forward_proj": "gated-gelu",
        "initializer_factor": 1.0,
        "is_encoder_decoder": True,
        "layer_norm_epsilon": 1e-06,
        "model_type": "t5",
        "n_positions": 512,
        "num_decoder_layers": 12,
        "num_heads": 12,
        "num_layers": 12,
        "output_past": True,
        "pad_token_id": 0,
        "relative_attention_max_distance": 128,
        "relative_attention_num_buckets": 32,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": 32128,
    },
}


def _moment_1_base_config() -> dict:
   

    return {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in _MOMENT_1_BASE_CONFIG.items()
    }


class MOMENTBase(nn.Module):
   
    def __init__(
        self,
        num_class: int = 2,
        physical_channels: int = 70,
        context_length: int = 512,
        model_name: str = "AutonLab/MOMENT-1-base",
        local_files_only: bool = False,
    ) -> None:
        super().__init__()

        self.num_class = int(num_class)
        self.physical_channels = int(physical_channels)


        self.input_channels = 2 * self.physical_channels

        self.context_length = int(context_length)
        self.model_name = str(model_name)

        model_kwargs = {
            "task_name": "classification",
            "n_channels": self.input_channels,
            "num_class": self.num_class,
        }

        try:
            self.model = MOMENTPipeline.from_pretrained(
                self.model_name,
                model_kwargs=model_kwargs,
                local_files_only=local_files_only,
            )
        except TypeError as exc:
            missing_config = "missing 1 required positional argument: 'config'"
            if missing_config not in str(exc):
                raise
            if self.model_name != "AutonLab/MOMENT-1-base":
                raise
            self.model = MOMENTPipeline.from_pretrained(
                self.model_name,
                config=_moment_1_base_config(),
                model_kwargs=model_kwargs,
                local_files_only=local_files_only,
            )


        self.model.init()

    def _prepare_input(
        self,
        value: torch.Tensor,
        input_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        

        if value.ndim != 4:
            raise ValueError(
                "MOMENTBase expects [B,2,C,T], "
                f"got {tuple(value.shape)}"
            )

        batch_size, chromophores, channels, time_points = value.shape

        if chromophores != 2:
            raise ValueError(
                f"expected HbO/HbR axis size=2, got {chromophores}"
            )

        if channels != self.physical_channels:
            raise ValueError(
                "physical channel mismatch: "
                f"model expects {self.physical_channels}, "
                f"input has {channels}"
            )

        if time_points > self.context_length:
            raise ValueError(
                f"input T={time_points} exceeds MOMENT "
                f"context length={self.context_length}. "
                "Do not silently truncate the downstream segment."
            )










        value = value.reshape(
            batch_size,
            self.input_channels,
            time_points,
        ).float()




        if input_mask is None:
            input_mask = torch.ones(
                batch_size,
                time_points,
                dtype=torch.long,
                device=value.device,
            )

        else:
            if input_mask.ndim == 3:
                if input_mask.shape[0] != batch_size or input_mask.shape[-1] != time_points:
                    raise ValueError(
                        "3-D input_mask must have shape [B,1,T] or [B,C,T], "
                        f"got {tuple(input_mask.shape)}"
                    )
                if input_mask.shape[1] == 1:
                    input_mask = input_mask[:, 0, :]
                elif input_mask.shape[1] == channels:
                    input_mask = input_mask.any(dim=1)
                else:
                    raise ValueError(
                        "3-D input_mask channel axis must be 1 or physical_channels, "
                        f"got {tuple(input_mask.shape)}"
                    )

            if input_mask.ndim != 2:
                raise ValueError(
                    "input_mask must be [B,T], [B,1,T], or [B,C,T], "
                    f"got {tuple(input_mask.shape)}"
                )

            if tuple(input_mask.shape) != (
                batch_size,
                time_points,
            ):
                raise ValueError(
                    "input_mask shape mismatch: "
                    f"expected {(batch_size, time_points)}, "
                    f"got {tuple(input_mask.shape)}"
                )

            input_mask = input_mask.to(
                device=value.device,
                dtype=torch.long,
            )







        pad_length = (
            self.context_length
            - time_points
        )

        if pad_length > 0:
            value = F.pad(
                value,
                (0, pad_length),
                value=0.0,
            )

            input_mask = F.pad(
                input_mask,
                (0, pad_length),
                value=0,
            )

        expected_value = (
            batch_size,
            self.input_channels,
            self.context_length,
        )

        expected_mask = (
            batch_size,
            self.context_length,
        )

        if tuple(value.shape) != expected_value:
            raise RuntimeError(
                f"internal MOMENT input error: "
                f"{tuple(value.shape)} != {expected_value}"
            )

        if tuple(input_mask.shape) != expected_mask:
            raise RuntimeError(
                f"internal MOMENT mask error: "
                f"{tuple(input_mask.shape)} != {expected_mask}"
            )

        return value, input_mask

    def forward(
        self,
        value: torch.Tensor,
        input_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        value, input_mask = self._prepare_input(
            value,
            input_mask,
        )

        output = self.model(
            x_enc=value,
            input_mask=input_mask,
        )

        logits = output.logits

        if logits is None:
            raise RuntimeError(
                "MOMENT classification mode returned logits=None"
            )

        if logits.ndim != 2:
            raise RuntimeError(
                f"expected MOMENT logits [B,K], got {tuple(logits.shape)}"
            )

        if logits.shape[-1] != self.num_class:
            raise RuntimeError(
                f"expected {self.num_class} classes, "
                f"got logits {tuple(logits.shape)}"
            )

        return logits


class MOMENTBaseClassifier(nn.Module):
 
    def __init__(
        self,
        backbone: MOMENTBase,
    ) -> None:
        super().__init__()

        self.backbone = backbone

    def feature_parameters(self):
        

        return [
            parameter
            for parameter in self.backbone.parameters()
            if parameter.requires_grad
        ]

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:

        hbo = batch["hbo"].float()
        hbr = batch["hbr"].float()

        if hbo.ndim != 3 or hbr.ndim != 3:
            raise ValueError(
                "expected hbo/hbr [B,C,T], "
                f"got {tuple(hbo.shape)} / {tuple(hbr.shape)}"
            )

        if hbo.shape != hbr.shape:
            raise ValueError(
                "HbO/HbR shape mismatch: "
                f"{tuple(hbo.shape)} vs {tuple(hbr.shape)}"
            )




        value = torch.stack(
            [hbo, hbr],
            dim=1,
        )

        input_mask = batch.get(
            "valid_time_mask",
            batch.get("valid"),
        )

        return self.backbone(
            value,
            input_mask=input_mask,
        )


class _LoRALinear(nn.Module):

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


def _install_linear_lora(
    module: nn.Module, rank: int, alpha: float, targets: str, prefix: str = ""
) -> list[str]:
    if targets != "all_nn_Linear":
        raise ValueError(f"unsupported canonical LoRA target policy: {targets}")
    installed: list[str] = []
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            setattr(module, name, _LoRALinear(child, rank=rank, alpha=alpha))
            installed.append(full_name)
        else:
            installed.extend(_install_linear_lora(child, rank, alpha, targets, full_name))
    return installed


def install_moment_lora(
    model: MOMENTBaseClassifier,
    rank: int = 4,
    alpha: float = 8.0,
) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    pipeline = model.backbone.model
    installed: list[str] = []
    installed += _install_linear_lora(
        pipeline.encoder, int(rank), float(alpha), "all_nn_Linear", "encoder"
    )
    installed += _install_linear_lora(
        pipeline.patch_embedding, int(rank), float(alpha), "all_nn_Linear",
        "patch_embedding",
    )

    for parameter in pipeline.head.parameters():
        parameter.requires_grad_(True)

    model._moment_lora_linear_names = installed
    return installed
