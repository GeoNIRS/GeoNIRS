from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from torch import nn

import model.ours as ours_module
from model.ours import FeatureProjection, Ours, ResidualProjectionBlock

__all__ = ["ScaledFeatureProjection", "OursLarge"]


class ScaledFeatureProjection(FeatureProjection):
    

    def __init__(
        self,
        input_dim: int = 192,
        output_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:


        nn.Module.__init__(self)

        if input_dim < 64 or input_dim % 4 != 0:
            raise ValueError("input_dim must be a multiple of 4 and at least 64")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")

        widths = (
            [192, 128, 96, 64]
            if input_dim == 192 and output_dim == 64
            else [input_dim, input_dim // 2, input_dim // 4, output_dim]
        )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.blocks = nn.ModuleList([
            ResidualProjectionBlock(widths[i], widths[i + 1], dropout)
            for i in range(len(widths) - 1)
        ])

@contextmanager
def scaled_projection() -> Iterator[None]:

    original = ours_module.FeatureProjection
    ours_module.FeatureProjection = ScaledFeatureProjection
    try:
        yield
    finally:
        ours_module.FeatureProjection = original


class OursLarge(Ours):
    

    def __init__(self, data_config: dict[str, Any], model_config: dict[str, Any]) -> None:
        with scaled_projection():
            super().__init__(data_config, model_config)
