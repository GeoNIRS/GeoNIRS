
from __future__ import annotations

import torch
from torch import nn


class LinearClassificationHead(nn.Linear):
    

    def __init__(self, input_dim: int, num_classes: int) -> None:
        if input_dim <= 0:
            raise ValueError("classification-head input_dim must be positive")
        if num_classes <= 1:
            raise ValueError("classification-head num_classes must exceed one")
        super().__init__(int(input_dim), int(num_classes))
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)


class HRFCropMixin:
   
    def _set_hrf_crop(
        self,
        sampling_rate_hz: float,
        pre_onset_seconds: float,
        hrf_start_seconds: float,
        hrf_end_seconds: float,
    ) -> None:
        rate = float(sampling_rate_hz)
        self.hrf_sample_start = round(
            (float(pre_onset_seconds) + float(hrf_start_seconds)) * rate
        )
        self.hrf_sample_stop = round(
            (float(pre_onset_seconds) + float(hrf_end_seconds)) * rate
        )
        if not 0 <= self.hrf_sample_start < self.hrf_sample_stop:
            raise ValueError("invalid HRF interval")

    def crop_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        start, stop = self.hrf_sample_start, self.hrf_sample_stop
        cropped = dict(batch)
        for key in ("hbo", "hbr", "valid"):
            cropped[key] = batch[key][..., start:stop]
        return cropped
