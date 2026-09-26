
from __future__ import annotations

import numpy as np

GEOMETRY_DIM = 6
SCALE_MM = 25.460582556839242
SCALE_M = SCALE_MM / 1000.0


def channel_endpoints(
    source_mm_or_m: np.ndarray,
    detector_mm_or_m: np.ndarray,
    channel_valid: np.ndarray | None = None,
) -> np.ndarray:
    source = np.asarray(source_mm_or_m, dtype=np.float64)
    detector = np.asarray(detector_mm_or_m, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3 or detector.shape != source.shape:
        raise ValueError(
            f"expected source/detector [C,3], got {source.shape}/{detector.shape}"
        )
    valid = (
        np.ones(source.shape[0], dtype=bool)
        if channel_valid is None
        else np.asarray(channel_valid, dtype=bool).reshape(-1)
    )
    if valid.shape[0] != source.shape[0]:
        raise ValueError("channel_valid length must equal source channel count")
    keep = valid & np.isfinite(source).all(axis=1) & np.isfinite(detector).all(axis=1)
    return np.concatenate([source[keep], detector[keep]], axis=0)


def common_center(source_mm_or_m, detector_mm_or_m, channel_valid=None) -> np.ndarray:
    endpoints = channel_endpoints(source_mm_or_m, detector_mm_or_m, channel_valid)
    if endpoints.shape[0] == 0:
        return np.zeros(3, dtype=np.float64)
    return endpoints.mean(axis=0)


def c_geometry(
    source_mm_or_m: np.ndarray,
    detector_mm_or_m: np.ndarray,
    channel_valid: np.ndarray | None = None,
    *,
    metres: bool = False,
) -> np.ndarray:
    source = np.asarray(source_mm_or_m, dtype=np.float64)
    detector = np.asarray(detector_mm_or_m, dtype=np.float64)
    center = common_center(source, detector, channel_valid)
    scale = SCALE_M if metres else SCALE_MM
    source = (source - center) / scale
    detector = (detector - center) / scale
    midpoint = (source + detector) / 2.0
    delta = detector - source
    geometry = np.concatenate([midpoint, delta], axis=1).astype(np.float32)
    if geometry.shape != (source.shape[0], GEOMETRY_DIM):
        raise RuntimeError(f"unexpected C geometry shape {geometry.shape}")
    return geometry
