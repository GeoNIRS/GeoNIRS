from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline


def features(signal: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 3:
        raise ValueError(f"expected [N,C,T], got {signal.shape}")
    keep = np.isfinite(signal) if valid is None else np.asarray(valid, dtype=bool) & np.isfinite(signal)
    if keep.shape != signal.shape:
        raise ValueError(f"signal/mask shape mismatch: {signal.shape} vs {keep.shape}")
    count = keep.sum(axis=-1).astype(np.float64)
    safe_count = np.maximum(count, 1.0)
    values = np.where(keep, signal, 0.0)
    mean = values.sum(axis=-1) / safe_count
    time = np.arange(signal.shape[-1], dtype=np.float64)
    mean_time = np.where(keep, time, 0.0).sum(axis=-1) / safe_count
    centered_time = np.where(keep, time - mean_time[..., None], 0.0)
    centered_signal = np.where(keep, values - mean[..., None], 0.0)
    denominator = np.square(centered_time).sum(axis=-1)
    slope = np.divide((centered_time * centered_signal).sum(axis=-1), denominator,
                      out=np.zeros_like(mean), where=denominator > 0)
    return np.stack((mean, slope), axis=-1).reshape(signal.shape[0], -1)


def paired_features(hbo: np.ndarray, hbr: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    return np.concatenate((features(hbo, valid), features(hbr, valid)), axis=-1)


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("classifier", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])
