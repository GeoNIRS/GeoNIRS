
from __future__ import annotations

import numpy as np
import warnings
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


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
    centered = np.where(keep, values - mean[..., None], 0.0)
    variance = np.square(centered).sum(axis=-1) / safe_count
    std = np.sqrt(np.maximum(variance, 0.0))
    rms = np.sqrt(np.square(values).sum(axis=-1) / safe_count)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        nan_values = np.where(keep, signal, np.nan)
        minimum = np.nanmin(nan_values, axis=-1)
        maximum = np.nanmax(nan_values, axis=-1)
        median = np.nanmedian(nan_values, axis=-1)
        q25 = np.nanpercentile(nan_values, 25.0, axis=-1)
        q75 = np.nanpercentile(nan_values, 75.0, axis=-1)
    time = np.arange(signal.shape[-1], dtype=np.float64)
    mean_time = np.where(keep, time, 0.0).sum(axis=-1) / safe_count
    centered_time = np.where(keep, time - mean_time[..., None], 0.0)
    denominator = np.square(centered_time).sum(axis=-1)



    slope = np.divide((centered_time * centered).sum(axis=-1), denominator,
                      out=np.zeros_like(mean), where=denominator > 0)
    third = np.power(centered, 3).sum(axis=-1) / safe_count
    fourth = np.power(centered, 4).sum(axis=-1) / safe_count
    skew = np.divide(third, np.power(std, 3), out=np.zeros_like(mean), where=std > 1e-12)
    kurt = np.divide(fourth, np.square(variance), out=np.zeros_like(mean), where=variance > 1e-12)
    kurt = np.where(variance > 1e-12, kurt - 3.0, 0.0)
    fraction = count / float(signal.shape[-1])
    out = np.stack((mean, std, minimum, maximum, median, q75 - q25, rms,
                    slope, skew, kurt, fraction), axis=-1)



    out[count == 0] = 0.0
    return out.reshape(signal.shape[0], -1)


def paired_features(hbo: np.ndarray, hbr: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    return np.concatenate((features(hbo, valid), features(hbr, valid)), axis=-1)


def build_pipeline() -> Pipeline:

    return Pipeline([
        ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                            class_weight="balanced", random_state=810)),
    ])
