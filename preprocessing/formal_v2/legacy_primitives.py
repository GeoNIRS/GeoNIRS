from __future__ import annotations

import math

import numpy as np
from scipy.signal import resample_poly


TARGET_FS = 5.0


def mask_segments(mask, t, fs):
    ids = np.flatnonzero(mask)
    groups = np.split(ids, np.where(np.diff(ids) > 1)[0] + 1) if ids.size else []
    return [{
        "start_sample": int(group[0]), "end_sample": int(group[-1]),
        "start_s": float(t[group[0]]), "end_s": float(t[group[-1]]),
        "duration_s": float(len(group) / fs),
    } for group in groups]


def qc(od, t, fs):
    n, channels = od.shape
    finite = np.isfinite(od)
    median = np.nanmedian(np.where(finite, od, np.nan), axis=0)
    mad = np.nanmedian(np.abs(np.where(finite, od, np.nan) - median), axis=0)
    valid_mad = np.isfinite(mad) & (mad > 0)
    robust_z = np.zeros_like(od)
    robust_z[:, valid_mad] = (
        od[:, valid_mad] - median[valid_mad]
    ) / (1.4826 * mad[valid_mad])
    outlier = finite & valid_mad[None, :] & (np.abs(robust_z) > 8)
    abnormal = (~finite) | outlier
    interpolated = 0
    time_bad = np.zeros(n, dtype=bool)
    for channel in range(channels):
        ids = np.flatnonzero(abnormal[:, channel])
        groups = np.split(ids, np.where(np.diff(ids) > 1)[0] + 1) if ids.size else []
        good = np.flatnonzero(np.isfinite(od[:, channel]) & ~outlier[:, channel])
        for group in groups:
            bounded = (
                group[0] > 0 and group[-1] < n - 1
                and np.isfinite(od[group[0] - 1, channel])
                and np.isfinite(od[group[-1] + 1, channel])
            )
            if len(group) <= 2 * fs and bounded:
                if good.size > 1:
                    od[group, channel] = np.interp(t[group], t[good], od[good, channel])
                    interpolated += int(group.size)
            else:
                time_bad[group] = True
    std = np.nanstd(od, axis=0)
    reference = np.nanmedian(std[np.isfinite(std) & (std > 0)])
    ratio = std / (reference if np.isfinite(reference) and reference > 0 else np.nan)
    constant = (~valid_mad) | (std <= 1e-12)
    invalid = constant | (~np.isfinite(ratio)) | (ratio > 10) | (ratio < 0.1)
    return {
        "od": od, "abnormal": abnormal, "time_bad": time_bad,
        "std_ratio": ratio, "constant": constant, "invalid": invalid,
        "segments": mask_segments(time_bad, t, fs),
        "nan": int(np.isnan(od).sum()), "inf": int(np.isinf(od).sum()),
        "outliers": int(outlier.sum()), "interpolated": interpolated,
        "invalid_time_samples": int(time_bad.sum()),
    }


def resample_valid_segments(arr, valid_time, source_fs, up, down, n_out):
    values = np.asarray(arr, dtype=np.float64)
    valid = np.asarray(valid_time, dtype=bool)
    if values.ndim != 2 or values.shape[0] != valid.size:
        raise ValueError(f"segment resample shape mismatch: arr={values.shape}, mask={valid.shape}")
    output = np.full((n_out, values.shape[1]), np.nan, dtype=np.float64)
    valid_output = np.zeros(n_out, dtype=bool)
    ids = np.flatnonzero(valid)
    groups = np.split(ids, np.where(np.diff(ids) > 1)[0] + 1) if ids.size else []
    singleton_target_ranges = []
    for group in groups:
        if group.size == 1:
            target_start = int(round(int(group[0]) * TARGET_FS / source_fs))
            target_length = (up + down - 1) // down
            target_stop = min(n_out, target_start + target_length)
            if target_stop > target_start:
                singleton_target_ranges.append((target_start, target_stop))
            continue
        start, stop = int(group[0]), int(group[-1]) + 1
        segment = resample_poly(values[start:stop], up, down, axis=0, padtype="line")
        target_start = int(round(start * TARGET_FS / source_fs))
        target_stop = min(n_out, target_start + segment.shape[0])
        length = target_stop - target_start
        if length <= 0:
            continue
        output[target_start:target_stop] = segment[:length]
        valid_output[target_start:target_stop] = True
    for target_start, target_stop in singleton_target_ranges:
        output[target_start:target_stop] = np.nan
        valid_output[target_start:target_stop] = False
    return output, valid_output


def streamed_global_stats(arr, valid_t, valid_c):
    count, total, total_sq = 0, 0.0, 0.0
    columns = np.flatnonzero(valid_c)
    for start in range(0, arr.shape[0], 2048):
        block = np.asarray(arr[start:start + 2048], dtype=np.float64)
        use = valid_t[start:start + len(block)]
        if use.any():
            values = block[use][:, columns]
            finite_values = values[np.isfinite(values)]
            count += int(finite_values.size)
            total += float(finite_values.sum())
            total_sq += float(np.square(finite_values).sum())
    if count < 2:
        raise RuntimeError("too few finite valid samples for recording-level z-score")
    mean = total / count
    std = math.sqrt(max(0.0, total_sq / count - mean * mean))
    if not np.isfinite(std) or std <= 1e-12:
        raise RuntimeError("recording-level z-score standard deviation is zero/nonfinite")
    return mean, std, count
