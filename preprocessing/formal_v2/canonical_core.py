from __future__ import annotations

import math
from collections import defaultdict
from fractions import Fraction
from typing import Callable, Iterable

import numpy as np
from mne.preprocessing.nirs._beer_lambert_law import _load_absorption

TARGET_FS = 5.0
PPF = 6.0
CHANNEL_MAPPING_SCHEMA = "n_wavelength_channel_mapping_v1"
from .legacy_primitives import (
    qc as canonical_qc,
    resample_valid_segments,
    streamed_global_stats,
)


def distance_to_m(distance: float, length_unit: str) -> float:
    unit = str(length_unit).strip().lower()
    if unit in ("m", "meter", "meters"):
        return float(distance)
    if unit in ("mm", "millimeter", "millimeters"):
        return float(distance) / 1000.0
    if unit in ("cm", "centimeter", "centimeters"):
        return float(distance) / 100.0
    raise ValueError(f"unsupported source-detector length unit: {length_unit!r}")


def positive_median_od(intensity: np.ndarray):
    values = np.asarray(intensity, dtype=np.float64)
    positive = np.isfinite(values) & (values > 0)
    baseline = np.nanmedian(np.where(positive, values, np.nan), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        od = -np.log(values / baseline)
    return od, baseline, positive


def solve_pair(
    optical_density: np.ndarray,
    wavelengths_nm: np.ndarray,
    distance_m: float,
    ppf: float = PPF,
):
    od = np.asarray(optical_density, dtype=np.float64)
    wavelengths = np.asarray(wavelengths_nm, dtype=np.float64).reshape(-1)
    if od.ndim != 2 or od.shape[1] != wavelengths.size:
        raise ValueError(f"OD/wavelength mismatch: {od.shape} vs {wavelengths.shape}")
    if wavelengths.size < 2:
        raise ValueError("MBLL requires at least two wavelengths")
    if not np.isfinite(wavelengths).all() or len(np.unique(wavelengths)) != wavelengths.size:
        raise ValueError("MBLL wavelengths must be finite and unique")
    if not np.isfinite(distance_m) or distance_m <= 0:
        raise ValueError(f"invalid source-detector distance: {distance_m}")
    if not np.isfinite(ppf) or ppf <= 0:
        raise ValueError(f"invalid PPF: {ppf}")
    extinction = _load_absorption(wavelengths)
    inverse = np.linalg.pinv(extinction * ppf)
    concentration = (inverse / distance_m) @ od.T
    return concentration.T, extinction, inverse


def _raw_groups(measurements: Iterable[dict], eligible: Callable[[dict], bool] | None):
    groups = defaultdict(list)
    ungrouped = []
    for measurement in measurements:
        item = dict(measurement)
        if eligible is not None and not eligible(item):
            item["adapter_exclusion_reason"] = "MEASUREMENT_NOT_ELIGIBLE"
            ungrouped.append(item)
            continue
        source = item.get("sourceIndex")
        detector = item.get("detectorIndex")
        if source is None or detector is None:
            item["adapter_exclusion_reason"] = "MISSING_SOURCE_OR_DETECTOR_INDEX"
            ungrouped.append(item)
            continue
        groups[(int(source), int(detector))].append(item)
    return groups, ungrouped


def structurally_solvable_columns(recording: dict):
    intensity = np.asarray(recording["intensity"], dtype=np.float64)
    source_positions = np.asarray(recording["source_positions"], dtype=float)
    detector_positions = np.asarray(recording["detector_positions"], dtype=float)
    groups, _ = _raw_groups(recording["measurements"], None)
    columns = []
    for (source, detector), items in groups.items():
        indices = [item.get("wavelengthIndex") for item in items]
        item_columns = [int(item["column"]) for item in items if "column" in item]
        unique_indices = {int(index) for index in indices if index is not None}
        mapping_ok = len(unique_indices) >= 2 and len(unique_indices) == len(indices) and len(item_columns) == len(items)
        geometry_ok = 1 <= source <= len(source_positions) and 1 <= detector <= len(detector_positions)
        if geometry_ok:
            a, b = source_positions[source - 1], detector_positions[detector - 1]
            geometry_ok = bool(np.isfinite(a).all() and np.isfinite(b).all() and np.linalg.norm(a - b) > 0)
        signal_ok = mapping_ok and all(
            0 <= column < intensity.shape[1] and np.any(np.isfinite(intensity[:, column]) & (intensity[:, column] > 0))
            for column in item_columns
        )
        if mapping_ok and geometry_ok and signal_ok:
            columns.extend(item_columns)
    columns = np.asarray(sorted(set(columns)), dtype=int)
    if columns.size == 0:
        raise RuntimeError("no structurally solvable measurement columns")
    return columns


def n_wavelength_mbll(
    optical_density: np.ndarray,
    measurements: Iterable[dict],
    wavelengths_nm: np.ndarray,
    source_positions: np.ndarray,
    detector_positions: np.ndarray,
    length_unit: str,
    invalid_measurement: np.ndarray | None = None,
    valid_time_mask: np.ndarray | None = None,
    ppf: float = PPF,
    eligible: Callable[[dict], bool] | None = None,
):
    od = np.asarray(optical_density, dtype=np.float64)
    wavelengths = np.asarray(wavelengths_nm, dtype=np.float64).reshape(-1)
    source_positions = np.asarray(source_positions, dtype=np.float64)
    detector_positions = np.asarray(detector_positions, dtype=np.float64)
    invalid = np.zeros(od.shape[1], dtype=bool) if invalid_measurement is None else np.asarray(invalid_measurement, dtype=bool)
    valid_time = np.ones(od.shape[0], dtype=bool) if valid_time_mask is None else np.asarray(valid_time_mask, dtype=bool)
    if od.ndim != 2 or invalid.shape != (od.shape[1],):
        raise ValueError("OD or invalid-measurement shape mismatch")
    if valid_time.shape != (od.shape[0],):
        raise ValueError("valid-time shape mismatch")
    if wavelengths.size < 2:
        raise ValueError("Formal V2 requires N >= 2 probe wavelengths")
    if source_positions.ndim != 2 or detector_positions.ndim != 2 or source_positions.shape[1] != detector_positions.shape[1]:
        raise ValueError("source/detector geometry shape mismatch")

    grouped, ungrouped = _raw_groups(measurements, eligible)
    hbo, hbr, valid, distances_native = [], [], [], []
    channels = []
    group_records = []
    inverse_cache = {}

    for pair in sorted(grouped):
        source, detector = pair
        items = grouped[pair]
        reasons = []
        by_wavelength = {}
        duplicates = []
        for item in items:
            index = item.get("wavelengthIndex")
            if index is None:
                reasons.append("MISSING_WAVELENGTH_INDEX")
                continue
            index = int(index)
            if index in by_wavelength:
                duplicates.append(index)
            else:
                by_wavelength[index] = item
        if duplicates:
            reasons.append("DUPLICATE_WAVELENGTH_INDEX:" + ",".join(map(str, sorted(set(duplicates)))))
        ordered = [by_wavelength[index] for index in sorted(by_wavelength)]
        indices = [int(item["wavelengthIndex"]) for item in ordered]
        columns = [int(item["column"]) for item in ordered]
        if len(ordered) < 2:
            reasons.append("FEWER_THAN_TWO_UNIQUE_WAVELENGTHS")
        if any(index < 1 or index > wavelengths.size for index in indices):
            reasons.append("WAVELENGTH_INDEX_OUT_OF_BOUNDS")
        if any(column < 0 or column >= od.shape[1] for column in columns):
            reasons.append("MEASUREMENT_COLUMN_OUT_OF_BOUNDS")
        pair_wavelengths = [float(wavelengths[index - 1]) for index in indices if 1 <= index <= wavelengths.size]
        signal_any_finite = bool(columns) and all(
            0 <= column < od.shape[1] and np.isfinite(od[:, column]).any() for column in columns
        )
        signal_finite_all = bool(columns) and all(
            0 <= column < od.shape[1] and np.isfinite(od[valid_time, column]).all() for column in columns
        )
        if columns and not signal_any_finite:
            reasons.append("NO_FINITE_OD_FOR_ONE_OR_MORE_WAVELENGTHS")

        distance_native = math.nan
        distance_m = math.nan
        if not (1 <= source <= len(source_positions) and 1 <= detector <= len(detector_positions)):
            reasons.append("GEOMETRY_INDEX_OUT_OF_BOUNDS")
        else:
            source_position = source_positions[source - 1]
            detector_position = detector_positions[detector - 1]
            if not np.isfinite(source_position).all() or not np.isfinite(detector_position).all():
                reasons.append("NONFINITE_GEOMETRY")
            else:
                distance_native = float(np.linalg.norm(source_position - detector_position))
                distance_m = distance_to_m(distance_native, length_unit)
                if not np.isfinite(distance_m) or distance_m <= 0:
                    reasons.append("NONPOSITIVE_OR_NONFINITE_DISTANCE")

        included = not reasons
        output_index = None
        if included:
            key = tuple(pair_wavelengths)
            if key not in inverse_cache:
                inverse_cache[key] = np.linalg.pinv(_load_absorption(pair_wavelengths) * ppf)
            concentration = (inverse_cache[key] / distance_m) @ od[:, columns].T
            output_index = len(channels)
            hbo.append(concentration[0])
            hbr.append(concentration[1])
            valid.append(bool(signal_finite_all and not invalid[columns].any()))
            distances_native.append(distance_native)
            channels.append({
                "channel_index": output_index,
                "source_index": source,
                "detector_index": detector,
                "measurement_columns": columns,
                "wavelength_indices": indices,
                "wavelength_nm": pair_wavelengths,
            })
        group_records.append({
            "source_index": source,
            "detector_index": detector,
            "measurement_columns": columns,
            "wavelength_indices": indices,
            "wavelength_nm": pair_wavelengths,
            "included_in_hb_output": included,
            "output_channel_index": output_index,
            "signal_valid": bool(signal_finite_all and columns and not invalid[columns].any()),
            "distance_valid": bool(np.isfinite(distance_m) and distance_m > 0),
            "distance_native": distance_native if np.isfinite(distance_native) else None,
            "distance_m": distance_m if np.isfinite(distance_m) else None,
            "exclusion_reasons": reasons,
        })

    for item in ungrouped:
        group_records.append({
            "source_index": item.get("sourceIndex"),
            "detector_index": item.get("detectorIndex"),
            "measurement_columns": [item.get("column")],
            "wavelength_indices": [item.get("wavelengthIndex")],
            "wavelength_nm": [],
            "included_in_hb_output": False,
            "output_channel_index": None,
            "signal_valid": False,
            "distance_valid": False,
            "distance_native": None,
            "distance_m": None,
            "exclusion_reasons": [item["adapter_exclusion_reason"]],
        })
    if not hbo:
        raise RuntimeError("N-wavelength MBLL produced no solvable physical channels")

    channel_mapping = {
        "schema": CHANNEL_MAPPING_SCHEMA,
        "source_detector_index_base": 1,
        "wavelength_index_base": 1,
        "measurement_column_index_base": 0,
        "channels": channels,
    }
    measurement_groups = {
        "schema": "n_wavelength_measurement_groups_v1",
        "source_detector_index_base": 1,
        "wavelength_index_base": 1,
        "measurement_column_index_base": 0,
        "groups": group_records,
    }
    return (
        np.asarray(hbo).T,
        np.asarray(hbr).T,
        np.asarray(valid, dtype=bool),
        np.asarray(distances_native, dtype=np.float64),
        channel_mapping,
        measurement_groups,
    )


def canonical_preprocess(recording: dict):
    intensity = np.asarray(recording["intensity"], dtype=np.float64)
    time = np.asarray(recording["time"], dtype=np.float64)
    fs = float(recording["sampling_rate"])
    od, baseline, positive = positive_median_od(intensity)
    qc_columns = structurally_solvable_columns(recording)
    selected_qc = canonical_qc(od[:, qc_columns].copy(), time, fs)
    qc_od = od.copy()
    qc_od[:, qc_columns] = selected_qc["od"]
    invalid_full = np.ones(od.shape[1], dtype=bool)
    invalid_full[qc_columns] = selected_qc["invalid"]
    qc = dict(selected_qc)
    qc["od"] = qc_od
    qc["invalid"] = invalid_full
    qc["structurally_solvable_measurement_columns"] = qc_columns
    hbo1, hbr1, valid_channel, distances, mapping, groups = n_wavelength_mbll(
        qc["od"], recording["measurements"], recording["wavelengths"],
        recording["source_positions"], recording["detector_positions"],
        recording["length_unit"], qc["invalid"], valid_time_mask=~qc["time_bad"], ppf=PPF,
    )
    ratio = Fraction(TARGET_FS / fs).limit_denominator(100000)
    n_out = math.ceil(hbo1.shape[0] * ratio.numerator / ratio.denominator)
    hbo, valid_time_hbo = resample_valid_segments(
        hbo1, ~qc["time_bad"], fs, ratio.numerator, ratio.denominator, n_out
    )
    hbr, valid_time_hbr = resample_valid_segments(
        hbr1, ~qc["time_bad"], fs, ratio.numerator, ratio.denominator, n_out
    )
    if not np.array_equal(valid_time_hbo, valid_time_hbr):
        raise RuntimeError("HbO/HbR valid-time mappings disagree")
    valid_time = valid_time_hbo
    hbo_mean, hbo_std, hbo_count = streamed_global_stats(hbo, valid_time, valid_channel)
    hbr_mean, hbr_std, hbr_count = streamed_global_stats(hbr, valid_time, valid_channel)
    hbo = ((hbo - hbo_mean) / hbo_std).astype("float32")
    hbr = ((hbr - hbr_mean) / hbr_std).astype("float32")
    events = []
    for original in recording.get("events", []):
        event = dict(original)
        try:
            event["sample_index_5hz"] = int(round(float(event["onset"]) * TARGET_FS))
            event["sample_index_in_bounds"] = bool(0 <= event["sample_index_5hz"] < n_out)
        except (KeyError, TypeError, ValueError):
            event["sample_index_5hz"] = None
            event["sample_index_in_bounds"] = False
        events.append(event)
    return {
        "od": od, "baseline": baseline, "positive": positive, "qc": qc,
        "hbo": hbo, "hbr": hbr, "valid_channel_mask": valid_channel,
        "valid_time_mask": valid_time, "source_detector_distance": distances,
        "channel_mapping_json": mapping, "measurement_groups_json": groups,
        "events": events, "output_samples": n_out,
        "normalization": {
            "hbo_mean": hbo_mean, "hbo_std": hbo_std, "hbo_count": hbo_count,
            "hbr_mean": hbr_mean, "hbr_std": hbr_std, "hbr_count": hbr_count,
            "count": 1,
        },
        "resample": {
            "up": ratio.numerator, "down": ratio.denominator,
            "invalid_samples_sent_to_resample_poly": 0,
        },
    }
