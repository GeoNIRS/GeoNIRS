from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


EXPECTED_WAVELENGTHS = {
    "DS44": (760.0, 850.0),
    "DS55": (757.0, 843.0),
    "DS58": (757.0, 843.0),
}


def _scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray):
        return _scalar(value.reshape(-1)[0]) if value.size else ""
    return value.item() if isinstance(value, np.generic) else value


def _numbered(keys, prefix):
    return sorted(
        (key for key in keys if key.startswith(prefix)),
        key=lambda key: int(re.sub(r"\D", "", key) or 0),
    )


def _sidecar_metadata(path: Path):
    sidecar = (
        path.with_name(path.name.replace("_nirs.snirf", "_nirs.json"))
        if path.name.endswith("_nirs.snirf")
        else path.with_suffix(".json")
    )
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise ValueError(f"invalid JSON sidecar {sidecar}: {exc}") from exc


def _stim_events(nirs_group):
    events = []
    for key in _numbered(nirs_group.keys(), "stim"):
        group = nirs_group[key]
        name = str(_scalar(group["name"][()])) if "name" in group else key
        values = np.asarray(group["data"][()], dtype=float) if "data" in group else np.empty((0, 3))
        if values.ndim == 1:
            values = values[None, :]
        for row in values:
            events.append({
                "event": name,
                "label": name,
                "onset": float(row[0]) if row.size else float("nan"),
                "duration": float(row[1]) if row.size > 1 else 0.0,
                "value": float(row[2]) if row.size > 2 else float("nan"),
                "source": "snirf_stim",
                "time_unit": "seconds",
            })
    return events


def _sidecar_events(path: Path):
    sidecar = (
        path.with_name(path.name.replace("_nirs.snirf", "_events.tsv"))
        if path.name.endswith("_nirs.snirf")
        else path.with_name(path.stem + "_events.tsv")
    )
    if not sidecar.exists():
        return []
    events = []
    with sidecar.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            event = {str(key): str(value) for key, value in row.items()}
            for key in ("onset", "duration"):
                try:
                    event[key] = float(event[key])
                except (KeyError, TypeError, ValueError):
                    pass
            event.update(source="events_tsv", time_unit="seconds")
            event.setdefault("event", event.get("trial_type", event.get("value", "")))
            event.setdefault("label", event.get("trial_type", event.get("value", "")))
            events.append(event)
    return events


def _identity(path: Path, dataset: str):
    text = path.as_posix()
    subject = ""
    session = ""
    run = ""
    task = ""
    match = re.search(r"/(sub-[^/]+)", text)
    if match:
        subject = match.group(1)
    match = re.search(r"[_/](ses-[^_/.]+)", text)
    if match:
        session = match.group(1)
    match = re.search(r"[_/](run-[^_/.]+)", text)
    if match:
        run = match.group(1)
    match = re.search(r"[_/](task-[^_/.]+)", text)
    if match:
        task = match.group(1)
    if dataset == "DS44":
        match = re.search(r"Subject(\d+)_day(\d+)_session(\d+)_raw", path.stem)
        if not match:
            raise ValueError(f"cannot parse DS44 identity: {path.name}")
        subject, session, run, task = (
            f"Subject{match.group(1)}",
            f"day{match.group(2)}",
            f"session{match.group(3)}",
            "stroop",
        )
    elif dataset == "DS55":
        subject = path.parent.name



        condition = path.parent.parent.parent.name
        session = condition
        task = condition
        recording_id = f"{subject}_{condition}_{path.stem}"
    elif dataset == "DS58":
        session = "Pre_Ctrl_Images" if "Pre_Ctrl_Images" in path.parts else "Pre_Interv_Images"
    identity = {"subject": subject, "session": session, "run": run, "task": task}
    if dataset == "DS55":
        identity["recording_id"] = recording_id
    return identity


def _validated_length_unit(value):
    raw = str(value).strip()
    key = raw.lower()
    aliases = {
        "m": "m", "meter": "m", "meters": "m",
        "mm": "mm", "millimeter": "mm", "millimeters": "mm",
        "cm": "cm", "centimeter": "cm", "centimeters": "cm",
    }
    if key not in aliases:
        raise ValueError(f"unsupported or ambiguous LengthUnit: {raw!r}")
    return raw, aliases[key]


@dataclass
class TwoWavelengthSnirfRecording:
    dataset: str
    path: Path
    intensity: np.ndarray
    time: np.ndarray
    sampling_rate: float
    sampling_rate_source: str
    measurements: list[dict]
    wavelengths: np.ndarray
    source_positions: np.ndarray
    detector_positions: np.ndarray
    geometry_dimension: int
    length_unit_raw: str
    length_unit: str
    events: list[dict]
    metadata: dict
    identity: dict
    physical_pairs: list[tuple[int, int]]

    @property
    def mbll_distance_geometry(self):
        return f"REAL_{self.geometry_dimension}D_WITH_LENGTHUNIT"

    @property
    def model_3d_geometry(self):
        return "REAL_3D_AVAILABLE" if self.geometry_dimension == 3 else "NOT_AVAILABLE"


def read_two_wavelength_snirf(path: str | Path, dataset: str):
    path = Path(path).resolve()
    if dataset not in EXPECTED_WAVELENGTHS:
        raise ValueError(f"unsupported pilot dataset: {dataset}")
    sidecar_metadata = _sidecar_metadata(path)
    with h5py.File(path, "r") as handle:
        if "nirs/data1" not in handle:
            raise ValueError("missing nirs/data1")
        data_group = handle["nirs/data1"]
        intensity = np.asarray(data_group["dataTimeSeries"][()], dtype=np.float64)
        time = np.asarray(data_group["time"][()], dtype=np.float64).reshape(-1)
        measurements = []
        for column, key in enumerate(_numbered(data_group.keys(), "measurementList")):
            group = data_group[key]
            item = {"column": column}
            for field in ("sourceIndex", "detectorIndex", "wavelengthIndex", "dataType", "dataTypeIndex"):
                if field in group:
                    item[field] = int(float(_scalar(group[field][()])))
            item["dataTypeLabel"] = str(_scalar(group["dataTypeLabel"][()])) if "dataTypeLabel" in group else ""
            item["dataUnit"] = str(_scalar(group["dataUnit"][()])) if "dataUnit" in group else ""
            measurements.append(item)
        probe = handle["nirs/probe"]
        wavelengths = np.asarray(probe["wavelengths"][()], dtype=float).reshape(-1)
        if "sourcePos3D" in probe and "detectorPos3D" in probe:
            source_positions = np.asarray(probe["sourcePos3D"][()], dtype=float)
            detector_positions = np.asarray(probe["detectorPos3D"][()], dtype=float)
            geometry_dimension = 3
        elif "sourcePos2D" in probe and "detectorPos2D" in probe:
            source_positions = np.asarray(probe["sourcePos2D"][()], dtype=float)
            detector_positions = np.asarray(probe["detectorPos2D"][()], dtype=float)
            geometry_dimension = 2
        else:
            raise ValueError("no complete real 3D or 2D source/detector geometry")
        if "nirs/metaDataTags/LengthUnit" not in handle:
            raise ValueError("missing LengthUnit; geometry distance is ambiguous")
        length_unit_raw, length_unit = _validated_length_unit(
            _scalar(handle["nirs/metaDataTags/LengthUnit"][()])
        )



        try:
            from adapters import normalise_geometry_unit
        except ImportError:
            from .formal_v2.adapters import normalise_geometry_unit
        source_positions, detector_positions, length_unit_raw = normalise_geometry_unit(
            dataset, source_positions, detector_positions, length_unit_raw
        )
        length_unit = length_unit_raw
        stim_events = _stim_events(handle["nirs"])

    if intensity.ndim != 2 or intensity.shape != (time.size, len(measurements)):
        raise ValueError(
            f"SNIRF dimension disagreement: data={intensity.shape}, time={time.shape}, "
            f"measurementList={len(measurements)}"
        )
    if time.size < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("time vector is not finite and strictly increasing")
    if wavelengths.size != 2 or not np.allclose(wavelengths, EXPECTED_WAVELENGTHS[dataset], atol=0.51):
        raise ValueError(
            f"wavelength mismatch for {dataset}: observed={wavelengths.tolist()} "
            f"expected={EXPECTED_WAVELENGTHS[dataset]}"
        )
    data_types = {item.get("dataType") for item in measurements}
    if data_types != {1}:
        raise ValueError(f"input is not unambiguous raw intensity: dataType={sorted(data_types, key=str)}")
    if source_positions.ndim != 2 or source_positions.shape[1] != geometry_dimension:
        raise ValueError("invalid source geometry shape")
    if detector_positions.ndim != 2 or detector_positions.shape[1] != geometry_dimension:
        raise ValueError("invalid detector geometry shape")
    if not np.isfinite(source_positions).all() or not np.isfinite(detector_positions).all():
        raise ValueError("non-finite geometry")

    by_pair = {}
    for item in measurements:
        required = ("sourceIndex", "detectorIndex", "wavelengthIndex")
        if any(field not in item for field in required):
            raise ValueError(f"incomplete measurementList item: {item}")
        pair = (item["sourceIndex"], item["detectorIndex"])
        by_pair.setdefault(pair, []).append(item["wavelengthIndex"])
    expected_indices = [1, 2]
    bad_pairs = {pair: sorted(indices) for pair, indices in by_pair.items() if sorted(indices) != expected_indices}
    if bad_pairs:
        raise ValueError(f"physical pairs do not have exactly one sample of each wavelength: {bad_pairs}")
    if any(si < 1 or si > len(source_positions) or di < 1 or di > len(detector_positions) for si, di in by_pair):
        raise ValueError("measurementList source/detector index is out of geometry bounds")

    inferred_fs = 1.0 / float(np.median(np.diff(time)))
    declared_fs = sidecar_metadata.get("SamplingFrequency")
    try:
        declared_fs = float(declared_fs)
    except (TypeError, ValueError):
        declared_fs = float("nan")
    if np.isfinite(declared_fs) and declared_fs > 0:
        sampling_rate = declared_fs
        sampling_rate_source = "sidecar_SamplingFrequency"
    else:
        sampling_rate = inferred_fs
        sampling_rate_source = "median_time_interval"

    events = _sidecar_events(path) or stim_events
    return TwoWavelengthSnirfRecording(
        dataset=dataset,
        path=path,
        intensity=intensity,
        time=time,
        sampling_rate=sampling_rate,
        sampling_rate_source=sampling_rate_source,
        measurements=measurements,
        wavelengths=wavelengths,
        source_positions=source_positions,
        detector_positions=detector_positions,
        geometry_dimension=geometry_dimension,
        length_unit_raw=length_unit_raw,
        length_unit=length_unit,
        events=events,
        metadata=sidecar_metadata,
        identity=_identity(path, dataset),
        physical_pairs=sorted(by_pair),
    )
