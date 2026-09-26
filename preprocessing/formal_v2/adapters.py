from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio

EXPECTED_WAVELENGTHS = {
    "DS2": (785.0, 830.0),
    "DS5": (760.0, 850.0),
    "DS8": (760.0, 850.0),
    "DS7": (730.0, 808.0, 850.0),
    "DS47": (780.0, 805.0, 830.0),
    "DS59": (757.0, 843.0),
    "DS51": (756.0, 853.0),
    "DS30": (760.0, 850.0),

    "DS36": (760.0, 850.0),
    "DS4": (760.0, 850.0),
    "DS39": (690.0, 830.0),
    "DS11": (695.0, 830.0),
    "DS49": (785.0, 830.0),
    "DS42": (690.0, 830.0),
    "DS19": (690.0, 830.0),
    "DS68": (760.0, 850.0),
    "DS69": (703.7, 829.4),


    "DS70": (760.0, 850.0),
}


















OPTICAL_DENSITY_DATASETS = {"DS51"}












CENTIMETRE_MISLABELLED = {"DS5", "DS49", "DS55", "DS58", "DS59"}
CENTIMETRE_TO_MM = 10.0


def normalise_geometry_unit(dataset, source_positions, detector_positions, length_unit):
    if dataset in CENTIMETRE_MISLABELLED and str(length_unit).strip().lower() == "mm":
        return (
            np.asarray(source_positions, dtype=float) * CENTIMETRE_TO_MM,
            np.asarray(detector_positions, dtype=float) * CENTIMETRE_TO_MM,
            "mm",
        )
    return source_positions, detector_positions, length_unit


def scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    array = np.asarray(value)
    if array.size == 1:
        return scalar(array.reshape(-1)[0]) if array.ndim else array.item()
    return value


def numbered(keys, prefix):
    return sorted(
        (key for key in keys if key.startswith(prefix)),
        key=lambda key: int(re.sub(r"\D", "", key) or 0),
    )


def sidecar_json(path):
    candidate = path.with_name(path.name.replace("_nirs.snirf", "_nirs.json"))
    if candidate == path or not candidate.exists():
        return {}
    return json.loads(candidate.read_text(encoding="utf-8-sig"))


def sidecar_events(path):
    candidate = path.with_name(path.name.replace("_nirs.snirf", "_events.tsv"))
    if candidate == path or not candidate.exists():
        return []
    events = []
    with candidate.open(newline="", encoding="utf-8-sig") as handle:
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


def stim_events(nirs):
    events = []
    for key in numbered(nirs.keys(), "stim"):
        group = nirs[key]
        name = str(scalar(group["name"][()])) if "name" in group else key
        data = np.asarray(group["data"][()], dtype=float) if "data" in group else np.empty((0, 3))
        if data.ndim == 1:
            data = data[None, :]
        for row in data:
            events.append({
                "event": name, "label": name,
                "onset": float(row[0]) if row.size else float("nan"),
                "duration": float(row[1]) if row.size > 1 else 0.0,
                "value": float(row[2]) if row.size > 2 else float("nan"),
                "source": "snirf_stim", "time_unit": "seconds",
            })
    return events


def geometry_unit(value):
    raw = str(value).strip()
    aliases = {
        "m": "m", "meter": "m", "meters": "m",
        "mm": "mm", "millimeter": "mm", "millimeters": "mm",
        "cm": "cm", "centimeter": "cm", "centimeters": "cm",
    }
    if raw.lower() not in aliases:
        raise ValueError(f"unsupported or ambiguous LengthUnit: {value!r}")
    return raw, aliases[raw.lower()]


def _bids_identity(path, dataset):
    if dataset == "DS68":



        return {
            "dataset": dataset,
            "subject": path.parent.name,
            "session": "",
            "run": "",
            "task": "",
            "recording_id": path.stem,
        }
    text = path.as_posix()
    subject_match = re.search(r"/(sub-[^/]+)", text)
    task_match = re.search(r"[_/](task-[^_/.]+)", text)
    session_match = re.search(r"/(ses-[^/]+)", text)
    run_match = re.search(r"[_/](run-[^_/.]+)", text)
    task = task_match.group(1) if task_match else ""
    return {
        "dataset": dataset,
        "subject": subject_match.group(1) if subject_match else "",
        "session": session_match.group(1) if session_match else "",
        "run": run_match.group(1) if run_match else "",
        "task": task,
        "recording_id": path.stem,
    }


def read_nirs(path: str | Path, dataset: str):
    def field(sd, name):
        arr = np.asarray(sd[name])
        while arr.dtype == object and arr.size == 1:
            arr = np.asarray(arr.ravel()[0])
        return arr

    def positions(sd, name):
        arr = field(sd, name).astype(float)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional")
        if arr.shape[1] == 3:
            return arr
        if arr.shape[0] == 3:
            return arr.T
        raise ValueError(f"{name} has neither 3 rows nor 3 columns: {arr.shape}")

    def measurement_table(sd):
        arr = field(sd, "MeasList").astype(float)
        if arr.ndim != 2:
            raise ValueError("MeasList must be two-dimensional")
        return arr if arr.shape[1] >= 4 else arr.T

    path = Path(path).resolve()
    metadata = sidecar_json(path)
    try:
        from scipy.io import loadmat
        mat = loadmat(path)
        data = np.asarray(mat["d"], dtype=np.float64)
        time = np.asarray(mat["t"], dtype=np.float64).reshape(-1)
        sd = mat["SD"][0, 0]
        wavelengths = field(sd, "Lambda").astype(float).reshape(-1)
        sources = positions(sd, "SrcPos")
        detectors = positions(sd, "DetPos")
        table = measurement_table(sd)
    except NotImplementedError:
        with h5py.File(path, "r") as handle:
            data = np.asarray(handle["d"][()], dtype=np.float64)
            time = np.asarray(handle["t"][()], dtype=np.float64).reshape(-1)
            sd = handle["SD"]
            wavelengths = field(sd, "Lambda").astype(float).reshape(-1)
            sources = positions(sd, "SrcPos")
            detectors = positions(sd, "DetPos")
            table = measurement_table(sd)
    if data.ndim != 2:
        raise ValueError("d must be two-dimensional")
    if data.shape[0] == time.size:
        pass
    elif data.shape[1] == time.size:
        data = data.T
    else:
        raise ValueError(f"nirs dimension mismatch: {data.shape} vs {time.size}")
    measurements = []
    for column in range(data.shape[1]):
        row = table[column] if column < len(table) else (0, 0, 0, 1)
        measurements.append({
            "column": column,
            "sourceIndex": int(row[0]),
            "detectorIndex": int(row[1]),
            "wavelengthIndex": int(row[3]),
            "dataType": 1,
            "dataTypeLabel": "",
        })
    if time.size < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("invalid time vector")
    if dataset not in EXPECTED_WAVELENGTHS or not np.allclose(wavelengths, EXPECTED_WAVELENGTHS[dataset], atol=0.51):
        raise ValueError(f"unexpected {dataset} wavelengths: {wavelengths.tolist()}")
    length_unit_raw, length_unit = geometry_unit("mm")
    sources, detectors, length_unit_raw = normalise_geometry_unit(
        dataset, sources, detectors, length_unit_raw
    )
    length_unit = length_unit_raw
    try:
        identity = _bids_identity(path, dataset)
    except Exception:
        identity = {}
    return {
        "dataset": dataset, "path": path, "intensity": data, "time": time,
        "sampling_rate": 1.0 / float(np.median(np.diff(time))),
        "measurements": measurements, "wavelengths": wavelengths,
        "source_positions": sources, "detector_positions": detectors,
        "geometry_dimension": 3 if sources.shape[1] >= 3 else 2,
        "length_unit": length_unit, "length_unit_raw": length_unit_raw,
        "events": sidecar_events(path) or [],
        "metadata": metadata, "identity": identity,
        "signal_type": "raw_intensity",
    }


def read_snirf(path: str | Path, dataset: str):
    path = Path(path).resolve()
    metadata = sidecar_json(path)
    with h5py.File(path, "r") as handle:
        group = handle["nirs/data1"]
        intensity = np.asarray(group["dataTimeSeries"][()], dtype=np.float64)
        time = np.asarray(group["time"][()], dtype=np.float64).reshape(-1)
        measurements = []
        for column, key in enumerate(numbered(group.keys(), "measurementList")):
            item = {"column": column}
            for field in ("sourceIndex", "detectorIndex", "wavelengthIndex", "dataType", "dataTypeIndex"):
                if field in group[key]:
                    item[field] = int(float(scalar(group[key][field][()])))
            item["dataTypeLabel"] = str(scalar(group[key]["dataTypeLabel"][()])) if "dataTypeLabel" in group[key] else ""
            measurements.append(item)
        probe = handle["nirs/probe"]
        wavelengths = np.asarray(probe["wavelengths"][()], dtype=float).reshape(-1)
        if "sourcePos3D" in probe and "detectorPos3D" in probe:
            source_positions = np.asarray(probe["sourcePos3D"][()], dtype=float)
            detector_positions = np.asarray(probe["detectorPos3D"][()], dtype=float)
            dimension = 3
        elif "sourcePos2D" in probe and "detectorPos2D" in probe:
            source_positions = np.asarray(probe["sourcePos2D"][()], dtype=float)
            detector_positions = np.asarray(probe["detectorPos2D"][()], dtype=float)
            dimension = 2
        else:
            raise ValueError("no complete real source/detector geometry")
        if "nirs/metaDataTags/LengthUnit" not in handle:
            raise ValueError("missing LengthUnit")
        length_unit_raw, length_unit = geometry_unit(scalar(handle["nirs/metaDataTags/LengthUnit"][()]))
        source_positions, detector_positions, length_unit_raw = normalise_geometry_unit(
            dataset, source_positions, detector_positions, length_unit_raw
        )
        length_unit = length_unit_raw
        internal_events = stim_events(handle["nirs"])
    if intensity.ndim != 2 or intensity.shape != (time.size, len(measurements)):
        raise ValueError(f"SNIRF dimension mismatch: {intensity.shape}, {time.shape}, {len(measurements)}")
    if time.size < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("invalid time vector")
    if dataset not in EXPECTED_WAVELENGTHS or not np.allclose(wavelengths, EXPECTED_WAVELENGTHS[dataset], atol=0.51):
        raise ValueError(f"unexpected {dataset} wavelengths: {wavelengths.tolist()}")
    data_types = {item.get("dataType") for item in measurements}
    if dataset in OPTICAL_DENSITY_DATASETS:

        if data_types & {1}:
            raise ValueError(f"{dataset} is declared as optical density but the "
                             "file carries raw intensity")
        stored_od = intensity
        intensity = np.exp(-stored_od)
        signal_type = "optical_density"
    else:
        if data_types != {1}:
            raise ValueError("input is not unambiguous raw intensity")
        signal_type = "raw_intensity"
    declared_fs = metadata.get("SamplingFrequency")
    try:
        declared_fs = float(declared_fs)
    except (TypeError, ValueError):
        declared_fs = float("nan")
    fs = declared_fs if np.isfinite(declared_fs) and declared_fs > 0 else 1.0 / float(np.median(np.diff(time)))
    return {
        "dataset": dataset, "path": path, "intensity": intensity, "time": time,
        "sampling_rate": fs, "measurements": measurements,
        "wavelengths": wavelengths, "source_positions": source_positions,
        "detector_positions": detector_positions, "geometry_dimension": dimension,
        "length_unit": length_unit, "length_unit_raw": length_unit_raw,
        "events": sidecar_events(path) or internal_events, "metadata": metadata,
        "identity": _bids_identity(path, dataset),
        "signal_type": signal_type,
    }


def read_ds8_zip_entry(archive: str | Path, entry: str):
    archive, entry = Path(archive), str(entry)
    with zipfile.ZipFile(archive) as z:
        info = z.getinfo(entry)
        payload = z.read(info)
    with h5py.File(io.BytesIO(payload), "r") as handle:
        group = handle["nirs/data1"]
        intensity = np.asarray(group["dataTimeSeries"][()], dtype=np.float64)
        time = np.asarray(group["time"][()], dtype=np.float64).reshape(-1)
        measurements = []
        for column, key in enumerate(numbered(group.keys(), "measurementList")):
            item = {"column": column}
            for field in ("sourceIndex", "detectorIndex", "wavelengthIndex", "dataType", "dataTypeIndex"):
                if field in group[key]: item[field] = int(float(scalar(group[key][field][()])))
            item["dataTypeLabel"] = str(scalar(group[key]["dataTypeLabel"][()])) if "dataTypeLabel" in group[key] else ""
            measurements.append(item)
        probe = handle["nirs/probe"]
        wavelengths = np.asarray(probe["wavelengths"][()], dtype=float).reshape(-1)
        source_positions = np.asarray(probe["sourcePos3D"][()], dtype=float)
        detector_positions = np.asarray(probe["detectorPos3D"][()], dtype=float)
        length_unit_raw, length_unit = geometry_unit(scalar(handle["nirs/metaDataTags/LengthUnit"][()]))
        internal_events = stim_events(handle["nirs"])
        metadata = {}
        if "nirs/metaDataTags" in handle:
            metadata = {k: str(scalar(handle["nirs/metaDataTags"][k][()])) for k in handle["nirs/metaDataTags"]}
    if intensity.ndim == 2 and intensity.shape == (len(measurements), time.size):
        intensity = intensity.T
    if intensity.ndim != 2 or intensity.shape != (time.size, len(measurements)): raise ValueError("DS8 SNIRF dimension mismatch")
    if time.size < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0): raise ValueError("DS8 invalid time axis")
    if not np.allclose(wavelengths, EXPECTED_WAVELENGTHS["DS8"], atol=0.51): raise ValueError(f"DS8 wavelengths {wavelengths.tolist()}")
    if {item.get("dataType") for item in measurements} != {1}: raise ValueError("DS8 input is not raw-DC/OD")
    distributed_od = intensity
    intensity = np.exp(-distributed_od)
    fs = 1.0 / float(np.median(np.diff(time)))
    stem = Path(entry).stem
    match = re.fullmatch(r"([LR]_\d+)_session(\d+)_(left|right)(\d+)", stem)
    if not match: raise ValueError(f"DS8 identity parse failed: {entry}")
    subject, session, side, run = match.groups()
    events = []


    event_entry = f"events/{stem}_evt.txt"
    with zipfile.ZipFile(archive) as z:
        if event_entry in z.namelist():
            for i, line in enumerate(z.read(event_entry).decode("utf-8").splitlines()):
                fields = line.split()
                if len(fields) >= 3:
                    sample = float(fields[0]); value = float(fields[2])
                    index = int(round(sample)); onset = float(time[index]) if 0 <= index < len(time) else sample / fs
                    events.append({"event": "paired_evt", "label": "paired_evt", "onset": onset, "duration": 0.0, "value": value, "sample_index_raw": sample, "event_entry": event_entry, "source": "paired_event_metadata", "time_unit": "seconds"})
    if not events:
        events = list(internal_events)
    return {"dataset":"DS8", "path":f"{archive}::{entry}", "intensity":intensity, "time":time, "sampling_rate":fs,
            "measurements":measurements, "wavelengths":wavelengths, "source_positions":source_positions,
            "detector_positions":detector_positions, "geometry_dimension":3, "length_unit":length_unit,
            "length_unit_raw":length_unit_raw, "events":events, "metadata":metadata,
            "identity":{"dataset":"DS8","subject":subject,"session":f"session{session}","run":f"run-{run}","task":side,"recording_id":stem},
            "signal_type":"raw_intensity", "archive_entry":entry, "archive_sha256":hashlib.sha256(payload).hexdigest()}


def read_ds7_nirs(path: str | Path):
    path = Path(path).resolve()
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False,
                      variable_names=["d", "t", "ml", "s", "SD"])
    intensity = np.asarray(mat["d"], dtype=np.float64)
    time = np.asarray(mat["t"], dtype=np.float64).reshape(-1)
    ml = np.asarray(mat["ml"], dtype=np.int64)
    sd = mat["SD"]
    if intensity.ndim != 2 or intensity.shape != (time.size, ml.shape[0]):
        raise ValueError(f"DS7 dimension mismatch: {intensity.shape}, {time.shape}, {ml.shape}")
    if not np.array_equal(ml, np.asarray(sd.MeasList, dtype=np.int64)):
        raise ValueError("DS7 ml and SD.MeasList disagree")
    wavelengths = np.asarray(sd.Lambda, dtype=float).reshape(-1)
    if not np.allclose(wavelengths, EXPECTED_WAVELENGTHS["DS7"], atol=0.01):
        raise ValueError(f"unexpected DS7 wavelengths: {wavelengths.tolist()}")
    measurements = [
        {"sourceIndex": int(row[0]), "detectorIndex": int(row[1]),
         "dataType": int(row[2]), "wavelengthIndex": int(row[3]), "column": column}
        for column, row in enumerate(ml)
    ]
    source_positions = np.asarray(sd.SrcPos, dtype=float)
    detector_positions = np.asarray(sd.DetPos, dtype=float)
    if source_positions.ndim != 2 or detector_positions.ndim != 2 or source_positions.shape[1] != detector_positions.shape[1]:
        raise ValueError("DS7 source/detector geometry mismatch")
    length_unit_raw, length_unit = geometry_unit(str(sd.SpatialUnit))
    stim = np.asarray(mat["s"])
    if stim.ndim == 1:
        stim = stim[:, None]
    events = []
    for column in range(stim.shape[1]):
        for index in np.flatnonzero(stim[:, column] != 0):
            events.append({
                "event": f"marker_{column + 1}", "label": f"marker_{column + 1}",
                "value": float(stim[index, column]), "onset": float(time[index]),
                "duration": 0.0, "sample_index_raw": int(index),
                "source": "matlab_s", "time_unit": "seconds",
            })
    events.sort(key=lambda event: (event["onset"], event["event"]))
    match = re.search(r"subject(\d+)", path.as_posix())
    if not match:
        raise ValueError(f"cannot parse DS7 subject: {path}")
    cohort = "Normals" if "Normals" in path.parts else "Patients"
    identity = {
        "dataset": "DS7", "subject": ("H" if cohort == "Normals" else "I") + match.group(1),
        "session": cohort, "run": path.stem, "task": "motor-imagery",
        "recording_id": f"{cohort}_subject{match.group(1)}_run{path.stem}",
    }
    return {
        "dataset": "DS7", "path": path, "intensity": intensity, "time": time,
        "sampling_rate": 1.0 / float(np.median(np.diff(time))),
        "measurements": measurements, "wavelengths": wavelengths,
        "source_positions": source_positions, "detector_positions": detector_positions,
        "geometry_dimension": source_positions.shape[1], "length_unit": length_unit,
        "length_unit_raw": length_unit_raw, "events": events, "metadata": {},
        "identity": identity, "signal_type": "raw_intensity",
    }


DS16_REPORTED_SEPARATION_M = 0.03


def ds16_geometry(subject_dir: Path):
    mnt = sio.loadmat(subject_dir / "mnt.mat", squeeze_me=True,
                      struct_as_record=False)["mnt"]
    source_all = np.asarray(mnt.source.pos_3d, dtype=float).T
    detector = np.asarray(mnt.detector.pos_3d, dtype=float).T
    sd_original = np.asarray(mnt.sd, dtype=int)
    used_sources = sorted(set(int(x) for x in sd_original[:, 0]))
    if any(not np.isfinite(source_all[i - 1]).all() for i in used_sources):
        raise ValueError("mnt.sd references a non-finite source placeholder")
    source = source_all[np.asarray(used_sources) - 1]
    source_remap = {old: new for new, old in enumerate(used_sources, start=1)}
    sd = sd_original.copy()
    sd[:, 0] = [source_remap[int(x)] for x in sd_original[:, 0]]
    pair_unit = np.linalg.norm(source[sd[:, 0] - 1] - detector[sd[:, 1] - 1], axis=1)
    median_unit = float(np.median(pair_unit))
    if not np.isfinite(median_unit) or median_unit <= 0:
        raise ValueError("invalid normalized source-detector geometry")
    scale = DS16_REPORTED_SEPARATION_M / median_unit
    return source * scale, detector * scale, sd, {
        "original_coordinate_unit": "normalized unit-head template",
        "reported_source_detector_separation_m": DS16_REPORTED_SEPARATION_M,
        "uniform_coordinate_scale_to_m": scale,
        "unscaled_pair_distance_median": median_unit,
    }


def ds16_events(marker):
    labels = [str(x) for x in np.atleast_1d(marker.className)]
    codes = np.asarray(marker.event.desc, dtype=int).reshape(-1)
    times_ms = np.asarray(marker.time, dtype=float).reshape(-1)
    if len(codes) != len(times_ms):
        raise ValueError("DS16 marker time/description length mismatch")
    return [
        {
            "event": labels[int(code) - 1],
            "label": labels[int(code) - 1],
            "onset": float(ms / 1000.0),
            "duration": 0.0,
            "source": "mrk.mat:event.desc",
            "time_unit": "seconds",
            "sample_index_native": int(round(float(ms) * 10.0 / 1000.0)),
            "duration_unknown": True,
        }
        for ms, code in zip(times_ms, codes)
    ]


def read_ds16_mat(source: str | Path):
    text = str(source)
    if "::" not in text or "_session-" not in text:
        raise ValueError("DS16 source must be '<subject dir>/cnt.mat::<identity>_session-<n>'")
    cnt_path = Path(text.split("::", 1)[0])
    session_index = int(text.rsplit("_session-", 1)[1])
    subject_dir = cnt_path.parent

    cnts = np.atleast_1d(sio.loadmat(cnt_path, squeeze_me=True,
                                     struct_as_record=False)["cnt"])
    mrks = np.atleast_1d(sio.loadmat(subject_dir / "mrk.mat", squeeze_me=True,
                                     struct_as_record=False)["mrk"])
    cnt, marker = cnts[session_index], mrks[session_index]
    intensity = np.asarray(cnt.x, dtype=np.float64)
    fs = float(cnt.fs)
    wavelengths = np.asarray(cnt.wavelengths, dtype=float).reshape(-1)
    if intensity.ndim != 2 or intensity.shape[1] != 72 or wavelengths.tolist() != [760.0, 850.0]:
        raise ValueError(f"unexpected DS16 input: shape={intensity.shape}, "
                         f"wavelengths={wavelengths.tolist()}")
    channels = [str(v) for v in np.asarray(cnt.clab).reshape(-1)]
    if not (all(v.endswith("lowWL") for v in channels[:36])
            and all(v.endswith("highWL") for v in channels[36:])
            and [v[:-5] for v in channels[:36]] == [v[:-6] for v in channels[36:]]):
        raise ValueError("DS16 wavelength-pair ordering validation failed")

    source_positions, detector_positions, sd, geometry_meta = ds16_geometry(subject_dir)
    measurements = []
    for wavelength_index, offset in ((1, 0), (2, 36)):
        for pair_index, (source_index, detector_index) in enumerate(sd):
            measurements.append({
                "column": offset + pair_index,
                "sourceIndex": int(source_index),
                "detectorIndex": int(detector_index),
                "wavelengthIndex": wavelength_index,
                "dataType": 1,
                "dataTypeLabel": "raw intensity",
                "dataUnit": str(cnt.yUnit),
            })

    subject = subject_dir.name.replace("subject ", "sub-")
    session = f"ses-{session_index + 1:02d}"
    task = str(cnt.title)
    return {
        "dataset": "DS16",
        "path": cnt_path,
        "intensity": intensity,
        "time": np.arange(intensity.shape[0], dtype=float) / fs,
        "sampling_rate": fs,
        "measurements": measurements,
        "wavelengths": wavelengths,
        "source_positions": source_positions,
        "detector_positions": detector_positions,
        "geometry_dimension": 3,
        "length_unit": "m",
        "length_unit_raw": "m",
        "events": ds16_events(marker),
        "metadata": {
            "format": "MAT",
            "cnt_file": str(cnt_path),
            "mrk_file": str(subject_dir / "mrk.mat"),
            "mnt_file": str(subject_dir / "mnt.mat"),
            "session_index_zero_based": session_index,
            "wavelengths_nm": wavelengths.tolist(),
            "input_y_unit": str(cnt.yUnit),
            "geometry_conversion": geometry_meta,
        },
        "identity": {
            "dataset": "DS16",
            "subject": subject,
            "session": session,
            "run": "",
            "task": task,
            "recording_id": f"{subject}_{session}_task-{task}",
        },
        "signal_type": "raw intensity",
    }













ETG4000_ROWS = 3
ETG4000_COLS = 11
ETG4000_OPTODES = ETG4000_ROWS * ETG4000_COLS
ETG4000_CHANNELS = 52


def etg4000_probe_layout():
    rows = np.repeat(np.arange(ETG4000_ROWS), ETG4000_COLS)
    cols = np.tile(np.arange(ETG4000_COLS), ETG4000_ROWS)
    is_source = ((rows + cols) % 2 == 0)
    edges = []
    for row in range(ETG4000_ROWS):
        for col in range(ETG4000_COLS):
            here = row * ETG4000_COLS + col
            if col + 1 < ETG4000_COLS:
                edges.append(((row, col + 0.5), here, here + 1))
            if row + 1 < ETG4000_ROWS:
                edges.append(((row + 0.5, col), here, here + ETG4000_COLS))
    edges.sort(key=lambda edge: edge[0])
    pairs = []
    for _, first, second in edges:
        pairs.append((first, second) if is_source[first] else (second, first))
    if len(pairs) != ETG4000_CHANNELS:
        raise RuntimeError(f"ETG-4000 probe yields {len(pairs)} channels, expected 52")
    return is_source, pairs


def read_ds69_pos(path: str | Path):
    text = Path(path).read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    blocks = re.findall(
        r"\[Probe1-ch(\d+)\]\s*\nX=([-\d.]+)\s*\nY=([-\d.]+)\s*\nZ=([-\d.]+)", text
    )
    if len(blocks) != ETG4000_OPTODES:
        raise ValueError(f"expected {ETG4000_OPTODES} probe positions, found {len(blocks)}")
    order = np.argsort([int(block[0]) for block in blocks])
    positions = np.array([[float(blocks[i][axis]) for axis in (1, 2, 3)] for i in order])
    if not np.isfinite(positions).all():
        raise ValueError("non-finite probe position")
    return positions


def read_ds69_etg(path: str | Path):
    path = Path(path).resolve()
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    sampling_period = None
    header_index = None
    for index, line in enumerate(lines):
        if line.startswith("Sampling Period[s],"):
            sampling_period = float(line.split(",")[1])
        if line.startswith("Probe1,") and header_index is None:
            header_index = index
    if sampling_period is None or not np.isfinite(sampling_period) or sampling_period <= 0:
        raise ValueError("missing or invalid Sampling Period[s]")
    if header_index is None:
        raise ValueError("missing Probe1 column header")
    labels = [column for column in lines[header_index].split(",")[1:] if column.startswith("CH")]
    wavelengths_per_column = []
    for label in labels:
        match = re.search(r"\(([\d.]+)\)", label)
        if match is None:
            raise ValueError(f"channel column without a wavelength: {label!r}")
        wavelengths_per_column.append(float(match.group(1)))
    if len(labels) != 2 * ETG4000_CHANNELS:
        raise ValueError(f"expected 104 channel columns, found {len(labels)}")
    rows = []
    for line in lines[header_index + 1:]:
        if not line.strip():
            continue
        cells = line.split(",")
        if len(cells) < 1 + len(labels):
            raise ValueError("short data row in ETG-4000 export")
        rows.append([float(value) for value in cells[1:1 + len(labels)]])
    if len(rows) < 2:
        raise ValueError("not enough samples")
    intensity = np.asarray(rows, dtype=np.float64)

    wavelengths = np.array([
        float(np.median(wavelengths_per_column[0::2])),
        float(np.median(wavelengths_per_column[1::2])),
    ])
    if not np.allclose(wavelengths, EXPECTED_WAVELENGTHS["DS69"], atol=0.51):
        raise ValueError(f"unexpected DS69 wavelengths: {wavelengths.tolist()}")

    is_source, lattice_pairs = etg4000_probe_layout()
    positions = read_ds69_pos(path.with_suffix(".pos"))
    source_lattice = np.flatnonzero(is_source)
    detector_lattice = np.flatnonzero(~is_source)


    source_rank = {int(cell): rank + 1 for rank, cell in enumerate(source_lattice)}
    detector_rank = {int(cell): rank + 1 for rank, cell in enumerate(detector_lattice)}
    measurements = []
    for channel, (source_cell, detector_cell) in enumerate(lattice_pairs):
        for wavelength_index in (1, 2):
            measurements.append({
                "column": 2 * channel + wavelength_index - 1,
                "sourceIndex": source_rank[int(source_cell)],
                "detectorIndex": detector_rank[int(detector_cell)],
                "wavelengthIndex": wavelength_index,
                "dataType": 1,
                "dataTypeLabel": "",
            })

    time = np.arange(intensity.shape[0], dtype=np.float64) * sampling_period
    return {
        "dataset": "DS69",
        "path": path,
        "intensity": intensity,
        "time": time,
        "sampling_rate": 1.0 / sampling_period,
        "measurements": measurements,
        "wavelengths": wavelengths,
        "source_positions": positions[source_lattice],
        "detector_positions": positions[detector_lattice],
        "geometry_dimension": 3,
        "length_unit": "mm",
        "length_unit_raw": "mm",
        "events": [],
        "metadata": {},
        "identity": {
            "dataset": "DS69",
            "subject": path.stem,
            "session": "",
            "run": "",
            "task": "",
            "recording_id": path.stem,
        },
        "signal_type": "raw_intensity",
    }


def read_recording(path: str | Path, dataset: str):
    if dataset == "DS7":
        return read_ds7_nirs(path)
    if dataset == "DS16":
        return read_ds16_mat(path)
    if dataset == "DS69":
        return read_ds69_etg(path)
    if dataset in ("DS2", "DS5", "DS47", "DS59", "DS51", "DS30",
                   "DS36", "DS42", "DS19", "DS4", "DS68", "DS70"):
        return read_snirf(path, dataset)
    if dataset in ("DS39", "DS11", "DS49"):
        return read_nirs(path, dataset)
    raise ValueError(f"no canonical adapter for {dataset}")
