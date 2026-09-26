from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import OrderedDict, defaultdict
from pathlib import Path

import h5py
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "benchmark" / "benchmark.yaml"
DEFAULT_OUTPUT = ROOT / "benchmark/fair_input/GEONIRS_FAIR_INPUT.h5"
DEFAULT_RECORDS = ROOT / "benchmark/fair_input/RECORDS.json"
DEFAULT_INDEX_DIR = ROOT / "benchmark/fair_input/index"

SCHEMA = "formal_v2_policy_b_fair_input_v1"
POLICY = "B"




POLICY_NAMES = {
    "union": "RECORDING_SPECIFIC_SUPPORT",
    "intersection": "CHANNEL_INTERSECTION",
}
POLICY_NAME = POLICY_NAMES["union"]
MISSING_VALUE_FILL = 0.0
GEOMETRY_VARIANT = "dataset_level_C_normalized_midpoint_delta"
GEOMETRY_SCALE_M = 25.460582556839242 / 1000.0

UNIT_FACTORS = {
    "m": 1.0, "meter": 1.0, "meters": 1.0,
    "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
    "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001,
}
SPLIT_ROLE = {"train": 0, "validation": 1, "test": 2}
SPLIT_NAMES = {0: "train", 1: "validation", 2: "test"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_authority() -> dict:
    return yaml.safe_load(AUTHORITY.read_text(encoding="utf-8"))


def read_manifest(config: dict, dataset: str):
    item = config["datasets"][dataset]
    path = (ROOT / item["manifest"]).resolve()
    if digest(path) != item["manifest_sha256"]:
        raise ValueError(f"{dataset} manifest hash mismatch")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != int(item["manifest_rows"]):
        raise ValueError(f"{dataset} manifest row count mismatch")
    return path, rows


def json_scalar(handle: h5py.File, name: str):
    value = handle[name][()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


def read_channel_mapping(handle: h5py.File):
    mapping = json_scalar(handle, "channel_mapping_json")["channels"]
    by_identity: dict[tuple[int, int], int] = {}
    columns = set()
    for item in mapping:
        identity = (int(item["source_index"]), int(item["detector_index"]))
        column = int(item["channel_index"])
        if identity in by_identity:
            raise ValueError("duplicate physical source/detector identity")
        if column in columns:
            raise ValueError("duplicate canonical channel column")
        by_identity[identity] = column
        columns.add(column)
    return by_identity


def position_arrays(handle: h5py.File):
    if "source_pos3d" in handle:
        source = np.asarray(handle["source_pos3d"], dtype=np.float64)
        detector = np.asarray(handle["detector_pos3d"], dtype=np.float64)
    elif "source_pos2d" in handle:
        source = np.pad(np.asarray(handle["source_pos2d"], dtype=np.float64), ((0, 0), (0, 1)))
        detector = np.pad(np.asarray(handle["detector_pos2d"], dtype=np.float64), ((0, 0), (0, 1)))
    else:
        raise ValueError("canonical recording has no source/detector geometry")
    provenance = json_scalar(handle, "provenance_json")
    unit = str(provenance["geometry_unit"]).strip().lower()
    factor = UNIT_FACTORS.get(unit)
    if factor is None:
        raise ValueError(f"unsupported geometry unit: {unit}")
    return source * factor, detector * factor


def accepted_rows(rows):
    accepted = []
    reasons = defaultdict(int)
    cache: dict[str, h5py.File] = {}
    try:
        for row in rows:
            path = row["canonical_hdf5_path"]
            if path not in cache:
                cache[path] = h5py.File((ROOT / path).resolve(), "r")
            handle = cache[path]
            start, end = int(row["window_start_sample"]), int(row["window_end_sample"])
            time_valid = np.asarray(handle["valid_time_mask"][start:end], dtype=bool)
            hbo = np.asarray(handle["hbo"][start:end], dtype=np.float32)
            hbr = np.asarray(handle["hbr"][start:end], dtype=np.float32)
            invalid_time = not time_valid.all()
            non_finite = not (np.isfinite(hbo).all() and np.isfinite(hbr).all())
            if invalid_time:
                reasons["INVALID_TIME"] += 1
            if non_finite:
                reasons["NAN_SIGNAL"] += 1
            if invalid_time or non_finite:
                continue
            accepted.append(row)
    finally:
        for handle in cache.values():
            handle.close()
    return accepted, reasons


def dataset_time_points(rows) -> int:
    points = {int(row["window_end_sample"]) - int(row["window_start_sample"]) for row in rows}
    if len(points) != 1:
        raise ValueError(f"non-uniform window length: {sorted(points)}")
    return int(points.pop())


def canonical_identities(dataset: str, recordings, by_identity_map,
                         channel_policy: str = "union", static_valid_map=None):
    if channel_policy not in ("union", "intersection"):
        raise ValueError(f"unknown channel policy: {channel_policy}")
    if channel_policy == "intersection":



        valid_sets = []
        for relative in recordings:
            valid = static_valid_map[relative]
            valid_sets.append({identity for identity, column
                               in by_identity_map[relative].items()
                               if bool(valid[column])})
        identities = sorted(set.intersection(*valid_sets))
        if not identities:
            raise ValueError(f"{dataset}: channel intersection is empty")
    else:
        union = set()
        for relative in recordings:
            union.update(by_identity_map[relative])
        identities = sorted(union)
    return identities


def dataset_geometry(identities, recordings, by_identity_map):
    accum_source = np.zeros((len(identities), 3), dtype=np.float64)
    accum_detector = np.zeros((len(identities), 3), dtype=np.float64)
    counts = np.zeros(len(identities), dtype=np.int64)
    slot_of = {identity: slot for slot, identity in enumerate(identities)}
    for relative in recordings:
        with h5py.File((ROOT / relative).resolve(), "r") as handle:
            mapping = read_channel_mapping(handle)
            source, detector = position_arrays(handle)
        for identity, column in mapping.items():
            slot = slot_of.get(identity)
            if slot is None:
                continue
            source_index, detector_index = identity
            accum_source[slot] += source[source_index - 1]
            accum_detector[slot] += detector[detector_index - 1]
            counts[slot] += 1
    present = counts > 0
    source = np.zeros_like(accum_source)
    detector = np.zeros_like(accum_detector)
    source[present] = accum_source[present] / counts[present, None]
    detector[present] = accum_detector[present] / counts[present, None]
    endpoints = np.concatenate((source[present], detector[present]), axis=0)
    center = endpoints.mean(axis=0) if len(endpoints) else np.zeros(3)
    source = (source - center) / GEOMETRY_SCALE_M
    detector = (detector - center) / GEOMETRY_SCALE_M
    geometry = np.concatenate(((source + detector) / 2.0, detector - source), axis=1)
    geometry[~present] = 0.0
    if not np.isfinite(geometry).all():
        raise ValueError("non-finite dataset geometry")
    return geometry.astype(np.float32)


def split_roles(config: dict, dataset: str) -> dict:
    folds = config["datasets"][dataset]["folds"]
    roles: dict[str, dict[int, int]] = defaultdict(dict)
    for fold_key, fold in folds.items():
        fold_number = int(fold_key)
        for subject in fold["train_subjects"]:
            roles[subject][fold_number] = SPLIT_ROLE["train"]
        for subject in fold["validation_subjects"]:
            roles[subject][fold_number] = SPLIT_ROLE["validation"]
        for subject in fold["test_subjects"]:
            roles[subject][fold_number] = SPLIT_ROLE["test"]
    return roles


def build_dataset(output: h5py.File, config: dict, dataset: str, index_dir: Path,
                  channel_policy: str = "union",
                  policy_name: str = POLICY_NAME):
    manifest_path, rows = read_manifest(config, dataset)
    accepted, reasons = accepted_rows(rows)
    if not accepted:
        raise ValueError(f"{dataset} has no Policy B accepted samples")
    time_points = dataset_time_points(accepted)

    recordings = list(OrderedDict.fromkeys(row["canonical_hdf5_path"] for row in accepted))
    canonical_root = (ROOT / config["datasets"][dataset]["canonical_root"]).resolve()
    by_identity_map: dict[str, dict] = {}
    static_valid_map: dict[str, np.ndarray] = {}
    for relative in recordings:
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(canonical_root):
            raise ValueError(f"{dataset} canonical path escapes its root")
        with h5py.File(path, "r") as handle:
            mapping = read_channel_mapping(handle)
            static_valid = np.asarray(handle["valid_channel_mask"], dtype=bool)
            if static_valid.shape[0] != handle["hbo"].shape[1]:
                raise ValueError("valid_channel_mask length mismatch")
        by_identity_map[relative] = mapping
        static_valid_map[relative] = static_valid

    identities = canonical_identities(dataset, recordings, by_identity_map,
                                      channel_policy, static_valid_map)
    channel_count = len(identities)
    slot_of = {identity: slot for slot, identity in enumerate(identities)}
    geometry = dataset_geometry(identities, recordings, by_identity_map)
    roles = split_roles(config, dataset)

    n = len(accepted)
    group = output.create_group(dataset)
    group.attrs.update(
        dataset=dataset,
        fair_input_policy=POLICY,
        fair_input_policy_name=policy_name,
        manifest_path=config["datasets"][dataset]["manifest"],
        manifest_sha256=config["datasets"][dataset]["manifest_sha256"],
        candidate_samples=len(rows),
        rows=n,
        channels=channel_count,
        time_points=time_points,
        tensor_shape=json.dumps([channel_count, time_points]),
        missing_value_fill=MISSING_VALUE_FILL,
        channel_order="ascending_(source_index,detector_index)_physical_identity",
        geometry_variant=GEOMETRY_VARIANT,
        geometry_scale_m=GEOMETRY_SCALE_M,
        geometry_scope="DATASET_LEVEL_FIXED",
        rejected_samples=len(rows) - n,
        rejection_reasons_json=json.dumps(dict(reasons)),
    )

    chunks = (min(16, n), channel_count, time_points)
    signal_kwargs = {"shape": (n, channel_count, time_points), "dtype": "f4",
                     "chunks": chunks, "compression": "lzf", "fillvalue": 0.0}
    hbo_out = group.create_dataset("hbo", **signal_kwargs)
    hbr_out = group.create_dataset("hbr", **signal_kwargs)
    valid_out = group.create_dataset(
        "valid", shape=(n, channel_count, time_points), dtype="u1",
        chunks=chunks, compression="lzf", fillvalue=0,
    )
    channel_out = group.create_dataset(
        "valid_channel_mask", shape=(n, channel_count), dtype="u1",
        chunks=(min(1024, n), channel_count), compression="lzf", fillvalue=0,
    )
    group.create_dataset("geometry", data=geometry)

    strings = h5py.string_dtype("utf-8")
    group.create_dataset("channel_ids", data=np.asarray([f"S{s}-D{d}" for s, d in identities], dtype=object), dtype=strings)
    group.create_dataset("sample_id", data=np.asarray([row["sample_id"] for row in accepted], dtype=object), dtype=strings)
    group.create_dataset("subject_id", data=np.asarray([row["subject_id"] for row in accepted], dtype=object), dtype=strings)
    group.create_dataset("canonical_hdf5_path", data=np.asarray([row["canonical_hdf5_path"] for row in accepted], dtype=object), dtype=strings)
    group.create_dataset("window_start_sample", data=np.asarray([int(row["window_start_sample"]) for row in accepted], dtype=np.int64))
    group.create_dataset("window_end_sample", data=np.asarray([int(row["window_end_sample"]) for row in accepted], dtype=np.int64))
    group.create_dataset("label", data=np.asarray([int(row["label_id"]) for row in accepted], dtype=np.int16))
    group.create_dataset("fold", data=np.asarray([int(row["fold"]) for row in accepted], dtype=np.int8))

    role_table = np.zeros((n, 5), dtype=np.int8)
    for index, row in enumerate(accepted):
        subject_roles = roles[row["subject_id"]]
        if len(subject_roles) != 5:
            raise ValueError(f"{dataset} subject {row['subject_id']} missing fold roles")
        for fold_number, role in subject_roles.items():
            role_table[index, fold_number - 1] = role
        if subject_roles[int(row["fold"])] != SPLIT_ROLE["test"]:
            raise ValueError(f"{dataset} sample {row['sample_id']} fold is not its test fold")
    group.create_dataset("split_role", data=role_table)

    recording_rows: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(accepted):
        recording_rows[row["canonical_hdf5_path"]].append(index)

    per_sample_padded_channels = np.zeros(n, dtype=np.int64)
    zero_slots = 0
    valid_slots = 0
    for relative, row_indices in recording_rows.items():
        mapping = by_identity_map[relative]
        static_valid = static_valid_map[relative]
        with h5py.File((ROOT / relative).resolve(), "r") as handle:
            hbo = np.asarray(handle["hbo"], dtype=np.float32)
            hbr = np.asarray(handle["hbr"], dtype=np.float32)
            time_valid = np.asarray(handle["valid_time_mask"], dtype=bool)
        count = len(row_indices)
        block_hbo = np.zeros((count, channel_count, time_points), dtype=np.float32)
        block_hbr = np.zeros_like(block_hbo)
        block_valid = np.zeros(block_hbo.shape, dtype=np.uint8)
        block_channel = np.zeros((count, channel_count), dtype=np.uint8)
        for local, row_index in enumerate(row_indices):
            row = accepted[row_index]
            start, end = int(row["window_start_sample"]), int(row["window_end_sample"])
            for identity, column in mapping.items():
                slot = slot_of.get(identity)
                if slot is None or not bool(static_valid[column]):
                    continue
                block_channel[local, slot] = 1
                joint = (time_valid[start:end]
                         & np.isfinite(hbo[start:end, column])
                         & np.isfinite(hbr[start:end, column]))
                block_valid[local, slot] = joint
                block_hbo[local, slot][joint] = hbo[start:end, column][joint]
                block_hbr[local, slot][joint] = hbr[start:end, column][joint]
            padded = channel_count - int(block_channel[local].sum())
            per_sample_padded_channels[row_index] = padded
            zero_slots += int(block_valid[local].size - block_valid[local].sum())
            valid_slots += int(block_valid[local].sum())
        selection = np.asarray(row_indices, dtype=np.int64)
        hbo_out[selection] = block_hbo
        hbr_out[selection] = block_hbr
        valid_out[selection] = block_valid
        channel_out[selection] = block_channel

    per_channel_zero = (n - np.asarray(channel_out[:], dtype=np.int64).sum(axis=0))

    index_rows = []
    for index, row in enumerate(accepted):
        subject_roles = roles[row["subject_id"]]
        index_rows.append({
            "row_index": index,
            "sample_id": row["sample_id"],
            "subject_id": row["subject_id"],
            "label_id": int(row["label_id"]),
            "label": row["label"],
            "test_fold": int(row["fold"]),
            "split_roles": "|".join(SPLIT_NAMES[subject_roles[fold]] for fold in range(1, 6)),
            "zero_padded_channels": int(per_sample_padded_channels[index]),
            "valid_channels": int(channel_count - per_sample_padded_channels[index]),
            "window_start_sample": int(row["window_start_sample"]),
            "window_end_sample": int(row["window_end_sample"]),
        })

    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / f"{dataset}.csv"
    with index_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(index_rows[0].keys()))
        writer.writeheader()
        writer.writerows(index_rows)

    label_names = list(config["datasets"][dataset]["task"]["labels"])
    class_counts = defaultdict(int)
    for row in accepted:
        class_counts[label_names[int(row["label_id"])]] += 1
    fold_counts = {fold: {name: 0 for name in SPLIT_NAMES.values()} for fold in range(1, 6)}
    for fold in range(1, 6):
        for role in role_table[:, fold - 1]:
            fold_counts[fold][SPLIT_NAMES[int(role)]] += 1

    records = {
        "dataset": dataset,
        "fair_input_policy": POLICY,
        "fair_input_policy_name": policy_name,
        "candidate_samples": len(rows),
        "n_samples": n,
        "rejected_samples": len(rows) - n,
        "rejection_reasons": dict(reasons),
        "tensor_shape": [channel_count, time_points],
        "n_channels": channel_count,
        "time_points": time_points,
        "channel_names": [f"S{s}-D{d}" for s, d in identities],
        "channel_identities": [[int(s), int(d)] for s, d in identities],
        "channel_order": "ascending (source_index, detector_index) physical identity",
        "class_counts": dict(class_counts),
        "split_counts": {str(fold): fold_counts[fold] for fold in range(1, 6)},
        "split_roles": {subject: {str(k): SPLIT_NAMES[v] for k, v in sorted(value.items())}
                        for subject, value in sorted(roles.items())},
        "zero_padding": {
            "fill_value": MISSING_VALUE_FILL,
            "zero_padded_scalar_fraction": round(zero_slots / float(zero_slots + valid_slots), 6),
            "zero_padded_scalar_values": zero_slots,
            "valid_scalar_values": valid_slots,
            "per_channel_zero_padded_sample_fraction": [round(v / n, 6) for v in per_channel_zero.tolist()],
            "mean_zero_padded_channels_per_sample": round(float(per_sample_padded_channels.mean()), 6),
            "max_zero_padded_channels_per_sample": int(per_sample_padded_channels.max()),
        },
        "geometry": {
            "variant": GEOMETRY_VARIANT,
            "scope": "DATASET_LEVEL_FIXED",
            "scale_m": GEOMETRY_SCALE_M,
            "shape": [channel_count, 6],
            "fields": ["midpoint_x", "midpoint_y", "midpoint_z", "delta_x", "delta_y", "delta_z"],
        },
        "manifest_path": config["datasets"][dataset]["manifest"],
        "manifest_sha256": config["datasets"][dataset]["manifest_sha256"],
        "index_csv": index_path.relative_to(ROOT).as_posix(),
        "sample_id_sha256": hashlib.sha256("\n".join(row["sample_id"] for row in accepted).encode()).hexdigest(),
        "tensor_sha256": None,
    }
    return records, reasons


def tensor_digests(output_path: Path, datasets) -> dict:
    digests = {}
    with h5py.File(output_path, "r") as handle:
        for dataset in datasets:
            group = handle[dataset]
            value = hashlib.sha256()
            for name in ("hbo", "hbr", "valid", "valid_channel_mask", "label", "fold", "sample_id", "split_role"):
                array = group[name][...]
                value.update(f"{name}|{array.dtype.str}|{array.shape}\n".encode("ascii"))
                if array.dtype.kind == "O":
                    value.update("\n".join(str(item) for item in array).encode("utf-8"))
                else:
                    value.update(np.ascontiguousarray(array).tobytes())
            digests[dataset] = value.hexdigest()
    return digests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--channel-policy", default="union",
                        choices=("union", "intersection"),
                        help="union: dataset-level union with zero fill (frozen "
                             "policy).  intersection: keep only channels valid in "
                             "every recording, so nothing is zero-filled.")
    parser.add_argument("--policy-name", default=None,
                        help="override the policy label written into the HDF5 "
                             "attributes and the records JSON (default: derived "
                             "from --channel-policy).")
    parser.add_argument("--skip-empty-intersection", action="store_true",
                        help="with --channel-policy intersection, report and skip "
                             "datasets whose intersection is empty instead of failing")
    args = parser.parse_args()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + f".tmp-{os.getpid()}")
    config = load_authority()
    policy_name = str(args.policy_name or POLICY_NAMES[args.channel_policy])
    datasets = args.datasets or list(config["paper_benchmark"]["datasets"])
    records = {}
    try:
        with h5py.File(temporary, "w") as output:
            output.attrs.update(
                schema=SCHEMA,
                authority="benchmark/benchmark.yaml",
                fair_input_policy=POLICY,
                fair_input_policy_name=policy_name,
                missing_value_fill=MISSING_VALUE_FILL,
                geometry_variant=GEOMETRY_VARIANT,
                geometry_scope="DATASET_LEVEL_FIXED",
                canonical_modified="NO",
                manifest_modified="NO",
            )
            for dataset in list(datasets):
                try:
                    item, _ = build_dataset(output, config, dataset,
                                            args.index_dir.resolve(),
                                            args.channel_policy,
                                            policy_name)
                except ValueError as error:
                    if args.skip_empty_intersection and "intersection is empty" in str(error):
                        print(dataset, "SKIPPED", error, flush=True)
                        datasets.remove(dataset)
                        continue
                    raise
                records[dataset] = item
                print(dataset, "rows", item["n_samples"], "shape", item["tensor_shape"],
                      "rejected", item["rejected_samples"], flush=True)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    digests = tensor_digests(output_path, datasets)
    for dataset in datasets:
        records[dataset]["tensor_sha256"] = digests[dataset]
    payload = {
        "schema": SCHEMA,
        "fair_input_policy": POLICY,
        "fair_input_policy_name": policy_name,
        "missing_value_fill": MISSING_VALUE_FILL,
        "path": output_path.relative_to(ROOT).as_posix(),
        "sha256": digest(output_path),
        "datasets": records,
    }
    args.records.resolve().write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("path", output_path)
    print("sha256", payload["sha256"])


if __name__ == "__main__":
    main()
