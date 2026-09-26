from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import h5py
import numpy as np
import torch
import yaml


REQUIRED_FIELDS = (
    "sample_id", "dataset", "label", "fold", "hbo", "hbr",
    "channel_ids", "time_support",
)
OPTIONAL_FIELDS = ("optional_geometry",)
FORBIDDEN_CLASSIFIER_FEATURES = frozenset({
    "valid_time_mask", "valid_channel_mask", "missingness_ratio",
    "missingness ratio", "valid_fraction", "valid fraction",
})
GEOMETRY_SCOPE = "DATASET_LEVEL_FIXED"
SPLIT_ROLE_NAMES = ("train", "validation", "test")


def _file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class FormalFairInputLoader:

    def __init__(self, dataset: str, project_root: str | Path | None = None,
                 materialized: bool = True):
        self.root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
        self.authority = yaml.safe_load(
            (self.root / "benchmark" / "benchmark.yaml").read_text(encoding="utf-8")
        )
        policy = self.authority["fair_input"]
        if float(policy["missing_value_fill"]) != 0.0:
            raise ValueError("the Frozen Fair Input must zero-fill, not impute")
        item, expected = self._resolve(dataset)
        self.dataset = dataset
        self.policy_name = str(item.get("policy_name"))
        self.records_sha256 = str(item["records_sha256"])
        self.tensor_sha256: str | None = None
        self.expected = expected
        self.tensor_shape = tuple(int(value) for value in self.expected["tensor_shape"])
        self.path = (self.root / policy["tensor"]).resolve()
        self.records = (self.root / policy["records"]).resolve()
        if not materialized:
            self.length = int(self.expected["n_samples"])
            self.channel_ids = list(self.expected["channel_names"])
            return
        if not self.path.is_relative_to(self.root) or not self.path.is_file():
            raise ValueError(
                f"the Frozen Fair Input tensors are not part of this repository; "
                f"build them with scripts/build_fair_input.py (expected {self.path})"
            )
        self.tensor_sha256 = _file_sha256(self.path)
        with h5py.File(self.path, "r") as handle:
            group = handle[dataset]
            self.length = int(group.attrs["rows"])
            shape = tuple(map(int, group["hbo"].shape[1:]))
            self.channel_ids = group["channel_ids"].asstr()[()].tolist()
            materialized_policy = str(group.attrs["fair_input_policy_name"])
        if self.length != int(self.expected["n_samples"]):
            raise ValueError("Fair Input row count disagrees with the authority")
        if shape != self.tensor_shape:
            raise ValueError("Fair Input tensor shape disagrees with the authority")
        if self.channel_ids != list(self.expected["channel_names"]):
            raise ValueError("Fair Input channel order disagrees with the authority")
        if materialized_policy != self.policy_name:
            raise ValueError("Fair Input file policy label disagrees with the authority")
        if policy.get("geometry_scope") != GEOMETRY_SCOPE:
            raise ValueError("Fair Input geometry is not dataset-level fixed")

    def _resolve(self, dataset: str):
        if dataset not in list(self.authority["paper_benchmark"]["datasets"]):
            raise ValueError(f"not a benchmark dataset: {dataset}")
        block = self.authority["fair_input"]
        records_path = (self.root / block["records"]).resolve()
        if not records_path.is_relative_to(self.root) or not records_path.is_file():
            raise ValueError("the Fair Input records must be a file inside the repository")
        if _file_sha256(records_path) != block["records_sha256"]:
            raise ValueError("Fair Input records SHA256 mismatch")
        records = json.loads(records_path.read_text(encoding="utf-8"))
        entry = records["datasets"][dataset]
        expected = {
            "n_samples": int(entry["n_samples"]),
            "tensor_shape": list(entry["tensor_shape"]),
            "channel_names": list(entry["channel_names"]),
        }
        return block, expected

    def __len__(self):
        return self.length

    def split_indices(self, fold: int, split: str) -> np.ndarray:
        if int(fold) not in range(1, 6):
            raise ValueError("fold must be in 1..5")
        if split not in SPLIT_ROLE_NAMES:
            raise ValueError(f"split must be one of {SPLIT_ROLE_NAMES}")
        with h5py.File(self.path, "r") as handle:
            roles = np.asarray(handle[self.dataset]["split_role"][:, int(fold) - 1])
        return np.nonzero(roles == SPLIT_ROLE_NAMES.index(split))[0]

    def load_batch(self, indices: Sequence[int], *, cv_fold: int | None = None):
        selected = np.asarray(list(indices), dtype=np.int64)
        if selected.ndim != 1 or not len(selected):
            raise ValueError("indices must be a non-empty one-dimensional sequence")
        if np.any(selected < 0) or np.any(selected >= self.length):
            raise IndexError("Fair Input index out of range")
        order = np.argsort(selected)
        sorted_indices = selected[order]
        if len(np.unique(sorted_indices)) != len(sorted_indices):
            raise ValueError("duplicate Fair Input indices")
        inverse = np.argsort(order)
        with h5py.File(self.path, "r") as handle:
            group = handle[self.dataset]
            hbo = np.asarray(group["hbo"][sorted_indices])[inverse]
            hbr = np.asarray(group["hbr"][sorted_indices])[inverse]
            valid = np.asarray(group["valid"][sorted_indices], dtype=bool)[inverse]
            channel_valid = np.asarray(
                group["valid_channel_mask"][sorted_indices], dtype=bool
            )[inverse]
            geometry = np.asarray(group["geometry"], dtype=np.float32)
            geometry = np.broadcast_to(geometry, (len(selected),) + geometry.shape).copy()
            split_role = np.asarray(group["split_role"][sorted_indices], dtype=np.int8)[inverse]
            labels = np.asarray(group["label"][sorted_indices])[inverse]
            folds = np.asarray(group["fold"][sorted_indices])[inverse]
            sample_ids = np.asarray(group["sample_id"].asstr()[sorted_indices])[inverse].tolist()
        if not (np.isfinite(hbo).all() and np.isfinite(hbr).all()
                and np.isfinite(geometry).all()):
            raise ValueError("materialized Fair Input is not finite")
        if np.any(hbo[~valid] != 0) or np.any(hbr[~valid] != 0):
            raise ValueError("invalid Fair Input values are not zero-filled")
        return {
            "hbo": torch.from_numpy(hbo),
            "hbr": torch.from_numpy(hbr),
            "valid": torch.from_numpy(valid),
            "valid_time_mask": torch.from_numpy(valid),
            "valid_channel_mask": torch.from_numpy(channel_valid),
            "channel_valid": torch.from_numpy(channel_valid),
            "geometry": torch.from_numpy(geometry),
            "split_role": torch.from_numpy(split_role),
            "label": torch.from_numpy(labels.astype(np.int64)),
            "fold": torch.from_numpy(folds.astype(np.int64)),
            "sample_id": sample_ids,
            "channel_ids": list(self.channel_ids),
            "dataset": self.dataset,
        }


@dataclass(frozen=True)
class FairInputV2Sample:
    sample_id: str
    dataset: str
    label: int
    fold: str | int
    hbo: Any
    hbr: Any
    channel_ids: Sequence[str]
    time_support: Sequence[float]
    optional_geometry: Any | None = None


class FairInputV2Loader(Protocol):

    def load_split(self, *, fold: str | int, split: str) -> Iterable[FairInputV2Sample]:
        ...


@dataclass(frozen=True)
class CanonicalSamplePreview:

    dataset: str
    sample_id: str
    label: int
    fold: int
    canonical_path: Path
    hbo: np.ndarray
    hbr: np.ndarray
    valid_channel_mask: np.ndarray
    valid_time_mask: np.ndarray
    provenance: Mapping[str, Any]
    events: Sequence[Mapping[str, Any]]


class CanonicalManifestLoader:

    def __init__(self, dataset: str, project_root: str | Path | None = None):
        self.root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
        authority_path = self.root / "benchmark" / "benchmark.yaml"
        self.authority = yaml.safe_load(authority_path.read_text(encoding="utf-8"))
        if dataset not in list(self.authority["paper_benchmark"]["datasets"]):
            raise ValueError(f"not a benchmark dataset: {dataset}")
        self.dataset = dataset
        self.dataset_config = self.authority["datasets"][dataset]
        manifest_config = self.dataset_config
        self.manifest_path = (self.root / manifest_config["manifest"]).resolve()
        if not self.manifest_path.is_relative_to(self.root):
            raise ValueError("manifest escapes the project root")
        actual_hash = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        if actual_hash != manifest_config["manifest_sha256"]:
            raise ValueError(f"{dataset} manifest SHA256 mismatch")
        with self.manifest_path.open(newline="", encoding="utf-8") as stream:
            self.rows = list(csv.DictReader(stream))
        if len(self.rows) != manifest_config["manifest_rows"]:
            raise ValueError(f"{dataset} manifest row count mismatch")



        self.canonical_root = self.root / "data" / "canonical" / dataset

    def load_preview(self, row_index: int = 0,
                     canonical_root: str | Path | None = None) -> CanonicalSamplePreview:
        row = self.rows[row_index]
        root = Path(canonical_root or self.canonical_root).resolve()


        canonical_path = (root / Path(row["canonical_hdf5_path"]).name).resolve()
        if not canonical_path.is_relative_to(root):
            raise ValueError("canonical sample escapes the dataset root")
        start, end = int(row["window_start_sample"]), int(row["window_end_sample"])
        if not canonical_path.is_file():
            raise FileNotFoundError(
                f"{canonical_path} is missing; the canonical HDF5 tree is not "
                f"part of this repository (see DATA.md)")
        with h5py.File(canonical_path, "r") as handle:
            if str(handle.attrs["dataset"]) != self.dataset:
                raise ValueError("canonical dataset identity mismatch")
            if not 0 <= start < end <= int(handle["hbo"].shape[0]):
                raise ValueError("manifest window is out of bounds")
            hbo = np.asarray(handle["hbo"][start:end])
            hbr = np.asarray(handle["hbr"][start:end])
            channel_valid = np.asarray(handle["valid_channel_mask"], dtype=bool)
            time_valid = np.asarray(handle["valid_time_mask"][start:end], dtype=bool)
            provenance = json.loads(handle["provenance_json"][()].decode("utf-8"))
            events = json.loads(handle["events_json"][()].decode("utf-8"))
        if hbo.shape != hbr.shape or hbo.shape[0] != end - start:
            raise ValueError("canonical signal shape mismatch")
        return CanonicalSamplePreview(
            dataset=self.dataset,
            sample_id=row["sample_id"],
            label=int(row["label_id"]),
            fold=int(row["fold"]),
            canonical_path=canonical_path,
            hbo=hbo,
            hbr=hbr,
            valid_channel_mask=channel_valid,
            valid_time_mask=time_valid,
            provenance=provenance,
            events=events,
        )


def _as_mapping(sample: FairInputV2Sample | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(sample, FairInputV2Sample):
        return sample.__dict__
    if not isinstance(sample, Mapping):
        raise TypeError("Fair Input V2 sample must be a mapping or FairInputV2Sample")
    return sample


def validate_sample(sample: FairInputV2Sample | Mapping[str, Any]) -> None:
    item = _as_mapping(sample)
    missing = [field for field in REQUIRED_FIELDS if field not in item]
    if missing:
        raise ValueError(f"missing Fair Input V2 fields: {missing}")
    forbidden = FORBIDDEN_CLASSIFIER_FEATURES.intersection(
        str(key).strip().lower() for key in item
    )
    if forbidden:
        raise ValueError(f"forbidden classifier feature(s): {sorted(forbidden)}")
    if not str(item["sample_id"]).strip() or not str(item["dataset"]).strip():
        raise ValueError("sample_id and dataset must be non-empty")
    if not isinstance(item["label"], (int, np.integer)):
        raise ValueError("label must be an integer class ID")
    if item["fold"] is None or not str(item["fold"]).strip():
        raise ValueError("fold must be explicit")

    hbo = np.asarray(item["hbo"])
    hbr = np.asarray(item["hbr"])
    if hbo.ndim != 2 or hbr.ndim != 2 or hbo.shape != hbr.shape:
        raise ValueError("hbo and hbr must be equal-shape [time, channel] arrays")
    if not np.isfinite(hbo).all() or not np.isfinite(hbr).all():
        raise ValueError("Fair Input V2 model signal must contain no NaN or Inf")
    if len(item["time_support"]) != hbo.shape[0]:
        raise ValueError("time_support length must equal signal time dimension")
    if not np.isfinite(np.asarray(item["time_support"], dtype=float)).all():
        raise ValueError("time_support must be finite")
    if len(item["channel_ids"]) != hbo.shape[1]:
        raise ValueError("channel_ids length must equal signal channel dimension")
    if len(set(map(str, item["channel_ids"]))) != len(item["channel_ids"]):
        raise ValueError("channel_ids must be unique")


def validate_collection(samples: Iterable[FairInputV2Sample | Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for sample in samples:
        validate_sample(sample)
        sample_id = str(_as_mapping(sample)["sample_id"])
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)


def validate_method_equivalence(
    reference: Iterable[FairInputV2Sample | Mapping[str, Any]],
    candidate: Iterable[FairInputV2Sample | Mapping[str, Any]],
) -> None:
    ref, cand = list(reference), list(candidate)
    validate_collection(ref)
    validate_collection(cand)
    if len(ref) != len(cand):
        raise ValueError("methods received different sample counts")
    for left_raw, right_raw in zip(ref, cand):
        left, right = _as_mapping(left_raw), _as_mapping(right_raw)
        for field in ("sample_id", "dataset", "label", "fold", "channel_ids", "time_support"):
            left_value = list(left[field]) if field in ("channel_ids", "time_support") else left[field]
            right_value = list(right[field]) if field in ("channel_ids", "time_support") else right[field]
            if left_value != right_value:
                raise ValueError(f"methods differ at {left['sample_id']}: {field}")
        for field in ("hbo", "hbr"):
            if not np.array_equal(np.asarray(left[field]), np.asarray(right[field])):
                raise ValueError(f"methods differ at {left['sample_id']}: {field}")


__all__ = [
    "CanonicalManifestLoader", "CanonicalSamplePreview",
    "FormalFairInputLoader",
    "FairInputV2Loader", "FairInputV2Sample", "FORBIDDEN_CLASSIFIER_FEATURES",
    "GEOMETRY_SCOPE", "OPTIONAL_FIELDS", "REQUIRED_FIELDS", "SPLIT_ROLE_NAMES",
    "validate_collection", "validate_method_equivalence", "validate_sample",
]
