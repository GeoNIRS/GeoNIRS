from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import h5py
import yaml


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = Path(os.environ.get("GEONIRS_CANONICAL_ROOT",
                                     ROOT / "data" / "canonical"))
MANIFEST_ROOT = ROOT / "benchmark" / "manifests"
AUTHORITY = ROOT / "benchmark" / "benchmark.yaml"

AUTHORITY_FOLD_DATASETS = ("DS2", "DS8")


SPLIT_ROOT = Path(os.environ.get("GEONIRS_SPLIT_ROOT", ROOT / "data" / "splits"))

EVENT_FIELDS = [
    "dataset_id", "sample_id", "subject_id", "canonical_recording_key",
    "canonical_hdf5_path", "recording_condition", "event_index",
    "event_label", "event_onset_s", "window_start_sample",
    "window_end_sample", "window_start_s", "window_end_s", "n_timepoints",
    "label", "label_id", "fold",
]


def events(handle: h5py.File) -> list[dict]:
    return json.loads(handle["events_json"][()].decode("utf-8"))


def subject_folds(split_file: Path, dataset: str) -> dict[int, dict[str, list[str]]]:
    payload = json.loads(split_file.read_text(encoding="utf-8"))
    folds = payload.get("folds")
    if folds is None:
        payload = payload["datasets"][dataset]
        folds = payload["folds"]
    if isinstance(folds, dict):
        return {int(str(k).replace("fold", "")): v for k, v in folds.items()}
    resolved = {}
    for entry in folds:
        fold = dict(entry)
        if "val_subjects" in fold and "validation_subjects" not in fold:
            fold["validation_subjects"] = fold["val_subjects"]
        resolved[int(fold["fold_id"])] = fold
    return resolved


def authority_folds(dataset: str) -> dict[int, dict[str, list[str]]]:
    document = yaml.safe_load(AUTHORITY.read_text(encoding="utf-8"))
    folds = document["datasets"][dataset]["folds"]
    return {int(key): {"train_subjects": list(value["train_subjects"]),
                       "validation_subjects": list(value["validation_subjects"]),
                       "test_subjects": list(value["test_subjects"])}
            for key, value in folds.items()}


def test_fold(folds: dict[int, dict], subject: str) -> int:
    key = "test_subjects"
    hits = [number for number, fold in folds.items() if subject in fold[key]]
    if len(hits) != 1:
        raise ValueError(f"{subject}: expected exactly one test fold, got {hits}")
    return hits[0]


def base_row(dataset: str, path: Path, handle: h5py.File, event: dict,
             event_index: int, start: int, points: int, label: str,
             label_id: int, fold: int, subject: str | None = None) -> dict:
    rel = path.relative_to(ROOT).as_posix()
    fs = float(handle.attrs["sampling_rate"])
    return {
        "dataset_id": dataset,
        "sample_id": f"{dataset}::{path.stem}::event{event_index}",
        "subject_id": str(handle.attrs["subject"]) if subject is None else subject,
        "canonical_recording_key": rel,
        "canonical_hdf5_path": rel,
        "recording_condition": str(handle.attrs.get("task", "")),
        "event_index": event_index,
        "event_label": str(event["label"]),
        "event_onset_s": float(event["onset"]),
        "window_start_sample": start,
        "window_end_sample": start + points,
        "window_start_s": start / fs,
        "window_end_s": (start + points) / fs,
        "n_timepoints": points,
        "label": label,
        "label_id": label_id,
        "fold": fold,
    }


def build_ds2(folds: dict[int, dict]) -> list[dict]:
    selection = {"animal_image": ("Animal", 0), "tool_image": ("Tool", 1)}
    offset_points, points = 15, 60
    rows = []
    for path in sorted((CANONICAL_ROOT / "DS2").glob("*.h5")):
        with h5py.File(path, "r") as handle:
            subject = str(handle.attrs["subject"])
            fold = test_fold(folds, subject)
            total = int(handle["hbo"].shape[0])
            for index, event in enumerate(events(handle)):
                name = str(event["label"])
                if name not in selection:
                    continue
                label, label_id = selection[name]
                start = int(event["sample_index_5hz"]) + offset_points
                if start + points > total:
                    continue



                rows.append(base_row("DS2", path, handle, event, index, start,
                                     points, label, label_id, fold))
    return rows


def build_ds16(folds: dict[int, dict], task: str = "MI") -> list[dict]:
    selection = {"LMI": ("Left", 0), "RMI": ("Right", 1)}
    points = 60
    dataset_id = "DS16"
    rows = []
    for path in sorted((CANONICAL_ROOT / "DS16").glob("*.h5")):
        with h5py.File(path, "r") as handle:
            if str(handle.attrs.get("task", "")) != task:
                continue
            subject = str(handle.attrs["subject"])
            fold = test_fold(folds, subject)
            total = int(handle["hbo"].shape[0])
            for index, event in enumerate(events(handle)):
                name = str(event["label"])
                if name not in selection:
                    continue
                label, label_id = selection[name]
                start = int(event["sample_index_5hz"]) if "sample_index_5hz" in event \
                    else int(round(float(event["onset"]) * 5.0))
                if start + points > total:
                    continue
                rows.append(base_row(dataset_id, path, handle, event, index, start,
                                     points, label, label_id, fold))
    return rows


def build_ds8(folds: dict[int, dict]) -> list[dict]:
    flash_names = {1.0: "flash_l", 2.0: "flash_r"}
    points = 50
    rows = []
    for path in sorted((CANONICAL_ROOT / "DS8").glob("*.h5")):
        with h5py.File(path, "r") as handle:
            side = str(handle.attrs["task"])
            subject = str(handle.attrs["subject"]).replace("_", "")
            fold = test_fold(folds, subject)
            total = int(handle["hbo"].shape[0])
            for index, event in enumerate(events(handle)):
                flash = flash_names.get(float(event.get("value", -1)))
                if flash is None:
                    continue
                start = int(event["sample_index_5hz"])
                if start + points > total:
                    continue
                flash_side = "left" if flash == "flash_l" else "right"
                label = "Movement" if flash_side == side else "No movement"
                labelled = dict(event, label=flash)
                rows.append(base_row("DS8", path, handle, labelled, index, start,
                                     points, label, 0 if label == "Movement" else 1,
                                     fold, subject=subject))
    return rows


BUILDERS = {
    "DS2": (build_ds2, "ds2_animal_tool_subject_5fold.json", "DS2"),
    "DS16": (build_ds16, "ds16_ds22_subject_5fold.json", "DS16"),
    "DS8": (build_ds8, "ds8_subject_5fold.json", None),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(BUILDERS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--split-root", type=Path, default=SPLIT_ROOT,
                        help="directory holding the subject-level split JSONs "
                             "(default: %(default)s)")
    args = parser.parse_args()

    builder, split_name, split_key = BUILDERS[args.dataset]
    folds = (authority_folds(args.dataset) if args.dataset in AUTHORITY_FOLD_DATASETS
             else subject_folds(args.split_root / split_name, split_key))
    rows = builder(folds)

    from collections import Counter
    counts = Counter(row["label"] for row in rows)
    subjects = {row["subject_id"] for row in rows}
    print(f"{args.dataset}: {len(rows)} 个样本, {len(subjects)} 个受试者")
    print("  类别分布:", dict(counts))
    print("  每折样本数:", {f: sum(r['fold'] == f for r in rows) for f in sorted(folds)})
    print("  每折测试受试者:", {f: sorted(folds[f]['test_subjects']) for f in sorted(folds)})
    per_subject = Counter(row["subject_id"] for row in rows)
    print("  每个受试者样本数:", sorted(set(per_subject.values())))
    if args.dry_run:
        return

    out = MANIFEST_ROOT / f"{args.dataset}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"\n已写出: {out.relative_to(ROOT)}  ({len(rows)} 行)")
    print("sha256:", digest)


if __name__ == "__main__":
    main()
