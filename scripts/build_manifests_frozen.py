from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import h5py
import yaml


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "benchmark" / "benchmark.yaml"
CANONICAL_ROOT = Path(os.environ.get("GEONIRS_CANONICAL_ROOT",
                                     ROOT / "data" / "canonical"))
MANIFEST_ROOT = ROOT / "benchmark" / "manifests"

DATASETS = ("DS7", "DS47")

FIELDS = ["dataset_id", "sample_id", "subject_id", "canonical_recording_key",
          "canonical_hdf5_path", "recording_condition", "event_index",
          "event_label", "event_onset_s", "window_start_sample",
          "window_end_sample", "window_start_s", "window_end_s", "n_timepoints",
          "label", "label_id", "fold"]


def events(handle: h5py.File) -> list[dict]:
    return json.loads(handle["events_json"][()].decode("utf-8"))


def authority_folds(dataset: str) -> dict[int, dict[str, list[str]]]:
    document = yaml.safe_load(AUTHORITY.read_text(encoding="utf-8"))
    folds = document["datasets"][dataset]["folds"]
    return {int(key): {"train_subjects": list(value["train_subjects"]),
                       "validation_subjects": list(value["validation_subjects"]),
                       "test_subjects": list(value["test_subjects"])}
            for key, value in folds.items()}


def test_fold(folds: dict[int, dict[str, list[str]]], subject: str) -> int:
    hits = [number for number, fold in folds.items() if subject in fold["test_subjects"]]
    if len(hits) != 1:
        raise ValueError(f"{subject}: expected exactly one test fold, got {hits}")
    return hits[0]


def base_row(dataset: str, path: Path, handle: h5py.File, event: dict,
             event_index: int, start: int, points: int, label: str,
             label_id: int, fold: int) -> dict:
    rel = path.relative_to(ROOT).as_posix()
    subject = str(handle.attrs["subject"])
    fs = float(handle.attrs["sampling_rate"])
    return {
        "dataset_id": dataset,
        "sample_id": f"{dataset}::{path.stem}::event{event_index}",
        "subject_id": subject,
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


def build_ds7(folds: dict[int, dict[str, list[str]]]) -> list[dict]:
    rows = []
    for path in sorted((CANONICAL_ROOT / "DS7").glob("*.h5")):
        with h5py.File(path, "r") as handle:
            subject = str(handle.attrs["subject"])
            label = "Healthy" if subject.startswith("H") else "ICH"
            fold = test_fold(folds, subject)
            sample_index = 0
            for event in events(handle):
                if event["label"] not in {"marker_9", "marker_11"}:
                    continue
                start = int(event["sample_index_5hz"]) + 10
                rows.append(base_row("DS7", path, handle, event, sample_index, start,
                                     50, label, 0 if label == "Healthy" else 1, fold))
                sample_index += 1
    return rows


def build_ds47(folds: dict[int, dict[str, list[str]]]) -> list[dict]:
    selection = {"AStim", "ANStim", "AVStim", "VStim"}
    rows = []
    for path in sorted((CANONICAL_ROOT / "DS47").glob("*.h5")):
        with h5py.File(path, "r") as handle:
            subject = str(handle.attrs["subject"])
            label = "NH" if subject.startswith("sub-C") else "CI"
            fold = test_fold(folds, subject)
            for index, event in enumerate(events(handle)):
                if event["label"] not in selection:
                    continue
                start = int(event["sample_index_5hz"])
                rows.append(base_row("DS47", path, handle, event, index, start, 60,
                                     label, 0 if label == "NH" else 1, fold))
    return rows


BUILDERS = {"DS7": build_ds7, "DS47": build_ds47}


def main() -> None:
    global CANONICAL_ROOT, MANIFEST_ROOT
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS),
                        help=f"datasets to build (default: {' '.join(DATASETS)})")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--canonical-root", type=Path, default=CANONICAL_ROOT,
                        help="canonical HDF5 tree (default: %(default)s)")
    parser.add_argument("--manifest-root", type=Path, default=MANIFEST_ROOT,
                        help="output directory (default: %(default)s)")
    args = parser.parse_args()
    CANONICAL_ROOT, MANIFEST_ROOT = args.canonical_root, args.manifest_root

    document = yaml.safe_load(AUTHORITY.read_text(encoding="utf-8"))
    for dataset in args.datasets:
        if dataset not in BUILDERS:
            raise SystemExit(f"unsupported dataset: {dataset} (expected {list(BUILDERS)})")
        rows = BUILDERS[dataset](authority_folds(dataset))
        counts = Counter(row["label"] for row in rows)
        print(f"{dataset}: {len(rows)} rows, {len({r['subject_id'] for r in rows})} subjects, "
              f"classes {dict(counts)}, folds "
              f"{ {f: sum(r['fold'] == f for r in rows) for f in sorted({r['fold'] for r in rows})} }")
        declared = document["datasets"][dataset]
        if int(declared.get("manifest_rows", len(rows))) != len(rows):
            print(f"  WARNING: {dataset} regenerated {len(rows)} rows, authority declares "
                  f"{declared['manifest_rows']} -- your canonical data differs from the release")
        if args.dry_run:
            continue
        out = MANIFEST_ROOT / f"{dataset}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        digest = hashlib.sha256(out.read_bytes()).hexdigest()
        print(f"wrote {out.relative_to(ROOT)}  sha256={digest}")
        if digest != declared.get("manifest_sha256"):
            print(f"  note: digest differs from the authority pin "
                  f"({declared.get('manifest_sha256')}); build_fair_input.py will refuse "
                  f"a manifest whose digest does not match benchmark.yaml")


if __name__ == "__main__":
    main()
