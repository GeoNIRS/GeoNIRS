from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import h5py
import yaml


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "benchmark" / "benchmark.yaml"
CANONICAL = Path(os.environ.get("GEONIRS_CANONICAL_ROOT",
                                ROOT / "data" / "canonical"))
MANIFESTS = ROOT / "benchmark" / "manifests"

FS = 5.0
WINDOW_SECONDS = 12.0
T = int(round(FS * WINDOW_SECONDS))
FOLD_SEED = 810
FOLD_COUNT = 5

FIELDS = ["dataset_id", "sample_id", "subject_id", "canonical_recording_key",
          "canonical_hdf5_path", "recording_condition", "event_index",
          "event_label", "event_onset_s", "window_start_sample",
          "window_end_sample", "window_start_s", "window_end_s", "n_timepoints",
          "label", "label_id", "fold"]

LABELS = {
    "DS51": ("Positive", "Negative"),
    "DS30": ("Left", "Right"),
    "DS36": ("Tinnitus", "Control"),
    "DS42": ("Observation", "Execution"),
    "DS19": ("Low pain", "High pain"),
    "DS70E": ("Happy", "Sad"),
}





DS36_PARTICIPANTS = Path(os.environ.get(
    "GEONIRS_DS36_PARTICIPANTS",
    ROOT / "data" / "raw" / "36_ds007990" / "participants.tsv"))


def ds36_subject_labels() -> dict[str, str]:
    mapping: dict[str, str] = {}
    with DS36_PARTICIPANTS.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            group = str(row.get("group", "")).strip().lower()
            if group == "experimental":
                mapping[str(row["participant_id"])] = "Tinnitus"
            elif group == "control":
                mapping[str(row["participant_id"])] = "Control"
    if not mapping:
        raise SystemExit("DS36 participants table yielded no groups")
    return mapping


def bbn_only(event: dict) -> bool:
    return str(event.get("trial_type", "")).strip().upper() == "BBN"


def mirror_label(event: dict) -> str | None:
    kind = str(event.get("trial_type", "")).strip()
    if kind.endswith("Observation"):
        return "Observation"
    if kind.endswith("Execution"):
        return "Execution"
    return None


def pain_label(event: dict) -> str | None:
    kind = str(event.get("trial_type", "")).replace(" ", "").upper()
    if kind.endswith("VAS3"):
        return "Low pain"
    if kind.endswith("VAS7"):
        return "High pain"
    return None


def valence_label(event: dict) -> str | None:
    kind = str(event.get("trial_type", "")).split("_")[-1].upper()
    if kind.endswith("PV"):
        return "Positive"
    if kind.endswith("NV"):
        return "Negative"
    return None


def attention_label(event: dict) -> str | None:
    if str(event.get("include", "1")) == "0":
        return None
    kind = str(event.get("trial_type", ""))
    if "Left" in kind:
        return "Left"
    if "Right" in kind:
        return "Right"
    return None


def threat_label(event: dict) -> str | None:
    kind = str(event.get("trial_type", "")).strip()
    if kind == "Threatening":
        return "Threatening"
    if kind == "Non-Threatening":
        return "NonThreatening"
    return None


def emotion_label(event: dict) -> str | None:
    kind = str(event.get("trial_type", "")).strip()
    if kind.endswith("Happy"):
        return "Happy"
    if kind.endswith("Sad"):
        return "Sad"
    return None


def music_label(event: dict) -> str | None:
    kind = str(event.get("trial_type", "")).strip()
    if kind.startswith("PosValence"):
        return "Positive"
    if kind.startswith("NegValence"):
        return "Negative"
    return None


TASKS = {

    "DS51": {"canonical_dataset": "DS51", "conditions": ("task-empe", "task-afim"),
             "classify": valence_label},
    "DS30": {"canonical_dataset": "DS30", "conditions": ("task-covert", "task-overt"),
             "classify": attention_label},
    "DS36": {"canonical_dataset": "DS36", "conditions": ("task-tinnitus",),
             "classify": None, "subject_labels": "participants",
             "event_filter": bbn_only},
    "DS42": {"canonical_dataset": "DS42", "conditions": ("task-MirrorNeuron",),
             "classify": mirror_label},
    "DS19": {"canonical_dataset": "DS19", "conditions": ("task-Electrical",),
             "classify": pain_label},


    "DS70E": {"canonical_dataset": "DS70", "conditions": ("task-emotion",),
              "classify": emotion_label},
}


def subject_folds(subjects) -> dict[int, dict[str, list[str]]]:
    ordered = sorted(subjects)
    random.Random(FOLD_SEED).shuffle(ordered)
    groups = [ordered[index::FOLD_COUNT] for index in range(FOLD_COUNT)]
    folds = {}
    for fold in range(1, FOLD_COUNT + 1):
        test = set(groups[fold - 1])
        validation = set(groups[fold % FOLD_COUNT])
        train = set(ordered) - test - validation
        folds[fold] = {"train_subjects": sorted(train),
                       "validation_subjects": sorted(validation),
                       "test_subjects": sorted(test)}
    return folds


def check_layout_against_frozen() -> None:
    config = yaml.safe_load(AUTHORITY.read_text(encoding="utf-8"))
    frozen = config["datasets"]["DS7"]["folds"]
    same = all(set(frozen[k]["test_subjects"]) == set(frozen[k - 1]["validation_subjects"])
               for k in range(2, 6))
    wrap = set(frozen[1]["test_subjects"]) == set(frozen[5]["validation_subjects"])
    if not (same and wrap):
        raise SystemExit("frozen DS7 folds do not follow the assumed rotation")


def build(dataset: str) -> tuple[list[dict], dict[int, dict[str, list[str]]]]:
    spec = TASKS[dataset]
    label_names = LABELS[dataset]
    rows: list[dict] = []
    subjects: set[str] = set()
    canonical_dataset = spec["canonical_dataset"]
    subject_labels = (ds36_subject_labels()
                      if spec.get("subject_labels") == "participants" else {})
    event_filter = spec.get("event_filter")
    for path in sorted((CANONICAL / canonical_dataset).glob("*.h5")):
        with h5py.File(path, "r") as handle:
            condition = str(handle.attrs.get("task", ""))
            if condition not in spec["conditions"]:
                continue
            recording = str(handle.attrs.get("recording_id", path.stem))
            subject = str(handle.attrs.get("subject", ""))
            length = int(handle["hbo"].shape[0])
            events = json.loads(handle["events_json"][()].decode())
        if subject not in subject_labels and spec.get("subject_labels"):
            raise SystemExit(f"{dataset}: no group for subject {subject!r}")
        subjects.add(subject)
        for index, event in enumerate(events):
            if event_filter is not None and not event_filter(event):
                continue
            label = (subject_labels[subject] if spec.get("subject_labels")
                     else spec["classify"](event))
            if label is None:
                continue
            start = int(round(float(event["onset"]) * FS))
            end = start + T
            if start < 0 or end > length:
                continue
            rows.append({
                "dataset_id": dataset,
                "sample_id": f"{dataset}::{recording}::event{index}",
                "subject_id": subject,
                "canonical_recording_key": recording,
                "canonical_hdf5_path": f"data/canonical/{canonical_dataset}/{path.name}",
                "recording_condition": condition,
                "event_index": index,
                "event_label": str(event.get("trial_type", "")),
                "event_onset_s": float(event["onset"]),
                "window_start_sample": start,
                "window_end_sample": end,
                "window_start_s": start / FS,
                "window_end_s": end / FS,
                "n_timepoints": T,
                "label": label,
                "label_id": label_names.index(label),
            })
    folds = subject_folds(sorted(subjects))
    test_of = {subject: fold for fold, item in folds.items()
               for subject in item["test_subjects"]}
    for row in rows:
        row["fold"] = test_of[row["subject_id"]]
    return rows, folds


def main() -> None:
    global CANONICAL, DS36_PARTICIPANTS
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="build only these datasets (default: all in TASKS). "
                             "Used to add datasets without rewriting the "
                             "manifests of the ones already frozen.")
    parser.add_argument("--canonical-root", type=Path, default=CANONICAL,
                        help="canonical HDF5 tree to read (default: %(default)s)")
    parser.add_argument("--ds36-participants", type=Path, default=DS36_PARTICIPANTS,
                        help="BIDS participants.tsv of DS36 (default: %(default)s)")
    args = parser.parse_args()

    CANONICAL, DS36_PARTICIPANTS = args.canonical_root, args.ds36_participants

    check_layout_against_frozen()
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    summary = {}
    for dataset in (args.datasets or list(TASKS)):
        if dataset not in TASKS:
            raise SystemExit(f"unknown dataset: {dataset}")
        rows, folds = build(dataset)
        counts = defaultdict(int)
        for row in rows:
            counts[row["label"]] += 1
        per_fold = defaultdict(int)
        for row in rows:
            per_fold[row["fold"]] += 1
        subjects = sorted({row["subject_id"] for row in rows})
        print(f"{dataset}: {len(rows)} 样本 / {len(subjects)} 受试者 / "
              f"类别 {dict(counts)} / 每折 {dict(sorted(per_fold.items()))}")
        summary[dataset] = {"rows": len(rows), "subjects": len(subjects),
                            "label_counts": dict(counts),
                            "fold_counts": {str(k): v for k, v in sorted(per_fold.items())},
                            "folds": folds,
                            "n_timepoints": T, "window_seconds": WINDOW_SECONDS,
                            "fold_seed": FOLD_SEED}
        if args.dry_run:
            continue
        path = MANIFESTS / f"{dataset}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  wrote {path}")
    if not args.dry_run:
        out = MANIFESTS / "_new_datasets_folds.json"


        merged = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
        merged.update(summary)
        out.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
