from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "benchmark" / "benchmark.yaml"
FAIR_INPUT_DIR = ROOT / "benchmark/fair_input"


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "formal_v2_fair_input_builder", ROOT / "scripts/build_fair_input.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_dataset(builder, config, records, dataset):
    entry = records["datasets"][dataset]
    path = ROOT / config["fair_input"]["tensor"]
    manifest_path, rows = builder.read_manifest(config, dataset)
    accepted, _ = builder.accepted_rows(rows)
    expected_ids = [row["sample_id"] for row in accepted]
    expected_label = {row["sample_id"]: int(row["label_id"]) for row in accepted}
    expected_fold = {row["sample_id"]: int(row["fold"]) for row in accepted}
    expected_subject = {row["sample_id"]: row["subject_id"] for row in accepted}
    roles = builder.split_roles(config, dataset)

    check = {}
    with h5py.File(path, "r") as handle:
        group = handle[dataset]
        sample_ids = group["sample_id"].asstr()[()].tolist()
        labels = np.asarray(group["label"], dtype=np.int64)
        folds = np.asarray(group["fold"], dtype=np.int64)
        subjects = group["subject_id"].asstr()[()].tolist()
        split_role = np.asarray(group["split_role"], dtype=np.int64)
        channel_ids = group["channel_ids"].asstr()[()].tolist()
        geometry = np.asarray(group["geometry"], dtype=np.float64)
        check["sample_id_sequence_matches_policy_b"] = sample_ids == expected_ids
        check["labels_match_manifest"] = all(
            int(value) == expected_label[sid] for sid, value in zip(sample_ids, labels))
        check["folds_match_manifest"] = all(
            int(value) == expected_fold[sid] for sid, value in zip(sample_ids, folds))
        check["subjects_match_manifest"] = all(
            value == expected_subject[sid] for sid, value in zip(sample_ids, subjects))
        check["split_roles_match_folds"] = all(
            split_role[index, fold - 1] == roles[subject][fold]
            for index, subject in enumerate(subjects)
            for fold in range(1, 6))
        check["channel_order_matches_authority"] = channel_ids == list(entry["channel_names"])
        check["tensor_shape_matches_authority"] = list(group["hbo"].shape[1:]) == list(entry["tensor_shape"])
        check["geometry_dataset_level"] = (geometry.ndim == 2 and geometry.shape[1] == 6
                                           and np.isfinite(geometry).all()
                                           and group.attrs["geometry_scope"] == "DATASET_LEVEL_FIXED")
        hbo = np.asarray(group["hbo"])
        hbr = np.asarray(group["hbr"])
        valid = np.asarray(group["valid"], dtype=bool)
        channel = np.asarray(group["valid_channel_mask"], dtype=bool)
        check["zero_fill_exact"] = bool((hbo[~valid] == 0).all() and (hbr[~valid] == 0).all())
        check["channel_mask_consistent"] = bool((channel == valid.any(axis=2)).all())
        check["values_finite"] = bool(np.isfinite(hbo).all() and np.isfinite(hbr).all())
        class_counts = Counter(group["label"][()].tolist())

    names = list(config["datasets"][dataset]["task"]["labels"])
    expected_classes = {names[index]: count for index, count in sorted(class_counts.items())}
    check["class_counts_match_records"] = expected_classes == records["datasets"][dataset]["class_counts"]
    return check


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FAIR_INPUT_DIR)
    parser.add_argument("--datasets", nargs="*")
    args = parser.parse_args()
    builder = load_builder()
    config = yaml.safe_load(AUTHORITY.read_text(encoding="utf-8"))
    records = json.loads((FAIR_INPUT_DIR / "RECORDS.json").read_text(encoding="utf-8"))
    datasets = args.datasets or list(config["paper_benchmark"]["datasets"])

    results = {}
    for dataset in datasets:
        results[dataset] = verify_dataset(builder, config, records, dataset)
        failed = [key for key, value in results[dataset].items() if not value]
        print(dataset, "PASS" if not failed else f"FAIL {failed}", flush=True)

    payload = {"authority": "benchmark/benchmark.yaml",
               "fair_input_records_sha256": config["fair_input"]["records_sha256"],
               "results": results}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "FAIR_INPUT_VERIFICATION.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = ["# Fair Input verification", "",
             f"Frozen input contract: `{config['fair_input']['records']}` sha256 `{config['fair_input']['records_sha256']}`",
             f"Policy **{config['fair_input']['policy']} {config['fair_input']['policy_name']}**", "",
             "Policy B accepted sample IDs were re-derived from the manifests and canonical",
             "HDF5 and compared against the frozen Fair Input.", "",
             "| Dataset | Checks | Failures |", "|---|---|---|"]
    for dataset in datasets:
        failed = [key for key, value in results[dataset].items() if not value]
        lines.append(f"| {dataset} | {len(results[dataset])} | {', '.join(failed) or 'none'} |")
    (args.output_dir / "FAIR_INPUT_VERIFICATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = ["# Fair Input report", "",
              f"Policy: **{config['fair_input']['policy']} {config['fair_input']['policy_name']}**  ",
              f"Tensors: `{config['fair_input']['tensor']}` (not distributed)  ",
              f"Records SHA256: `{config['fair_input']['records_sha256']}`  ",
              f"Missing-value fill: `{config['fair_input']['missing_value_fill']}` (owned by {config['fair_input']['zero_padding_owner']})  ",
              f"Geometry: {config['fair_input']['geometry_scope']} `{config['fair_input']['geometry_variant']}`", ""]
    for dataset in datasets:
        item = records["datasets"][dataset]
        report += [
            f"## {dataset}", "",
            f"- N samples: **{item['n_samples']}** (candidates {item['candidate_samples']}, "
            f"rejected {item['rejected_samples']} {item['rejection_reasons']})",
            f"- Tensor shape (HbO/HbR): **{item['tensor_shape']}**",
            f"- Channel order: {item['channel_order']}",
            f"- Channels: {', '.join(item['channel_names'])}",
            f"- Zero-padded scalar fraction: {item['zero_padding']['zero_padded_scalar_fraction']} "
            f"(mean padded channels/sample {item['zero_padding']['mean_zero_padded_channels_per_sample']}, "
            f"max {item['zero_padding']['max_zero_padded_channels_per_sample']})",
            f"- Class counts: {item['class_counts']}",
            f"- Split counts: {item['split_counts']}",
            f"- Sample IDs / split IDs: `{item['index_csv']}`",
            f"- Sample ID SHA256: `{item['sample_id_sha256']}`",
            f"- Tensor SHA256: `{item['tensor_sha256']}`",
            "- Geometry: dataset-level fixed `" + config["fair_input"]["geometry_variant"] +
            "` stored separately as [C,6] " + str(config["fair_input"]["geometry_fields"]), "",
        ]
    (args.output_dir / "FAIR_INPUT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("wrote", args.output_dir / "FAIR_INPUT_REPORT.md")


if __name__ == "__main__":
    main()
