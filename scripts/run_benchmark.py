from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DOWNSTREAM = ROOT / "benchmark" / "downstream" / "downstream.yaml"
TRAIN = ROOT / "benchmark" / "downstream" / "train.py"
SHARDS = ROOT / "benchmark" / "results" / "_shards_v2"
MERGED = ROOT / "benchmark" / "results" / "matrix_v2"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.authority import datasets as benchmark_datasets
from benchmark.authority import load as load_benchmark

OURS_METHODS = ("ours_lp", "ours_lora", "ours_ft")
BASELINE_METHODS = ("cnn_1d", "fnirs_t", "fnirs_4d", "fnirsnet")




COST_PER_ROW = {
    "ours_lp": 0.225, "ours_lora": 0.337, "ours_ft": 0.258,
    "cnn_1d": 0.140, "fnirs_t": 0.169, "fnirs_4d": 0.181, "fnirsnet": 0.116,
    "mean_slope_lda": 0.002, "statistical_rbf_svm": 0.035,
    "jung_mae_lp": 0.225, "jung_mae_lora": 0.337, "jung_mae_ft": 0.258,
}


def methods() -> list[str]:
    config = yaml.safe_load(DOWNSTREAM.read_text(encoding="utf-8"))
    return (list(config["ours_methods"]) + list(config["baseline_methods"])
            + list(config["classical_methods"]) + list(config.get("mae_methods", [])))


def mae_methods() -> list[str]:
    config = yaml.safe_load(DOWNSTREAM.read_text(encoding="utf-8"))
    return list(config.get("mae_methods", []))


def datasets() -> list[str]:
    return benchmark_datasets(ROOT)


def protocol() -> dict:
    return yaml.safe_load(DOWNSTREAM.read_text(encoding="utf-8"))["protocol"]


def train_rows(dataset: str, fold: int) -> int:
    import h5py
    import numpy as np

    config = load_benchmark(ROOT)
    path = ROOT / config["fair_input"]["tensor"]
    with h5py.File(path, "r") as handle:
        roles = np.asarray(handle[dataset]["split_role"][:, fold - 1])
    return int((roles == 0).sum())


def shard_dir(seed: int, dataset: str, fold: int) -> Path:
    return SHARDS / f"seed-{seed}" / f"{dataset}_fold{fold}"


def shard_path(seed: int, dataset: str, fold: int, method: str) -> Path:
    return shard_dir(seed, dataset, fold) / f"{method}.json"


def merged_path(seed: int, dataset: str, fold: int) -> Path:
    return MERGED / f"seed-{seed}" / f"{dataset}_fold{fold}.json"





def run_stage(args) -> None:
    grid = list(methods())
    if args.methods:
        grid = [m for m in grid if m in set(args.methods)]
    datasets_ = args.datasets or datasets()
    seeds = args.seeds

    jobs = []
    for dataset in datasets_:
        for fold in range(1, 6):
            rows = train_rows(dataset, fold)
            for seed in seeds:
                for method in grid:
                    out = shard_path(seed, dataset, fold, method)
                    if out.is_file():
                        continue
                    jobs.append({
                        "dataset": dataset, "fold": fold, "seed": seed,
                        "method": method, "out": out,
                        "cost": rows * COST_PER_ROW.get(method, 0.2),
                    })
    jobs.sort(key=lambda job: -job["cost"])
    print(f"[run] {len(jobs)} jobs to do "
          f"({','.join(datasets_)} x 5 folds x {len(seeds)} seeds x {len(grid)} methods)")
    if not jobs:
        return

    proto = protocol()
    running: list[tuple[subprocess.Popen, int, dict]] = []
    gpu_load = [0] * args.gpus
    finished = failed = 0
    started = time.time()

    def start(job):
        gpu = min(range(args.gpus), key=lambda index: gpu_load[index])
        gpu_load[gpu] += 1
        job["out"].parent.mkdir(parents=True, exist_ok=True)
        log = job["out"].with_suffix(".log").open("w", encoding="utf-8")
        command = [
            args.python, "-u", str(TRAIN),
            "--dataset", job["dataset"], "--fold", str(job["fold"]),
            "--method", job["method"],
            "--epochs", str(proto["epochs"]),
            "--patience", str(proto["early_stopping_patience"]),
            "--batch-size", str(proto["batch_size"]),
            "--seed", str(job["seed"]),
            "--device", f"cuda:{gpu}",
            "--output-dir", str(job["out"].parent),
            "--output-name", job["out"].name,
        ]
        if job["method"] in mae_methods():
            command += ["--enable-mae-training",
                        "--mae-checkpoint", str(args.mae_checkpoint)]
        env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=env
        )
        running.append((process, gpu, job))

    pending = list(jobs)
    while pending or running:
        while pending and len(running) < args.concurrency:
            start(pending.pop(0))
        time.sleep(5)
        for entry in list(running):
            process, gpu, job = entry
            if process.poll() is None:
                continue
            running.remove(entry)
            gpu_load[gpu] -= 1
            if job["out"].is_file():
                finished += 1
            else:
                failed += 1
                print(f"[run] FAILED {job['dataset']} fold{job['fold']} "
                      f"seed{job['seed']} {job['method']} (exit {process.returncode})")
            total = finished + failed
            if total % 25 == 0 or total == len(jobs):
                rate = total / max(time.time() - started, 1)
                left = (len(jobs) - total) / rate if rate else 0
                print(f"[run] {total}/{len(jobs)} done, {failed} failed, "
                      f"in flight {len(running)}, eta {left/60:.0f} min", flush=True)
    print(f"[run] finished={finished} failed={failed}")





def merge_cell(seed: int, dataset: str, fold: int, grid: list[str]) -> dict | None:
    shards = {}
    for method in grid:
        path = shard_path(seed, dataset, fold, method)
        if not path.is_file():
            return None
        shards[method] = json.loads(path.read_text(encoding="utf-8"))

    head = next(iter(shards.values()))
    result = {key: head[key] for key in
              ("dataset", "fold", "num_classes", "tensor_shape", "patch_count",
               "split_sizes", "split_digest", "split_index_digest", "seed",
               "protocol")}
    result["methods"] = {}
    fairness_of = {}
    runtimes = {}
    for method in grid:
        result["methods"][method] = shards[method]["methods"][method]
        fairness_of[method] = shards[method]["fairness"]
        runtimes[method] = shards[method].get("runtime", {})

    evaluated = [m for m in grid
                 if "error" not in result["methods"][m]
                 and result["methods"][m].get("status") != "INCOMPATIBLE_WINDOW"]
    reference = fairness_of[evaluated[0]] if evaluated else fairness_of[grid[0]]
    observed = {}
    for method in evaluated:
        observed.update(fairness_of[method].get("per_method_row_index_digest", {}))
    declared = reference["declared_row_index_digest"]
    identical = bool(observed) and all(v == declared for v in observed.values())
    overlaps = reference["split_overlap_counts"]

    result["fairness"] = {
        "identical_input_rows": identical,
        "identical_split_indices": identical,
        "splits_are_disjoint": all(v == 0 for v in overlaps.values()),
        "identical_protocol_across_all_methods":
            len({json.dumps(result["methods"][m]["protocol"], sort_keys=True)
                 for m in result["methods"]}) == 1,
        "gradient_trained_methods":
            [m for m in evaluated if m in OURS_METHODS + BASELINE_METHODS],
        "same_budget_for_all_gradient_methods":
            all(entry["protocol"] == result["protocol"]
                for entry in result["methods"].values()),
        "evaluated_methods": evaluated,
        "declared_row_count": reference["declared_row_count"],
        "declared_row_index_digest": declared,
        "declared_row_bytes_digest": reference["declared_row_bytes_digest"],
        "per_method_row_index_digest": observed,
        "split_overlap_counts": overlaps,
        "distinct_split_digests": reference["distinct_split_digests"],
        "byte_level_cross_model_audit": reference["byte_level_cross_model_audit"],
        "sharded_run": {
            "path": f"benchmark/results/_shards_v2/seed-{seed}/{dataset}_fold{fold}/<method>.json",
            "note": ("produced as one process per method; the per-method row "
                     "digests above were recorded independently by each "
                     "process and merged here"),
        },
    }

    if runtimes:
        head_runtime = next(iter(runtimes.values()))


        union_files: dict[str, str] = {}
        for value in runtimes.values():
            union_files.update(value.get("code_files") or {})
        import hashlib as _hashlib
        result["runtime"] = {
            key: head_runtime.get(key)
            for key in ("python", "torch", "numpy", "cuda",
                        "torch_cuda_available", "seed_applied_to")
        }
        result["runtime"]["code_sha256"] = _hashlib.sha256(
            "\n".join(f"{k}\0{v}" for k, v in sorted(union_files.items())).encode()
        ).hexdigest()
        result["runtime"]["code_files"] = union_files
        result["runtime"]["per_method_code_sha256"] = {
            method: value.get("code_sha256") for method, value in runtimes.items()
        }
        result["runtime"]["per_method_device"] = {
            method: value.get("device") for method, value in runtimes.items()
        }
    return result


def merge_stage(args) -> None:
    grid = methods()
    datasets_ = args.datasets or datasets()
    merged = 0
    for seed in args.seeds:
        for dataset in datasets_:
            for fold in range(1, 6):
                result = merge_cell(seed, dataset, fold, grid)
                if result is None:
                    done = sum(shard_path(seed, dataset, fold, m).is_file() for m in grid)
                    if done:
                        print(f"[merge] {dataset} fold{fold} seed{seed}: {done}/{len(grid)} shards")
                    continue
                out = merged_path(seed, dataset, fold)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                merged += 1
    print(f"[merge] wrote {merged} cells")





def summarize_stage(args) -> None:
    grid = methods()
    datasets_ = args.datasets or datasets()
    rows = []
    for dataset in datasets_:
        for method in grid:
            per_seed = []
            for seed in args.seeds:
                folds = []
                for fold in range(1, 6):
                    path = merged_path(seed, dataset, fold)
                    if not path.is_file():
                        continue
                    entry = json.loads(path.read_text(encoding="utf-8"))["methods"][method]
                    if "test" in entry:
                        folds.append(entry["test"]["balanced_accuracy"])
                if folds:
                    per_seed.append(statistics.mean(folds))
            if not per_seed:
                continue
            rows.append({
                "dataset": dataset,
                "method": method,
                "n_seeds": len(per_seed),
                "test_balanced_accuracy_mean": statistics.mean(per_seed),
                "test_balanced_accuracy_std_across_seeds":
                    statistics.pstdev(per_seed) if len(per_seed) > 1 else 0.0,
                "per_seed": ";".join(f"{value:.4f}" for value in per_seed),
            })

    MERGED.mkdir(parents=True, exist_ok=True)
    csv_path = MERGED / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Formal V2 downstream summary", "",
             f"Seeds: {', '.join(str(s) for s in args.seeds)}. "
             "Values are test balanced accuracy, averaged over the 5 folds "
             "within a seed, then mean ± population std across seeds.", "",
             "| dataset | method | mean | std | per seed |",
             "|---|---|---:|---:|---|"]
    for dataset in datasets_:
        subset = [r for r in rows if r["dataset"] == dataset]
        for row in sorted(subset, key=lambda r: -r["test_balanced_accuracy_mean"]):
            lines.append(f"| {row['dataset']} | `{row['method']}` | "
                         f"{row['test_balanced_accuracy_mean']:.4f} | "
                         f"{row['test_balanced_accuracy_std_across_seeds']:.4f} | "
                         f"{row['per_seed']} |")
    (MERGED / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[summarize] wrote {csv_path} and {MERGED / 'summary.md'} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default="all",
                        choices=("run", "merge", "summarize", "all"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[810])
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=96)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--mae-checkpoint", type=Path,
                        default=ROOT / "pretrain_checkpoint" / "geonirs_mae.pt",
                        help="released Jung-inspired MAE checkpoint read only by "
                             "the jung_mae_* arms (default: %(default)s)")
    args = parser.parse_args()

    if args.stage in ("run", "all"):
        run_stage(args)
    if args.stage in ("merge", "all"):
        merge_stage(args)
    if args.stage in ("summarize", "all"):
        summarize_stage(args)


if __name__ == "__main__":
    main()
