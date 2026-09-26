from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "benchmark" / "benchmark.yaml"
DOWNSTREAM = ROOT / "benchmark" / "downstream" / "downstream.yaml"
SUMMARY = ROOT / "benchmark" / "results" / "matrix_v2" / "summary.csv"


def dataset_order() -> list[str]:
    document = yaml.safe_load(AUTHORITY.read_text(encoding="utf-8"))
    return list(document["paper_benchmark"]["datasets"])


def method_order() -> list[str]:
    config = yaml.safe_load(DOWNSTREAM.read_text(encoding="utf-8"))
    return (list(config["ours_methods"]) + list(config["baseline_methods"])
            + list(config["classical_methods"]) + list(config.get("mae_methods", [])))


def metric_columns(rows: list[dict]) -> tuple[str, str | None]:
    fields = list(rows[0].keys())
    mean = next((f for f in fields if f.endswith("_mean")), None)
    if mean is None:
        raise SystemExit("summary.csv has no *_mean metric column")
    std = mean.replace("_mean", "_std_across_seeds")
    return mean, (std if std in fields else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark" / "results")
    args = parser.parse_args()
    if not args.summary.is_file():
        raise SystemExit(f"missing {args.summary}; run scripts/run_benchmark.py --stage all first")

    rows = list(csv.DictReader(args.summary.open(newline="", encoding="utf-8")))
    if not rows:
        raise SystemExit(f"{args.summary} is empty; run scripts/run_benchmark.py --stage all first")
    mean_column, std_column = metric_columns(rows)

    by_cell = {(row["dataset"], row["method"]): row for row in rows}
    datasets = [d for d in dataset_order() if any(key[0] == d for key in by_cell)]
    datasets += sorted({key[0] for key in by_cell} - set(datasets))
    methods = [m for m in method_order() if any(key[1] == m for key in by_cell)]
    methods += sorted({key[1] for key in by_cell} - set(methods))

    def cell(dataset: str, method: str) -> str:
        row = by_cell.get((dataset, method))
        if row is None:
            return ""
        mean = float(row[mean_column])
        if std_column is None:
            return f"{mean:.4f}"
        return f"{mean:.4f} ± {float(row[std_column]):.4f}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "main_table.csv"
    md_path = args.output_dir / "main_table.md"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", *methods])
        for dataset in datasets:
            writer.writerow([dataset, *[cell(dataset, method) for method in methods]])

    lines = ["# Main table", "",
             f"Metric: `{mean_column}` (5 folds averaged within a seed, then mean ± "
             f"population std across seeds), from `{args.summary.relative_to(ROOT)}`.", "",
             "| dataset | " + " | ".join(methods) + " |",
             "|---|" + "---|" * len(methods)]
    for dataset in datasets:
        lines.append(f"| {dataset} | " + " | ".join(cell(dataset, method) for method in methods) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"datasets": datasets, "methods": methods,
                      "csv": str(csv_path.relative_to(ROOT)),
                      "markdown": str(md_path.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
