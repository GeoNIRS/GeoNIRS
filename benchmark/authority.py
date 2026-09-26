from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "benchmark" / "benchmark.yaml"


def load(root: str | Path | None = None) -> dict:
    path = Path(root or ROOT) / "benchmark" / "benchmark.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def datasets(root: str | Path | None = None) -> list[str]:
    return list(load(root)["paper_benchmark"]["datasets"])


def folds(root: str | Path | None = None) -> list[int]:
    count = int(load(root)["paper_benchmark"]["folds"])
    return list(range(1, count + 1))


def seeds(root: str | Path | None = None) -> list[int]:
    return [int(seed) for seed in load(root)["paper_benchmark"]["seeds"]]


def selection(names: list[str] | tuple[str, ...] | None,
              root: str | Path | None = None) -> list[str]:
    allowed = datasets(root)
    if not names:
        return allowed
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise SystemExit(
            f"not benchmark datasets: {' '.join(unknown)}\n"
            f"benchmark/benchmark.yaml declares: {' '.join(allowed)}")
    return list(names)
