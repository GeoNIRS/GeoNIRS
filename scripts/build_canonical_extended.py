from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py


ROOT = Path(__file__).resolve().parents[1]
LEGACY_PRODUCER = ROOT / "scripts" / "legacy" / "preprocessing" / "run_full_canonical_v2.py"
CANONICAL_ROOT = Path(os.environ.get("GEONIRS_CANONICAL_ROOT",
                                     ROOT / "data" / "canonical"))




RAW_SUBDIRS = {
    "DS51": "51/nemo-bids/nemo-bids",
    "DS30": "30_ds007738",
    "DS36": "36_ds007990",
    "DS42": "42_ds005935",
    "DS19": "19_ds005777",
    "DS4": "4_ds004541",
    "DS39": "39_speech_b",
    "DS11": "11_cfdsz_pre",
    "DS49": "49_hyperscanning",
    "DS68": "68_osf_hmvsu",
    "DS69": "69",
    "DS70": "70/extracted/MULPA_v01",
}
RAW_ROOT = Path(os.environ.get("GEONIRS_RAW_ROOT", ROOT / "data" / "raw"))


def _raw(name: str) -> Path:
    return Path(os.environ.get(f"GEONIRS_RAW_{name}", RAW_ROOT / RAW_SUBDIRS[name]))


DS51_ROOT = _raw("DS51")
DS30_ROOT = _raw("DS30")
DS30_ATTENTION_TASKS = ("covert", "overt")



DS36_ROOT = _raw("DS36")
DS42_ROOT = _raw("DS42")
DS19_ROOT = _raw("DS19")

DS4_ROOT = _raw("DS4")
DS39_ROOT = _raw("DS39")
DS11_ROOT = _raw("DS11")
DS49_ROOT = _raw("DS49")



DS68_ROOT = _raw("DS68")


DS69_ROOT = _raw("DS69")




DS70_ROOT = _raw("DS70")


def _nirs_readable(path: Path) -> bool:
    try:
        from scipy.io import loadmat
        loaded = loadmat(str(path))
        loaded["d"]; loaded["t"]; loaded["SD"]
        return True
    except NotImplementedError:
        try:
            import h5py
            with h5py.File(path, "r") as handle:
                return "d" in handle and "t" in handle
        except Exception:
            return False
    except Exception:
        return False


def load_producer():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    for bare, package in (
        ("adapters", "preprocessing.formal_v2.adapters"),
        ("canonical_core", "preprocessing.formal_v2.canonical_core"),
        ("two_wavelength_snirf_adapter", "preprocessing.two_wavelength_snirf_adapter"),
    ):
        sys.modules.setdefault(bare, importlib.import_module(package))
    spec = importlib.util.spec_from_file_location("full_canonical_v2", LEGACY_PRODUCER)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("full_canonical_v2", module)
    spec.loader.exec_module(module)
    return module


def ds51_jobs(limit: int | None) -> list[tuple[str, str]]:
    paths = sorted(DS51_ROOT.glob("sub-*/nirs/*_nirs.snirf"))
    return [("DS51", str(path)) for path in paths[:limit]]


def ds30_jobs(limit: int | None) -> list[tuple[str, str]]:
    paths = []
    for task in DS30_ATTENTION_TASKS:
        paths += sorted(DS30_ROOT.glob(f"sub-*/nirs/*_task-{task}_run-*_nirs.snirf"))
    paths = sorted(set(paths))
    return [("DS30", str(path)) for path in paths[:limit]]


def ds36_jobs(limit: int | None) -> list[tuple[str, str]]:
    paths = sorted(DS36_ROOT.glob("sub-*/nirs/*_task-tinnitus_nirs.snirf"))
    return [("DS36", str(path)) for path in paths[:limit]]


def ds42_jobs(limit: int | None) -> list[tuple[str, str]]:
    paths = sorted(DS42_ROOT.glob("sub-*/ses-*/nirs/*_task-MirrorNeuron_run-*_nirs.snirf"))
    return [("DS42", str(path)) for path in paths[:limit]]


def ds19_jobs(limit: int | None) -> list[tuple[str, str]]:
    paths = sorted(DS19_ROOT.glob("sub-*/ses-*/nirs/*_task-Electrical_run-*_nirs.snirf"))
    return [("DS19", str(path)) for path in paths[:limit]]


def ds4_jobs(limit: int | None) -> list[tuple[str, str]]:
    paths = sorted(DS4_ROOT.glob("sub-*/ses-*/nirs/*_task-anesthesia_nirs.snirf"))
    return [("DS4", str(path)) for path in paths[:limit]]


def ds39_jobs(limit: int | None) -> list[tuple[str, str]]:
    paths = [p for p in sorted(DS39_ROOT.glob("*.nirs")) if _nirs_readable(p)]
    return [("DS39", str(path)) for path in paths[:limit]]


def ds11_jobs(limit: int | None) -> list[tuple[str, str]]:
    paths = [p for p in sorted(DS11_ROOT.glob("*.nirs")) if _nirs_readable(p)]
    return [("DS11", str(path)) for path in paths[:limit]]


def ds49_jobs(limit: int | None) -> list[tuple[str, str]]:
    paths = [p for p in sorted(DS49_ROOT.rglob("NIRS.nirs")) if _nirs_readable(p)]
    return [("DS49", str(path)) for path in paths[:limit]]


def ds68_jobs(limit: int | None) -> list[tuple[str, str]]:
    paths = [p for p in sorted(DS68_ROOT.glob("data/fNIRS data/*/*.snirf"))
             if not p.name.startswith("._")]
    return [("DS68", str(path)) for path in paths[:limit]]


def ds69_jobs(limit: int | None) -> list[tuple[str, str]]:
    paths = sorted(DS69_ROOT.glob("Subject*.csv"))
    paths = [p for p in paths if p.with_suffix(".pos").is_file()]
    return [("DS69", str(path)) for path in paths[:limit]]


def ds70_jobs(limit: int | None) -> list[tuple[str, str]]:
    paths = sorted(DS70_ROOT.glob("sub-*/nirs/*_nirs.snirf"))
    return [("DS70", str(path)) for path in paths[:limit]]


JOB_BUILDERS = {
    "DS51": ds51_jobs, "DS30": ds30_jobs, "DS36": ds36_jobs,
    "DS42": ds42_jobs, "DS19": ds19_jobs, "DS4": ds4_jobs,
    "DS39": ds39_jobs, "DS11": ds11_jobs, "DS49": ds49_jobs,
    "DS68": ds68_jobs, "DS69": ds69_jobs, "DS70": ds70_jobs,
}


def main() -> None:
    global CANONICAL_ROOT, RAW_ROOT
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(JOB_BUILDERS), required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT,
                        help="root holding the raw releases (default: %(default)s)")
    parser.add_argument("--dataset-root", action="append", default=[],
                        metavar="DS=PATH", help="raw directory of one dataset, repeatable")
    parser.add_argument("--canonical-root", type=Path, default=CANONICAL_ROOT,
                        help="canonical HDF5 tree to write (default: %(default)s)")
    args = parser.parse_args()

    CANONICAL_ROOT, RAW_ROOT = args.canonical_root, args.raw_root
    for name, sub in RAW_SUBDIRS.items():
        if f"GEONIRS_RAW_{name}" not in os.environ:
            globals()[f"{name}_ROOT"] = args.raw_root / sub
    for item in args.dataset_root:
        name, _, path = item.partition("=")
        if name not in RAW_SUBDIRS or not path:
            raise SystemExit(f"--dataset-root expects DS=PATH, got {item!r}")
        globals()[f"{name}_ROOT"] = Path(path)

    jobs = JOB_BUILDERS[args.dataset](args.limit)
    producer = load_producer()
    producer.OUT = CANONICAL_ROOT
    print(f"{args.dataset}: {len(jobs)} recordings -> {CANONICAL_ROOT / args.dataset}")
    if not jobs:
        raise SystemExit("no input files found")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(producer.run, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index % 20 == 0 or index == len(jobs):
                print(f"  {index}/{len(jobs)}", flush=True)





    if args.dataset == "DS51":
        fixed = 0
        for record in results:
            if record.get("status") != "PASS":
                continue
            with h5py.File(record["output"], "r+") as handle:
                handle.attrs["signal_type"] = "optical_density_as_intensity"
                handle.attrs["signal_type_note"] = (
                    "source distributes OD (dataTypeLabel=dOD); the adapter stores "
                    "exp(-OD) as intensity, which the canonical core turns back "
                    "into OD up to a per-column constant")
                fixed += 1
        print(f"  corrected signal_type on {fixed} DS51 files")

    passed = sum(1 for r in results if r.get("status") == "PASS")
    exists = sum(1 for r in results if r.get("status") == "EXISTS")
    print(f"[{args.dataset}] PASS {passed}  EXISTS {exists}  total {len(results)}")
    summary = CANONICAL_ROOT / f"{args.dataset}_build.json"
    summary.write_text(json.dumps({
        "dataset": args.dataset,
        "producer": "scripts/legacy/preprocessing/run_full_canonical_v2.py (unmodified)",
        "canonical_core": "preprocessing/formal_v2/canonical_core.py",
        "n_jobs": len(jobs), "n_pass": passed, "n_exists": exists,
        "records": results,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
