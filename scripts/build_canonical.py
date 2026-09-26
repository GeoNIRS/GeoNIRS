from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from zipfile import ZipFile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LEGACY_PRODUCER = ROOT / "scripts" / "legacy" / "preprocessing" / "run_full_canonical_v2.py"
CANONICAL_ROOT = Path(os.environ.get("GEONIRS_CANONICAL_ROOT",
                                     ROOT / "data" / "canonical"))




RAW_ROOT = Path(os.environ.get("GEONIRS_RAW_ROOT", ROOT / "data" / "raw"))
DS2_ROOT = Path(os.environ.get("GEONIRS_RAW_DS2", RAW_ROOT / "2_ds004514"))
DS16_ROOT = Path(os.environ.get("GEONIRS_RAW_DS16", RAW_ROOT / "16_NIRS_01-29"))
DS8_ZIP = Path(os.environ.get("GEONIRS_DS8_ARCHIVE", RAW_ROOT / "DS8" / "FNIRS_SNIRF.zip"))


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


def ds2_jobs(limit: int | None) -> list[tuple[str, str]]:
    paths = sorted(DS2_ROOT.glob("sub-*/nirs/*_nirs.snirf"))
    return [("DS2", str(path)) for path in paths[:limit]]


def ds8_jobs(limit: int | None) -> list[tuple[str, str]]:
    with ZipFile(DS8_ZIP) as archive:
        entries = sorted(name for name in archive.namelist()
                         if name.startswith("FNIRS_SNIRF/") and name.endswith(".snirf"))
    return [("DS8", f"{DS8_ZIP}::{entry}") for entry in entries[:limit]]


def ds16_jobs(limit: int | None) -> list[tuple[str, str]]:
    import scipy.io as sio

    jobs = []
    for subject_dir in sorted(DS16_ROOT.glob("subject *")):
        cnt = subject_dir / "cnt.mat"
        if not cnt.is_file():
            continue
        subject = subject_dir.name.replace("subject ", "sub-")
        sessions = np.atleast_1d(sio.loadmat(cnt, squeeze_me=True,
                                             struct_as_record=False)["cnt"])
        for session_index in range(6):
            task = str(sessions[session_index].title)
            identity = f"{subject}_ses-{session_index + 1:02d}_task-{task}"
            jobs.append(("DS16", f"{cnt}::{identity}_session-{session_index}"))
    return jobs[:limit] if limit else jobs


JOB_BUILDERS = {"DS2": ds2_jobs, "DS16": ds16_jobs, "DS8": ds8_jobs}


def main() -> None:
    global CANONICAL_ROOT, DS2_ROOT, DS16_ROOT, DS8_ZIP
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(JOB_BUILDERS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--serial", action="store_true",
                        help="run jobs in this process (no multiprocessing)")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT,
                        help="root holding the raw releases (default: %(default)s)")
    parser.add_argument("--ds2-root", type=Path, default=DS2_ROOT)
    parser.add_argument("--ds16-root", type=Path, default=DS16_ROOT)
    parser.add_argument("--ds8-archive", type=Path, default=DS8_ZIP)
    parser.add_argument("--canonical-root", type=Path, default=CANONICAL_ROOT,
                        help="canonical HDF5 tree to write (default: %(default)s)")
    args = parser.parse_args()

    CANONICAL_ROOT, DS2_ROOT, DS16_ROOT, DS8_ZIP = (
        args.canonical_root, args.ds2_root, args.ds16_root, args.ds8_archive)

    producer = load_producer()
    producer.OUT = CANONICAL_ROOT
    jobs = JOB_BUILDERS[args.dataset](args.limit)
    print(f"{args.dataset}: {len(jobs)} 个作业 -> {producer.OUT}")

    results = []

    def note(index: int, record: dict) -> None:
        results.append(record)
        if index % 5 == 0 or index == len(jobs) or record["status"] == "FAIL":
            print(f"  {index}/{len(jobs)} {record['status']} "
                  f"{Path(record['source'].split('::')[-1]).name}", flush=True)

    if args.serial:
        for index, job in enumerate(jobs, 1):
            try:
                record = producer.run(job)
            except Exception as error:
                record = {"dataset": job[0], "source": job[1], "status": "FAIL",
                          "error": f"{type(error).__name__}: {error}"}
            note(index, record)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(producer.run, job): job for job in jobs}
            for index, future in enumerate(as_completed(futures), 1):
                job = futures[future]
                try:
                    record = future.result()
                except Exception as error:
                    record = {"dataset": job[0], "source": job[1], "status": "FAIL",
                              "error": f"{type(error).__name__}: {error}"}
                note(index, record)

    passed = [r for r in results if r["status"] == "PASS"]
    failed = [r for r in results if r["status"] == "FAIL"]
    print(f"\n完成: PASS={len(passed)} EXISTS={sum(r['status']=='EXISTS' for r in results)} FAIL={len(failed)}")
    for record in failed[:10]:
        print("  FAIL", Path(record["source"].split("::")[-1]).name, record.get("error"))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
        print("报告:", args.report)


if __name__ == "__main__":
    main()
