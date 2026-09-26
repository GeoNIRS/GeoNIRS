[README.md](https://github.com/user-attachments/files/32681947/README.md)
# GeoNIRS — downstream main-table reproduction

This repository contains the code needed to reproduce the paper's main table
from public downstream fNIRS data and the released GeoNIRS checkpoint:

```text
raw downstream data
  -> unified canonical preprocessing
  -> sample manifests
  -> Frozen Fair Input
  -> released pretrained checkpoint
  -> eight downstream tasks
  -> main table
```

It deliberately does **not** contain the pre-training pipeline, the ablation /
robustness / label-efficiency / scaling / random-initialisation experiments, or
any result-aggregation code beyond the summary and table formatting below.
Pre-training is not re-run: the released checkpoint is downloaded.

## Environment

Python 3.10 with `torch`, `numpy`, `h5py`, `scipy`, `scikit-learn`, `mne`,
`pyyaml` and `einops`; the `moment_lora` / `moment_base` arms additionally need
the public `momentfm` package and the `AutonLab/MOMENT-1-base` weights (fetched
from Hugging Face on first use, or pre-cached). A CUDA GPU is required for the
downstream runners (`--device cuda:N`).

## Step 1 — Download the downstream datasets

Create `data/raw/` in the repository root and place each public release in the
directory below. Download every dataset from its official page and unpack it
there; raw data is never part of this repository.

```text
data/raw/
├── DS7/
│   └── source_data/                 # DS7 raw .nirs files
├── DS47/
│   └── snirf_local-bids/            # DS47 BIDS/SNIRF files
├── 36_ds007990/
│   ├── participants.tsv
│   └── sub-*/nirs/
├── 19_ds005777/
│   └── sub-*/ses-*/nirs/
├── 70/
│   └── extracted/
│       └── MULPA_v01/
│           └── sub-*/nirs/
├── 2_ds004514/
│   └── sub-*/nirs/
├── DS8/
│   └── FNIRS_SNIRF.zip
└── 42_ds005935/
    └── sub-*/ses-*/nirs/
```

| Main-table task | Raw location |
|---|---|
| DS7 (Healthy vs ICH) | `data/raw/DS7/source_data/` |
| DS47 (CI vs NH) | `data/raw/DS47/snirf_local-bids/` |
| DS36 (Tinnitus vs Control) | `data/raw/36_ds007990/` |
| DS19 (pain intensity) | `data/raw/19_ds005777/` |
| DS70E (emotion, happy vs sad) | `data/raw/70/extracted/MULPA_v01/` |
| DS2 (animal vs tool) | `data/raw/2_ds004514/` |
| DS8 (motor execution vs rest) | `data/raw/DS8/FNIRS_SNIRF.zip` |
| DS42 (observation vs execution) | `data/raw/42_ds005935/` |

DS70E is the emotion task of the DS70 release, so its canonical source is
`data/canonical/DS70/`.

## Step 2 — Preprocess and build the benchmark input

```bash
bash scripts/prepare_downstream.sh
```

The wrapper only calls the Python entry points below, in order, and implements
no science of its own:

1. **Canonical preprocessing** — raw data to `data/canonical/<DS>/`

   ```bash
   # DS7 and DS47 (frozen Formal V2 producer)
   python scripts/legacy/preprocessing/run_full_canonical_v2.py \
       --raw-root data/raw --output data/canonical
   # DS2 and DS8
   python scripts/build_canonical.py --dataset DS2 \
       --ds2-root data/raw/2_ds004514 --canonical-root data/canonical
   python scripts/build_canonical.py --dataset DS8 \
       --ds8-archive data/raw/DS8/FNIRS_SNIRF.zip --canonical-root data/canonical
   # DS36, DS19, DS42 and the DS70 source behind DS70E
   python scripts/build_canonical_extended.py --dataset DS70 \
       --raw-root data/raw --canonical-root data/canonical
   ```

   All eight datasets use the same pipeline: OD conversion, QC, N-wavelength
   MBLL, resampling to 5 Hz, z-scoring and 6-D source–detector geometry.

2. **Sample manifests** — canonical data to `benchmark/manifests/<DS>.csv`

   ```bash
   python scripts/build_manifests_frozen.py                       # DS7, DS47
   python scripts/build_manifests.py --dataset DS2
   python scripts/build_manifests.py --dataset DS8
   python scripts/build_manifests_extended.py --datasets DS36 DS19 DS42 DS70E
   ```

   Sample definition, event selection, window length, label mapping and the
   subject-level 5-fold split are the frozen benchmark values.

3. **Frozen Fair Input** — manifests + canonical data to `benchmark/fair_input/`

   ```bash
   python scripts/build_fair_input.py \
       --datasets DS7 DS47 DS36 DS19 DS70E DS2 DS8 DS42 \
       --channel-policy intersection
   ```

   `--channel-policy intersection` is policy **B / CHANNEL_INTERSECTION**, the
   policy the paper's Fair Input uses: one dataset-level channel space, no
   zero-filled channels, one dataset-level geometry.

4. **Independent verification**

   ```bash
   python scripts/verify_fair_input.py
   ```

Every model in the main table reads this single Fair Input through
`benchmark/downstream/fair_input.py`. No baseline may re-preprocess, re-window,
interpolate, re-normalise or select different samples; `train.py` asserts that
all methods received identical rows.

## Step 3 — Pretrained checkpoints

Both released checkpoints are included in this repository; keep them at these
relative paths:

```text
pretrain_checkpoint/geonirs_base.pt    # GeoNIRS-base, read by ours_lp / ours_lora / ours_ft
pretrain_checkpoint/geonirs_mae.pt     # Jung-inspired MAE, read by jung_mae_lp / jung_mae_lora / jung_mae_ft
```

Their weights are unchanged; only machine-specific paths inside the checkpoint
metadata were normalised. No GeoNIRS pre-training is required or supported by
this repository.

## Step 4 — Reproduce the main table

```bash
python scripts/run_benchmark.py --stage all
```

or explicitly:

```bash
python scripts/run_benchmark.py --stage all \
    --datasets DS7 DS47 DS36 DS19 DS70E DS2 DS8 DS42 \
    --seeds 810 811
```

The runner schedules the whole grid itself — 8 datasets x 5 folds x 2 seeds x
every main-table method — one process per cell, resumable, and merges and
summarises the results afterwards. Nothing has to be run fold by fold.

## Outputs

```text
benchmark/results/matrix_v2/summary.csv     # long table: dataset x method
benchmark/results/matrix_v2/summary.md
benchmark/results/main_table.csv            # paper layout (dataset x method)
benchmark/results/main_table.md
```

`summary.csv` is one row per dataset and method: the test metric averaged over
the 5 folds within each seed, then the mean and population standard deviation
across the seeds. `scripts/make_main_table.py` only pivots that file into the
paper's Table 1 layout (it never recomputes a metric):

```bash
python scripts/make_main_table.py
```

## Notes

* The paper's main table aggregates 5 folds x 2 seeds per task, selecting each
  run's checkpoint on validation balanced accuracy and reporting test balanced
  accuracy, macro-F1 and AUROC.
* Build artifacts are digest-pinned in `benchmark/benchmark.yaml`
  (`manifest_sha256`, `fair_input.records_sha256`). If your canonical files
  differ from the released processing (different raw release version, or a
  different HDF5/h5py build), the builders and the loader refuse to continue
  rather than silently scoring different data.
* The repository ships no raw data, no canonical HDF5, no Fair Input tensor and
  no result tree; those are produced as described above. The two released
  checkpoints are included under `pretrain_checkpoint/`.
