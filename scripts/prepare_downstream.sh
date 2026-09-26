#!/usr/bin/env bash
set -euo pipefail
PYTHON=python3
SKIP_CANONICAL=0
SKIP_MANIFESTS=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --python) PYTHON="$2"; shift 2 ;;
        --skip-canonical) SKIP_CANONICAL=1; shift ;;
        --skip-manifests) SKIP_MANIFESTS=1; shift ;;
        -h|--help)
            echo "usage: bash scripts/prepare_downstream.sh [--python PYTHON] [--skip-canonical] [--skip-manifests]"
            echo "  builds data/canonical, benchmark/manifests, benchmark/fair_input and verifies the Fair Input"
            exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RAW=data/raw
CANONICAL=data/canonical
DATASETS=(DS7 DS47 DS36 DS19 DS70E DS2 DS8 DS42)
if [ "$SKIP_CANONICAL" -eq 0 ]; then
    echo "=== [1/4] canonical preprocessing -> $CANONICAL ==="
    "$PYTHON" scripts/legacy/preprocessing/run_full_canonical_v2.py \
        --raw-root "$RAW" --output "$CANONICAL"
    "$PYTHON" scripts/build_canonical.py --dataset DS2 \
        --ds2-root "$RAW/2_ds004514" --canonical-root "$CANONICAL"
    "$PYTHON" scripts/build_canonical.py --dataset DS8 \
        --ds8-archive "$RAW/DS8/FNIRS_SNIRF.zip" --canonical-root "$CANONICAL"
    for dataset in DS36 DS19 DS42 DS70; do
        "$PYTHON" scripts/build_canonical_extended.py --dataset "$dataset" \
            --raw-root "$RAW" --canonical-root "$CANONICAL"
    done
else
    echo "=== [1/4] canonical preprocessing skipped (--skip-canonical) ==="
fi
if [ "$SKIP_MANIFESTS" -eq 0 ]; then
    echo "=== [2/4] manifests -> benchmark/manifests ==="
    "$PYTHON" scripts/build_manifests_frozen.py
    "$PYTHON" scripts/build_manifests.py --dataset DS2
    "$PYTHON" scripts/build_manifests.py --dataset DS8
    "$PYTHON" scripts/build_manifests_extended.py --datasets DS36 DS19 DS42 DS70E
else
    echo "=== [2/4] manifests skipped (--skip-manifests) ==="
fi
echo "=== [3/4] Fair Input -> benchmark/fair_input ==="
"$PYTHON" scripts/build_fair_input.py --datasets "${DATASETS[@]}" \
    --channel-policy intersection
echo "=== [4/4] independent verification ==="
"$PYTHON" scripts/verify_fair_input.py --datasets "${DATASETS[@]}"
echo
echo "Done. Fair Input: benchmark/fair_input/GEONIRS_FAIR_INPUT.h5"
