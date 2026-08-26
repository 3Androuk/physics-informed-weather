#!/usr/bin/env bash
# Download the WB2 dataset on a BriCS LOGIN node — deliberately not a Slurm job.
#
#   scripts/download_login.sh config/wb2_20var.yaml
#
# WHY THE LOGIN NODE: Isambard-AI has no CPU-only partition, so a Slurm download
# would hold GPUs — billed in node-hours — while doing nothing but wait on the
# network. Login nodes are not a Slurm allocation, so they should not be billed
# at all (worth confirming against the project balance the first time). They do
# cap each user at ~4 GiB RAM, which is why the flags below keep the footprint
# small: data.download_era5 streams each batch straight into an on-disk memmap
# (~0.8 GiB peak at --batch 16), rather than buffering a whole year (~19 GiB at
# 20 channels, which the cap would kill).
#
# Resumable: each year is cached separately and a rerun skips finished years,
# so just run it again if the link drops.
#
# Env overrides: ERA5_VENV (default ./.venv), BATCH (16), CHUNK_TIME (4),
# TIMEOUT (120), RETRIES (8).
set -euo pipefail

CONFIG=${1:?usage: $0 <config yaml>   e.g. config/wb2_20var.yaml}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

VENV="${ERA5_VENV:-$REPO/.venv}"
if [[ -f "$VENV/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

echo "[download] $CONFIG on $(hostname) — login node, no node-hours charged"
echo "[download] run under tmux/screen; it is resumable if it drops"

exec python -m data.download_era5 \
    --config "$CONFIG" \
    --batch "${BATCH:-16}" \
    --chunk-time "${CHUNK_TIME:-4}" \
    --timeout "${TIMEOUT:-120}" \
    --max-retries "${RETRIES:-8}"
