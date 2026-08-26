#!/usr/bin/env bash
# Download the WB2 dataset on a BriCS LOGIN node — deliberately not a Slurm job.
#
#   scripts/download_login.sh config/wb2_20var.yaml
#
# WHY THE LOGIN NODE: Isambard-AI has no CPU-only partition, so a Slurm download
# would hold GPUs — billed in node-hours — while doing nothing but wait on the
# network. Login nodes are not a Slurm allocation, so they should not be billed
# at all (worth confirming against the project balance the first time). They do
# cap each user at 4 GiB RAM (cgroup MemoryMax, measured), which shapes
# everything below.
#
# ONE PROCESS PER YEAR. fsspec keeps filesystem instances and block caches
# globally, so a single long-lived process creeps upward in RSS across years and
# is eventually OOM-killed (observed: 2007 completed, 2008 died 48/366 in, exit
# 137). A fresh process per year resets that, and because each year is cached on
# completion nothing in flight is ever lost. The merge runs on the final pass,
# once every configured year is on disk.
#
# NOTE: there is no tmux or screen on these login nodes, and `loginctl
# enable-linger` is denied, so a detached nohup/setsid process is reaped when
# your last SSH session closes. Keep a session open for the duration, or rerun
# this script — it resumes from the last completed year.
#
# Env overrides: ERA5_VENV (default ./.venv), BATCH (4), CHUNK_TIME (4), DASK_THREADS (4),
# TIMEOUT (120), RETRIES (8).
set -uo pipefail

CONFIG=${1:?usage: $0 <config yaml>   e.g. config/wb2_20var.yaml}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

VENV="${ERA5_VENV:-$REPO/.venv}"
if [[ -f "$VENV/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

# Configured years, in order, from the config itself.
mapfile -t YEARS < <(python - "$CONFIG" <<'PY'
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd()))
from utils import load_config
d = load_config(sys.argv[1])["data"]
ys = []
for key in ("train_years", "test_years"):
    lo, hi = d[key]
    ys += list(range(lo, hi + 1))
print("\n".join(str(y) for y in sorted(set(ys))))
PY
)

if [[ ${#YEARS[@]} -eq 0 ]]; then
    echo "[download] could not read years from $CONFIG" >&2
    exit 1
fi

echo "[download] $CONFIG on $(hostname) — login node, no node-hours charged"
echo "[download] ${#YEARS[@]} years, one process each: ${YEARS[*]}"

run_one() {   # $1 = --years argument, or empty for the final merge pass
    local sel=("$@")
    python -u -m data.download_era5 \
        --config "$CONFIG" \
        --batch "${BATCH:-4}" \
        --chunk-time "${CHUNK_TIME:-4}" \
        --timeout "${TIMEOUT:-120}" \
        --max-retries "${RETRIES:-8}" \
        --dask-threads "${DASK_THREADS:-4}" \
        "${sel[@]}"
}

for y in "${YEARS[@]}"; do
    echo "[download] ===== year $y ====="
    for attempt in 1 2 3; do
        # Capture the status directly: after `if cmd; then ...; fi` with no else,
        # $? is the `if` statement's own 0, not the command's.
        run_one --years "$y"
        rc=$?
        [[ $rc -eq 0 ]] && break
        echo "[download] year $y exited $rc (137 = OOM-killed) — attempt $attempt/3"
        if [[ $attempt -eq 3 ]]; then
            echo "[download] giving up on $y; rerun the script to retry" >&2
            exit "$rc"
        fi
        sleep 10
    done
done

echo "[download] ===== all years cached; merging ====="
run_one
