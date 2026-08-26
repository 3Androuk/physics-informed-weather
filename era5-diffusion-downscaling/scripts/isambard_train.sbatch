#!/bin/bash
# Train on BriCS Isambard-AI: ONE whole node, all 4 GH200s, auto-chained across
# the 24h walltime.
#
#   sbatch scripts/isambard_train.sbatch train.train_flow_matching \
#       --config config/wb2_20var.yaml --geo
#
# Submit from era5-diffusion-downscaling/ ($SLURM_SUBMIT_DIR is used to find the
# repo and this script again when chaining). Works with any trainer module:
# train.train_diffusion, train.train_directmap, train.train_residual,
# train.train_flow_matching, train.train_stochastic_interpolant.
#
# WHY A WHOLE NODE: BriCS bills in NODE-hours (the docs do not say whether a
# partial-node job is charged for the whole node — check TRESBillingWeights via
# `scontrol show partition`). Taking all 4 GH200s and running 4-way DDP means
# every GPU billed is a GPU working, which is right under either rule. Note
# scripts/train_multinode.sh defaults to 4 nodes x 1 GPU — right for a
# 1-GPU-per-node cluster, four nodes' worth of billing here for the same 4 GPUs.
#
# Env overrides:
#   ERA5_VENV    path to the venv           (default $SLURM_SUBMIT_DIR/.venv)
#   ERA5_CHAIN   0 to disable auto-chaining (default 1)
#   SBATCH_ACCOUNT / --account=<proj>       if the site needs it explicitly
#
#SBATCH --job-name=era5
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=%x-%j.out

set -uo pipefail

MODULE=${1:?usage: sbatch $0 <train module> [trainer args...]}
shift
ARGS=("$@")

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
SELF="$REPO/scripts/isambard_train.sbatch"
cd "$REPO" || exit 1

VENV="${ERA5_VENV:-$REPO/.venv}"
if [[ -f "$VENV/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
else
    echo "[warn] no venv at $VENV — using whatever python is on PATH"
fi

echo "[node]   $(hostname)  job $SLURM_JOB_ID"
echo "[gpus]   ${SLURM_GPUS_ON_NODE:-?} visible"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# ── Chain on walltime ONLY ────────────────────────────────────────────────
# Slurm sends USR1 600s before the limit; we queue the continuation then, so it
# starts from the last per-epoch checkpoint. A crash (non-zero exit without the
# signal) is deliberately NOT chained — that would just replay the crash and
# burn node-hours doing it.
chained=0
chain() {
    if [[ "${ERA5_CHAIN:-1}" == "1" && $chained -eq 0 ]]; then
        chained=1
        echo "[chain] walltime approaching — queueing continuation after $SLURM_JOB_ID"
        sbatch --dependency=afterany:"$SLURM_JOB_ID" "$SELF" "$MODULE" "${ARGS[@]}"
    fi
}
trap chain USR1

# One process per GH200. batch_size is PER PROCESS, so the global batch is
# train.batch_size x 4 (see train/distributed.py).
PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "[run]    torchrun 1 node x 4 procs | $MODULE ${ARGS[*]} --resume"

torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="localhost:$PORT" \
    --rdzv_id="era5-$SLURM_JOB_ID" \
    -m "$MODULE" "${ARGS[@]}" --resume &
TRAIN_PID=$!

# `wait` returns 128+signum when our USR1 trap fires; keep waiting for the real
# exit after handling it.
status=0
while true; do
    wait "$TRAIN_PID"
    status=$?
    (( status > 128 )) && kill -0 "$TRAIN_PID" 2>/dev/null && continue
    break
done

if (( status == 0 )); then
    echo "[done]  training finished cleanly — no continuation queued"
elif (( chained == 1 )); then
    echo "[exit $status] stopped at walltime — continuation is queued"
else
    echo "[exit $status] FAILED before walltime — not chaining; check the log above"
fi
exit "$status"
