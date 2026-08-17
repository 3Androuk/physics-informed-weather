#!/usr/bin/env bash
# Split ONE training run across multiple GPU nodes with torchrun (DDP).
#
# Run this script once on EVERY node, from era5-diffusion-downscaling/,
# giving each node its index. Defaults to 4 nodes x 1 GPU each:
#
#   # on node 0 (also the rendezvous host):
#   MASTER_ADDR=node0.example NODE_RANK=0 \
#     scripts/train_multinode.sh train.train_diffusion --config config/t2m.yaml
#
#   # on nodes 1..3 (same command, different NODE_RANK):
#   MASTER_ADDR=node0.example NODE_RANK=1 \
#     scripts/train_multinode.sh train.train_diffusion --config config/t2m.yaml
#
# Works with any trainer module: train.train_diffusion, train.train_directmap,
# train.train_residual, train.train_flow_matching,
# train.train_stochastic_interpolant — extra args are passed through.
#
# Overrides:
#   NNODES          number of nodes            (default 4)
#   NPROC_PER_NODE  GPUs per node              (default 1)
#   MASTER_PORT     rendezvous port on node 0  (default 29500)
#   RDZV_ID         unique job name, needed if several jobs share node 0
#
# Single node, 4 GPUs:
#   NNODES=1 NPROC_PER_NODE=4 NODE_RANK=0 MASTER_ADDR=localhost \
#     scripts/train_multinode.sh train.train_diffusion --config config/t2m.yaml
#
# All nodes must see the same datasets/ (shared filesystem or identical copies);
# checkpoints/results are written by node 0 only. See train/distributed.py for
# the run semantics (global batch = train.batch_size x total processes).
set -euo pipefail

NNODES=${NNODES:-4}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_PORT=${MASTER_PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:?set MASTER_ADDR to the hostname/IP of the rank-0 node}
NODE_RANK=${NODE_RANK:?set NODE_RANK to the index of this node (0..NNODES-1)}

MODULE=${1:?usage: MASTER_ADDR=<host> NODE_RANK=<i> $0 <train module> [args...]}
shift

exec torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --node_rank="$NODE_RANK" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_id="${RDZV_ID:-era5-ddp}" \
    -m "$MODULE" "$@"
