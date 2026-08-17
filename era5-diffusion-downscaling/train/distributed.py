"""Optional distributed data-parallel (DDP) support for the trainers.

Every trainer runs single-process by default — nothing here activates unless
the script is launched under torchrun, which sets RANK / WORLD_SIZE /
LOCAL_RANK in the environment. Launched that way, the SAME training run is
split across the participating processes (e.g. 4 GPU nodes, one process per
GPU):

    torchrun --nnodes=4 --nproc_per_node=1 --node_rank=<i> \
        --rdzv_backend=c10d --rdzv_endpoint=<node0-host>:29500 \
        -m train.train_diffusion --config config/t2m.yaml

(scripts/train_multinode.sh wraps exactly this). Semantics:

- Each process draws a disjoint shard of every epoch (DistributedSampler with
  a shared shuffle seed), so one epoch still visits each patch once.
- DDP averages gradients every step: the run is equivalent to a single-process
  run with global batch = train.batch_size x world_size (batch_size stays
  PER PROCESS; scale the LR yourself if you want large-batch scaling rules).
- Only rank 0 validates, logs (stdout/tensorboard/wandb), and writes
  checkpoints. Checkpoints are saved from the unwrapped module, so they are
  byte-identical in format to single-process ones and load anywhere.
"""

import builtins
import os

import torch
import torch.distributed as td
from torch.utils.data import DataLoader, DistributedSampler


class DistContext:
    """World description for one training process.

    `enabled` is False when not launched via torchrun (or world size is 1), in
    which case every helper below degrades to a no-op and the trainers behave
    exactly as before.
    """

    def __init__(self, enabled: bool, rank: int, local_rank: int,
                 world_size: int, device: torch.device):
        self.enabled = enabled
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_distributed() -> DistContext:
    """Join the torchrun process group, or return a disabled no-op context.

    Must be called once, before any other helper here. On non-main ranks
    print() is silenced (pass force=True to override) so the interleaved logs
    of 4+ processes don't shred the console.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistContext(False, 0, 0, 1, _default_device())

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK {local_rank} >= visible GPUs "
                f"{torch.cuda.device_count()} — check --nproc_per_node")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    td.init_process_group(backend=backend)
    _silence_nonmain_prints(rank)
    print(f"distributed: rank {rank}/{world_size} (local {local_rank}) | "
          f"backend={backend} | device={device}")
    return DistContext(True, rank, local_rank, world_size, device)


def _silence_nonmain_prints(rank: int) -> None:
    if rank == 0:
        return
    orig_print = builtins.print

    def print_main_only(*args, force=False, **kwargs):
        if force:
            orig_print(*args, **kwargs)

    builtins.print = print_main_only


def make_train_loader(dataset, batch_size: int, num_workers: int,
                      ctx: DistContext, seed: int = 0) -> DataLoader:
    """The trainers' shared training DataLoader, sharded per rank under DDP.

    The sampler seed must be the BASE config seed (identical on every rank):
    all ranks then agree on each epoch's permutation and take disjoint slices
    of it. Callers must invoke set_epoch(loader, epoch) each epoch so the
    permutation actually reshuffles.
    """
    sampler = None
    if ctx.enabled:
        sampler = DistributedSampler(
            dataset, num_replicas=ctx.world_size, rank=ctx.rank,
            shuffle=True, seed=seed, drop_last=True)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=sampler is None,
        sampler=sampler, num_workers=num_workers, pin_memory=True,
        drop_last=True, persistent_workers=num_workers > 0)


def set_epoch(loader: DataLoader, epoch: int) -> None:
    """Reshuffle the distributed shard for a new epoch (no-op otherwise)."""
    if isinstance(getattr(loader, "sampler", None), DistributedSampler):
        loader.sampler.set_epoch(epoch)


def wrap_model(model: torch.nn.Module, ctx: DistContext, cfg: dict | None = None):
    """Wrap in DistributedDataParallel when enabled; identity otherwise.

    Call AFTER any --resume state load so weights restore into the raw module,
    and keep a reference to the raw module for EMA/checkpointing — the DDP
    state_dict would gain 'module.'-prefixed keys. DDP broadcasts rank-0
    weights at construction, so all ranks start (or resume) identically.

    broadcast_buffers=False: every buffer in these models is a constant
    (hash-grid primes, sinusoidal freqs, ...), identical on all ranks by
    construction — skipping the broadcast removes a per-forward collective and
    keeps rank-0-only validation through the raw module safe. (torch >= 2.13
    warns that the flag is renamed to forward_sync_buffers; the old name is
    kept for the torch >= 2.2 floor in requirements.txt and works on both.)
    """
    if not ctx.enabled:
        return model
    dcfg = (cfg or {}).get("distributed", {})
    device_ids = [ctx.device.index] if ctx.device.type == "cuda" else None
    return torch.nn.parallel.DistributedDataParallel(
        model, device_ids=device_ids, broadcast_buffers=False,
        find_unused_parameters=dcfg.get("find_unused_parameters", False))


def cleanup(ctx: DistContext) -> None:
    """Leave the process group (barrier first so rank 0 finishes writing)."""
    if ctx.enabled and td.is_initialized():
        td.barrier()
        td.destroy_process_group()
