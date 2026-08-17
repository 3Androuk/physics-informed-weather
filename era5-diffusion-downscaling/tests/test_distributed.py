"""CPU unit tests for the optional DDP helpers (train/distributed.py).

The two-process test exercises the real code path — env-var handshake,
DistributedSampler sharding, DDP weight broadcast and gradient averaging —
over the gloo backend, so it runs without GPUs.
"""

import os
import socket
import unittest

import torch
import torch.distributed as td
import torch.multiprocessing as mp
from torch.utils.data import DistributedSampler

from train.distributed import (cleanup, init_distributed, make_train_loader,
                               set_epoch, wrap_model)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class DisabledContextTests(unittest.TestCase):
    """Without torchrun env vars everything must degrade to a no-op."""

    def setUp(self):
        for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
            os.environ.pop(var, None)

    def test_single_process_passthrough(self):
        ctx = init_distributed()
        self.assertFalse(ctx.enabled)
        self.assertTrue(ctx.is_main)
        self.assertEqual((ctx.rank, ctx.world_size), (0, 1))

        model = torch.nn.Linear(2, 2)
        self.assertIs(wrap_model(model, ctx), model)

        data = torch.arange(8, dtype=torch.float32).unsqueeze(1)
        loader = make_train_loader(data, batch_size=2, num_workers=0,
                                   ctx=ctx, seed=0)
        self.assertNotIsInstance(loader.sampler, DistributedSampler)
        set_epoch(loader, 3)  # must be a silent no-op
        self.assertEqual(sum(len(b) for b in loader), 8)
        cleanup(ctx)  # must be a silent no-op


def _worker(rank, world_size, port):
    os.environ.update({
        "RANK": str(rank), "LOCAL_RANK": str(rank),
        "WORLD_SIZE": str(world_size),
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
    })
    ctx = init_distributed()
    assert ctx.enabled and ctx.world_size == world_size and ctx.rank == rank

    # Epoch shards must be equal-sized, disjoint, and cover the dataset.
    data = torch.arange(12, dtype=torch.float32).unsqueeze(1)
    loader = make_train_loader(data, batch_size=2, num_workers=0, ctx=ctx, seed=7)
    set_epoch(loader, 1)
    mine = sorted(int(v) for batch in loader for v in batch)
    assert len(mine) == 12 // world_size
    shards = [None] * world_size
    td.all_gather_object(shards, mine)
    assert sorted(i for shard in shards for i in shard) == list(range(12))

    # DDP: rank-0 broadcast erases deliberately different inits, and averaged
    # gradients keep every rank's weights identical after a step on different
    # per-rank batches. The raw module's state_dict keys must stay unprefixed.
    torch.manual_seed(100 + rank)
    model = torch.nn.Linear(4, 4)
    wrapped = wrap_model(model, ctx)
    assert wrapped is not model
    assert not any(k.startswith("module.") for k in model.state_dict())
    opt = torch.optim.SGD(wrapped.parameters(), lr=0.1)
    torch.manual_seed(rank)
    wrapped(torch.randn(3, 4)).sum().backward()
    opt.step()
    flat = torch.cat([p.detach().flatten() for p in model.parameters()])
    gathered = [torch.empty_like(flat) for _ in range(world_size)]
    td.all_gather(gathered, flat)
    assert all(torch.allclose(gathered[0], g) for g in gathered[1:])

    cleanup(ctx)


class TwoProcessGlooTests(unittest.TestCase):
    @unittest.skipUnless(td.is_available() and td.is_gloo_available(),
                         "torch.distributed gloo backend unavailable")
    def test_sharding_broadcast_and_grad_averaging(self):
        mp.spawn(_worker, args=(2, _free_port()), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
