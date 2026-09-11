"""Training checks: memorising one batch (step 5), and the learning-rate schedule.

Run from the repo root:

    python -m pytest -v
"""

import pytest
import torch

from src.config import ModelConfig, TrainConfig
from src.train import get_lr, overfit_one_batch


def test_overfit_one_batch():
    # A tiny model and random tokens keep this fast and free of data files.
    # The code path is the real one: GPT, AdamW, zero_grad, backward, step.
    mc = ModelConfig(n_layer=2, n_head=2, n_embd=64, block_size=32)
    seq = torch.randint(0, mc.vocab_size, (4, mc.block_size + 1),
                        generator=torch.Generator().manual_seed(0))
    x, y = seq[:, :-1], seq[:, 1:]  # slices, so y is not contiguous: the loss must cope
    losses = overfit_one_batch(x, y, mc, lr=3e-3, steps=100)
    assert losses[0] > 8.5   # starts near ln(8000), knowing nothing
    assert losses[-1] < 0.1  # and memorises the batch


def test_lr_schedule():
    tc = TrainConfig(learning_rate=1e-3, min_lr=1e-4, warmup_iters=100, max_iters=1000)
    lrs = [get_lr(step, tc) for step in range(tc.max_iters + 1)]
    assert lrs[0] == pytest.approx(1e-5)                      # warmup starts near 0
    assert lrs[99] == pytest.approx(1e-3)                     # and reaches the peak
    assert all(a < b for a, b in zip(lrs[:99], lrs[1:100]))   # rising during warmup
    assert all(a >= b for a, b in zip(lrs[100:], lrs[101:]))  # never rising after it
    assert lrs[550] == pytest.approx((1e-3 + 1e-4) / 2)       # cosine: halfway at the midpoint
    assert lrs[-1] == pytest.approx(1e-4)                     # ends on the floor
    assert min(lrs[100:]) >= 1e-4 - 1e-12                     # and never goes below it
