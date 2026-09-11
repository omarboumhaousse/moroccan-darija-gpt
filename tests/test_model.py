"""The three checks that must pass before any training.

Run from the repo root:

    python -m pytest -v
"""

import math

import torch

from src.config import ModelConfig
from src.model import GPT


def make_model():
    torch.manual_seed(0)
    return GPT(ModelConfig()).eval()


def test_shapes():
    cfg = ModelConfig()
    model = make_model()
    idx = torch.randint(0, cfg.vocab_size, (2, 16))

    logits, loss = model(idx)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss is None

    logits, loss = model(idx, targets=idx)
    assert loss.shape == ()  # a single number


def test_position_t_cannot_see_t_plus_1():
    cfg = ModelConfig()
    model = make_model()
    idx = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        before, _ = model(idx)
        for t in range(15):
            changed = idx.clone()
            changed[0, t + 1:] = (changed[0, t + 1:] + 1) % cfg.vocab_size  # every token after t
            after, _ = model(changed)
            assert torch.equal(after[0, :t + 1], before[0, :t + 1]), f"position {t} saw the future"
            assert not torch.equal(after[0, t + 1], before[0, t + 1])  # the change did reach t+1


def test_untrained_loss_is_ln_vocab():
    cfg = ModelConfig()
    model = make_model()
    idx = torch.randint(0, cfg.vocab_size, (8, 64))
    targets = torch.randint(0, cfg.vocab_size, (8, 64))
    with torch.no_grad():
        _, loss = model(idx, targets)
    # knowing nothing means about 1/vocab_size on every token, so loss ~ ln(vocab_size)
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.2
