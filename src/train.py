"""Training the model."""

import math

import numpy as np
import torch

from src.config import DataConfig, ModelConfig, TrainConfig
from src.model import GPT


def get_batch(split, batch_size, block_size, device="cpu"):
    """Random windows from train.bin or val.bin; y is x shifted by one token."""
    data_cfg = DataConfig()
    path = data_cfg.train_bin if split == "train" else data_cfg.val_bin
    # Re-opened on every call, as in nanoGPT, to avoid a known memory leak
    # with long-lived memmaps.
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)  # (B, T) each, int64


@torch.no_grad()
def estimate_loss(model, tc, mc, device):
    """Mean loss over tc.eval_iters random batches of each split."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(tc.eval_iters)
        for k in range(tc.eval_iters):
            x, y = get_batch(split, tc.batch_size, mc.block_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(step, tc):
    """Linear warmup to the peak, then cosine decay to min_lr at max_iters."""
    if step < tc.warmup_iters:
        return tc.learning_rate * (step + 1) / tc.warmup_iters
    if step >= tc.max_iters:
        return tc.min_lr
    progress = (step - tc.warmup_iters) / (tc.max_iters - tc.warmup_iters)  # 0 -> 1
    coeff = 0.5 * (1 + math.cos(math.pi * progress))                        # 1 -> 0
    return tc.min_lr + coeff * (tc.learning_rate - tc.min_lr)


def train(mc=ModelConfig(), tc=TrainConfig()):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)
    model = GPT(mc).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.learning_rate)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, on {device}")

    history = []
    for step in range(tc.max_iters):
        lr = get_lr(step, tc)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if step % tc.eval_interval == 0 or step == tc.max_iters - 1:
            losses = estimate_loss(model, tc, mc, device)
            history.append({"step": step, "lr": lr, **losses})
            print(f"step {step}: train loss {losses['train']:.4f}, "
                  f"val loss {losses['val']:.4f}, lr {lr:.2e}")

        x, y = get_batch("train", tc.batch_size, mc.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # rescale all gradients together so their total norm is at most grad_clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        optimizer.step()
    return model, history


def overfit_one_batch(x, y, mc=ModelConfig(), lr=3e-4, steps=300):
    """Train on one fixed batch: if the loss can't get near zero, something is broken."""
    torch.manual_seed(1337)
    model = GPT(mc).to(x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


if __name__ == "__main__":
    train()
