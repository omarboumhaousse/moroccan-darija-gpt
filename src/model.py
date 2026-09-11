"""GPT model: a decoder-only transformer, written in the style of Karpathy's
"Let's build GPT" video.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, config, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        # Lower-triangular matrix of ones. A buffer: it moves to the GPU with
        # the model but is never trained. persistent=False keeps it out of
        # checkpoints, since it is rebuilt from block_size anyway.
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.block_size, config.block_size)),
            persistent=False,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        # attention scores: how much each position t attends to each position s
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # no looking ahead
        wei = F.softmax(wei, dim=-1)  # (B, T, T), each row sums to 1
        wei = self.dropout(wei)
        v = self.value(x)  # (B, T, hs)
        out = wei @ v      # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """Several heads in parallel, concatenated, then mixed by a projection."""

    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(config, config.head_dim) for _ in range(config.n_head)]
        )
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """A small MLP, applied to each position on its own."""

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)  # (B, T, n_embd) -> (B, T, 4*n_embd) -> (B, T, n_embd)


class Block(nn.Module):
    """Transformer block: communication (attention), then computation (MLP).

    Each sub-layer reads a normalised copy of x and adds its result back to x.
    """

    def __init__(self, config):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # (B, T, n_embd)
        x = x + self.ffwd(self.ln2(x))  # (B, T, n_embd)
        return x


class GPT(nn.Module):
    """The whole model: embeddings, the stack of blocks, and the output head."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)  # pre-norm leaves the stream unnormalised
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: the head and the token embedding share one
        # (vocab_size, n_embd) matrix, saving vocab_size * n_embd parameters.
        self.token_embedding_table.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # One small init for every weight, as in the video's code and GPT-2.
        # PyTorch's own defaults would start the position embedding at N(0, 1).
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"{T} tokens, but the context is {self.config.block_size}"
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb     # (B, T, n_embd): pos_emb is broadcast over the batch
        x = self.blocks(x)        # (B, T, n_embd)
        x = self.ln_f(x)          # (B, T, n_embd)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss
