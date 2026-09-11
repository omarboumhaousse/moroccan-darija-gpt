"""Single source of truth for every number in this project.

Nothing else in the codebase should hardcode a vocab size, a context length or
a layer count. If a number matters, it lives here.
"""

from dataclasses import dataclass
from pathlib import Path

# Defined once, shared by the tokenizer, the data pipeline and the model.
# If these ever disagree, the model's embedding table and the token ids in
# train.bin silently stop matching, which is the kind of bug that shows up as
# "loss is weird" three hours into a run.
VOCAB_SIZE = 8_000

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class DataConfig:
    """Everything about turning Atlaset into train.bin / val.bin."""

    hf_repo: str = "atlasia/Atlaset"
    text_column: str = "text"          # the only column we read; `metadata` is
                                       # 32% of the file and we never need it

    vocab_size: int = VOCAB_SIZE

    # Filtering. The Arabic check is close to a no-op on this corpus (99.9% of
    # documents already score above 0.9) but it is cheap insurance against a
    # future shard of Latin-script junk.
    min_arabic_ratio: float = 0.5
    min_doc_chars: int = 32            # drops the 5-to-12-character rows that
                                       # some sources are full of

    # Split by document, never mid-document, so no document has its head in
    # train and its tail in val.
    val_fraction: float = 0.005        # ~1.2M tokens of validation, plenty
    seed: int = 1337                   # fixes the split so it is reproducible

    tokenizer_path: Path = DATA_DIR / "tokenizer.json"
    train_bin: Path = DATA_DIR / "train.bin"
    val_bin: Path = DATA_DIR / "val.bin"

    # uint16 holds 0..65535. Our vocab is 8000, so it fits with room to spare,
    # and it halves the file size versus uint32.
    dtype: str = "uint16"

    def __post_init__(self):
        assert self.vocab_size < 65_536, "vocab does not fit in uint16"


@dataclass(frozen=True)
class ModelConfig:
    """The transformer itself. These are nanoGPT's baby-GPT numbers."""

    vocab_size: int = VOCAB_SIZE
    block_size: int = 256              # context length; attention is O(T^2), so
                                       # this is the main lever on a weak GPU
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384                  # 384 / 6 heads = 64 dims per head
    dropout: float = 0.0               # ~240M tokens against ~14M params means
                                       # we cannot overfit in one pass; raise
                                       # this only if we start reusing data

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, (
            f"n_embd={self.n_embd} must divide evenly into n_head={self.n_head}"
        )

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def estimated_params(self) -> dict:
        """Parameter count from the formula, before any PyTorch exists.

        Per block: 4*d^2 for attention (q, k, v, output projection) plus 8*d^2
        for the MLP (d -> 4d -> d), so 12*d^2. LayerNorms are a few hundred
        parameters each and are ignored here. The output head is free because
        it shares its weights with the token embedding.
        """
        tok_emb = self.vocab_size * self.n_embd
        pos_emb = self.block_size * self.n_embd
        blocks = 12 * self.n_layer * self.n_embd ** 2
        return {
            "token_embedding": tok_emb,
            "position_embedding": pos_emb,
            "blocks": blocks,
            "output_head (tied)": 0,
            "total": tok_emb + pos_emb + blocks,
        }


@dataclass(frozen=True)
class TrainConfig:
    """The training loop's settings."""

    batch_size: int = 64          # sequences per step: 64 x 256 = 16,384 tokens
    learning_rate: float = 1e-3   # peak, reached after warmup (nanoGPT's baby-GPT value)
    min_lr: float = 1e-4          # floor of the cosine decay, a tenth of the peak
    warmup_iters: int = 100       # linear warmup from almost 0 up to the peak
    max_iters: int = 5000         # the decay ends here; revisited for the full run
    grad_clip: float = 1.0        # cap on the total gradient norm (nanoGPT's value)
    eval_interval: int = 500      # steps between loss estimates
    eval_iters: int = 200         # batches averaged per estimate


if __name__ == "__main__":
    cfg = ModelConfig()
    parts = cfg.estimated_params()
    total = parts["total"]
    print(f"vocab {cfg.vocab_size}, {cfg.n_layer} layers, {cfg.n_head} heads, "
          f"d_model {cfg.n_embd} ({cfg.head_dim}/head), context {cfg.block_size}\n")
    for name, n in parts.items():
        share = f"{100 * n / total:5.1f}%" if name != "total" else ""
        print(f"  {name:<22} {n:>12,}  {share}")
