"""Train and load the byte-level BPE tokenizer.

Run from the project root:

    python -m src.tokenizer

Also owns reading Atlaset and deciding which documents to keep, because both
this file and data/prepare.py need exactly that.
"""

import argparse
import time

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from src.config import DataConfig

# Exact, from reading the parquet footers and the light columns. Used to turn
# a per-document or per-word average into a corpus-wide estimate.
TOTAL_TRAIN_DOCS = 1_171_515
TOTAL_TRAIN_WORDS = 153_612_577

# The separator we put between documents in the flat token stream. Without it
# the model learns to run the end of one document straight into the start of
# the next.
EOT = "<|endoftext|>"


# --- Arabic script check -----------------------------------------------------
ARABIC_RANGES = [
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
]


def arabic_ratio(text: str) -> float:
    """Fraction of the letters in `text` that are Arabic.

    Digits, punctuation and whitespace are ignored: a document that is mostly
    numbers should not be judged as containing no Arabic.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    hits = sum(
        any(lo <= ord(c) <= hi for lo, hi in ARABIC_RANGES) for c in letters
    )
    return hits / len(letters)


def keep_document(text: str, cfg: DataConfig) -> bool:
    if not text or len(text) < cfg.min_doc_chars:
        return False
    return arabic_ratio(text) >= cfg.min_arabic_ratio


# --- Reading the corpus ------------------------------------------------------
def iter_rows(cfg: DataConfig, columns: list[str] | None = None,
              stride: int = 1, limit: int | None = None, offset: int = 0):
    """Yield kept rows from Atlaset's train split, as dicts.

    `stride` selects every Nth row group, starting at `offset`. Row groups hold
    1000 rows each, so striding gives an even sample of the corpus rather than
    a prefix.

    Shards are read **round-robin**, one row group from each in turn, so that
    any prefix of this iterator is balanced across all five. Reading them
    sequentially instead means a `limit` never reaches shards 1 to 4, and the
    shards are not interchangeable: shard 0 averages 842 bytes per row against
    about 2,000 for the corpus.

    `offset` lets a second pass pick row groups the first pass never touched,
    which is how we measure compression on text the tokenizer has not seen.

    Only the columns asked for are read. Parquet is columnar, so the `metadata`
    column (32% of the file) never crosses the network.
    """
    cols = list(dict.fromkeys([cfg.text_column, *(columns or [])]))

    fs = HfFileSystem()
    paths = sorted(fs.glob(f"datasets/{cfg.hf_repo}/**/*.parquet"))
    paths = [p for p in paths if "train" in p.rsplit("/", 1)[-1]]
    shards = [pq.ParquetFile(p, filesystem=fs) for p in paths]
    max_groups = max(s.metadata.num_row_groups for s in shards)

    yielded = 0
    for rg in range(offset, max_groups, stride):
        for shard in shards:
            if rg >= shard.metadata.num_row_groups:
                continue
            for row in shard.read_row_group(rg, columns=cols).to_pylist():
                text = row[cfg.text_column]
                if text and keep_document(text, cfg):
                    yield row
                    yielded += 1
                    if limit is not None and yielded >= limit:
                        return


def iter_documents(cfg: DataConfig, stride: int = 1, limit: int | None = None,
                   offset: int = 0):
    """Just the text, for callers that do not need the other columns."""
    for row in iter_rows(cfg, None, stride, limit, offset):
        yield row[cfg.text_column]


# --- Training ----------------------------------------------------------------
def build_tokenizer(vocab_size: int) -> tuple[Tokenizer, trainers.BpeTrainer]:
    tok = Tokenizer(models.BPE())

    # ByteLevel means the tokenizer operates on UTF-8 bytes, not characters.
    # Every possible input is representable, so there is no UNK token and no
    # way to feed it something it cannot encode. Arabic characters are 2 bytes
    # each in UTF-8, so the first few dozen merges just rebuild the alphabet.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT],
        # Seed the vocab with all 256 byte values so that even a byte the
        # training data never contained still has an id.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    return tok, trainer


def train(cfg: DataConfig, stride: int, limit: int | None) -> Tokenizer:
    tok, trainer = build_tokenizer(cfg.vocab_size)

    print(f"training {cfg.vocab_size}-token BPE (row group stride {stride})...")
    t0 = time.time()
    tok.train_from_iterator(iter_documents(cfg, stride, limit), trainer=trainer)
    print(f"done in {time.time() - t0:,.0f}s")

    cfg.tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(cfg.tokenizer_path))
    print(f"saved to {cfg.tokenizer_path}")
    return tok


def load(cfg: DataConfig) -> Tokenizer:
    return Tokenizer.from_file(str(cfg.tokenizer_path))


# --- Checking it worked ------------------------------------------------------
def report(tok: Tokenizer, cfg: DataConfig, stride: int, n_docs: int = 5000):
    """Measure compression on a fresh sample and round-trip a few documents."""
    print(f"\nvocab size: {tok.get_vocab_size()}")
    print(f"{EOT} id: {tok.token_to_id(EOT)}")

    # Offset so these row groups are ones training never touched. Training used
    # 0, stride, 2*stride, ...; starting at stride//2+1 with a wider stride
    # lands on different row groups entirely.
    chars = tokens = words = docs = 0
    samples = []
    for row in iter_rows(cfg, ["word_count"], stride=stride * 3,
                         limit=n_docs, offset=stride // 2 + 1):
        text = row[cfg.text_column]
        ids = tok.encode(text).ids
        chars += len(text)
        tokens += len(ids)
        words += row["word_count"] or 0
        docs += 1
        if len(samples) < 2 and 100 < len(text) < 400:
            samples.append(text)

    print(f"\nmeasured on {docs:,} held-out documents across all 5 shards:")
    print(f"  chars per token   {chars / tokens:>10.2f}")
    print(f"  tokens per word   {tokens / words:>10.2f}")
    print(f"  tokens per doc    {tokens / docs:>10.1f}")

    # Two independent estimators. Trust the word-based one: word_count is known
    # exactly for all 1.17M rows, and tokens-per-word barely moves with
    # document length. The document-based one is a cross-check. If they
    # disagree badly, the sample is not representative of the corpus.
    by_words = tokens / words * TOTAL_TRAIN_WORDS
    by_docs = tokens / docs * TOTAL_TRAIN_DOCS
    print(f"\n  corpus estimate, via words {by_words / 1e6:>9.1f}M tokens")
    print(f"  corpus estimate, via docs  {by_docs / 1e6:>9.1f}M tokens")
    print(f"  tokens per parameter       {by_words / 13_787_136:>9.1f}")

    print("\nround trip:")
    for s in samples:
        ids = tok.encode(s).ids
        back = tok.decode(ids)
        ok = "OK" if back == s else "MISMATCH"
        print(f"  [{ok}] {len(s)} chars -> {len(ids)} tokens -> {len(back)} chars")
        if back != s:
            print(f"    original: {s[:120]}")
            print(f"    decoded:  {back[:120]}")

    if samples:
        # Show the split using character offsets into the original string.
        # Printing enc.tokens instead would show the byte-level alphabet, in
        # which every Arabic character appears as two Latin-ish symbols.
        print("\nhow a sentence gets split:")
        piece = samples[0][:120]
        enc = tok.encode(piece)
        print(f"  {piece}")
        print(f"  {' | '.join(piece[a:b] for a, b in enc.offsets)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stride", type=int, default=25,
                   help="use every Nth parquet row group (lower = more data)")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after this many documents")
    p.add_argument("--report-only", action="store_true",
                   help="load the saved tokenizer and re-measure, no training")
    args = p.parse_args()

    cfg = DataConfig()
    tok = load(cfg) if args.report_only else train(cfg, args.stride, args.limit)
    report(tok, cfg, args.stride)


if __name__ == "__main__":
    main()
