"""Turn Atlaset into train.bin and val.bin.

Run from the project root:

    python -m data.prepare                    # the whole corpus
    python -m data.prepare --stride 50        # 1/50th of it, for a smoke test

Output is two flat uint16 arrays, no header, no structure: just token ids laid
end to end with <|endoftext|> after each document. The Dataset in train.py will
memory-map them and slice random windows, which is why they are flat.
"""

import argparse
import hashlib
import json
import time

import numpy as np

from src.config import DataConfig
from src.tokenizer import EOT, TOTAL_TRAIN_DOCS, iter_documents, load

REPORT_EVERY = 50_000  # documents


def doc_key(text: str, seed: int) -> int:
    """A stable 64-bit fingerprint of a document.

    Python's built-in hash() is salted per process, so it would give different
    answers on different runs and the train/val split would not be
    reproducible. blake2b is deterministic, and keying it with the config seed
    means changing the seed reshuffles the split.
    """
    digest = hashlib.blake2b(
        text.encode("utf-8"), digest_size=8, key=str(seed).encode()
    ).digest()
    return int.from_bytes(digest, "big")


def batched(iterable, n):
    """Group an iterator into lists of n.

    Worth doing because encode_batch tokenizes a whole list in parallel in
    Rust, which is far faster than calling encode() once per document.
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def prepare(cfg: DataConfig, stride: int, limit: int | None, batch_size: int):
    tok = load(cfg)

    # If tokenizer.json was built with a different vocab size than the config
    # believes, every downstream shape is wrong and nothing crashes until the
    # model is training on ids it has no embeddings for.
    assert tok.get_vocab_size() == cfg.vocab_size, (
        f"tokenizer has {tok.get_vocab_size()} tokens, config says {cfg.vocab_size}"
    )
    eot_id = tok.token_to_id(EOT)

    seen = set()          # ~1.17M 64-bit ints, roughly 100 MB at full scale
    tokens = {"train": 0, "val": 0}
    docs = {"train": 0, "val": 0}
    n_read = n_dup = 0
    last_report = 0
    t0 = time.time()

    cfg.train_bin.parent.mkdir(parents=True, exist_ok=True)

    # "wb" truncates, so a rerun replaces rather than appends.
    with open(cfg.train_bin, "wb") as f_train, open(cfg.val_bin, "wb") as f_val:
        handles = {"train": f_train, "val": f_val}

        for batch in batched(iter_documents(cfg, stride=stride, limit=limit),
                             batch_size):
            # Deduplicate before tokenizing, so we never pay to encode a
            # document we are about to throw away.
            keep, splits = [], []
            for text in batch:
                n_read += 1
                key = doc_key(text, cfg.seed)
                if key in seen:
                    n_dup += 1
                    continue
                seen.add(key)
                # Reuse the same fingerprint to assign the split. Hash-based
                # assignment is order-independent: a document lands in the same
                # split no matter when it is read, or how many others there are.
                keep.append(text)
                splits.append("val" if (key >> 32) / 2**32 < cfg.val_fraction
                              else "train")

            if keep:
                buf = {"train": [], "val": []}
                for split, enc in zip(splits, tok.encode_batch(keep)):
                    buf[split].append(
                        np.array(enc.ids + [eot_id], dtype=np.uint16)
                    )
                    docs[split] += 1
                for split, arrays in buf.items():
                    if arrays:
                        flat = np.concatenate(arrays)
                        flat.tofile(handles[split])
                        tokens[split] += len(flat)

            if n_read - last_report >= REPORT_EVERY:
                last_report = n_read
                done = tokens["train"] + tokens["val"]
                secs = time.time() - t0
                pct = 100 * n_read / (TOTAL_TRAIN_DOCS / max(stride, 1))
                print(f"  {n_read:>9,} docs  {done / 1e6:>7.1f}M tokens  "
                      f"{n_read / secs:>6,.0f} docs/s  {pct:5.1f}%  "
                      f"{secs / 60:5.1f}m elapsed")

    return {
        "documents_read": n_read,
        "duplicates_dropped": n_dup,
        "train_documents": docs["train"],
        "val_documents": docs["val"],
        "train_tokens": tokens["train"],
        "val_tokens": tokens["val"],
        "seconds": round(time.time() - t0, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stride", type=int, default=1,
                   help="use every Nth row group; 1 means the whole corpus")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after this many documents")
    p.add_argument("--batch-size", type=int, default=1000,
                   help="documents per encode_batch call")
    args = p.parse_args()

    cfg = DataConfig()
    print(f"tokenizer: {cfg.tokenizer_path}")
    print(f"writing:   {cfg.train_bin}\n           {cfg.val_bin}\n")

    stats = prepare(cfg, args.stride, args.limit, args.batch_size)

    total = stats["train_tokens"] + stats["val_tokens"]
    print(f"\ndocuments read      {stats['documents_read']:>14,}")
    print(f"duplicates dropped  {stats['duplicates_dropped']:>14,} "
          f"({100 * stats['duplicates_dropped'] / max(stats['documents_read'], 1):.1f}%)")
    print(f"train               {stats['train_documents']:>14,} docs  "
          f"{stats['train_tokens']:>14,} tokens")
    print(f"val                 {stats['val_documents']:>14,} docs  "
          f"{stats['val_tokens']:>14,} tokens")
    print(f"total                              {total:>14,} tokens")
    print(f"val share           {100 * stats['val_tokens'] / max(total, 1):>13.2f}%")
    print(f"took                {stats['seconds'] / 60:>13.1f} min")

    for name, path in [("train", cfg.train_bin), ("val", cfg.val_bin)]:
        size = path.stat().st_size
        print(f"{name}.bin {size / 1e6:>10,.1f} MB "
              f"({size // 2:,} tokens at 2 bytes each)")

    # Provenance. Two .bin files with no record of which tokenizer or which
    # filters produced them are a debugging trap later.
    meta = {
        **stats,
        "vocab_size": cfg.vocab_size,
        "eot_token": EOT,
        "min_doc_chars": cfg.min_doc_chars,
        "min_arabic_ratio": cfg.min_arabic_ratio,
        "val_fraction": cfg.val_fraction,
        "seed": cfg.seed,
        "stride": args.stride,
        "hf_repo": cfg.hf_repo,
    }
    meta_path = cfg.train_bin.parent / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nwrote {meta_path}")


if __name__ == "__main__":
    main()
