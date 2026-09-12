"""Generate Darija from a trained checkpoint.

    python -m src.sample --checkpoint s3://omarboumhaousse/darija-gpt/ckpt.pt
    python -m src.sample --checkpoint ckpt.pt --prompt "المغرب" --temperature 0.7
"""

import argparse
import sys

import torch

from src.config import DataConfig, ModelConfig
from src.model import GPT
from src.tokenizer import EOT, load
from src.train import filesystem


def load_model(path, device):
    """Rebuild the model the checkpoint describes, then fill in its weights."""
    with filesystem(path).open(path, "rb") as f:
        ckpt = torch.load(f, map_location=device, weights_only=True)
    # The architecture comes from the file, not from config.py, so a checkpoint
    # can still be read after the config has moved on.
    model = GPT(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()          # generate() leaves the mode alone, so it is set here
    return model, ckpt["step"]


def main():
    p = argparse.ArgumentParser(description="Sample from a Darija GPT checkpoint")
    p.add_argument("--checkpoint", required=True, help="a local path or an s3:// URL")
    p.add_argument("--prompt", default="", help="text to continue; empty starts a new document")
    p.add_argument("--tokens", type=int, default=200, help="tokens generated per sample")
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50, help="0 keeps the whole distribution")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    # Windows consoles default to cp1252, which cannot print Arabic at all.
    sys.stdout.reconfigure(encoding="utf-8")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model, step = load_model(args.checkpoint, device)
    tok = load(DataConfig())

    # With no prompt, start from the token the model saw between documents.
    ids = tok.encode(args.prompt).ids if args.prompt else [tok.token_to_id(EOT)]
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]  # (1, T)
    x = x.repeat(args.samples, 1)          # (samples, T): the rows run in parallel

    out = model.generate(x, args.tokens, temperature=args.temperature,
                         top_k=args.top_k or None)

    print(f"step {step}, {args.samples} samples of {args.tokens} tokens, "
          f"temperature {args.temperature}, top-k {args.top_k or 'off'}, on {device}")
    start = 0 if args.prompt else 1        # hide the <|endoftext|> we seeded
    for i, row in enumerate(out):
        print(f"\n--- {i + 1} " + "-" * 56)
        print(tok.decode(row[start:].tolist(), skip_special_tokens=False))


if __name__ == "__main__":
    main()
