"""Generate the README figures as SVG.

    python -m docs.figures

Writes a light and a dark variant of each figure into docs/. The README pairs
them with <picture> so GitHub picks the one matching the reader's theme.

Everything is hand-written SVG rather than matplotlib: these are three specific
pictures, not a plotting problem, and hand-written SVG stays crisp at any zoom
and carries no dependency.
"""

import json
from pathlib import Path

from src.config import ModelConfig
from src.tokenizer import load
from src.config import DataConfig

OUT = Path(__file__).resolve().parent
STATS = OUT / "corpus_stats.json"

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
ARABIC_FONT = '"Segoe UI", "Noto Sans Arabic", "Geeza Pro", Tahoma, sans-serif'

# From the reference data-viz palette. Both modes are selected, not flipped:
# the dark column is the same hues re-stepped for a dark surface.
THEME = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "rule": "#e1e0d9",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
        "accent": "#2a78d6",
        "neutral": "#c3c2b7",
        "chip": ["#cde2fb", "#9ec5f4"],
        "chip_ink": "#0d366b",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "rule": "#2c2c2a",
        "series": ["#3987e5", "#d95926", "#199e70"],
        "accent": "#3987e5",
        "neutral": "#383835",
        "chip": ["#184f95", "#256abf"],
        "chip_ink": "#e8f1fd",
    },
}


# --- tiny SVG helpers --------------------------------------------------------
def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def text(x, y, s, *, fill, size=12, weight=400, anchor="start", font=FONT,
         extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family=\'{font}\' '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}"{extra}>{esc(s)}</text>')


def bar(x, y, w, h, r, fill):
    """A bar with its far (data) end rounded and its baseline end square."""
    if w <= r:
        return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w,0.5):.1f}" height="{h}" fill="{fill}"/>'
    return (f'<path d="M{x:.1f},{y} H{x + w - r:.1f} '
            f'A{r},{r} 0 0 1 {x + w:.1f},{y + r} V{y + h - r} '
            f'A{r},{r} 0 0 1 {x + w - r:.1f},{y + h} H{x:.1f} Z" fill="{fill}"/>')


def frame(w, h, t, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" role="img">'
            f'<rect width="{w}" height="{h}" rx="10" fill="{t["surface"]}"/>'
            f'{body}</svg>')


def write(name, mode, svg):
    path = OUT / f"{name}-{mode}.svg"
    path.write_text(svg, encoding="utf-8")
    return path


# --- data --------------------------------------------------------------------
def corpus_stats():
    """Token totals per source. Cached, because it is a network read."""
    if STATS.exists():
        return json.loads(STATS.read_text(encoding="utf-8"))

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    cfg = DataConfig()
    fs = HfFileSystem()
    paths = sorted(fs.glob(f"datasets/{cfg.hf_repo}/**/*.parquet"))
    paths = [p for p in paths if "train" in p.rsplit("/", 1)[-1]]
    table = pq.read_table(paths, columns=["dataset_source", "token_count"],
                          filesystem=fs)
    df = table.to_pandas()
    grouped = (df.groupby("dataset_source")["token_count"].sum()
                 .sort_values(ascending=False))
    stats = {"total": int(grouped.sum()),
             "sources": {str(k): int(v) for k, v in grouped.items()}}
    STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


# --- figure 1: corpus composition -------------------------------------------
def fig_corpus(mode):
    t = THEME[mode]
    stats = corpus_stats()
    total = stats["total"]
    items = list(stats["sources"].items())

    TOP = 9
    rows = items[:TOP]
    other = sum(v for _, v in items[TOP:])
    rows.append((f"{len(items) - TOP} smaller sources", other))

    W, H = 760, 420
    x0, x1 = 210, 660          # label gutter, bar area right edge
    y0, row_h, bar_h = 96, 30, 18
    scale = (x1 - x0) / max(v for _, v in rows)

    out = [
        text(28, 38, "What the model reads", fill=t["ink"], size=18, weight=600),
        text(28, 60, f"Atlaset train split, {total / 1e6:.0f}M words across "
                     f"{len(items)} sources", fill=t["ink2"], size=13),
        f'<line x1="28" y1="76" x2="{W - 28}" y2="76" stroke="{t["rule"]}" stroke-width="1"/>',
    ]

    for i, (name, value) in enumerate(rows):
        y = y0 + i * row_h
        is_other = i == len(rows) - 1
        fill = t["neutral"] if is_other else t["accent"]
        label = name if len(name) <= 30 else name[:28] + ".."
        out.append(text(x0 - 12, y + bar_h - 4, label, fill=t["ink2"], size=12,
                        anchor="end"))
        out.append(bar(x0, y, value * scale, bar_h, 4, fill))
        out.append(text(x0 + value * scale + 10, y + bar_h - 4,
                        f"{100 * value / total:.1f}%", fill=t["muted"], size=11))

    return frame(W, H, t, "".join(out))


# --- figure 2: where the parameters live ------------------------------------
def fig_params(mode):
    t = THEME[mode]
    c = ModelConfig()
    d, L = c.n_embd, c.n_layer

    segments = [
        ("Embeddings", c.vocab_size * d + c.block_size * d, t["series"][0]),
        ("Attention", L * 4 * d * d, t["series"][1]),
        ("MLP", L * 8 * d * d, t["series"][2]),
    ]
    total = sum(v for _, v, _ in segments)

    W, H = 760, 250
    x0, x1 = 28, 732
    bar_y, bar_h, GAP = 110, 46, 2
    span = x1 - x0

    out = [
        text(28, 38, "Where the 13.8M parameters live", fill=t["ink"], size=18,
             weight=600),
        text(28, 60, "The output head is free: it shares weights with the token "
                     "embedding", fill=t["ink2"], size=13),
        f'<line x1="28" y1="76" x2="{W - 28}" y2="76" stroke="{t["rule"]}" stroke-width="1"/>',
    ]

    x = x0
    for i, (name, value, color) in enumerate(segments):
        w = span * value / total
        draw_w = w - (GAP if i < len(segments) - 1 else 0)
        last = i == len(segments) - 1
        # Data end rounded, baseline end square; only the final segment owns
        # the data end, so the others are plain rectangles.
        if last:
            out.append(bar(x, bar_y, draw_w, bar_h, 4, color))
        else:
            out.append(f'<rect x="{x:.1f}" y="{bar_y}" width="{draw_w:.1f}" '
                       f'height="{bar_h}" fill="{color}"/>')
        # Labels sit under the bar, not inside it: the smallest segment is 23%
        # of the width but "Embeddings" plus a percentage does not fit in it.
        cx = x + draw_w / 2
        out.append(text(cx, bar_y + bar_h + 26, name, fill=t["ink"], size=13,
                        weight=600, anchor="middle"))
        out.append(text(cx, bar_y + bar_h + 44,
                        f"{value / 1e6:.1f}M   {100 * value / total:.0f}%",
                        fill=t["muted"], size=12, anchor="middle"))
        x += w

    out.append(text(28, bar_y - 16,
                    f"{c.n_layer} layers  ·  {c.n_head} heads  ·  {c.n_embd} dim  "
                    f"·  {c.block_size} context  ·  {c.vocab_size:,} vocab",
                    fill=t["muted"], size=12))
    return frame(W, H, t, "".join(out))


# --- figure 3: the tokenizer -------------------------------------------------
# Arabic glyph advance is hard to predict without shaping the text, and an SVG
# rendered inside <img> cannot use foreignObject to let the browser lay it out.
# So chip widths are estimated generously: Arabic letters join, which makes
# shaped text narrower than the sum of its glyphs, so overestimating leaves
# roomy padding rather than overflowing text.
CHAR_W = 9.6
CHIP_PAD = 13


# A real sentence from Atlaset rather than one composed for the occasion, so
# the Darija is authentic. Picked to fill one row of chips at 98% width.
SENTENCE = "هاد المشاريع متوقع باش تزيد بزاف من اقتصاد المدينة"


def fig_tokens(mode, sentence=SENTENCE):
    t = THEME[mode]
    tok = load(DataConfig())
    enc = tok.encode(sentence)
    pieces = [sentence[a:b] for a, b in enc.offsets]

    W, left, right = 760, 28, 732
    max_row = right - left
    chip_h, row_gap, y0 = 46, 10, 110

    # Pass 1: pack chips into rows. Widths have to be known before anything is
    # drawn, because each row is centred on its own total width.
    rows, row, row_w = [], [], 0.0
    for piece in pieces:
        shown = piece.strip() or "␣"
        w = len(shown) * CHAR_W + 2 * CHIP_PAD
        if row and row_w + w > max_row:
            rows.append((row, row_w))
            row, row_w = [], 0.0
        row.append((shown, w))
        row_w += w
    if row:
        rows.append((row, row_w))

    foot = y0 + len(rows) * (chip_h + row_gap) + 22
    H = foot + 44

    out = [
        text(28, 38, "How Darija tokenizes", fill=t["ink"], size=18, weight=600),
        text(28, 60, "8,000-token byte-level BPE, trained on Atlaset",
             fill=t["ink2"], size=13),
        f'<line x1="28" y1="76" x2="{W - 28}" y2="76" stroke="{t["rule"]}" stroke-width="1"/>',
    ]

    # Pass 2: draw right to left within each centred row.
    n = 0
    for r, (chips, total_w) in enumerate(rows):
        x = W / 2 + total_w / 2
        cy = y0 + r * (chip_h + row_gap)
        for shown, w in chips:
            out.append(f'<rect x="{x - w:.1f}" y="{cy}" width="{w - 2:.1f}" '
                       f'height="{chip_h}" rx="5" fill="{t["chip"][n % 2]}"/>')
            out.append(text(x - w / 2, cy + 30, shown, fill=t["chip_ink"],
                            size=17, anchor="middle", font=ARABIC_FONT,
                            extra=' direction="rtl"'))
            x -= w
            n += 1

    out.append(text(28, foot,
                    f"{len(sentence)} characters  →  {len(enc.ids)} tokens",
                    fill=t["ink2"], size=14, weight=600))
    out.append(text(28, foot + 20,
                    "1.85 tokens per word across the corpus, so 256 tokens of "
                    "context is about 138 words", fill=t["muted"], size=12))
    return frame(W, H, t, "".join(out))


# --- figures 4 and 5: inside the model ---------------------------------------
# Computed from src/model.py itself, untrained, with a fixed seed. After the
# training run these can be regenerated from the real checkpoint.

# Sequential blue from the same palette, lightest to darkest. On a light
# surface more weight is darker; on a dark surface it is lighter, so "more"
# always means "more contrast with the background".
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]


def ramp(mode, w):
    if mode == "light":
        return BLUE_RAMP[round(w * 12)]
    return BLUE_RAMP[round((1 - w) * 10)]


def sentence_tokens():
    tok = load(DataConfig())
    enc = tok.encode(SENTENCE)
    return enc.ids, [SENTENCE[a:b].strip() for a, b in enc.offsets]


def untrained_parts(seed=0):
    """An untrained embedding table, attention layer and MLP."""
    import torch  # only the model figures need PyTorch
    from src.model import FeedForward, MultiHeadAttention

    torch.manual_seed(seed)
    c = ModelConfig()
    return (torch.nn.Embedding(c.vocab_size, c.n_embd),
            MultiHeadAttention(c), FeedForward(c))


def attention_weights(ids):
    """The first head's weights on the sentence, as a T x T list of lists."""
    import torch
    from torch.nn import functional as F

    emb, mha, _ = untrained_parts()
    head, T = mha.heads[0], len(ids)
    with torch.no_grad():
        x = emb(torch.tensor([ids]))
        q, k = head.query(x), head.key(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(head.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
    return wei[0].tolist()


def outputs_that_move(ids, k, new_id):
    """Swap the token at position k: which outputs of each sub-layer change?"""
    import torch

    emb, mha, ff = untrained_parts()
    swapped = list(ids)
    swapped[k] = new_id
    with torch.no_grad():
        x, x2 = emb(torch.tensor([ids])), emb(torch.tensor([swapped]))
        att = (mha(x) - mha(x2)).abs().amax(dim=-1)[0] > 0
        mlp = (ff(x) - ff(x2)).abs().amax(dim=-1)[0] > 0
    return att.tolist(), mlp.tolist()


def fig_attention(mode):
    t = THEME[mode]
    ids, pieces = sentence_tokens()
    w = attention_weights(ids)
    T = len(ids)

    W, cell = 760, 40
    grid = cell * T
    gap, label_w = 10, 64
    gx1 = (W + grid + gap + label_w) / 2 - gap - label_w  # grid's right edge
    lx = gx1 + gap + label_w / 2                          # row labels' centre
    y0 = 124

    out = [
        text(28, 38, "No peeking ahead", fill=t["ink"], size=18, weight=600),
        text(28, 60, "One attention head before training: each row is a token "
                     "choosing what to read", fill=t["ink2"], size=13),
        f'<line x1="28" y1="76" x2="{W - 28}" y2="76" stroke="{t["rule"]}" stroke-width="1"/>',
    ]

    # Right to left, like the text: token 0 is the top row and the right column.
    for s, piece in enumerate(pieces):
        out.append(text(gx1 - (s + 0.5) * cell, y0 - 12, piece, fill=t["ink2"],
                        size=12, anchor="middle", font=ARABIC_FONT,
                        extra=' direction="rtl"'))
    for r in range(T):
        y = y0 + r * cell
        out.append(text(lx, y + cell / 2 + 4, pieces[r], fill=t["ink2"], size=12,
                        anchor="middle", font=ARABIC_FONT, extra=' direction="rtl"'))
        for s in range(T):
            x = gx1 - (s + 1) * cell
            fill = ramp(mode, w[r][s]) if s <= r else t["rule"]
            out.append(f'<rect x="{x + 1:.1f}" y="{y + 1:.1f}" width="{cell - 2}" '
                       f'height="{cell - 2}" rx="3" fill="{fill}"/>')

    ly = y0 + grid + 36
    bx = 28 + 112
    out.append(text(28, ly, "attention weight", fill=t["ink2"], size=12))
    out.append(text(bx - 8, ly, "0", fill=t["muted"], size=11, anchor="end"))
    for i in range(11):
        out.append(f'<rect x="{bx + i * 14}" y="{ly - 10}" width="12" height="12" '
                   f'rx="2" fill="{ramp(mode, i / 10)}"/>')
    out.append(text(bx + 11 * 14 + 6, ly, "1", fill=t["muted"], size=11))
    mx = bx + 11 * 14 + 40
    out.append(f'<rect x="{mx}" y="{ly - 10}" width="12" height="12" rx="2" '
               f'fill="{t["rule"]}"/>')
    out.append(text(mx + 20, ly, "masked: the future", fill=t["ink2"], size=12))
    out.append(text(28, ly + 30, "Darija reads right to left, so the first token "
                    "sits at the top right and the triangle opens that way.",
                    fill=t["muted"], size=12))
    return frame(W, ly + 54, t, "".join(out))


def fig_mixing(mode, k=4):
    t = THEME[mode]
    ids, pieces = sentence_tokens()
    att, mlp = outputs_that_move(ids, k, (ids[k] + 1) % ModelConfig().vocab_size)
    faint = t["rule"] if mode == "light" else t["neutral"]
    chip = "#f0efec" if mode == "light" else "#383835"
    swap = t["series"][1]

    W = 760
    widths = [len(p) * 8.0 + 18 for p in pieces]
    lefts, x = [], W - 28          # right to left, like the text
    for w in widths:
        x -= w
        lefts.append(x)

    out = [
        text(28, 38, "Attention mixes tokens. The MLP doesn't.", fill=t["ink"],
             size=18, weight=600),
        text(28, 60, "Swap one token for another and see which outputs change",
             fill=t["ink2"], size=13),
        f'<line x1="28" y1="76" x2="{W - 28}" y2="76" stroke="{t["rule"]}" stroke-width="1"/>',
    ]

    rows = [("Input", "one token swapped", 98, 38),
            ("Attention", f"{sum(att)} of {len(att)} outputs change", 156, 18),
            ("MLP", f"{sum(mlp)} of {len(mlp)} outputs change", 194, 18)]
    for label, note, y, h in rows:
        out.append(text(28, y + h / 2 - 1, label, fill=t["ink"], size=13, weight=600))
        out.append(text(28, y + h / 2 + 13, note, fill=t["muted"], size=11))

    for i, (x, w, piece) in enumerate(zip(lefts, widths, pieces)):
        out.append(f'<rect x="{x + 1:.1f}" y="98" width="{w - 2:.1f}" height="38" '
                   f'rx="5" fill="{swap if i == k else chip}"/>')
        out.append(text(x + w / 2, 123, piece,
                        fill="#0b0b0b" if i == k else t["ink"], size=15,
                        anchor="middle", font=ARABIC_FONT, extra=' direction="rtl"'))
        for moved, y in ((att[i], 156), (mlp[i], 194)):
            out.append(f'<rect x="{x + 1:.1f}" y="{y}" width="{w - 2:.1f}" '
                       f'height="18" rx="4" fill="{t["accent"] if moved else faint}"/>')

    ly, sx = 252, 28
    for color, label in [(swap, "swapped token"), (t["accent"], "output changed"),
                         (faint, "output identical")]:
        out.append(f'<rect x="{sx}" y="{ly - 10}" width="12" height="12" rx="2" '
                   f'fill="{color}"/>')
        out.append(text(sx + 20, ly, label, fill=t["ink2"], size=12))
        sx += 20 + len(label) * 6.6 + 28
    out.append(text(28, ly + 30, "Tokens before the swap can't see it. The MLP "
                    "handles each token on its own.", fill=t["muted"], size=12))
    return frame(W, ly + 54, t, "".join(out))


# --- figure 6: one transformer block -----------------------------------------
MONO = 'ui-monospace, "Cascadia Code", Consolas, Menlo, monospace'


def short(n):
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    return f"{n / 1e3:.0f}K" if n >= 1e3 else str(n)


def fig_block(mode):
    from src.model import Block  # parameter counts come from the real module

    t = THEME[mode]
    c = ModelConfig()
    b = Block(c)
    n_att = sum(p.numel() for p in b.sa.parameters())
    n_mlp = sum(p.numel() for p in b.ffwd.parameters())
    n_ln = sum(p.numel() for p in b.ln1.parameters())
    n_all = sum(p.numel() for p in b.parameters())

    W, S = 760, 222          # canvas width, y of the residual stream
    top, h = 122, 46         # the sub-layer boxes
    mid = top + h / 2
    chip = "#f0efec" if mode == "light" else "#383835"

    def marker(name, color, size):
        return (f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" '
                f'markerWidth="{size}" markerHeight="{size}" markerUnits="userSpaceOnUse" '
                f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>')

    def box(x0, x1, fill, ink, title, sub):
        cx = (x0 + x1) / 2
        return (f'<rect x="{x0}" y="{top}" width="{x1 - x0}" height="{h}" rx="8" fill="{fill}"/>'
                + text(cx, mid - 3, title, fill=ink, size=13, weight=600, anchor="middle")
                + text(cx, mid + 13, sub, fill=ink, size=11, anchor="middle"))

    branch = f'fill="none" stroke="{t["muted"]}" stroke-width="1.75" marker-end="url(#ab)"'

    def side_road(split, ln, sub, join):
        return (f'<circle cx="{split}" cy="{S}" r="3.5" fill="{t["ink2"]}"/>'
                f'<path d="M{split},{S} V{mid} H{ln[0] - 1}" {branch}/>'
                f'<path d="M{ln[1]},{mid} H{sub[0] - 1}" {branch}/>'
                f'<path d="M{sub[1]},{mid} H{join} V{S - 12}" {branch}/>')

    def add_node(x):
        return (f'<circle cx="{x}" cy="{S}" r="11" fill="{t["surface"]}" '
                f'stroke="{t["ink2"]}" stroke-width="2"/>'
                + text(x, S + 5.5, "+", fill=t["ink"], size=16, weight=600, anchor="middle"))

    stream = f'stroke="{t["ink2"]}" stroke-width="3" stroke-linecap="round"'

    out = [
        f'<defs>{marker("ab", t["muted"], 8)}{marker("as", t["ink2"], 11)}</defs>',
        text(28, 38, "One transformer block", fill=t["ink"], size=18, weight=600),
        text(28, 60, f"The model stacks {c.n_layer} of these, {short(n_all)} "
                     "parameters each", fill=t["ink2"], size=13),
        f'<line x1="28" y1="76" x2="{W - 28}" y2="76" stroke="{t["rule"]}" stroke-width="1"/>',
        # the residual stream, interrupted only by the two additions
        f'<line x1="56" y1="{S}" x2="341" y2="{S}" {stream}/>',
        f'<line x1="363" y1="{S}" x2="649" y2="{S}" {stream}/>',
        f'<line x1="671" y1="{S}" x2="700" y2="{S}" {stream} marker-end="url(#as)"/>',
        text(40, S + 5, "x", fill=t["ink"], size=15, anchor="middle", font=MONO),
        text(720, S + 5, "x", fill=t["ink"], size=15, anchor="middle", font=MONO),
        text(208, S - 10, "residual stream", fill=t["muted"], size=11, anchor="middle"),
        # branch 1: attention
        side_road(76, (92, 172), (190, 330), 352),
        box(92, 172, chip, t["ink"], "LayerNorm", f"{n_ln} params"),
        box(190, 330, t["series"][1], "#0b0b0b", "Attention",
            f"{c.n_head} heads · {short(n_att)} params"),
        add_node(352),
        # branch 2: MLP
        side_road(384, (400, 480), (498, 638), 660),
        box(400, 480, chip, t["ink"], "LayerNorm", f"{n_ln} params"),
        box(498, 638, t["series"][2], "#0b0b0b", "MLP", f"{short(n_mlp)} params"),
        add_node(660),
        # the two lines of Block.forward, each under the branch it describes
        text(214, S + 34, "x = x + attention(layernorm(x))", fill=t["ink2"], size=12,
             anchor="middle", font=MONO),
        text(522, S + 34, "x = x + mlp(layernorm(x))", fill=t["ink2"], size=12,
             anchor="middle", font=MONO),
        text(28, S + 70, "LayerNorm sits on the branches, not on the stream, so the "
                         "path from input to output is a plain sum.",
             fill=t["muted"], size=12),
    ]
    return frame(W, S + 94, t, "".join(out))


# --- figure 7: the whole model -----------------------------------------------
def fig_model(mode):
    from src.model import GPT  # parameter counts come from the real module

    t = THEME[mode]
    c = ModelConfig()
    m = GPT(c)

    def count(module):
        return sum(p.numel() for p in module.parameters())

    n_tok, n_pos = count(m.token_embedding_table), count(m.position_embedding_table)
    n_blocks, n_all = count(m.blocks), count(m)

    W, Y = 760, 160          # canvas width, y of the main row
    top, h = Y - 28, 56      # main-row boxes
    chip = "#f0efec" if mode == "light" else "#383835"
    blue = t["series"][0]
    on_blue = "#ffffff" if mode == "light" else "#0b0b0b"
    pos_fill, pos_ink = ((BLUE_RAMP[2], BLUE_RAMP[12]) if mode == "light"
                         else (BLUE_RAMP[9], "#ffffff"))
    arrow = f'fill="none" stroke="{t["muted"]}" stroke-width="1.75" marker-end="url(#am)"'

    def box(x0, x1, y0, hh, fill, ink, title, sub):
        cx = (x0 + x1) / 2
        return (f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{hh}" rx="8" fill="{fill}"/>'
                + text(cx, y0 + hh / 2 - 3, title, fill=ink, size=13, weight=600, anchor="middle")
                + text(cx, y0 + hh / 2 + 13, sub, fill=ink, size=11, anchor="middle"))

    out = [
        '<defs><marker id="am" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" '
        'markerHeight="8" markerUnits="userSpaceOnUse" orient="auto">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{t["muted"]}"/></marker></defs>',
        text(28, 38, "The whole model", fill=t["ink"], size=18, weight=600),
        text(28, 60, f"{short(n_all)} parameters. At every position it scores all "
                     f"{c.vocab_size:,} possible next tokens.", fill=t["ink2"], size=13),
        f'<line x1="28" y1="76" x2="{W - 28}" y2="76" stroke="{t["rule"]}" stroke-width="1"/>',
        # weight tying: one matrix, used at both ends of the model
        f'<path d="M149,{top - 2} C149,{top - 40} 580,{top - 40} 580,{top - 2}" fill="none" '
        f'stroke="{blue}" stroke-width="1.5" stroke-dasharray="5 4"/>',
        f'<rect x="286" y="{top - 39}" width="156" height="16" fill="{t["surface"]}"/>',
        text(364, top - 27, "same matrix: weight tying", fill=t["ink2"], size=11,
             anchor="middle"),
        # main row, left to right
        text(28, Y + 4, "tokens", fill=t["ink2"], size=12),
        f'<path d="M72,{Y} H85" {arrow}/>',
        box(86, 212, top, h, blue, on_blue, "Token embedding",
            f"{c.vocab_size:,} × {c.n_embd} · {short(n_tok)}"),
        f'<path d="M212,{Y} H222" {arrow}/>',
        f'<circle cx="234" cy="{Y}" r="11" fill="{t["surface"]}" stroke="{t["ink2"]}" '
        'stroke-width="2"/>',
        text(234, Y + 5.5, "+", fill=t["ink"], size=16, weight=600, anchor="middle"),
        box(159, 309, Y + 56, 44, pos_fill, pos_ink, "Position embedding",
            f"{c.block_size} × {c.n_embd} · {short(n_pos)}"),
        f'<path d="M234,{Y + 56} V{Y + 12}" {arrow}/>',
        f'<path d="M245,{Y} H265" {arrow}/>',
    ]

    # the stack of blocks: two cards peeking out behind the front one
    bx0, bx1 = 266, 392
    for k in (2, 1):
        out.append(f'<rect x="{bx0 + 4 * k}" y="{top - 4 * k}" width="{bx1 - bx0}" '
                   f'height="{h}" rx="8" fill="{t["rule"]}"/>')
    out.append(box(bx0, bx1, top, h, chip, t["ink"], f"{c.n_layer} × Block",
                   f"{short(n_blocks)} params"))
    inner = bx1 - bx0 - 20   # a thin stripe echoing the block figure: attention 1/3, MLP 2/3
    out.append(f'<rect x="{bx0 + 10}" y="{top + h - 9}" width="{inner / 3 - 1:.1f}" '
               f'height="4" rx="2" fill="{t["series"][1]}"/>')
    out.append(f'<rect x="{bx0 + 10 + inner / 3 + 1:.1f}" y="{top + h - 9}" '
               f'width="{inner * 2 / 3 - 1:.1f}" height="4" rx="2" fill="{t["series"][2]}"/>')

    out += [
        f'<path d="M401,{Y} H415" {arrow}/>',
        box(416, 496, top, h, chip, t["ink"], "LayerNorm", "final"),
        f'<path d="M496,{Y} H517" {arrow}/>',
        box(518, 642, top, h, blue, on_blue, "Output head", "tied: no new params"),
        f'<path d="M642,{Y} H653" {arrow}/>',
        text(658, Y - 3, "next-token", fill=t["ink2"], size=12),
        text(658, Y + 13, "scores", fill=t["ink2"], size=12),
        text(28, Y + 136, "The output head reuses the token embedding matrix, so it "
                          "adds no parameters.", fill=t["muted"], size=12),
    ]
    return frame(W, Y + 160, t, "".join(out))


# --- figure 8: the pre-training checks ----------------------------------------
def check_values():
    """The three quantities tests/test_model.py asserts on, measured."""
    import math

    import torch
    from src.model import GPT

    torch.manual_seed(0)
    c = ModelConfig()
    m = GPT(c).eval()
    idx = torch.randint(0, c.vocab_size, (1, 16))
    with torch.no_grad():
        logits, _ = m(idx)
        moved = 0.0
        for t in range(15):
            changed = idx.clone()
            changed[0, t + 1:] = (changed[0, t + 1:] + 1) % c.vocab_size
            after, _ = m(changed)
            moved = max(moved, (logits[0, :t + 1] - after[0, :t + 1]).abs().max().item())
        x = torch.randint(0, c.vocab_size, (8, 64))
        y = torch.randint(0, c.vocab_size, (8, 64))
        _, loss = m(x, y)
    return logits.shape[-1], moved, loss.item(), math.log(c.vocab_size)


def fig_checks(mode):
    t = THEME[mode]
    n_out, moved, loss, ln_v = check_values()
    tile = "#f4f3ef" if mode == "light" else "#242423"
    good = "#006300" if mode == "light" else "#0ca30c"   # success text, never alone: "✓ passes"

    W, y0, h, gap = 760, 96, 150, 16
    w = (W - 56 - 2 * gap) / 3
    tiles = [
        ("Output shape", f"{n_out:,}", ["scores at every position,", "one for each token"]),
        ("No peeking ahead", "0" if moved == 0 else f"{moved:.1e}",
         ["how far earlier outputs move", "when later tokens change"]),
        ("Loss before training", f"{loss:.2f}",
         [f"ln {n_out:,} = {ln_v:.2f}: the model", "knows nothing yet, as it should"]),
    ]
    out = [
        text(28, 38, "Checked before any training", fill=t["ink"], size=18, weight=600),
        text(28, 60, "Three tests in tests/test_model.py, all passing",
             fill=t["ink2"], size=13),
        f'<line x1="28" y1="76" x2="{W - 28}" y2="76" stroke="{t["rule"]}" stroke-width="1"/>',
    ]
    for i, (label, value, caption) in enumerate(tiles):
        x = 28 + i * (w + gap)
        out.append(f'<rect x="{x:.1f}" y="{y0}" width="{w:.1f}" height="{h}" rx="10" '
                   f'fill="{tile}"/>')
        out.append(text(x + 18, y0 + 28, label, fill=t["ink2"], size=13, weight=600))
        out.append(text(x + 18, y0 + 76, value, fill=t["ink"], size=40, weight=600))
        for j, line in enumerate(caption):
            out.append(text(x + 18, y0 + 100 + 16 * j, line, fill=t["muted"], size=12))
        out.append(text(x + 18, y0 + h - 12, "✓ passes", fill=good, size=12, weight=600))
    return frame(W, y0 + h + 28, t, "".join(out))


# --- figure 9: training pairs ---------------------------------------------------
def fig_batch(mode, k=4):
    t = THEME[mode]
    _, pieces = sentence_tokens()
    xs, ys = pieces[:-1], pieces[1:]   # the same slicing as get_batch: y is x shifted by one
    n = len(xs)
    chip = "#f0efec" if mode == "light" else "#383835"
    seen, seen_ink = ((BLUE_RAMP[1], BLUE_RAMP[12]) if mode == "light"
                      else (BLUE_RAMP[10], "#ffffff"))
    target = t["series"][1]

    W = 760
    widths = [max(len(a), len(b)) * 8.0 + 18 for a, b in zip(xs, ys)]
    lefts, x = [], W - 28              # right to left, like the text
    for w in widths:
        x -= w
        lefts.append(x)

    r1, r2, h = 100, 166, 36           # tops of the two rows, chip height
    out = [
        '<defs><marker id="ad" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" markerUnits="userSpaceOnUse" orient="auto">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{t["muted"]}"/></marker></defs>',
        text(28, 38, f"One sentence, {n} predictions", fill=t["ink"], size=18, weight=600),
        text(28, 60, "A training pair is a window and the same window shifted by one token",
             fill=t["ink2"], size=13),
        f'<line x1="28" y1="76" x2="{W - 28}" y2="76" stroke="{t["rule"]}" stroke-width="1"/>',
        text(28, r1 + h / 2 - 1, "Input", fill=t["ink"], size=13, weight=600),
        text(28, r1 + h / 2 + 13, "x", fill=t["muted"], size=11, font=MONO),
        text(28, r2 + h / 2 - 1, "Target", fill=t["ink"], size=13, weight=600),
        text(28, r2 + h / 2 + 13, "y = x shifted by one", fill=t["muted"], size=11),
    ]
    for i, (left, w) in enumerate(zip(lefts, widths)):
        cx = left + w / 2
        fx, ix = (seen, seen_ink) if i <= k else (chip, t["ink"])
        fy, iy = (target, "#0b0b0b") if i == k else (chip, t["ink"])
        out.append(f'<rect x="{left + 1:.1f}" y="{r1}" width="{w - 2:.1f}" height="{h}" '
                   f'rx="5" fill="{fx}"/>')
        out.append(text(cx, r1 + 24, xs[i], fill=ix, size=15, anchor="middle",
                        font=ARABIC_FONT, extra=' direction="rtl"'))
        out.append(f'<path d="M{cx:.1f},{r1 + h + 4} V{r2 - 5}" fill="none" '
                   f'stroke="{t["muted"]}" stroke-width="1.5" marker-end="url(#ad)"/>')
        out.append(f'<rect x="{left + 1:.1f}" y="{r2}" width="{w - 2:.1f}" height="{h}" '
                   f'rx="5" fill="{fy}"/>')
        out.append(text(cx, r2 + 24, ys[i], fill=iy, size=15, anchor="middle",
                        font=ARABIC_FONT, extra=' direction="rtl"'))

    ly, sx = r2 + h + 34, 28
    for color, label in [(seen, f"what position {k + 1} has read"),
                         (target, "what it must predict next")]:
        out.append(f'<rect x="{sx}" y="{ly - 10}" width="12" height="12" rx="2" '
                   f'fill="{color}"/>')
        out.append(text(sx + 20, ly, label, fill=t["ink2"], size=12))
        sx += 20 + len(label) * 6.6 + 28
    out.append(text(28, ly + 30, "Every position does this at once, in one forward pass. "
                    "A real window is 256 tokens, so 256 predictions each.",
                    fill=t["muted"], size=12))
    return frame(W, ly + 54, t, "".join(out))


# --- figure 10: the first training run ------------------------------------------
FIRST_RUN = OUT / "first_run.json"
FIRST_RUN_CFG = dict(batch_size=8, max_iters=60, eval_interval=10, eval_iters=4)


def first_run():
    """Loss history of a short CPU run of the real training loop. Cached."""
    if FIRST_RUN.exists():
        return json.loads(FIRST_RUN.read_text(encoding="utf-8"))
    from src.config import TrainConfig
    from src.train import train

    _, history = train(tc=TrainConfig(**FIRST_RUN_CFG))
    run = {"config": FIRST_RUN_CFG, "history": history}
    FIRST_RUN.write_text(json.dumps(run, indent=2), encoding="utf-8")
    return run


def fig_loss(mode):
    import math

    t = THEME[mode]
    run = first_run()
    hist, cfg = run["history"], run["config"]
    ln_v = math.log(ModelConfig().vocab_size)
    last = hist[-1]["step"]

    W, x0, x1, y0, y1 = 760, 64, 560, 104, 304        # plot area
    lo = min(min(h["train"], h["val"]) for h in hist)
    y_min, y_max = math.floor((lo - 0.1) * 2) / 2, 9.5

    def X(s):
        return x0 + (x1 - x0) * s / last

    def Y(v):
        return y1 - (y1 - y0) * (v - y_min) / (y_max - y_min)

    out = [
        text(28, 38, "The loss starts to fall", fill=t["ink"], size=18, weight=600),
        text(28, 60, f"The real training loop, {cfg['max_iters']} steps on the laptop CPU "
                     f"(batch {cfg['batch_size']}, smoke-test data)", fill=t["ink2"], size=13),
        f'<line x1="28" y1="76" x2="{W - 28}" y2="76" stroke="{t["rule"]}" stroke-width="1"/>',
    ]
    v = y_min                                          # gridlines and y labels
    while v <= y_max + 1e-9:
        out.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" '
                   f'stroke="{t["rule"]}" stroke-width="1"/>')
        out.append(text(x0 - 10, Y(v) + 4, f"{v:.1f}", fill=t["muted"], size=11, anchor="end"))
        v += 0.5
    for h in hist:                                     # x labels at every evaluation
        out.append(text(X(h["step"]), y1 + 18, str(h["step"]), fill=t["muted"], size=11,
                        anchor="middle"))
    out.append(text(x1, y1 + 36, "step", fill=t["muted"], size=11, anchor="end"))
    # the loss of a model that knows nothing
    out.append(f'<line x1="{x0}" y1="{Y(ln_v):.1f}" x2="{x1}" y2="{Y(ln_v):.1f}" '
               f'stroke="{t["muted"]}" stroke-width="1"/>')
    out.append(text(x1 + 8, Y(ln_v) + 4, f"ln 8,000 = {ln_v:.2f}", fill=t["muted"], size=11))

    for key, color in (("train", t["series"][0]), ("val", t["series"][1])):
        pts = " ".join(f"{X(h['step']):.1f},{Y(h[key]):.1f}" for h in hist)
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" '
                   'stroke-linejoin="round" stroke-linecap="round"/>')
        for h in hist:
            out.append(f'<circle cx="{X(h["step"]):.1f}" cy="{Y(h[key]):.1f}" r="4" '
                       f'fill="{color}" stroke="{t["surface"]}" stroke-width="2"/>')

    # direct labels at the line ends, merged into one if they would collide
    yt, yv = Y(hist[-1]["train"]), Y(hist[-1]["val"])
    if abs(yt - yv) >= 16:
        out.append(text(x1 + 12, yt + 4, f"train {hist[-1]['train']:.2f}", fill=t["ink"], size=12))
        out.append(text(x1 + 12, yv + 4, f"val {hist[-1]['val']:.2f}", fill=t["ink"], size=12))
    else:
        out.append(text(x1 + 12, (yt + yv) / 2 + 4,
                        f"train {hist[-1]['train']:.2f} · val {hist[-1]['val']:.2f}",
                        fill=t["ink"], size=12))
    lx = x0                                            # legend
    for key, color in (("train", t["series"][0]), ("val", t["series"][1])):
        out.append(f'<line x1="{lx}" y1="92" x2="{lx + 16}" y2="92" stroke="{color}" '
                   'stroke-width="2" stroke-linecap="round"/>')
        out.append(text(lx + 22, 96, f"{key} loss", fill=t["ink2"], size=12))
        lx += 110
    out.append(text(28, y1 + 62, "Step 0 starts at ln 8,000, the loss of a model that knows "
                    "nothing. The GPU run will replace this curve.", fill=t["muted"], size=12))
    return frame(W, y1 + 86, t, "".join(out))


# --- figure 11: overfitting one batch -------------------------------------------
OVERFIT_RUN = OUT / "overfit_run.json"
OVERFIT_CFG = dict(batch_size=2, steps=301, lr=3e-4)   # 301 losses: after 0..300 updates


def overfit_run():
    """Loss at every step of the real model trained over and over on one batch. Cached."""
    if OVERFIT_RUN.exists():
        return json.loads(OVERFIT_RUN.read_text(encoding="utf-8"))
    import torch
    from src.train import get_batch, overfit_one_batch

    c = ModelConfig()
    torch.manual_seed(0)
    x, y = get_batch("train", OVERFIT_CFG["batch_size"], c.block_size)
    losses = overfit_one_batch(x, y, c, lr=OVERFIT_CFG["lr"], steps=OVERFIT_CFG["steps"])
    for s in list(range(0, len(losses), 25)) + [len(losses) - 1]:
        print(f"overfit, after {s} updates: loss {losses[s]:.4f}")
    run = {"config": OVERFIT_CFG, "losses": losses}
    OVERFIT_RUN.write_text(json.dumps(run), encoding="utf-8")
    return run


def unigram_entropy():
    """Loss of a model that only knows how often each token appears.

    Entropy of the token distribution in train.bin. Any real model has to beat
    this to be worth anything, so it is the honest floor to compare against.
    """
    import numpy as np
    from src.config import DataConfig

    d = np.memmap(DataConfig().train_bin, dtype=np.uint16, mode="r")
    counts = np.bincount(d, minlength=ModelConfig().vocab_size).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def fig_overfit(mode):
    import math

    from src.model import GPT

    t = THEME[mode]
    run = overfit_run()
    losses, cfg = run["losses"], run["config"]
    c = ModelConfig()
    n_params = sum(p.numel() for p in GPT(c).parameters())
    ln_v = math.log(c.vocab_size)
    uni = unigram_entropy()

    last = len(losses) - 1
    W, x0, x1, y0, y1, y_max = 760, 64, 540, 96, 296, 9.5

    def X(s):
        return x0 + (x1 - x0) * s / last

    def Y(v):
        return y1 - (y1 - y0) * v / y_max

    out = [
        text(28, 38, "It can memorise one batch", fill=t["ink"], size=18, weight=600),
        text(28, 60, f"The real {short(n_params)}-parameter model, trained {last} times on "
                     f"the same {cfg['batch_size']} × {c.block_size} tokens",
             fill=t["ink2"], size=13),
        f'<line x1="28" y1="76" x2="{W - 28}" y2="76" stroke="{t["rule"]}" stroke-width="1"/>',
    ]
    for v in (0, 2, 4, 6, 8):
        out.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" '
                   f'stroke="{t["rule"]}" stroke-width="1"/>')
        out.append(text(x0 - 10, Y(v) + 4, str(v), fill=t["muted"], size=11, anchor="end"))
    for s in range(0, last + 1, 50):
        out.append(text(X(s), y1 + 18, str(s), fill=t["muted"], size=11, anchor="middle"))
    out.append(text(x1, y1 + 36, "step", fill=t["muted"], size=11, anchor="end"))
    for v, label in ((ln_v, "knows nothing"), (uni, "knows token frequencies")):
        out.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" '
                   f'stroke="{t["muted"]}" stroke-width="1"/>')
        out.append(text(x1 + 8, Y(v) + 4, f"{v:.2f} · {label}", fill=t["muted"], size=11))

    pts = " ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in enumerate(losses))
    out.append(f'<polyline points="{pts}" fill="none" stroke="{t["series"][0]}" '
               'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    out.append(f'<circle cx="{X(last):.1f}" cy="{Y(losses[-1]):.1f}" r="4" '
               f'fill="{t["series"][0]}" stroke="{t["surface"]}" stroke-width="2"/>')
    out.append(text(x1 + 12, Y(losses[-1]) - 6, f"{losses[-1]:.3f} after {last} steps",
                    fill=t["ink"], size=12))
    out.append(text(28, y1 + 62, "Fresh batches stall near 7.7 for now. One repeated batch "
                    "goes almost to zero: the model and loop can learn.",
                    fill=t["muted"], size=12))
    return frame(W, y1 + 86, t, "".join(out))


# --- figure 12: the learning-rate schedule --------------------------------------
def sci(v):
    """1e-3 -> '1e-3', 2.5e-4 -> '2.5e-4', 0 -> '0'."""
    if v == 0:
        return "0"
    mantissa, exponent = f"{v:.1e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"


def fig_lr(mode):
    from src.config import TrainConfig
    from src.train import get_lr

    t = THEME[mode]
    tc = TrainConfig()
    blue = t["series"][0]
    steps = list(range(tc.warmup_iters)) + list(range(tc.warmup_iters, tc.max_iters + 1, 25))
    lrs = [get_lr(s, tc) for s in steps]

    W, x0, x1, y0, y1 = 760, 72, 580, 96, 286
    y_max = tc.learning_rate * 1.1

    def X(s):
        return x0 + (x1 - x0) * s / tc.max_iters

    def Y(v):
        return y1 - (y1 - y0) * v / y_max

    out = [
        text(28, 38, "Warm up, then glide down", fill=t["ink"], size=18, weight=600),
        text(28, 60, f"Learning rate per step: linear warmup for {tc.warmup_iters} steps, "
                     f"cosine decay to {sci(tc.min_lr)} by step {tc.max_iters:,}",
             fill=t["ink2"], size=13),
        f'<line x1="28" y1="76" x2="{W - 28}" y2="76" stroke="{t["rule"]}" stroke-width="1"/>',
    ]
    for k in range(5):
        v = tc.learning_rate * k / 4
        out.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" '
                   f'stroke="{t["rule"]}" stroke-width="1"/>')
        out.append(text(x0 - 10, Y(v) + 4, sci(v), fill=t["muted"], size=11, anchor="end"))
    for s in range(0, tc.max_iters + 1, 1000):
        out.append(text(X(s), y1 + 18, f"{s:,}", fill=t["muted"], size=11, anchor="middle"))
    out.append(text(x1, y1 + 36, "step", fill=t["muted"], size=11, anchor="end"))

    curve = " ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in zip(steps, lrs))
    out.append(f'<polygon points="{X(0):.1f},{Y(0):.1f} {curve} {X(tc.max_iters):.1f},{Y(0):.1f}" '
               f'fill="{blue}" fill-opacity="0.12"/>')
    out.append(f'<polyline points="{curve}" fill="none" stroke="{blue}" stroke-width="2" '
               'stroke-linejoin="round" stroke-linecap="round"/>')

    peak_s = tc.warmup_iters - 1
    for s, v in ((peak_s, get_lr(peak_s, tc)), (tc.max_iters, tc.min_lr)):
        out.append(f'<circle cx="{X(s):.1f}" cy="{Y(v):.1f}" r="4" fill="{blue}" '
                   f'stroke="{t["surface"]}" stroke-width="2"/>')
    out.append(text(X(peak_s) + 10, Y(tc.learning_rate) - 10,
                    f"peak {sci(tc.learning_rate)}, reached after {tc.warmup_iters} steps",
                    fill=t["ink"], size=12))
    out.append(text(x1 + 12, Y(tc.min_lr) - 2, f"floor {sci(tc.min_lr)}", fill=t["ink"], size=12))
    out.append(text(x1 + 12, Y(tc.min_lr) + 13, f"at step {tc.max_iters:,}", fill=t["muted"],
                    size=11))
    out.append(text(28, y1 + 62, "The warmup keeps the first steps gentle while AdamW's running "
                    "averages are still unreliable.", fill=t["muted"], size=12))
    out.append(text(28, y1 + 80, "The slow decay lets the model settle into a minimum at the "
                    "end instead of bouncing around it.", fill=t["muted"], size=12))
    return frame(W, y1 + 104, t, "".join(out))


# --- figure 13: gradient clipping --------------------------------------------------
CLIP_RUN = OUT / "clip_run.json"
CLIP_CFG = dict(batch_size=4, steps=101, max_iters=5000)  # 101 norms: steps 0..100.
# max_iters is pinned so the figure stays reproducible when the config's default changes.


def clip_run():
    """Gradient norm at every step of the real loop's first steps. Cached."""
    if CLIP_RUN.exists():
        return json.loads(CLIP_RUN.read_text(encoding="utf-8"))
    import torch
    from src.config import TrainConfig
    from src.model import GPT
    from src.train import get_batch, get_lr

    mc = ModelConfig()
    tc = TrainConfig(batch_size=CLIP_CFG["batch_size"], max_iters=CLIP_CFG["max_iters"])
    torch.manual_seed(1337)
    model = GPT(mc)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.learning_rate)
    norms = []
    for step in range(CLIP_CFG["steps"]):   # the same step as train(), recording the norm
        for group in optimizer.param_groups:
            group["lr"] = get_lr(step, tc)
        x, y = get_batch("train", tc.batch_size, mc.block_size)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip).item())
        optimizer.step()
    print("clip run, norms at steps 0, 10, ..., 100: "
          + " ".join(f"{norms[s]:.2f}" for s in range(0, len(norms), 10)))
    run = {"config": CLIP_CFG, "grad_clip": tc.grad_clip, "norms": norms}
    CLIP_RUN.write_text(json.dumps(run), encoding="utf-8")
    return run


def fig_clip(mode):
    t = THEME[mode]
    run = clip_run()
    norms, cap = run["norms"], run["grad_clip"]
    last = len(norms) - 1
    clipped = [i for i, v in enumerate(norms) if v > cap]
    top = max(max(norms), cap) * 1.12
    tick = next(s for s in (0.25, 0.5, 1, 2, 5, 10, 20) if top / s <= 6)
    blue, orange = t["series"][0], t["series"][1]

    W, x0, x1, y0, y1 = 760, 64, 560, 104, 294

    def X(s):
        return x0 + (x1 - x0) * s / last

    def Y(v):
        return y1 - (y1 - y0) * v / top

    out = [
        text(28, 38, "A cap on the step size", fill=t["ink"], size=18, weight=600),
        text(28, 60, f"Gradient norm at every step from 0 to {last}, during the warmup, "
                     f"before clipping at {cap:g}", fill=t["ink2"], size=13),
        f'<line x1="28" y1="76" x2="{W - 28}" y2="76" stroke="{t["rule"]}" stroke-width="1"/>',
        f'<line x1="{x0}" y1="92" x2="{x0 + 16}" y2="92" stroke="{blue}" stroke-width="2" '
        'stroke-linecap="round"/>',
        text(x0 + 22, 96, "gradient norm, before clipping", fill=t["ink2"], size=12),
        f'<circle cx="{x0 + 238}" cy="92" r="4" fill="{orange}" stroke="{t["surface"]}" '
        'stroke-width="2"/>',
        text(x0 + 248, 96, "steps scaled down to the cap", fill=t["ink2"], size=12),
    ]
    v = 0.0
    while v <= top + 1e-9:
        out.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" '
                   f'stroke="{t["rule"]}" stroke-width="1"/>')
        out.append(text(x0 - 10, Y(v) + 4, f"{v:g}", fill=t["muted"], size=11, anchor="end"))
        v += tick
    for s in range(0, last + 1, 25):
        out.append(text(X(s), y1 + 18, str(s), fill=t["muted"], size=11, anchor="middle"))
    out.append(text(x1, y1 + 36, "step", fill=t["muted"], size=11, anchor="end"))

    out.append(f'<line x1="{x0}" y1="{Y(cap):.1f}" x2="{x1}" y2="{Y(cap):.1f}" '
               f'stroke="{t["ink2"]}" stroke-width="1.5"/>')
    out.append(text(x1 + 8, Y(cap) + 4, f"cap {cap:g}", fill=t["ink"], size=12))
    pts = " ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in enumerate(norms))
    out.append(f'<polyline points="{pts}" fill="none" stroke="{blue}" stroke-width="2" '
               'stroke-linejoin="round" stroke-linecap="round"/>')
    for i in clipped:
        out.append(f'<circle cx="{X(i):.1f}" cy="{Y(norms[i]):.1f}" r="4" fill="{orange}" '
                   f'stroke="{t["surface"]}" stroke-width="2"/>')
    peak = max(range(len(norms)), key=lambda i: norms[i])
    out.append(text(X(peak) + 10, Y(norms[peak]) + 4, f"{norms[peak]:.2f} at step {peak}",
                    fill=t["ink"], size=12))

    if clipped:
        note = (f"{len(clipped)} of {last + 1} steps went over the cap and were scaled down to "
                f"length {cap:g}; the rest passed through unchanged.")
    else:
        note = ("No step went over the cap here: clipping is insurance against the bad batch "
                "that hasn't come yet.")
    out.append(text(28, y1 + 62, note, fill=t["muted"], size=12))
    return frame(W, y1 + 86, t, "".join(out))


# --- figure 14: mixed precision ---------------------------------------------------
AMP_RUN = OUT / "amp_run.json"


def fig_amp(mode):
    t = THEME[mode]
    run = json.loads(AMP_RUN.read_text(encoding="utf-8"))
    fp32, fp16 = run["runs"]["fp32"], run["runs"]["fp16"]
    speedup = fp32["ms_per_step"] / fp16["ms_per_step"]
    per_step = run["batch_size"] * run["block_size"]
    minutes = round(run["corpus_tokens"] / per_step * fp16["ms_per_step"] / 1000 / 60)

    W, x0, x1 = 760, 200, 600   # x1 leaves room for the value labels at the bar tips
    scale = (x1 - x0) / fp32["ms_per_step"]
    rows = [("fp32", "no autocast", fp32["ms_per_step"], t["neutral"], ""),
            ("fp16", "autocast + GradScaler", fp16["ms_per_step"], t["series"][0],
             f"   {speedup:.1f}x faster")]

    out = [
        text(28, 38, "Half precision, twice the speed", fill=t["ink"], size=18, weight=600),
        text(28, 60, f"{run['steps']} real training steps at batch {run['batch_size']} on the "
                     f"pod's {run['device']}, same seed and same data",
             fill=t["ink2"], size=13),
        f'<line x1="28" y1="76" x2="{W - 28}" y2="76" stroke="{t["rule"]}" stroke-width="1"/>',
    ]
    for i, (name, sub, ms, color, extra) in enumerate(rows):
        y = 106 + i * 58
        out.append(text(28, y + 16, name, fill=t["ink"], size=14, weight=600))
        out.append(text(28, y + 32, sub, fill=t["muted"], size=11))
        out.append(bar(x0, y, ms * scale, 34, 4, color))
        out.append(text(x0 + ms * scale + 10, y + 23, f"{ms} ms per step", fill=t["ink"],
                        size=13))
        if extra:
            out.append(text(x0 + ms * scale + 122, y + 23, extra, fill=t["ink2"], size=13,
                            weight=600))

    facts = [
        f"Identical results: train loss {fp16['train_loss_step39']:.4f} and val loss "
        f"{fp16['val_loss_step39']:.4f} after {run['steps']} steps, in both runs.",
        f"Peak GPU memory {fp32['peak_gb']:.2f} GB in fp32, {fp16['peak_gb']:.2f} GB in fp16: "
        "the tensors kept in fp32 sit alongside their fp16 copies.",
        f"At this speed one pass over the {run['corpus_tokens'] / 1e6:.0f}M-token corpus takes "
        f"about {int(minutes // 60)} h {int(minutes % 60)} min.",
    ]
    for i, line in enumerate(facts):
        out.append(text(28, 244 + i * 20, line, fill=t["muted"], size=12))
    return frame(W, 244 + len(facts) * 20 + 16, t, "".join(out))


# --- figure 15: what a checkpoint carries -----------------------------------------
CKPT_RUN = OUT / "checkpoint_run.json"


def fig_ckpt(mode):
    t = THEME[mode]
    run = json.loads(CKPT_RUN.read_text(encoding="utf-8"))
    segments = [("Model weights", run["model_mb"], t["series"][0]),
                ("Optimizer state", run["optimizer_mb"], t["series"][2])]
    total = run["total_mb"]

    W, x0, x1 = 760, 28, 732
    bar_y, bar_h, gap = 112, 46, 2
    span = x1 - x0

    out = [
        text(28, 38, "What a checkpoint carries", fill=t["ink"], size=18, weight=600),
        text(28, 60, f"{total:.1f} MB written to the bucket every "
                     f"{run['checkpoint_every']} steps, and once at the end",
             fill=t["ink2"], size=13),
        f'<line x1="28" y1="76" x2="{W - 28}" y2="76" stroke="{t["rule"]}" stroke-width="1"/>',
    ]
    x = x0
    for i, (name, mb, color) in enumerate(segments):
        w = span * mb / total
        draw = w - (gap if i < len(segments) - 1 else 0)
        if i == len(segments) - 1:
            out.append(bar(x, bar_y, draw, bar_h, 4, color))
        else:
            out.append(f'<rect x="{x:.1f}" y="{bar_y}" width="{draw:.1f}" height="{bar_h}" '
                       f'fill="{color}"/>')
        cx = x + draw / 2
        out.append(text(cx, bar_y + bar_h + 26, name, fill=t["ink"], size=13, weight=600,
                        anchor="middle"))
        out.append(text(cx, bar_y + bar_h + 44, f"{mb:.1f} MB   {100 * mb / total:.0f}%",
                        fill=t["muted"], size=12, anchor="middle"))
        x += w

    facts = [
        f"AdamW keeps two running averages per parameter, so its state is twice the model's "
        f"{run['model_mb']:.0f} MB.",
        f"The rest (step number, model config, loss history, scaler state) is "
        f"{run['other_mb']:.1f} MB.",
        "The tied matrix appears twice in the state dict but is stored once in the file, "
        "and comes back shared.",
    ]
    for i, line in enumerate(facts):
        out.append(text(28, 240 + i * 20, line, fill=t["muted"], size=12))
    return frame(W, 240 + len(facts) * 20 + 16, t, "".join(out))


# --- figure 16: the real training run ---------------------------------------------
TRAIN_RUN = OUT / "train_run.json"


def train_run():
    """The 17,000-step run's history, transcribed from the pod's train.log.

    Not regenerable on this laptop: it is 78 minutes of GPU time, and the real
    train.bin lives on the pod. The same history is inside the checkpoint.
    """
    return json.loads(TRAIN_RUN.read_text(encoding="utf-8"))


def fig_train(mode):
    import math

    t = THEME[mode]
    run = train_run()
    hist, cfg = run["history"], run["config"]
    final = hist[-1]
    per_step = cfg["batch_size"] * cfg["block_size"]

    W, x0, x1, y0, y1 = 760, 64, 540, 112, 342
    y_min, y_max = 3.5, 6.0        # the data's own range: the baselines it beats
                                   # are far above it and live in the callout

    def X(s):
        return x0 + (x1 - x0) * s / cfg["max_iters"]

    def Y(v):
        return y1 - (y1 - y0) * (v - y_min) / (y_max - y_min)

    out = [
        text(28, 38, f"One pass over {run['train_tokens'] / 1e6:.0f} million tokens",
             fill=t["ink"], size=18, weight=600),
        text(28, 60, f"{cfg['max_iters']:,} steps on a single T4 in about {run['minutes']} "
                     f"minutes, {cfg['batch_size']} × {cfg['block_size']} = {per_step:,} "
                     f"tokens per step", fill=t["ink2"], size=13),
        f'<line x1="28" y1="76" x2="{W - 28}" y2="76" stroke="{t["rule"]}" stroke-width="1"/>',
    ]
    v = y_min                                          # gridlines and y labels
    while v <= y_max + 1e-9:
        out.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" '
                   f'stroke="{t["rule"]}" stroke-width="1"/>')
        out.append(text(x0 - 10, Y(v) + 4, f"{v:.1f}", fill=t["muted"], size=11, anchor="end"))
        v += 0.5
    for s in range(0, cfg["max_iters"], 2000):
        out.append(text(X(s), y1 + 18, f"{s:,}", fill=t["muted"], size=11, anchor="middle"))
    out.append(text(x1, y1 + 36, "step", fill=t["muted"], size=11, anchor="end"))

    # The curve falls away from the top right, so the comparison goes there.
    cx0, cx1, cy = 300, x1, y0 + 24
    out.append(text(cx0, cy, "validation loss, for scale", fill=t["ink2"], size=12, weight=600))
    out.append(f'<line x1="{cx0}" y1="{cy + 10}" x2="{cx1}" y2="{cy + 10}" '
               f'stroke="{t["rule"]}" stroke-width="1"/>')
    rows = [("a model that knows nothing", math.log(ModelConfig().vocab_size), False),
            ("a model that knows token frequencies", unigram_entropy(), False),
            (f"this one, after {cfg['max_iters']:,} steps", final["val"], True)]
    for i, (label, value, strong) in enumerate(rows):
        y = cy + 30 + i * 19
        out.append(text(cx0, y, label, fill=t["ink"] if strong else t["muted"], size=12,
                        weight=600 if strong else 400))
        out.append(text(cx1, y, f"{value:.2f}", fill=t["ink"] if strong else t["muted"],
                        size=12, weight=600 if strong else 400, anchor="end"))

    for key, color in (("train", t["series"][0]), ("val", t["series"][1])):
        pts = " ".join(f"{X(h['step']):.1f},{Y(h[key]):.1f}" for h in hist)
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" '
                   'stroke-linejoin="round" stroke-linecap="round"/>')
        out.append(f'<circle cx="{X(final["step"]):.1f}" cy="{Y(final[key]):.1f}" r="4" '
                   f'fill="{color}" stroke="{t["surface"]}" stroke-width="2"/>')
    # the two lines end 0.13 apart, far too close for one label each
    out.append(text(x1 + 12, (Y(final["train"]) + Y(final["val"])) / 2,
                    f"train {final['train']:.2f} · val {final['val']:.2f}",
                    fill=t["ink"], size=12))
    out.append(text(x1 + 12, (Y(final["train"]) + Y(final["val"])) / 2 + 16,
                    f"perplexity {math.exp(final['val']):.0f}", fill=t["muted"], size=11))

    lx = x0                                            # legend
    for key, color in (("train", t["series"][0]), ("val", t["series"][1])):
        out.append(f'<line x1="{lx}" y1="{y0 - 20}" x2="{lx + 16}" y2="{y0 - 20}" '
                   f'stroke="{color}" stroke-width="2" stroke-linecap="round"/>')
        out.append(text(lx + 22, y0 - 16, f"{key} loss", fill=t["ink2"], size=12))
        lx += 110
    out.append(text(28, y1 + 62, f"Perplexity {math.exp(final['val']):.0f}: out of 8,000 tokens, "
                    "the model has narrowed the next one down to about "
                    f"{math.exp(final['val']):.0f} likely candidates.",
                    fill=t["muted"], size=12))
    out.append(text(28, y1 + 80, f"Train and validation end {final['val'] - final['train']:.2f} "
                    "apart. At 20 tokens per parameter in a single pass, there is nothing to "
                    "memorise.", fill=t["muted"], size=12))
    return frame(W, y1 + 104, t, "".join(out))


# --- figure 17: what the trained model writes -------------------------------------
SAMPLES = OUT / "samples.json"


def rtl(x, y, s, *, fill, size=15, weight=400):
    """One line of Arabic, anchored at its right edge.

    direction="rtl" makes "start" the right side and keeps the quotes and
    commas on the correct end of the line.
    """
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family=\'{ARABIC_FONT}\' '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'direction="rtl" text-anchor="start">{esc(s)}</text>')


def fig_samples(mode):
    t = THEME[mode]
    run = json.loads(SAMPLES.read_text(encoding="utf-8"))
    panels = run["panels"]

    W, x0, x1 = 760, 28, 732
    top, pitch, box_h = 96, 104, 66

    out = [
        text(28, 38, "What it writes", fill=t["ink"], size=18, weight=600),
        text(28, 60, f"The finished model, step {run['step']:,}, top-k {run['top_k']}. "
                     "Temperature is the only thing that changes.",
             fill=t["ink2"], size=13),
        f'<line x1="28" y1="76" x2="{W - 28}" y2="76" stroke="{t["rule"]}" stroke-width="1"/>',
    ]
    for i, p in enumerate(panels):
        y = top + i * pitch
        head = f"temperature {p['temperature']}"
        if p["prompt"]:
            head += ", prompted"
        out.append(text(x0, y + 14, head, fill=t["ink"], size=13, weight=600))
        out.append(text(x1, y + 14, p["verdict"], fill=t["muted"], size=12, anchor="end"))
        out.append(f'<rect x="{x0}" y="{y + 24}" width="{x1 - x0}" height="{box_h}" rx="6" '
                   f'fill="none" stroke="{t["rule"]}" stroke-width="1"/>')
        for j, line in enumerate(p["lines"]):
            out.append(rtl(x1 - 16, y + 52 + j * 26, line, fill=t["ink"]))

    y = top + len(panels) * pitch
    out.append(text(28, y + 18, "The two unprompted samples both open with the formula the "
                    "Moroccan news source in the corpus starts its articles with.",
                    fill=t["muted"], size=12))
    out.append(text(28, y + 36, "In longer samples it also uses <|endoftext|> the way it was "
                    "trained to: end the document, start an unrelated new one.",
                    fill=t["muted"], size=12))
    return frame(W, y + 60, t, "".join(out))


def main():
    figures = {"corpus": fig_corpus, "params": fig_params, "tokens": fig_tokens,
               "attention": fig_attention, "mixing": fig_mixing, "block": fig_block,
               "model": fig_model, "checks": fig_checks, "batch": fig_batch,
               "loss": fig_loss, "overfit": fig_overfit, "lr": fig_lr, "clip": fig_clip,
               "amp": fig_amp, "ckpt": fig_ckpt, "train": fig_train,
               "samples": fig_samples}
    for name, fn in figures.items():
        for mode in ("light", "dark"):
            print(f"wrote {write(name, mode, fn(mode))}")


if __name__ == "__main__":
    main()
