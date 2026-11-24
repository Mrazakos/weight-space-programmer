"""Image and text data: loading, encoding, profiling, and rendering examples for the agent.

Images are stored as uint8 arrays (N, C, H, W); the network divides by 255 and applies per-channel
normalization. Text is stored as int32 token ids (N, L) with 0 = <pad>, 1 = <unk>; the vocabulary
is saved in the run directory (vocab.json).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
PAD, UNK = "<pad>", "<unk>"


# ------------------------------------------------------------------------------------------------
# images
# ------------------------------------------------------------------------------------------------
def _pil():
    try:
        from PIL import Image
        return Image
    except ImportError as e:
        raise ImportError("image data needs Pillow: pip install pillow") from e


def scan_image_folder(root: str):
    """root/<class>/<image>. Returns (paths, labels)."""
    root = Path(root)
    paths, labels = [], []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(d.rglob("*")):
            if f.suffix.lower() in IMG_EXT:
                paths.append(str(f.resolve()))
                labels.append(d.name)
    if not paths:
        raise ValueError(f"no images found under {root}/<class>/ (extensions {sorted(IMG_EXT)})")
    return paths, labels


def choose_image_format(paths: list, size: int | None, grayscale: bool | None):
    Image = _pil()
    sample = paths[:: max(1, len(paths) // 50)][:50]
    dims, modes = [], set()
    for p in sample:
        with Image.open(p) as im:
            dims.append(im.size)
            modes.add(im.mode)
    if grayscale is None:
        grayscale = modes <= {"L", "1", "I", "I;16", "LA"}
    if size is None:
        if len(set(dims)) == 1 and max(dims[0]) <= 64:
            w, h = dims[0]
            return (h, w), grayscale
        med = int(np.median([max(d) for d in dims]))
        size = 32 if med <= 48 else 64
    return (size, size), grayscale


def read_images(paths: list, hw: tuple, grayscale: bool) -> np.ndarray:
    Image = _pil()
    C = 1 if grayscale else 3
    out = np.zeros((len(paths), C, hw[0], hw[1]), dtype=np.uint8)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            im = im.convert("L" if grayscale else "RGB")
            if im.size != (hw[1], hw[0]):
                im = im.resize((hw[1], hw[0]), Image.BILINEAR)
            a = np.asarray(im, dtype=np.uint8)
        out[i] = a[None] if grayscale else a.transpose(2, 0, 1)
    return out


def array_to_images(X: np.ndarray) -> np.ndarray:
    """Accept (N,H,W), (N,H,W,C) or (N,C,H,W), float [0,1] or uint8 -> uint8 (N,C,H,W)."""
    X = np.asarray(X)
    if X.ndim == 3:
        X = X[:, None]
    elif X.ndim == 4 and X.shape[-1] in (1, 3, 4) and X.shape[1] not in (1, 3, 4):
        X = X.transpose(0, 3, 1, 2)
    if X.ndim != 4:
        raise ValueError(f"image array must be (N,H,W), (N,H,W,C) or (N,C,H,W); got {X.shape}")
    if X.shape[1] == 4:
        X = X[:, :3]
    if X.dtype != np.uint8:
        X = X.astype(float)
        if X.max() <= 1.0 + 1e-6:
            X = X * 255.0
        X = np.clip(np.round(X), 0, 255).astype(np.uint8)
    return X


def image_norm_stats(X: np.ndarray) -> dict:
    x = X.astype(np.float64) / 255.0
    mean = x.mean(axis=(0, 2, 3))
    std = x.std(axis=(0, 2, 3))
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean.round(6).tolist(), "std": std.round(6).tolist()}


def image_profile(X: np.ndarray, y: np.ndarray, meta: dict) -> str:
    N, C, H, W = X.shape
    lines = [f"samples={N}  image shape (C,H,W)=({C},{H},{W})  task={meta['task']}"]
    if meta["task"] == "classification":
        counts = np.bincount(y, minlength=len(meta["classes"]))
        lines.append("classes: " + ", ".join(f"{i}={c} (n={counts[i]})" for i, c in enumerate(meta["classes"])))
    x = X.astype(np.float64) / 255.0
    lines.append("per-channel mean: " + ", ".join(f"{m:.3f}" for m in x.mean(axis=(0, 2, 3)))
                 + "   std: " + ", ".join(f"{s:.3f}" for s in x.std(axis=(0, 2, 3))))
    if meta["task"] == "classification":
        lines.append("per-class mean intensity (by channel) and spatial centre of mass (row, col):")
        rr, cc = np.mgrid[0:H, 0:W]
        for k, c in enumerate(meta["classes"][:20]):
            xs = x[y == k]
            if not len(xs):
                continue
            m = xs.mean(axis=0)  # (C,H,W)
            g = m.mean(axis=0)
            tot = g.sum() + 1e-12
            lines.append(f"  {c[:20]:<20} intensity " + " ".join(f"{v:.3f}" for v in m.mean(axis=(1, 2)))
                         + f"  com=({(g * rr).sum() / tot:.1f},{(g * cc).sum() / tot:.1f})")
    if meta["task"] == "classification" and C * H * W <= 256 and len(meta["classes"]) <= 12:
        lines.append("per-class mean images (channel-averaged, 0-9 scale = intensity 0..1):")
        for k, c in enumerate(meta["classes"]):
            xs = x[y == k]
            if not len(xs):
                continue
            g = xs.mean(axis=(0, 1))
            lines.append(f"  class {c}:")
            for row in g:
                lines.append("    " + "".join(str(int(min(9, v * 10))) for v in row))
    lines.append("view examples: `wsp sample <run>` writes a labelled contact sheet; class-mean images are in "
                 "dumps/class_means.png (Read them; they are images)")
    return "\n".join(lines)


def save_contact_sheet(X: np.ndarray, labels: list, path: Path, cols: int = 8, scale: int | None = None) -> str:
    Image = _pil()
    from PIL import ImageDraw
    N, C, H, W = X.shape
    scale = scale or max(1, 96 // max(H, W))
    cw, ch = W * scale, H * scale + 14
    rows = int(np.ceil(N / cols))
    sheet = Image.new("RGB", (cols * cw, max(rows, 1) * ch), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i in range(N):
        a = X[i]
        im = Image.fromarray(a[0] if C == 1 else a.transpose(1, 2, 0), "L" if C == 1 else "RGB").convert("RGB")
        im = im.resize((cw, H * scale), Image.NEAREST)
        r, c = divmod(i, cols)
        sheet.paste(im, (c * cw, r * ch))
        draw.text((c * cw + 2, r * ch + H * scale + 1), str(labels[i])[:16], fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return str(path)


def save_class_means(X, y, classes, path: Path) -> str:
    means = []
    for k in range(len(classes)):
        xs = X[y == k]
        means.append(xs.astype(np.float64).mean(axis=0) if len(xs) else np.zeros(X.shape[1:]))
    M = np.clip(np.round(np.stack(means)), 0, 255).astype(np.uint8)
    return save_contact_sheet(M, [f"mean {c}" for c in classes], path)


# ------------------------------------------------------------------------------------------------
# text
# ------------------------------------------------------------------------------------------------
WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")


def tokenize(s: str, level: str) -> list:
    s = str(s)
    if level == "char":
        return list(s)
    return WORD_RE.findall(s.lower())


def build_vocab(texts: list, level: str, max_vocab: int = 20000, min_freq: int = 2) -> list:
    cnt = Counter()
    for t in texts:
        cnt.update(tokenize(t, level))
    if level == "char":
        min_freq = 1
    toks = [w for w, c in cnt.most_common() if c >= min_freq][: max_vocab - 2]
    return [PAD, UNK] + toks


def encode_texts(texts: list, vocab: list, level: str, max_len: int) -> np.ndarray:
    idx = {w: i for i, w in enumerate(vocab)}
    out = np.zeros((len(texts), max_len), dtype=np.int32)
    for i, t in enumerate(texts):
        ids = [idx.get(w, 1) for w in tokenize(t, level)][:max_len]
        out[i, : len(ids)] = ids
    return out


def choose_max_len(texts: list, level: str) -> int:
    lens = [len(tokenize(t, level)) for t in texts[:5000]]
    cap = 512 if level == "char" else 256
    return int(max(8, min(cap, np.percentile(lens, 95) if lens else 8)))


def detect_text_column(df, target: list) -> str | None:
    cands = []
    for c in df.columns:
        if c in target:
            continue
        s = df[c]
        import pandas as pd
        if not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            st = s.dropna().astype(str)
            if len(st) and st.str.len().mean() >= 20 and st.str.split().str.len().mean() >= 3:
                cands.append((st.str.len().mean(), c))
    return max(cands)[1] if cands else None


def text_profile(texts: list, X: np.ndarray, y: np.ndarray, vocab: list, meta: dict, top: int = 15) -> str:
    lens = (X != 0).sum(axis=1)
    lines = [f"samples={len(X)}  task={meta['task']}  tokenization={meta.get('text_level')}  vocab={len(vocab)}  "
             f"max_len={X.shape[1]} (95th pct; longer texts truncated)",
             f"tokens per text: mean={lens.mean():.1f} median={np.median(lens):.0f} max={lens.max()}  "
             f"<unk> rate={(X == 1).sum() / max(lens.sum(), 1):.3f}"]
    V = len(vocab)
    # document frequency per class (descriptive, no fitting)
    present = np.zeros((len(X), V), dtype=bool) if V * len(X) <= 2e8 else None
    if present is None:
        lines.append("(corpus too large for the per-token table)")
        return "\n".join(lines)
    rows = np.repeat(np.arange(len(X)), X.shape[1])
    present[rows, X.reshape(-1)] = True
    present[:, :2] = False
    if meta["task"] == "classification":
        classes = meta["classes"]
        counts = np.bincount(y, minlength=len(classes))
        lines.append("classes: " + ", ".join(f"{i}={c} (n={counts[i]})" for i, c in enumerate(classes)))
        df_all = present.sum(axis=0) + 1.0
        for k, c in enumerate(classes[:12]):
            inside = present[y == k].sum(axis=0) + 0.5
            outside = present[y != k].sum(axis=0) + 0.5
            nk, no = max((y == k).sum(), 1), max((y != k).sum(), 1)
            lo = np.log(inside / nk) - np.log(outside / no)
            lo[df_all < 4] = -np.inf
            best = np.argsort(-lo)[:top]
            lines.append(f"  tokens most indicative of {c!r} (log-odds, doc count): "
                         + ", ".join(f"{vocab[j]!r} {lo[j]:+.2f} ({int(inside[j] - .5)})" for j in best if np.isfinite(lo[j])))
    else:
        yy = y.reshape(len(y), -1)[:, 0]
        dfc = present.sum(axis=0)
        ok = dfc >= 5
        mean_with = np.where(ok, (present * yy[:, None]).sum(axis=0) / np.maximum(dfc, 1), np.nan)
        diff = mean_with - yy.mean()
        order = np.argsort(-np.nan_to_num(diff, nan=-np.inf))
        lines.append(f"target mean={yy.mean():.4g} std={yy.std():.4g}")
        lines.append("  tokens with highest mean target: " + ", ".join(f"{vocab[j]!r} {diff[j]:+.3g}" for j in order[:top]))
        order = np.argsort(np.nan_to_num(diff, nan=np.inf))
        lines.append("  tokens with lowest mean target: " + ", ".join(f"{vocab[j]!r} {diff[j]:+.3g}" for j in order[:top]))
    lines.append("most frequent tokens: " + ", ".join(repr(w) for w in vocab[2:32]))
    return "\n".join(lines)


def save_json(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj), encoding="utf-8")


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
