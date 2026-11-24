"""Serialize weights to text for the agent to read, chunked so each file fits one Read call."""
from __future__ import annotations

import numpy as np

from . import meter
from .netspec import Network

CHUNK_TOKENS = 20_000


def fmt(v: float, precision: int) -> str:
    s = f"{v:.{precision}f}"
    if float(s) == 0.0:
        return "0"
    return s


def parse_slice(s: str | None, n: int) -> range:
    if s is None or s == "":
        return range(n)
    if ":" in s:
        a, b = s.split(":", 1)
        return range(int(a) if a else 0, min(int(b), n) if b else n)
    i = int(s)
    return range(i, i + 1)


def row_labels(net: Network, name: str, vocab: list | None = None):
    """Human labels for the rows of a tensor."""
    if net.spec.get("arch") == "net":
        a = net.W.get(name)
        if name.endswith(".E") and vocab:
            return [repr(t) for t in vocab[: a.shape[0]]]
        if a is not None and a.ndim > 2:
            return [f"filter{j}" for j in range(a.shape[0])]
        first = next((L for L in net.layers if L["type"] == "dense"), None)
        if (first is not None and name == f"{first['name']}.W" and net.spec["input"].get("modality") == "tabular"
                and net.layers[0] is first):
            return [f"in:{n}" for n in net.spec.get("feature_names") or []]
        return None
    if not net.is_mlp or not name.startswith("W"):
        return None
    i = int(name[1:])
    if i == 0:
        fn = net.spec.get("feature_names") or []
        return [f"in:{n}" for n in fn]
    return [f"h{i-1}.{j}" for j in range(net.layers[i]["in"])]


def format_tensor(net: Network, name: str, precision: int = 3, rows: str | None = None,
                  cols: str | None = None, vocab: list | None = None) -> str:
    a = net.W[name]
    shape = a.shape
    lines = []
    if a.ndim == 1:
        cr = parse_slice(cols or rows, a.shape[0])
        lines.append(f"## {name} shape={shape} (bias per unit; showing [{cr.start}:{cr.stop}])")
        vals = [fmt(a[j], precision) for j in cr]
        for k in range(0, len(vals), 16):
            lines.append(f"[{cr.start + k}] " + " ".join(vals[k:k + 16]))
        return "\n".join(lines)
    if a.ndim > 2:
        a = a.reshape(a.shape[0], -1)
        lines.append(f"## {name} shape={shape} (one row per output filter, flattened in C-order to {a.shape})")
    else:
        if net.is_mlp:
            lines.append(f"## {name} shape={shape} (row r = input r, column c = unit c of this layer)")
        else:
            lines.append(f"## {name} shape={shape}")
    rr = parse_slice(rows, a.shape[0])
    cr = parse_slice(cols, a.shape[1])
    if cr.start != 0 or cr.stop != a.shape[1] or rr.start != 0 or rr.stop != a.shape[0]:
        lines.append(f"(rows [{rr.start}:{rr.stop}], columns [{cr.start}:{cr.stop}])")
    labels = row_labels(net, name, vocab)
    for r in rr:
        lab = f" {labels[r]}" if labels and r < len(labels) else ""
        lines.append(f"[{r}{lab}] " + " ".join(fmt(a[r, c], precision) for c in cr))
    return "\n".join(lines)


def header(net: Network) -> str:
    h = [f"# network {net.arch_str()}  params={net.n_params()}"]
    if net.is_mlp:
        h.append("# convention: h_{i+1} = act_i(h_i @ W{i} + b{i}); W{i}[r, c] = weight from input r to unit c")
        if net.spec.get("input_norm"):
            h.append("# inputs are standardized first: x' = (x - mean) / std  (see `wsp status` / run.json input_norm)")
        if net.spec.get("output_norm"):
            h.append("# regression output is de-standardized: y = out * out_std + out_mean")
        h.append("# layers: " + "; ".join(f"L{i}: {L['in']}->{L['out']} {L['act']}" for i, L in enumerate(net.layers)))
    elif net.spec.get("arch") == "net":
        from .graph import infer
        h.append("# convention: dense y = x @ W + b (W[r, c]: input r -> unit c); conv W[out, in, kh, kw] "
                 "(PyTorch layout); embedding E[token, dim]")
        for r in infer(net.spec):
            L = r["layer"]
            h.append(f"#   {L['name']:<10} {L['type']:<12} {str(r['in_shape']):<16} -> {str(r['out_shape']):<16} "
                     + (L.get("act") or ""))
    return "\n".join(h)


def estimate_full_dump_tokens(net: Network, precision: int = 3) -> int:
    # ~ (precision + 3) chars per number incl. sign/space, plus row labels.
    chars = sum(v.size * (precision + 3.2) + (v.shape[0] * 14 if v.ndim > 1 else 0) for v in net.W.values())
    return int(chars / 2.5) + 200


def build_dump(net: Network, tensors: list, precision: int, rows=None, cols=None, vocab=None) -> str:
    parts = [header(net)]
    for t in tensors:
        parts.append(format_tensor(net, t, precision, rows, cols, vocab))
    return "\n\n".join(parts) + "\n"


def chunk_text(text: str, chunk_tokens: int = CHUNK_TOKENS) -> list:
    max_chars = int(chunk_tokens * 2.5)
    chunks, cur, size = [], [], 0
    for line in text.splitlines(keepends=True):
        if size + len(line) > max_chars and cur:
            chunks.append("".join(cur)); cur, size = [], 0
        cur.append(line); size += len(line)
    if cur:
        chunks.append("".join(cur))
    return chunks


def tensor_order(net: Network) -> list:
    if net.spec.get("arch") == "net":
        from .graph import infer
        return [k for r in infer(net.spec) for k in r["params"]]
    if net.is_mlp:
        out = []
        for i in range(len(net.layers)):
            out += [f"W{i}", f"b{i}"]
        return out
    return list(net.W.keys())


def select_tensors(net: Network, layer: int | None, tensors: list | None) -> list:
    if tensors:
        bad = [t for t in tensors if t not in net.W]
        if bad:
            raise KeyError(f"unknown tensor(s) {bad}; available: {list(net.W)}")
        return tensors
    if layer is not None:
        if net.spec.get("arch") == "net":
            names = [k for k in tensor_order(net) if k.startswith(f"{layer}.")]
            if not names:
                raise KeyError(f"layer {layer!r} has no tensors; layers: {[L['name'] for L in net.layers]}")
            return names
        return [f"W{layer}", f"b{layer}"]
    return tensor_order(net)


def check_fits(session: dict, tokens: int) -> bool:
    return tokens <= meter.context_remaining(session)
