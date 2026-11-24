"""General layer-graph networks (arch="net"), executed with PyTorch.

Spec
----
{"arch": "net",
 "input": {"modality": "image", "shape": [3, 32, 32]}       # or text: {"modality":"text","shape":[L],"vocab_size":V}
                                                            # or tabular: {"modality":"tabular","shape":[D]}
 "layers": [{"type": "conv2d", "name": "c1", "out_ch": 16, "k": 3, "act": "relu"},
            {"type": "maxpool2d", "k": 2},
            {"type": "global_pool", "op": "mean"},
            {"type": "dense", "name": "out", "out": 10, "act": "softmax"}]}

Tensors are named "<layer name>.<param>". Shapes are inferred from the input shape.

Layer types (shapes exclude the batch dimension)
  dense        {out, act}                 (..., D) -> (..., out)      W (D, out), b (out)      y = x @ W + b
  conv2d       {out_ch, k, stride, pad, act}  (C,H,W) -> (out_ch,H',W') W (out_ch, C, k, k), b (out_ch)
  maxpool2d / avgpool2d {k, stride}       (C,H,W) -> (C,H/k,W/k)
  flatten      {}                         (C,H,W) or (L,D) -> (C*H*W) or (L*D)
  global_pool  {op: mean|max|sum}         (C,H,W) -> (C);  (L,D) -> (D)  (padding tokens are masked)
  embedding    {dim}                      token ids (L) -> (L, dim)  E (vocab, dim); id 0 = <pad>, 1 = <unk>
  posembed     {}                         (L,D) -> (L,D)               P (L, D) added per position
  conv1d       {out_ch, k, act}           (L,D) -> (L,out_ch)          W (out_ch, D, k), b (out_ch)
  transformer  {heads, ff, act}           (L,D) -> (L,D)  pre-LN block: x += Attn(LN1 x); x += FF(LN2 x)
               tensors ln1.g ln1.b Wq bq Wk bk Wv bv Wo bo (D,D)/(D) ln2.g ln2.b W1 (D,ff) b1 W2 (ff,D) b2
  layernorm    {}                         (..., D) -> (..., D)         g (D), b (D)
  act          {act}                      elementwise activation
"""
from __future__ import annotations

import math

import numpy as np

from .netspec import ACTIVATIONS, NetworkError

NET_ACTIVATIONS = tuple(ACTIVATIONS) + ("gelu",)
LAYER_TYPES = ("dense", "conv2d", "maxpool2d", "avgpool2d", "flatten", "global_pool", "embedding", "posembed",
               "conv1d", "transformer", "layernorm", "act")
PARAM_LAYERS = ("dense", "conv2d", "embedding", "posembed", "conv1d", "transformer", "layernorm")


def _torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError("this network type needs PyTorch: pip install torch "
                          "(or pip install 'weight-space-programmer[all]')") from e


# ------------------------------------------------------------------------------------------------
# shape inference
# ------------------------------------------------------------------------------------------------
def assign_names(layers: list) -> list:
    seen = set()
    out = []
    for i, L in enumerate(layers):
        L = dict(L)
        if L.get("type") not in LAYER_TYPES:
            raise NetworkError(f"layer {i}: unknown type {L.get('type')!r}; use one of {LAYER_TYPES}")
        name = L.get("name") or f"{L['type']}{i}"
        if name in seen:
            raise NetworkError(f"duplicate layer name {name!r}")
        if "." in name:
            raise NetworkError(f"layer name {name!r} must not contain '.'")
        seen.add(name)
        L["name"] = name
        out.append(L)
    return out


def infer(spec: dict) -> list:
    """Walk the layers. Returns [{layer, in_shape, out_shape, params: {tensor: shape}}]."""
    inp = spec["input"]
    shape = tuple(inp["shape"])
    modality = inp.get("modality", "tabular")
    seq_ids = modality == "text"          # still token ids until an embedding layer
    rows = []
    for i, L in enumerate(spec["layers"]):
        t, n = L["type"], L["name"]
        params = {}
        act = L.get("act")
        if act is not None and act not in NET_ACTIVATIONS:
            raise NetworkError(f"layer {n}: unknown activation {act!r}; use one of {NET_ACTIVATIONS}")
        if seq_ids and t not in ("embedding",):
            raise NetworkError(f"layer {n}: text input is token ids; the first layer must be an embedding")
        if t == "dense":
            D = shape[-1]
            params = {f"{n}.W": (D, int(L["out"])), f"{n}.b": (int(L["out"]),)}
            if len(shape) == 3:
                raise NetworkError(f"layer {n}: dense needs a vector or sequence input; add flatten or global_pool")
            new = shape[:-1] + (int(L["out"]),)
        elif t == "conv2d":
            if len(shape) != 3:
                raise NetworkError(f"layer {n}: conv2d needs (C,H,W) input, got {shape}")
            C, H, W = shape
            k, s = int(L["k"]), int(L.get("stride", 1))
            p = int(L.get("pad", k // 2))
            Ho, Wo = (H + 2 * p - k) // s + 1, (W + 2 * p - k) // s + 1
            if Ho < 1 or Wo < 1:
                raise NetworkError(f"layer {n}: spatial size collapses to {Ho}x{Wo}")
            params = {f"{n}.W": (int(L["out_ch"]), C, k, k), f"{n}.b": (int(L["out_ch"]),)}
            new = (int(L["out_ch"]), Ho, Wo)
        elif t in ("maxpool2d", "avgpool2d"):
            if len(shape) != 3:
                raise NetworkError(f"layer {n}: {t} needs (C,H,W) input")
            k = int(L.get("k", 2)); s = int(L.get("stride", k))
            C, H, W = shape
            Ho, Wo = (H - k) // s + 1, (W - k) // s + 1
            if Ho < 1 or Wo < 1:
                raise NetworkError(f"layer {n}: pooling collapses the image ({H}x{W}, k={k})")
            new = (C, Ho, Wo)
        elif t == "flatten":
            new = (int(np.prod(shape)),)
        elif t == "global_pool":
            if L.get("op", "mean") not in ("mean", "max", "sum"):
                raise NetworkError(f"layer {n}: op must be mean|max|sum")
            if len(shape) == 3:
                new = (shape[0],)
            elif len(shape) == 2:
                new = (shape[1],)
            else:
                raise NetworkError(f"layer {n}: global_pool needs an image or sequence input")
        elif t == "embedding":
            if not seq_ids:
                raise NetworkError(f"layer {n}: embedding must come first, on token-id input")
            V = int(inp["vocab_size"])
            params = {f"{n}.E": (V, int(L["dim"]))}
            new = (shape[0], int(L["dim"]))
            seq_ids = False
        elif t == "posembed":
            if len(shape) != 2:
                raise NetworkError(f"layer {n}: posembed needs (L,D) input")
            params = {f"{n}.P": shape}
            new = shape
        elif t == "conv1d":
            if len(shape) != 2:
                raise NetworkError(f"layer {n}: conv1d needs (L,D) input")
            k = int(L["k"])
            params = {f"{n}.W": (int(L["out_ch"]), shape[1], k), f"{n}.b": (int(L["out_ch"]),)}
            new = (shape[0], int(L["out_ch"]))
        elif t == "transformer":
            if len(shape) != 2:
                raise NetworkError(f"layer {n}: transformer needs (L,D) input")
            D = shape[1]
            h = int(L.get("heads", 1))
            if D % h:
                raise NetworkError(f"layer {n}: model dim {D} not divisible by heads {h}")
            ff = int(L.get("ff", 4 * D))
            params = {f"{n}.ln1.g": (D,), f"{n}.ln1.b": (D,), f"{n}.ln2.g": (D,), f"{n}.ln2.b": (D,)}
            for m in ("Wq", "Wk", "Wv", "Wo"):
                params[f"{n}.{m}"] = (D, D)
                params[f"{n}.b{m[1]}"] = (D,)
            params.update({f"{n}.W1": (D, ff), f"{n}.b1": (ff,), f"{n}.W2": (ff, D), f"{n}.b2": (D,)})
            new = shape
        elif t == "layernorm":
            D = shape[-1]
            params = {f"{n}.g": (D,), f"{n}.b": (D,)}
            new = shape
        elif t == "act":
            if act is None:
                raise NetworkError(f"layer {n}: act layer needs 'act'")
            new = shape
        rows.append({"layer": L, "in_shape": shape, "out_shape": new, "params": params})
        shape = new
    if seq_ids:
        raise NetworkError("text network has no embedding layer")
    return rows


def output_shape(spec: dict) -> tuple:
    rows = infer(spec)
    return rows[-1]["out_shape"] if rows else tuple(spec["input"]["shape"])


def validate(spec: dict, W: dict) -> None:
    if not spec.get("layers"):
        raise NetworkError("net has no layers")
    if spec["layers"][-1]["type"] != "dense":
        raise NetworkError("the last layer must be dense (it produces the outputs)")
    for L in spec["layers"]:
        if L.get("act") == "softmax" and L is not spec["layers"][-1]:
            raise NetworkError("softmax is only allowed on the output layer")
    rows = infer(spec)
    if len(rows[-1]["out_shape"]) != 1:
        raise NetworkError(f"network output must be a vector, got shape {rows[-1]['out_shape']}; "
                           "add global_pool or flatten before the output layer")
    expected = {}
    for r in rows:
        expected.update(r["params"])
    for k, shp in expected.items():
        if k not in W:
            raise NetworkError(f"missing tensor {k}")
        if tuple(W[k].shape) != tuple(shp):
            raise NetworkError(f"{k} has shape {tuple(W[k].shape)}, expected {tuple(shp)}")
    extra = set(W) - set(expected)
    if extra:
        raise NetworkError(f"unexpected tensors: {sorted(extra)[:10]}")


def init_params(spec: dict, zero: bool = True, seed: int = 0) -> dict:
    """Zero init (LayerNorm gains = 1 so signals pass), or scaled-normal random init."""
    rng = np.random.default_rng(seed)
    W = {}
    for r in infer(spec):
        for k, shp in r["params"].items():
            last = k.rsplit(".", 1)[-1]
            if last == "g":
                W[k] = np.ones(shp)
            elif zero or last.startswith("b") and len(shp) == 1:
                W[k] = np.zeros(shp)
            elif last in ("E", "P"):
                W[k] = rng.normal(0, 0.1 if last == "E" else 0.02, shp)
            else:
                fan_in = shp[0] if len(shp) == 2 else int(np.prod(shp[1:]))
                gain = 2.0 if r["layer"].get("act") in ("relu", "leaky_relu", "gelu") else 1.0
                W[k] = rng.normal(0, math.sqrt(gain / max(fan_in, 1)), shp)
    return W


def describe(spec: dict) -> str:
    parts = []
    for L in spec["layers"]:
        t = L["type"]
        a = f" {L['act']}" if L.get("act") else ""
        if t == "dense":
            parts.append(f"{L['name']}:dense({L['out']}{a})")
        elif t == "conv2d":
            parts.append(f"{L['name']}:conv2d({L['out_ch']},k{L['k']}{a})")
        elif t == "conv1d":
            parts.append(f"{L['name']}:conv1d({L['out_ch']},k{L['k']}{a})")
        elif t in ("maxpool2d", "avgpool2d"):
            parts.append(f"{t}({L.get('k', 2)})")
        elif t == "global_pool":
            parts.append(f"gpool({L.get('op', 'mean')})")
        elif t == "embedding":
            parts.append(f"{L['name']}:embed({L['dim']})")
        elif t == "transformer":
            parts.append(f"{L['name']}:transformer(h{L.get('heads', 1)},ff{L.get('ff', '4D')})")
        else:
            parts.append(f"{L['name']}:{t}{a}")
    inp = spec["input"]
    return f"{inp.get('modality')}{list(inp['shape'])} -> " + " -> ".join(parts)


# ------------------------------------------------------------------------------------------------
# torch forward
# ------------------------------------------------------------------------------------------------
def t_act(name: str, x, torch):
    F = torch.nn.functional
    if name is None or name == "identity":
        return x
    if name == "relu":
        return torch.relu(x)
    if name == "leaky_relu":
        return F.leaky_relu(x, 0.01)
    if name == "tanh":
        return torch.tanh(x)
    if name == "sigmoid":
        return torch.sigmoid(x)
    if name == "step":
        hard = (x > 0).to(x.dtype)
        return hard + (x - x.detach()) * (x.abs() < 1).to(x.dtype)  # straight-through gradient
    if name == "softmax":
        return torch.softmax(x, dim=-1)
    if name == "abs":
        return x.abs()
    if name == "square":
        return x * x
    if name == "sin":
        return torch.sin(x)
    if name == "gelu":
        return F.gelu(x)
    raise NetworkError(f"unknown activation {name}")


def prepare_input(spec: dict, X, torch, device="cpu"):
    """numpy batch -> (torch tensor, mask). Applies input normalization."""
    inp = spec["input"]
    mod = inp.get("modality", "tabular")
    if mod == "text":
        ids = torch.as_tensor(np.asarray(X), dtype=torch.long, device=device)
        return ids, ids != 0
    x = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    norm = spec.get("input_norm")
    if mod == "image":
        x = x / 255.0 if inp.get("dtype", "uint8") == "uint8" else x
        if norm:
            m = torch.tensor(norm["mean"], dtype=torch.float32, device=device).view(1, -1, 1, 1)
            s = torch.tensor(norm["std"], dtype=torch.float32, device=device).view(1, -1, 1, 1)
            x = (x - m) / s
    elif norm:
        x = (x - torch.tensor(norm["mean"], dtype=torch.float32, device=device)) / \
            torch.tensor(norm["std"], dtype=torch.float32, device=device)
    return x, None


def forward_torch(spec: dict, P: dict, x, mask=None, capture: bool = False, logits: bool = False):
    """P: name -> torch tensor. Returns (output, [(layer name, activation tensor)])."""
    torch = _torch()
    F = torch.nn.functional
    caps = []
    layers = spec["layers"]
    for i, L in enumerate(layers):
        t, n = L["type"], L["name"]
        last = i == len(layers) - 1
        act = None if (last and logits) else L.get("act")
        if t == "dense":
            x = t_act(act, x @ P[f"{n}.W"] + P[f"{n}.b"], torch)
            if mask is not None and x.dim() == 3:
                x = x * mask.unsqueeze(-1)
        elif t == "conv2d":
            k = int(L["k"])
            x = t_act(act, F.conv2d(x, P[f"{n}.W"], P[f"{n}.b"], stride=int(L.get("stride", 1)),
                                    padding=int(L.get("pad", k // 2))), torch)
        elif t == "maxpool2d":
            k = int(L.get("k", 2)); x = F.max_pool2d(x, k, int(L.get("stride", k)))
        elif t == "avgpool2d":
            k = int(L.get("k", 2)); x = F.avg_pool2d(x, k, int(L.get("stride", k)))
        elif t == "flatten":
            x = x.flatten(1); mask = None
        elif t == "global_pool":
            op = L.get("op", "mean")
            if x.dim() == 4:
                x = {"mean": lambda v: v.mean(dim=(2, 3)), "max": lambda v: v.amax(dim=(2, 3)),
                     "sum": lambda v: v.sum(dim=(2, 3))}[op](x)
            else:
                if mask is None:
                    mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
                m = mask.unsqueeze(-1).to(x.dtype)
                if op == "max":
                    x = x.masked_fill(~mask.unsqueeze(-1), -1e9).amax(dim=1)
                    x = torch.where(mask.any(dim=1, keepdim=True), x, torch.zeros_like(x))
                elif op == "sum":
                    x = (x * m).sum(dim=1)
                else:
                    x = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            mask = None
        elif t == "embedding":
            x = F.embedding(x, P[f"{n}.E"])
        elif t == "posembed":
            x = x + P[f"{n}.P"][: x.shape[1]]
        elif t == "conv1d":
            k = int(L["k"])
            y = F.conv1d(x.transpose(1, 2), P[f"{n}.W"], P[f"{n}.b"], padding=k // 2)
            x = t_act(act, y[:, :, : x.shape[1]].transpose(1, 2), torch)
            if mask is not None:
                x = x * mask.unsqueeze(-1)
        elif t == "transformer":
            x = _block(L, P, x, mask, torch)
        elif t == "layernorm":
            x = F.layer_norm(x, (x.shape[-1],), P[f"{n}.g"], P[f"{n}.b"])
        elif t == "act":
            x = t_act(act, x, torch)
        if capture:
            caps.append((n, x, mask))
    return x, caps


def _block(L, P, x, mask, torch):
    F = torch.nn.functional
    n = L["name"]
    N, T, D = x.shape
    H = int(L.get("heads", 1))
    dh = D // H
    h = F.layer_norm(x, (D,), P[f"{n}.ln1.g"], P[f"{n}.ln1.b"])
    q = (h @ P[f"{n}.Wq"] + P[f"{n}.bq"]).view(N, T, H, dh).transpose(1, 2)
    k = (h @ P[f"{n}.Wk"] + P[f"{n}.bk"]).view(N, T, H, dh).transpose(1, 2)
    v = (h @ P[f"{n}.Wv"] + P[f"{n}.bv"]).view(N, T, H, dh).transpose(1, 2)
    att = q @ k.transpose(-1, -2) / math.sqrt(dh)
    if mask is not None:
        att = att.masked_fill(~mask[:, None, None, :], -1e9)
    att = torch.softmax(att, dim=-1)
    o = (att @ v).transpose(1, 2).reshape(N, T, D)
    x = x + o @ P[f"{n}.Wo"] + P[f"{n}.bo"]
    h = F.layer_norm(x, (D,), P[f"{n}.ln2.g"], P[f"{n}.ln2.b"])
    x = x + t_act(L.get("act", "gelu"), h @ P[f"{n}.W1"] + P[f"{n}.b1"], torch) @ P[f"{n}.W2"] + P[f"{n}.b2"]
    if mask is not None:
        x = x * mask.unsqueeze(-1)
    return x


# ------------------------------------------------------------------------------------------------
# presets (starting points the agent can pick, modify, or replace)
# ------------------------------------------------------------------------------------------------
def preset_layers(name: str, input_spec: dict, out_dim: int, out_act: str, width: int = 16, depth: int = 2,
                  dim: int = 32, heads: int = 4, blocks: int = 2, hidden: int = 0) -> list:
    mod = input_spec.get("modality", "tabular")
    shape = input_spec["shape"]
    head = ([{"type": "dense", "name": "hid", "out": hidden, "act": "relu"}] if hidden else []) + \
           [{"type": "dense", "name": "out", "out": out_dim, "act": out_act}]
    if mod == "image":
        if name == "linear":
            return [{"type": "flatten", "name": "flat"}] + head
        if name == "cnn":
            layers, side = [], min(shape[1], shape[2])
            for i in range(depth):
                layers.append({"type": "conv2d", "name": f"c{i+1}", "out_ch": width * 2 ** i, "k": 3, "act": "relu"})
                if side >= 4:
                    layers.append({"type": "maxpool2d", "name": f"pool{i+1}", "k": 2})
                    side //= 2
            return layers + [{"type": "global_pool", "name": "gp", "op": "mean"}] + head
    if mod == "text":
        if name == "bow":
            return [{"type": "embedding", "name": "emb", "dim": dim},
                    {"type": "global_pool", "name": "gp", "op": "mean"}] + head
        if name == "cnn1d":
            return [{"type": "embedding", "name": "emb", "dim": dim},
                    {"type": "conv1d", "name": "conv", "out_ch": width, "k": 3, "act": "relu"},
                    {"type": "global_pool", "name": "gp", "op": "max"}] + head
        if name == "transformer":
            return ([{"type": "embedding", "name": "emb", "dim": dim}, {"type": "posembed", "name": "pos"}]
                    + [{"type": "transformer", "name": f"blk{i+1}", "heads": heads, "ff": 4 * dim, "act": "gelu"}
                       for i in range(blocks)]
                    + [{"type": "layernorm", "name": "lnf"}, {"type": "global_pool", "name": "gp", "op": "mean"}]
                    + head)
    if mod == "tabular" and name == "mlp":
        layers = [{"type": "dense", "name": f"h{i+1}", "out": width, "act": "relu"} for i in range(depth)]
        return layers + [{"type": "dense", "name": "out", "out": out_dim, "act": out_act}]
    raise NetworkError(f"unknown preset {name!r} for {mod} input; image: linear|cnn, text: bow|cnn1d|transformer, "
                       "tabular: mlp")


PRESETS = {"image": ("linear", "cnn"), "text": ("bow", "cnn1d", "transformer"), "tabular": ("mlp",)}
