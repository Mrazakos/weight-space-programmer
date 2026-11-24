"""Execution backends behind one interface.

  arch "mlp"           numpy (wsp/forward.py, wsp/train.py)
  arch "net"           PyTorch layer graph (wsp/graph.py)
  arch "torch_module"  an imported PyTorch module (TorchScript or pickled nn.Module) whose state_dict
                       tensors are the editable weights
  arch "opaque"        tensors only: inspect/edit, no forward pass
"""
from __future__ import annotations

import time

import numpy as np

from .evaluate import compute_metrics, primary_metric
from .netspec import Network, NetworkError

RUNNABLE = ("mlp", "net", "torch_module")


def output_act(spec: dict) -> str:
    if spec.get("output_act"):
        return spec["output_act"]
    layers = spec.get("layers") or []
    return layers[-1].get("act", "identity") if layers else "identity"


def is_runnable(net: Network) -> bool:
    return net.spec.get("arch") in RUNNABLE


def _torch():
    from .graph import _torch as t
    return t()


def device():
    torch = _torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def reduce_units(a, mask=None):
    """Collapse spatial / sequence positions so each channel or unit becomes one number per sample."""
    if a.dim() == 4:
        return a.mean(dim=(2, 3))
    if a.dim() == 3:
        if mask is None:
            return a.mean(dim=1)
        m = mask.unsqueeze(-1).to(a.dtype)
        return (a * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
    return a


# ------------------------------------------------------------------------------------------------
# imported torch modules
# ------------------------------------------------------------------------------------------------
def load_module(spec: dict):
    import warnings
    torch = _torch()
    src = spec["source"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        if src.get("loader") == "jit":
            return torch.jit.load(src["path"], map_location="cpu")
        if src.get("loader") == "export":
            return torch.export.load(src["path"]).module()
        return torch.load(src["path"], map_location="cpu", weights_only=False)


def module_with_weights(net: Network, dev: str, train: bool = False):
    torch = _torch()
    m = load_module(net.spec)
    sd = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in net.W.items()}
    missing = m.load_state_dict(sd, strict=False)
    m.to(dev)
    if net.spec["source"].get("loader") != "export":
        m.train(train)
    return m


def _module_out(o):
    if isinstance(o, (tuple, list)):
        o = o[0]
    if isinstance(o, dict):
        o = o.get("logits", next(iter(o.values())))
    return o


# ------------------------------------------------------------------------------------------------
# forward
# ------------------------------------------------------------------------------------------------
def forward(net: Network, X, capture: bool = False, batch: int = 256):
    """Returns (output ndarray, [(layer name, per-unit activations ndarray (N, units))])."""
    arch = net.spec.get("arch")
    if arch == "mlp":
        from .forward import forward as np_forward
        out, (pre, post) = np_forward(net, X, capture)
        caps = [(f"L{i}", post[i]) for i in range(len(post) - 1)] if capture else []
        return out, caps
    if arch == "net":
        from .graph import forward_torch, prepare_input
        torch = _torch()
        dev = device()
        P = {k: torch.as_tensor(v, dtype=torch.float32, device=dev) for k, v in net.W.items()}
        outs, caps_acc, names = [], {}, []
        with torch.no_grad():
            for s in range(0, len(X), batch):
                x, mask = prepare_input(net.spec, X[s:s + batch], torch, dev)
                o, caps = forward_torch(net.spec, P, x, mask, capture=capture)
                outs.append(o.float().cpu().numpy())
                for n, a, m in caps[:-1]:
                    if n not in caps_acc:
                        caps_acc[n] = []
                        names.append(n)
                    caps_acc[n].append(reduce_units(a, m).float().cpu().numpy())
        out = np.concatenate(outs) if outs else np.zeros((0, 1))
        return out, [(n, np.concatenate(caps_acc[n])) for n in names]
    if arch == "torch_module":
        from .graph import prepare_input
        torch = _torch()
        dev = device()
        m = module_with_weights(net, dev)
        outs = []
        with torch.no_grad():
            for s in range(0, len(X), batch):
                x, _ = prepare_input(net.spec, X[s:s + batch], torch, dev)
                try:
                    o = _module_out(m(x))
                except Exception as e:
                    raise NetworkError(f"the imported module failed on input of shape {tuple(x.shape)}: "
                                       f"{type(e).__name__}: {str(e)[:300]}. If it was exported with a fixed batch "
                                       "size, re-export it with a dynamic batch dimension.") from e
                outs.append(o.float().cpu().numpy().reshape(len(x), -1))
        return np.concatenate(outs), []
    raise NetworkError("this network is opaque: wsp cannot run it (inspect/slice/svd/patch/write still work). "
                       "Import it as a TorchScript or full torch module (--trust) to make it runnable.")


def to_prediction(spec: dict, out: np.ndarray):
    if spec["task"] == "regression":
        on = spec.get("output_norm")
        y = out * np.asarray(on["std"]) + np.asarray(on["mean"]) if on else out
        return y, None
    act = output_act(spec)
    if out.shape[1] == 1:
        thr = 0.5 if act in ("sigmoid", "step") else 0.0
        pred = (out[:, 0] > thr).astype(int)
        prob = np.clip(out[:, 0], 0, 1) if act == "sigmoid" else None
        return pred, prob
    pred = out.argmax(axis=1)
    return pred, (out if act == "softmax" else None)


def predict(net: Network, X):
    out, _ = forward(net, X)
    return to_prediction(net.spec, out)


# ------------------------------------------------------------------------------------------------
# training (torch backends; MLPs use the numpy trainer)
# ------------------------------------------------------------------------------------------------
def fit(net: Network, Xtr, ytr, Xva, yva, epochs=30, lr=1e-3, batch=64, weight_decay=0.0, freeze=None,
        patience=5, seed=0, max_minutes=10.0):
    arch = net.spec.get("arch")
    if arch == "mlp":
        from .train import fit as np_fit
        fr = [int(f) for f in (freeze or [])]
        return np_fit(net, Xtr, ytr, Xva, yva, epochs=epochs, lr=lr, batch=batch, weight_decay=weight_decay,
                      freeze=fr, patience=patience, seed=seed)
    if arch not in ("net", "torch_module"):
        raise NetworkError("cannot train an opaque network")
    torch = _torch()
    from .graph import forward_torch, prepare_input, t_act
    F = torch.nn.functional
    torch.manual_seed(seed)
    dev = device()
    spec = net.spec
    freeze = [str(f) for f in (freeze or [])]

    def frozen(name):
        return any(name == f or name.startswith(f + ".") for f in freeze)

    if arch == "net":
        P = {k: torch.tensor(v, dtype=torch.float32, device=dev, requires_grad=not frozen(k)) for k, v in net.W.items()}
        trainable = [p for k, p in P.items() if not frozen(k)]

        def run(xb, logits=True):
            x, mask = prepare_input(spec, xb, torch, dev)
            return forward_torch(spec, P, x, mask, logits=logits)[0]

        def snapshot():
            return {k: v.detach().cpu().numpy().astype(np.float64) for k, v in P.items()}

        def set_mode(train):
            pass
        final_is_logits = True
    else:
        m = module_with_weights(net, dev, train=True)
        for n, p in m.named_parameters():
            p.requires_grad_(not frozen(n))
        trainable = [p for p in m.parameters() if p.requires_grad]

        def run(xb, logits=True):
            x, _ = prepare_input(spec, xb, torch, dev)
            return _module_out(m(x)).reshape(len(xb), -1)

        def snapshot():
            return {k: v.detach().cpu().numpy().astype(np.float64) for k, v in m.state_dict().items()
                    if k in net.W}

        def set_mode(train):
            if net.spec["source"].get("loader") != "export":
                m.train(train)
        final_is_logits = False
    if not trainable:
        raise NetworkError("everything is frozen")

    task = spec["task"]
    act = output_act(spec)
    if task == "regression":
        T = np.asarray(ytr, float).reshape(len(ytr), -1)
        on = spec.get("output_norm")
        if on:
            T = (T - np.asarray(on["mean"])) / np.asarray(on["std"])
    else:
        T = np.asarray(ytr).astype(int)

    def loss_fn(o, t):
        if task == "classification":
            k = o.shape[1]
            if final_is_logits or act == "identity":
                if k == 1:
                    return F.binary_cross_entropy_with_logits(o[:, 0], t.float())
                return F.cross_entropy(o, t)
            if act == "softmax":
                return F.nll_loss(torch.log(o.clamp_min(1e-12)), t)
            if act == "sigmoid":
                return F.binary_cross_entropy(o[:, 0].clamp(1e-7, 1 - 1e-7), t.float())
            return F.mse_loss(o[:, 0], t.float())
        out = t_act(act, o, torch) if final_is_logits else o
        return F.mse_loss(out, t)

    opt = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    key = primary_metric(task)
    k_cls = len(spec.get("classes") or []) or None
    best = (-np.inf, snapshot(), 0)
    curve = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        set_mode(True)
        perm = rng.permutation(len(Xtr))
        losses = []
        for s in range(0, len(perm), batch):
            idx = perm[s:s + batch]
            tt = torch.as_tensor(T[idx], device=dev)
            if task == "regression":
                tt = tt.float()
            loss = loss_fn(run(Xtr[idx]), tt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            if time.time() - t0 > max_minutes * 60:
                break
        set_mode(False)
        with torch.no_grad():
            outs = []
            for s in range(0, len(Xva), 512):
                o = run(Xva[s:s + 512], logits=False)
                outs.append(o.float().cpu().numpy())
        out = np.concatenate(outs)
        pred, prob = to_prediction(spec, out)
        val = compute_metrics(task, yva, pred, prob, k_cls)[key]
        curve.append((ep, float(np.mean(losses)), float(val)))
        if val > best[0] + 1e-9:
            best = (val, snapshot(), ep)
        elif ep - best[2] >= patience:
            break
        if time.time() - t0 > max_minutes * 60:
            break
    new = Network(dict(spec), best[1])
    return new, curve, best[2]
