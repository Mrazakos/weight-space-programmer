"""Import networks trained elsewhere, and export wsp networks back out.

Supported imports:
  .npz          keys W0,b0,... (wsp layout, W: (in,out)) or torch-style '<name>.weight'/'<name>.bias' (W: (out,in))
  .pkl/.joblib  scikit-learn MLPClassifier / MLPRegressor, optionally inside a Pipeline with
                StandardScaler / MinMaxScaler (folded into input_norm). Pickles execute code when loaded:
                only import files you trust (requires --trust).
  .pt/.pth      torch state_dict (needs torch installed); loaded with weights_only=True
  .safetensors  needs the `safetensors` package

Sequential Linear stacks become runnable MLPs. Anything else is imported as an 'opaque' network
(inspect / slice / svd / patch / write work, forward passes do not).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .netspec import ACTIVATIONS, Network

SK_ACT = {"relu": "relu", "tanh": "tanh", "logistic": "sigmoid", "identity": "identity", "softmax": "softmax"}


def _default_acts(n_layers: int, out_dim: int, task: str) -> list:
    if task == "classification":
        out = "sigmoid" if out_dim == 1 else "softmax"
    else:
        out = "identity"
    return ["relu"] * (n_layers - 1) + [out]


def _parse_state(arrays: dict, transpose_torch: bool = True):
    """Try to interpret an ordered dict of arrays as a sequential MLP. Returns (Ws, bs, keymap) or None."""
    keys = list(arrays)
    if all(k[0] in "Wb" and k[1:].isdigit() for k in keys):
        n = len([k for k in keys if k.startswith("W")])
        Ws = [arrays[f"W{i}"] for i in range(n)]
        bs = [arrays.get(f"b{i}", np.zeros(Ws[i].shape[1])) for i in range(n)]
        return Ws, bs, [{"W": f"W{i}", "b": f"b{i}", "transpose": False} for i in range(n)]
    Ws, bs, keymap = [], [], []
    used = set()
    for k in keys:
        if k.endswith("weight") and arrays[k].ndim == 2:
            prefix = k[: -len("weight")]
            bk = prefix + "bias"
            W = arrays[k].T if transpose_torch else arrays[k]
            b = arrays[bk] if bk in arrays else np.zeros(W.shape[1])
            Ws.append(W); bs.append(b)
            keymap.append({"W": k, "b": bk if bk in arrays else None, "transpose": transpose_torch})
            used |= {k, bk}
    leftovers = [k for k in keys if k not in used]
    if not Ws or leftovers:
        return None
    for i in range(1, len(Ws)):
        if Ws[i].shape[0] != Ws[i - 1].shape[1]:
            return None
    return Ws, bs, keymap


def _opaque(arrays: dict, base_spec: dict, source: dict) -> Network:
    spec = dict(base_spec)
    spec.update({"arch": "opaque", "layers": None, "source": source,
                 "tensors": {k: list(np.shape(v)) for k, v in arrays.items()}})
    return Network(spec, {k: np.asarray(v, dtype=np.float64) for k, v in arrays.items()})


def _mlp(Ws, bs, acts, base_spec, source) -> Network:
    layers = [{"in": int(W.shape[0]), "out": int(W.shape[1]), "act": a} for W, a in zip(Ws, acts)]
    spec = dict(base_spec)
    spec.update({"arch": "mlp", "layers": layers, "source": source})
    W = {}
    for i, (w, b) in enumerate(zip(Ws, bs)):
        W[f"W{i}"] = np.asarray(w, dtype=np.float64)
        W[f"b{i}"] = np.asarray(b, dtype=np.float64).reshape(-1)
    return Network(spec, W)


def import_model(path: str, base_spec: dict, acts: list | None = None, trust: bool = False,
                 keep_run_norm: bool = False):
    """Returns (Network, notes:list[str])."""
    p = Path(path)
    ext = p.suffix.lower()
    notes = []
    spec = dict(base_spec)
    if not keep_run_norm:
        spec["input_norm"] = None
        spec["output_norm"] = None
        notes.append("input/output normalization from the run is NOT applied (the imported model expects raw "
                     "features unless it carried its own scaler); use --keep-run-norm to override")
    source = {"path": str(p.resolve()), "format": ext.lstrip(".")}

    if ext in (".pkl", ".pickle", ".joblib"):
        if not trust:
            raise PermissionError("pickle/joblib files can execute arbitrary code when loaded. Re-run with --trust "
                                  "only if you trust where this file came from.")
        try:
            import joblib
            obj = joblib.load(p)
        except Exception:
            with open(p, "rb") as f:
                obj = pickle.load(f)
        return _from_sklearn(obj, spec, source, notes)

    if ext == ".npz":
        with np.load(p, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    elif ext in (".pt", ".pth", ".bin"):
        try:
            import torch
        except ImportError as e:
            raise ImportError("importing .pt needs torch: pip install torch") from e
        sd = torch.load(p, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        arrays = {k: v.detach().cpu().numpy() for k, v in sd.items() if hasattr(v, "detach")}
        source["torch"] = True
    elif ext == ".safetensors":
        try:
            from safetensors.numpy import load_file
        except ImportError as e:
            raise ImportError("importing .safetensors needs: pip install safetensors") from e
        arrays = load_file(str(p))
    else:
        raise ValueError(f"unsupported model format {ext}")

    parsed = _parse_state(arrays)
    if parsed is None:
        notes.append("not a plain sequential Linear stack: imported as OPAQUE (inspect/patch/write only, "
                     "no forward passes). Use the sandbox inspection tools.")
        return _opaque(arrays, spec, source), notes
    Ws, bs, keymap = parsed
    source["keymap"] = keymap
    if acts is None:
        acts = _default_acts(len(Ws), Ws[-1].shape[1], spec["task"])
        notes.append(f"activations are not stored in {ext} files; ASSUMED {','.join(acts)} "
                     "(pass --act to set them; wrong activations make the import meaningless)")
    if len(acts) != len(Ws):
        raise ValueError(f"--act needs {len(Ws)} activations, got {len(acts)}")
    for a in acts:
        if a not in ACTIVATIONS:
            raise ValueError(f"unknown activation {a}")
    return _mlp(Ws, bs, acts, spec, source), notes


def _from_sklearn(obj, spec, source, notes):
    steps = obj.steps if hasattr(obj, "steps") else [("model", obj)]
    norm = None
    for name, step in steps[:-1]:
        cls = type(step).__name__
        if cls == "StandardScaler":
            mean = getattr(step, "mean_", None)
            scale = getattr(step, "scale_", None)
            n = step.n_features_in_
            norm = {"mean": (mean if mean is not None else np.zeros(n)).tolist(),
                    "std": (scale if scale is not None else np.ones(n)).tolist()}
        elif cls == "MinMaxScaler":
            norm = {"mean": (-step.min_ / step.scale_).tolist(), "std": (1.0 / step.scale_).tolist()}
        else:
            raise ValueError(f"unsupported pipeline step {cls}; only StandardScaler/MinMaxScaler before the MLP")
    est = steps[-1][1]
    cls = type(est).__name__
    if cls not in ("MLPClassifier", "MLPRegressor"):
        raise ValueError(f"unsupported sklearn estimator {cls}; expected MLPClassifier/MLPRegressor")
    acts = [SK_ACT[est.activation]] * (len(est.coefs_) - 1) + [SK_ACT[est.out_activation_]]
    if norm:
        spec["input_norm"] = norm
        notes.append("folded the pipeline scaler into input_norm")
    if cls == "MLPClassifier":
        sk_classes = [str(c) for c in est.classes_]
        run_classes = spec.get("classes") or []
        if sk_classes != list(run_classes):
            notes.append(f"WARNING: model classes {sk_classes} differ from run classes {run_classes}")
    source["sklearn"] = cls
    net = _mlp(est.coefs_, est.intercepts_, acts, spec, source)
    return net, notes


def export_model(net: Network, out: str, trust: bool = False) -> str:
    p = Path(out)
    ext = p.suffix.lower()
    src = net.spec.get("source") or {}
    if ext == ".npz":
        np.savez(p, **net.W)
        return f"wrote {p} (wsp layout: W{{i}} is (in,out))"
    if ext in (".pt", ".pth"):
        import torch
        sd = {}
        if src.get("keymap") and net.is_mlp and len(src["keymap"]) == len(net.layers):
            for i, km in enumerate(src["keymap"]):
                W = net.W[f"W{i}"]
                sd[km["W"]] = torch.tensor(W.T if km["transpose"] else W, dtype=torch.float32)
                if km["b"]:
                    sd[km["b"]] = torch.tensor(net.W[f"b{i}"], dtype=torch.float32)
        elif net.is_mlp:
            for i in range(len(net.layers)):
                sd[f"layers.{i}.weight"] = torch.tensor(net.W[f"W{i}"].T, dtype=torch.float32)
                sd[f"layers.{i}.bias"] = torch.tensor(net.W[f"b{i}"], dtype=torch.float32)
        else:
            sd = {k: torch.tensor(v, dtype=torch.float32) for k, v in net.W.items()}
        torch.save(sd, p)
        return f"wrote torch state_dict {p}"
    if ext == ".safetensors":
        from safetensors.numpy import save_file
        save_file({k: v.astype(np.float32) for k, v in net.W.items()}, str(p))
        return f"wrote {p}"
    if ext in (".pkl", ".joblib"):
        if not src.get("sklearn"):
            raise ValueError("pickle export is only available for networks imported from scikit-learn")
        if not trust:
            raise PermissionError("re-loading the original pickle executes code; pass --trust")
        import joblib
        obj = joblib.load(src["path"])
        est = obj.steps[-1][1] if hasattr(obj, "steps") else obj
        shapes = [tuple(c.shape) for c in est.coefs_]
        if shapes != [tuple(net.W[f"W{i}"].shape) for i in range(len(net.layers))]:
            raise ValueError("architecture changed since import (layer shapes differ); cannot write back into the "
                             "sklearn object. Export to .npz/.pt instead.")
        if [L["act"] for L in net.layers[:-1]] and any(L["act"] != net.layers[0]["act"] for L in net.layers[:-1]):
            raise ValueError("sklearn MLPs use one hidden activation for all layers; this network mixes them")
        est.coefs_ = [net.W[f"W{i}"].copy() for i in range(len(net.layers))]
        est.intercepts_ = [net.W[f"b{i}"].copy() for i in range(len(net.layers))]
        joblib.dump(obj, p)
        return f"wrote scikit-learn model {p} (original pipeline with edited weights)"
    raise ValueError("export format must be .npz, .pt, .safetensors, or .pkl")
