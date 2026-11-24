"""Numpy forward pass for wsp MLPs."""
from __future__ import annotations

import numpy as np

from .netspec import Network, NetworkError


def activate(name: str, z: np.ndarray) -> np.ndarray:
    if name == "relu":
        return np.maximum(z, 0.0)
    if name == "leaky_relu":
        return np.where(z > 0, z, 0.01 * z)
    if name == "tanh":
        return np.tanh(z)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
    if name == "step":
        return (z > 0).astype(np.float64)
    if name == "identity":
        return z
    if name == "softmax":
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    if name == "abs":
        return np.abs(z)
    if name == "square":
        return z * z
    if name == "sin":
        return np.sin(z)
    raise NetworkError(f"unknown activation {name}")


def normalize_input(spec: dict, X: np.ndarray) -> np.ndarray:
    norm = spec.get("input_norm")
    if not norm:
        return X
    return (X - np.asarray(norm["mean"])) / np.asarray(norm["std"])


def forward(net: Network, X: np.ndarray, capture: bool = False):
    """Returns (output, (pre_activations, post_activations)). Lists are empty unless capture."""
    if not net.is_mlp:
        raise NetworkError(
            "forward pass is only supported for MLP networks; this network is 'opaque' "
            "(inspect / slice / svd / patch / write still work)"
        )
    h = normalize_input(net.spec, np.asarray(X, dtype=np.float64))
    pre, post = [], []
    for i, L in enumerate(net.layers):
        z = h @ net.W[f"W{i}"] + net.W[f"b{i}"]
        h = activate(L["act"], z)
        if capture:
            pre.append(z)
            post.append(h)
    return h, (pre, post)


def to_prediction(spec: dict, out: np.ndarray):
    """Map raw network output to (prediction, probabilities-or-None)."""
    if spec["task"] == "regression":
        on = spec.get("output_norm")
        y = out * np.asarray(on["std"]) + np.asarray(on["mean"]) if on else out
        return y, None
    from .backend import output_act
    last = output_act(spec)
    if out.shape[1] == 1:
        thr = 0.5 if last in ("sigmoid", "step") else 0.0
        pred = (out[:, 0] > thr).astype(int)
        prob = np.clip(out[:, 0], 0, 1) if last == "sigmoid" else None
        return pred, prob
    pred = out.argmax(axis=1)
    prob = out if last == "softmax" else None
    return pred, prob


def predict(net: Network, X: np.ndarray):
    out, _ = forward(net, X)
    return to_prediction(net.spec, out)
