"""Numpy backprop + Adam for wsp MLPs.

Used by `wsp fit` (hybrid mode only) and by the gradient baseline in `wsp report`.
Pure-mode agents never get access to this module through the CLI.
"""
from __future__ import annotations

import numpy as np

from .evaluate import compute_metrics, primary_metric
from .forward import activate, normalize_input, to_prediction
from .netspec import Network


def _act_grad(name: str, z: np.ndarray, h: np.ndarray) -> np.ndarray:
    if name == "relu":
        return (z > 0).astype(np.float64)
    if name == "leaky_relu":
        return np.where(z > 0, 1.0, 0.01)
    if name == "tanh":
        return 1.0 - h * h
    if name == "sigmoid":
        return h * (1.0 - h)
    if name == "step":
        return (np.abs(z) < 1).astype(np.float64)  # straight-through estimator
    if name == "identity":
        return np.ones_like(z)
    if name == "abs":
        return np.sign(z)
    if name == "square":
        return 2.0 * z
    if name == "sin":
        return np.cos(z)
    raise ValueError(name)


def _targets(net: Network, y: np.ndarray):
    spec = net.spec
    if spec["task"] == "regression":
        T = np.asarray(y, float).reshape(len(y), -1)
        on = spec.get("output_norm")
        if on:
            T = (T - np.asarray(on["mean"])) / np.asarray(on["std"])
        return T
    k = net.layers[-1]["out"]
    y = np.asarray(y).astype(int)
    if k == 1:
        return y.reshape(-1, 1).astype(float)
    T = np.zeros((len(y), k))
    T[np.arange(len(y)), y] = 1.0
    return T


def loss_and_grads(net: Network, X: np.ndarray, T: np.ndarray, weight_decay: float = 0.0):
    hs = [normalize_input(net.spec, X)]
    zs = []
    for i, L in enumerate(net.layers):
        z = hs[-1] @ net.W[f"W{i}"] + net.W[f"b{i}"]
        zs.append(z)
        hs.append(activate(L["act"], z))
    B = len(X)
    out_act = net.layers[-1]["act"]
    cls = net.spec["task"] == "classification"
    zL, hL = zs[-1], hs[-1]
    eps = 1e-12
    if cls and out_act == "softmax":
        loss = -np.mean(np.sum(T * np.log(hL + eps), axis=1))
        dz = (hL - T) / B
    elif cls and out_act == "sigmoid":
        loss = -np.mean(T * np.log(hL + eps) + (1 - T) * np.log(1 - hL + eps))
        dz = (hL - T) / B
    elif cls and out_act == "identity":
        if zL.shape[1] == 1:
            p = 1 / (1 + np.exp(-np.clip(zL, -60, 60)))
            loss = -np.mean(T * np.log(p + eps) + (1 - T) * np.log(1 - p + eps))
        else:
            p = activate("softmax", zL)
            loss = -np.mean(np.sum(T * np.log(p + eps), axis=1))
        dz = (p - T) / B
    else:
        diff = hL - T
        loss = np.mean(np.sum(diff ** 2, axis=1))
        dz = 2 * diff / B * _act_grad(out_act, zL, hL)
    grads = {}
    for i in reversed(range(len(net.layers))):
        grads[f"W{i}"] = hs[i].T @ dz + weight_decay * net.W[f"W{i}"]
        grads[f"b{i}"] = dz.sum(axis=0)
        if i > 0:
            dh = dz @ net.W[f"W{i}"].T
            dz = dh * _act_grad(net.layers[i - 1]["act"], zs[i - 1], hs[i])
    return float(loss), grads


def _score(net: Network, X, y) -> float:
    h = normalize_input(net.spec, X)
    for i, L in enumerate(net.layers):
        h = activate(L["act"], h @ net.W[f"W{i}"] + net.W[f"b{i}"])
    pred, prob = to_prediction(net.spec, h)
    k = len(net.spec.get("classes") or []) or None
    return compute_metrics(net.spec["task"], y, pred, prob, k)[primary_metric(net.spec["task"])]


def fit(net: Network, Xtr, ytr, Xva=None, yva=None, epochs: int = 200, lr: float = 1e-3,
        batch: int = 64, weight_decay: float = 0.0, freeze: list | None = None, patience: int = 30,
        seed: int = 0, log_every: int = 0):
    """Adam training with early stopping on val. Returns (net, curve)."""
    rng = np.random.default_rng(seed)
    net = net.copy()
    frozen = set()
    for f in freeze or []:
        frozen |= {f"W{f}", f"b{f}"}
    T = _targets(net, ytr)
    m = {k: np.zeros_like(v) for k, v in net.W.items()}
    v2 = {k: np.zeros_like(v) for k, v in net.W.items()}
    b1, b2, t = 0.9, 0.999, 0
    best = (-np.inf, net.copy(), 0)
    curve = []
    for ep in range(1, epochs + 1):
        perm = rng.permutation(len(Xtr))
        losses = []
        for s in range(0, len(perm), batch):
            idx = perm[s:s + batch]
            loss, g = loss_and_grads(net, Xtr[idx], T[idx], weight_decay)
            losses.append(loss)
            t += 1
            for k in net.W:
                if k in frozen:
                    continue
                m[k] = b1 * m[k] + (1 - b1) * g[k]
                v2[k] = b2 * v2[k] + (1 - b2) * g[k] ** 2
                mh = m[k] / (1 - b1 ** t)
                vh = v2[k] / (1 - b2 ** t)
                net.W[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        val = _score(net, Xva, yva) if Xva is not None else -float(np.mean(losses))
        curve.append((ep, float(np.mean(losses)), float(val)))
        if val > best[0] + 1e-9:
            best = (val, net.copy(), ep)
        elif ep - best[2] >= patience:
            break
    return best[1], curve, best[2]
