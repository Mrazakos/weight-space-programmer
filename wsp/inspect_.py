"""Sandbox inspection: understand a network without reading every weight.

Weight-only tools (stats, unit, slice, svd) are free of forward-pass cost.
`probe` runs the network on data and therefore costs one metered forward pass.
"""
from __future__ import annotations

import numpy as np

from .dump import format_tensor, row_labels
from .forward import forward
from .netspec import Network


def stats_text(net: Network) -> str:
    lines = [f"network {net.arch_str()}  params={net.n_params()}",
             f"{'tensor':<24}{'shape':<16}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}{'%zero':>7}{'norm':>9}"]
    for k, v in net.W.items():
        z = 100.0 * np.mean(np.abs(v) < 1e-12) if v.size else 0.0
        lines.append(f"{k[:23]:<24}{str(v.shape):<16}{v.mean():9.3g}{v.std():9.3g}{v.min():9.3g}"
                     f"{v.max():9.3g}{z:7.1f}{np.linalg.norm(v):9.3g}")
    if net.is_mlp:
        for i, L in enumerate(net.layers):
            W, b = net.W[f"W{i}"], net.W[f"b{i}"]
            in_norm = np.linalg.norm(W, axis=0)
            dead_in = int(np.sum(in_norm < 1e-12))
            line = (f"L{i} {L['in']}->{L['out']} {L['act']}: units with zero incoming weights={dead_in}, "
                    f"incoming-norm range=[{in_norm.min():.3g}, {in_norm.max():.3g}], bias range=[{b.min():.3g}, {b.max():.3g}]")
            if i + 1 < len(net.layers):
                out_norm = np.linalg.norm(net.W[f"W{i+1}"], axis=1)
                line += f", units with zero outgoing weights={int(np.sum(out_norm < 1e-12))}"
            lines.append(line)
    return "\n".join(lines)


def _labels_for_units(net: Network, layer: int) -> list:
    if layer == len(net.layers) - 1:
        classes = net.spec.get("classes")
        if classes and net.layers[-1]["out"] == len(classes):
            return [f"out:{c}" for c in classes]
        return [f"out{j}" for j in range(net.layers[-1]["out"])]
    return [f"h{layer}.{j}" for j in range(net.layers[layer]["out"])]


def unit_text(net: Network, layer: int, unit: int, top: int = 12) -> str:
    if not net.is_mlp:
        raise ValueError("unit inspection needs an MLP; use slice/svd for opaque networks")
    W, b = net.W[f"W{layer}"], net.W[f"b{layer}"]
    labels = row_labels(net, f"W{layer}") or [f"in{r}" for r in range(W.shape[0])]
    w = W[:, unit]
    order = np.argsort(-np.abs(w))[:top]
    name = _labels_for_units(net, layer)[unit]
    lines = [f"unit {name} (layer {layer}, act={net.layers[layer]['act']}): bias={b[unit]:.4g}, "
             f"incoming |w| sum={np.abs(w).sum():.4g}",
             "  top incoming weights: " + ", ".join(f"{labels[r]}={w[r]:.3f}" for r in order)]
    if layer + 1 < len(net.layers):
        w2 = net.W[f"W{layer+1}"][unit, :]
        outl = _labels_for_units(net, layer + 1)
        o2 = np.argsort(-np.abs(w2))[:top]
        lines.append("  top outgoing weights: " + ", ".join(f"{outl[c]}={w2[c]:.3f}" for c in o2))
    return "\n".join(lines)


def slice_text(net: Network, tensor: str, rows: str | None, cols: str | None, precision: int) -> str:
    if tensor not in net.W:
        raise KeyError(f"unknown tensor {tensor}; available: {list(net.W)}")
    return format_tensor(net, tensor, precision, rows, cols)


def svd_text(net: Network, tensor: str, k: int = 5, top: int = 8) -> str:
    a = net.W[tensor]
    if a.ndim < 2:
        raise ValueError("svd needs a 2-D (or higher) tensor")
    a2 = a.reshape(a.shape[0], -1)
    U, S, Vt = np.linalg.svd(a2, full_matrices=False)
    energy = S ** 2 / max((S ** 2).sum(), 1e-30)
    lines = [f"svd of {tensor} {a.shape}: top singular values " +
             ", ".join(f"{s:.4g} ({e*100:.1f}%)" for s, e in zip(S[:k], energy[:k])),
             f"effective rank (99% energy): {int(np.searchsorted(np.cumsum(energy), 0.99) + 1)} of {len(S)}"]
    labels = row_labels(net, tensor) or [f"r{r}" for r in range(a2.shape[0])]
    for i in range(min(k, len(S))):
        ru = np.argsort(-np.abs(U[:, i]))[:top]
        cv = np.argsort(-np.abs(Vt[i]))[:top]
        lines.append(f"  comp {i}: rows " + ", ".join(f"{labels[r]}={U[r, i]:.2f}" for r in ru) +
                     " | cols " + ", ".join(f"c{c}={Vt[i, c]:.2f}" for c in cv))
    return "\n".join(lines)


def probe_text(net: Network, X: np.ndarray, y: np.ndarray, task: str, layer: int | None = None,
               max_units: int = 64) -> str:
    _, (pre, post) = forward(net, X, capture=True)
    lines = [f"activation probe on {len(X)} samples"]
    layers = range(len(net.layers) - 1) if layer is None else [layer]
    classes = net.spec.get("classes") or []
    for li in layers:
        act = net.layers[li]["act"]
        h, z = post[li], pre[li]
        active = z > 0
        frac = active.mean(axis=0)
        dead = int(np.sum(frac == 0))
        always = int(np.sum(frac == 1))
        lines.append(f"L{li} ({act}, {h.shape[1]} units): never-active(z<=0)={dead}, always-active={always}")
        if act == "tanh":
            lines.append(f"   saturated (|h|>0.95 on >90% samples): {int(np.sum((np.abs(h) > 0.95).mean(axis=0) > 0.9))}")
        lines.append(f"   {'unit':<8}{'mean':>9}{'std':>9}{'%z>0':>7}  relation to target")
        units = range(h.shape[1])
        if h.shape[1] > max_units:
            # show the most variable units
            units = np.argsort(-h.std(axis=0))[:max_units]
            lines.append(f"   (showing the {max_units} most variable units of {h.shape[1]})")
        for j in units:
            col = h[:, j]
            if task == "classification" and len(classes):
                mus = [col[y == c].mean() if np.any(y == c) else np.nan for c in range(len(classes))]
                c_hi = int(np.nanargmax(mus))
                rel = f"highest for class {classes[c_hi]} ({mus[c_hi]:.3g} vs mean {np.nanmean(mus):.3g})"
            else:
                yy = y.reshape(len(y), -1)[:, 0]
                c = np.corrcoef(col, yy)[0, 1] if col.std() > 0 and yy.std() > 0 else 0.0
                rel = f"corr(y)={c:.3f}"
            lines.append(f"   h{li}.{j:<5}{col.mean():9.3g}{col.std():9.3g}{100*frac[j]:7.1f}  {rel}")
    return "\n".join(lines)
