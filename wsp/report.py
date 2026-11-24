"""Final held-out evaluation and comparison against a gradient-trained baseline."""
from __future__ import annotations

import time

import numpy as np

from .backend import fit, predict
from .evaluate import compute_metrics, primary_metric
from .netspec import Network, init_weights


def _metrics(run, net: Network, X, y) -> dict:
    pred, prob = predict(net, X)
    k = len(run.meta.get("classes") or []) or None
    return compute_metrics(run.task, y, pred, prob, k)


def baseline(run, net: Network, seeds=(0, 1, 2), epochs: int = 400, max_minutes: float = 6.0) -> dict | None:
    """Same architecture, random init, Adam + early stopping on val. Best of the seeds by val."""
    arch = net.spec.get("arch")
    if arch not in ("mlp", "net"):
        return None
    Xtr, ytr = run.load_split("train")
    Xva, yva = run.load_split("val")
    layers = [dict(L) for L in net.layers]
    for L in layers:
        if L.get("act") == "step":
            L["act"] = "sigmoid"
    best = None
    key = primary_metric(run.task)
    if arch == "net":
        seeds = seeds[:2]
    for s in seeds:
        if arch == "mlp":
            b = Network({**net.spec, "layers": layers}, init_weights(layers, zero=False, seed=s))
            b, _, ep = fit(b, Xtr, ytr, Xva, yva, epochs=epochs, lr=3e-3, batch=64, patience=40, seed=s)
        else:
            from .graph import init_params
            spec = {**net.spec, "layers": layers}
            b = Network(spec, init_params(spec, zero=False, seed=s))
            b, _, ep = fit(b, Xtr, ytr, Xva, yva, epochs=60, lr=1e-3, batch=64, patience=6, seed=s,
                           max_minutes=max_minutes / len(seeds))
        val = _metrics(run, b, Xva, yva)
        if best is None or val[key] > best[1][key]:
            best = (b, val, ep, s)
    return {"net": best[0], "val": best[1], "epochs": best[2], "seed": best[3]}


def make_report(run, version: str | None = None, with_baseline: bool = True) -> str:
    v = version or run.best_version()
    net = run.load_net(v)
    key = primary_metric(run.task)
    Xte, yte = run.load_split("test", allow_test=True)
    Xva, yva = run.load_split("val")
    lines = [f"# Report: {run.meta['name']}", "",
             f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"- mode: **{run.mode}** ({'weights written by reasoning only, no gradients' if run.mode == 'pure' else 'reasoning + gradient training allowed'})",
             f"- task: {run.task}" + (f", classes: {run.meta['classes']}" if run.meta.get('classes') else ""),
             f"- source: {run.meta.get('source')}",
             f"- reported version: **{v}**, architecture {net.arch_str()}, {net.n_params()} parameters",
             f"- versions created: {len(run.versions())}; sessions: {run.session.get('index')}",
             ""]
    hist = run.history()
    fwd = sum(1 for e in hist if e.get("event") in ("eval", "probe"))
    fits = sum(1 for e in hist if e.get("event") == "create" and e.get("op") == "fit")
    lines.append(f"- metered forward passes used (all sessions): {fwd}; gradient fits: {fits}")
    if run.meta.get("test_seen"):
        lines.append(f"- NOTE: the test split was already evaluated before (first at {run.meta['test_seen']}); "
                     "later improvements may be biased toward it")
    lines.append("")
    from .backend import is_runnable
    if is_runnable(net):
        val = _metrics(run, net, Xva, yva)
        test = _metrics(run, net, Xte, yte)
        lines += ["| model | split | " + " | ".join(test) + " |", "|---|---|" + "---|" * len(test)]
        lines.append(f"| {v} | val | " + " | ".join(f"{val[m]:.4f}" for m in test) + " |")
        lines.append(f"| {v} | **test** | " + " | ".join(f"**{test[m]:.4f}**" for m in test) + " |")
        if with_baseline:
            b = baseline(run, net)
            if b:
                bt = _metrics(run, b["net"], Xte, yte)
                lines.append(f"| gradient baseline (same arch, Adam, {b['epochs']} ep) | val | "
                             + " | ".join(f"{b['val'][m]:.4f}" for m in test) + " |")
                lines.append("| gradient baseline | test | " + " | ".join(f"{bt[m]:.4f}" for m in test) + " |")
    else:
        lines.append("Opaque network: wsp cannot run it, so no metrics. Export it and evaluate in its own framework.")
    lines += ["", "## Version history", ""]
    scores = run.val_scores()
    for e in hist:
        if e.get("event") == "create":
            s = scores.get(e["version"])
            lines.append(f"- {e['version']} <- {e.get('parent')} [{e['op']}] {e.get('arch')}"
                         + (f" val {key}={s:.4f}" if s is not None else "") + (f": {e['note']}" if e.get("note") else ""))
    if not run.meta.get("test_seen"):
        run.meta["test_seen"] = time.strftime("%Y-%m-%d %H:%M:%S")
        run.save()
    text = "\n".join(lines) + "\n"
    (run.path / "report.md").write_text(text, encoding="utf-8")
    run.log_event({"event": "report", "version": v})
    return text
