"""Architecture sizing: candidate networks with parameter counts and context cost.

The agent makes the final call. This module supplies the numbers it needs to decide:
how many parameters the data can support, and whether a full weight dump fits in context.
"""
from __future__ import annotations

import numpy as np

from .dump import estimate_full_dump_tokens
from .netspec import Network, init_weights, mlp_layers

HIDDEN_OPTIONS = [[], [2], [4], [8], [16], [32], [64], [128], [256],
                  [8, 8], [16, 16], [32, 32], [64, 64], [128, 128], [64, 64, 64], [256, 256]]


def output_layer(task: str, n_classes: int | None, n_targets: int = 1):
    if task == "classification":
        if n_classes == 2:
            return 1, "sigmoid"
        return n_classes, "softmax"
    return n_targets, "identity"


def n_params(sizes: list) -> int:
    return sum((sizes[i] + 1) * sizes[i + 1] for i in range(len(sizes) - 1))


def candidates(n_in: int, task: str, n_classes: int | None, n_targets: int, n_train: int,
               context_remaining: int, mode: str) -> list:
    out_dim, out_act = output_layer(task, n_classes, n_targets)
    rows = []
    for hidden in HIDDEN_OPTIONS:
        sizes = [n_in] + hidden + [out_dim]
        acts = ["relu"] * len(hidden) + [out_act]
        layers = mlp_layers(sizes, acts)
        net = Network({"arch": "mlp", "layers": layers}, init_weights(layers))
        p = n_params(sizes)
        tok = estimate_full_dump_tokens(net)
        rows.append({
            "arch": "-".join(map(str, sizes)), "acts": ",".join(acts), "params": p,
            "params_per_sample": p / max(n_train, 1), "dump_tokens": tok,
            "dump_frac": tok / max(context_remaining, 1),
        })
    return rows


def recommend(rows: list, mode: str) -> dict:
    """Heuristic default. Pure: smallest net with a hidden layer whose full dump is <=10% of the
    remaining context (you must be able to read and reason about every weight). Hybrid: largest net
    with params <= ~2x training samples whose dump still fits in 50% of the remaining context."""
    if mode == "pure":
        ok = [r for r in rows if r["dump_frac"] <= 0.10 and r["arch"].count("-") >= 2]
        return min(ok, key=lambda r: r["params"]) if ok else min(rows, key=lambda r: r["params"])
    ok = [r for r in rows if r["params_per_sample"] <= 2.0 and r["dump_frac"] <= 0.5]
    return max(ok, key=lambda r: r["params"]) if ok else min(rows, key=lambda r: r["params"])


def table_text(rows: list, rec: dict, meta: dict, n_train: int, mode: str) -> str:
    lines = [f"inputs={len(meta['feature_names'])}  task={meta['task']}"
             + (f"  classes={len(meta['classes'])}" if meta.get("classes") else "")
             + f"  train samples={n_train}  mode={mode}",
             f"{'arch':<22}{'acts':<30}{'params':>9}{'p/sample':>10}{'dump tok':>10}{'%ctx':>7}"]
    for r in rows:
        mark = "  <= heuristic default" if r is rec else ""
        lines.append(f"{r['arch']:<22}{r['acts']:<30}{r['params']:>9}{r['params_per_sample']:>10.2f}"
                     f"{r['dump_tokens']:>10}{100*r['dump_frac']:>6.1f}%{mark}")
    lines += [
        "",
        "How to decide (you choose; justify the choice in log.md):",
        " - Pure mode: count the features/rules/regions you intend to *design*. Each hidden unit should have a job",
        "   you can state (a hyperplane, a threshold, an AND/OR gate, a bump). Width = number of such units.",
        "   Prefer a full dump under ~10% of context so you can read every weight. Wider is not better",
        "   unless you have a plan for every unit; unused units stay at zero.",
        " - Hybrid mode: params/sample <= ~1-2 generalizes well for tabular data; deeper only for clearly",
        "   compositional tasks. Keep the dump under ~50% of context so hand edits stay possible.",
        " - Activations: relu for piecewise-linear features, step/sigmoid for logic, abs/square for distance",
        "   or radial structure, sin for periodic structure. Any listed activation can be set per layer.",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------
# image / text sizing (layer-graph presets)
# ------------------------------------------------------------------------------------------------
def preset_grid(modality: str) -> list:
    if modality == "image":
        return ([("linear", {})] + [("linear", {"hidden": h}) for h in (16, 64)]
                + [("cnn", {"width": w, "depth": d}) for d in (2, 3) for w in (8, 16, 32, 64)])
    if modality == "text":
        return ([("bow", {"dim": d}) for d in (1, 4, 16, 64)]
                + [("cnn1d", {"dim": d, "width": w}) for d, w in ((32, 64), (64, 128))]
                + [("transformer", {"dim": d, "blocks": b, "heads": 4}) for d, b in ((32, 1), (64, 2), (128, 2))])
    return []


def net_candidates(input_spec: dict, task: str, n_classes, n_targets: int, n_train: int,
                   context_remaining: int) -> list:
    from .graph import assign_names, infer, preset_layers
    out_dim, out_act = output_layer(task, n_classes, n_targets)
    rows = []
    for name, kw in preset_grid(input_spec["modality"]):
        try:
            layers = assign_names(preset_layers(name, input_spec, out_dim, out_act, **kw))
            spec = {"arch": "net", "input": input_spec, "layers": layers}
            p = sum(int(np.prod(s)) for r in infer(spec) for s in r["params"].values())
        except Exception:
            continue
        tok = int(p * 6.2 / 2.5) + 400
        opts = " ".join(f"--{k} {v}" for k, v in kw.items())
        rows.append({"preset": name, "opts": opts, "params": p, "params_per_sample": p / max(n_train, 1),
                     "dump_tokens": tok, "dump_frac": tok / max(context_remaining, 1)})
    return rows


def net_recommend(rows: list, modality: str, mode: str, n_train: int) -> dict:
    def find(preset, **kw):
        want = " ".join(f"--{k} {v}" for k, v in kw.items())
        return next((r for r in rows if r["preset"] == preset and r["opts"] == want), rows[0])
    if mode == "pure":
        return find("linear") if modality == "image" else find("bow", dim=1)
    if modality == "image":
        if n_train < 2000:
            return find("cnn", width=16, depth=2)
        if n_train < 20000:
            return find("cnn", width=32, depth=3)
        return find("cnn", width=64, depth=3)
    if n_train < 2000:
        return find("bow", dim=16)
    if n_train < 20000:
        return find("cnn1d", dim=64, width=128)
    return find("transformer", dim=64, blocks=2, heads=4)


def net_table_text(rows: list, rec: dict, modality: str, mode: str, n_train: int, run_name: str) -> str:
    lines = [f"modality={modality}  train samples={n_train}  mode={mode}",
             f"{'preset':<12}{'options':<36}{'params':>10}{'p/sample':>10}{'dump tok':>11}{'%ctx':>7}"]
    for r in rows:
        mark = "  <= heuristic default" if r is rec else ""
        lines.append(f"{r['preset']:<12}{r['opts']:<36}{r['params']:>10}{r['params_per_sample']:>10.2f}"
                     f"{r['dump_tokens']:>11}{100 * r['dump_frac']:>6.1f}%{mark}")
    lines += [
        "",
        f"create one with: python -m wsp init {run_name} --preset <preset> <options>   "
        "(or --layers layers.json for any custom stack; see graph layer types in `wsp init -h`)",
        "How to decide (you choose; justify it in log.md):",
    ]
    if modality == "image":
        lines += [
            " - Programmable by hand: `linear` (template matching: class-mean images as weights, see",
            "   dumps/class_means.png) or a tiny cnn whose filters you design (edges, blobs, colour detectors).",
            " - Learned: cnn width 16-64, depth 2-3; images share weights, so params/sample > 1 is fine.",
        ]
    else:
        lines += [
            " - Programmable by hand: `bow --dim 1` is a lexicon scorer (one number per token, mean-pooled);",
            "   `bow --dim <classes>` gives one score per class per token. Use the indicative-token table in `profile`.",
            " - Learned: bow for small data, cnn1d for local phrases, transformer for order/long-range structure",
            "   with lots of data (> ~20k samples).",
        ]
    return "\n".join(lines)
