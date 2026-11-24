"""wsp command line: the metered toolset the weight-space-programming agents use.

    python -m wsp <command> <run> [options]

Run `python -m wsp -h` or `python -m wsp <command> -h` for details.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from . import data as D
from . import meter
from .dump import (build_dump, check_fits, chunk_text, estimate_full_dump_tokens, select_tensors)
from .evaluate import compute_metrics, confusion_matrix, primary_metric, worst_examples
from .forward import forward, to_prediction
from .inspect_ import probe_text, slice_text, stats_text, svd_text, unit_text
from .netspec import Network, NetworkError, init_weights, mlp_layers, parse_arch
from .program import ProgramRejected, run_program
from .run import Run, resolve, runs_root
from .sizing import candidates, output_layer, recommend, table_text


class CLIError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
def _run(name: str) -> Run:
    return Run(resolve(name))


def _parse_index(s: str | None):
    if s is None or s == "":
        return (slice(None),)
    parts = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if tok in (":", ""):
            parts.append(slice(None))
        elif ":" in tok:
            a, b = tok.split(":", 1)
            parts.append(slice(int(a) if a else None, int(b) if b else None))
        else:
            parts.append(int(tok))
    return tuple(parts)


def _index_from_json(idx):
    if idx is None:
        return (slice(None),)
    if isinstance(idx, (int, str)):
        idx = [idx]
    return _parse_index(",".join(":" if i is None else str(i) for i in idx))


def _n_classes(run: Run) -> int | None:
    c = run.meta.get("classes")
    return len(c) if c else None


def _check_io(run: Run, net: Network) -> None:
    if not net.is_mlp:
        return
    if net.layers[0]["in"] != run.meta["n_in"]:
        raise CLIError(f"network expects {net.layers[0]['in']} inputs but the run's data has {run.meta['n_in']}")
    out = net.layers[-1]["out"]
    if run.task == "classification":
        k = _n_classes(run)
        if not (out == k or (k == 2 and out == 1)):
            raise CLIError(f"classification with {k} classes needs {k} outputs (or 1 sigmoid output if binary), got {out}")
    elif out != run.meta.get("n_targets", 1):
        raise CLIError(f"regression with {run.meta.get('n_targets', 1)} target(s) needs that many outputs, got {out}")


def _changed_summary(old: Network, new: Network) -> str:
    lines = []
    for k, v in new.W.items():
        if k not in old.W or old.W[k].shape != v.shape:
            lines.append(f"  {k}: new/reshaped {v.shape}, nonzero={int(np.count_nonzero(v))}")
            continue
        d = np.abs(v - old.W[k]) > 1e-12
        if d.any():
            lines.append(f"  {k}: changed {int(d.sum())}/{v.size} entries, nonzero={int(np.count_nonzero(v))}")
    for k in old.W:
        if k not in new.W:
            lines.append(f"  {k}: removed")
    return "\n".join(lines) if lines else "  (no weights changed)"


def _session_line(run: Run) -> str:
    s = run.session
    return (f"session {s['index']}: forward passes {s['forward_used']}/{s['forward_budget']} used, "
            f"context ~{s['context_used']:,}/{s['context_limit']:,} tokens used "
            f"(window {s['context_window']:,}, 25% reserved)")


# ------------------------------------------------------------------------------------------------
# commands
# ------------------------------------------------------------------------------------------------
def cmd_new(a):
    if bool(a.data) == bool(a.task):
        raise CLIError("give exactly one of --data FILE (with --target) or --task task.py")
    if a.data:
        if not a.target:
            raise CLIError("--target COLUMN is required with --data")
        df = D.load_table(a.data)
        X, y, meta = D.encode_table(df, [t.strip() for t in a.target.split(",")], a.task_type,
                                    [d.strip() for d in (a.drop or "").split(",") if d.strip()])
        source = {"data": str(Path(a.data).resolve()), "target": a.target}
        norm = True if a.norm is None else a.norm
    else:
        X, y, meta = D.load_task(a.task, a.n, a.seed)
        if a.task_type:
            meta["task"] = a.task_type
        source = {"task": str(Path(a.task).resolve()), "n": a.n}
        norm = False if a.norm is None else a.norm
    splits = D.split(X, y, meta["task"], seed=a.seed)
    Xtr, ytr = splits["train"]
    path = Path(a.name) if ("/" in a.name or "\\" in a.name) else runs_root() / a.name
    n_targets = 1 if meta["task"] == "classification" or ytr.ndim == 1 else ytr.shape[1]
    run_meta = {
        "name": path.name, "mode": a.mode, "task": meta["task"], "classes": meta["classes"],
        "feature_names": meta["feature_names"], "n_in": int(X.shape[1]), "n_targets": int(n_targets),
        "source": source, "description": meta.get("description", ""), "notes": meta["notes"],
        "input_norm": D.norm_stats(Xtr) if norm else None,
        "output_norm": None,
        "sizes": {k: int(len(v[0])) for k, v in splits.items()},
    }
    if norm and meta["task"] == "regression":
        yy = ytr.reshape(len(ytr), -1)
        sd = yy.std(axis=0)
        run_meta["output_norm"] = {"mean": yy.mean(axis=0).tolist(), "std": np.where(sd < 1e-12, 1, sd).tolist()}
    run_meta["session"] = meter.new_session(None, a.budget, a.context_tokens)
    run = Run.create(path, run_meta, splits)
    run.log_event({"event": "new", "source": source, "mode": a.mode})
    run.emit(f"created run {run.path} (mode={a.mode}; train/val/test = "
             f"{run_meta['sizes']['train']}/{run_meta['sizes']['val']}/{run_meta['sizes']['test']}; test is held out)\n"
             f"input standardization: {'on' if norm else 'off'}\n" + _session_line(run) + "\n\n"
             + D.profile_text(Xtr, ytr, {**meta}))


def cmd_session(a):
    run = _run(a.run)
    run.meta["session"] = meter.new_session(run.meta.get("session"), a.budget, a.context_tokens)
    run.save()
    run.log_event({"event": "session", "budget": run.session["forward_budget"]})
    cmd_status(a, run=run)
    h = run.path / "handoff.md"
    if h.exists():
        run.emit("\n--- handoff.md from the previous session ---\n" + h.read_text(encoding="utf-8"))


def cmd_status(a, run: Run | None = None):
    run = run or _run(a.run)
    lines = [f"run {run.path}  mode={run.mode}  task={run.task}  inputs={run.meta['n_in']}"
             + (f"  classes={run.meta['classes']}" if run.meta.get("classes") else ""),
             _session_line(run)]
    if run.meta.get("input_norm"):
        lines.append("inputs are standardized with the run's train mean/std (stored in run.json input_norm)")
    cur = run.meta.get("current")
    if cur:
        net = run.load_net()
        best = run.best_version()
        scores = run.val_scores()
        key = primary_metric(run.task)
        lines.append(f"current={cur} ({net.arch_str()}, {net.n_params()} params)"
                     + (f" val {key}={scores[cur]:.4f}" if cur in scores else " (not evaluated on full val)"))
        if best and best in scores:
            lines.append(f"best={best} val {key}={scores[best]:.4f}")
        tok = estimate_full_dump_tokens(net)
        rem = meter.context_remaining(run.session)
        if tok <= 0.5 * rem:
            lines.append(f"full weight dump ~{tok:,} tokens: FITS (read all weights with `wsp dump`)")
        else:
            lines.append(f"full weight dump ~{tok:,} tokens vs ~{rem:,} remaining: TOO LARGE -> use sandbox "
                         "inspection (inspect stats/unit/svd/slice/probe, dump --layer/--rows)")
    else:
        lines.append("no network yet: `wsp size` then `wsp init` (or `wsp import`)")
    run.emit("\n".join(lines))


def cmd_profile(a):
    run = _run(a.run)
    X, y = run.load_split("train")
    run.emit(D.profile_text(X, y, {"task": run.task, "feature_names": run.meta["feature_names"],
                                   "classes": run.meta.get("classes"), "description": run.meta.get("description"),
                                   "notes": run.meta.get("notes", [])}, max_features=a.max_features))


def cmd_sample(a):
    run = _run(a.run)
    X, y = run.load_split(a.split)
    rng = np.random.default_rng(a.seed)
    idx = rng.choice(len(X), size=min(a.n, len(X)), replace=False)
    names = run.meta["feature_names"]
    cols = list(range(min(len(names), a.max_features)))
    lines = [f"{len(idx)} random {a.split} rows (raw feature values)",
             "idx | target | " + " ".join(names[c][:10] for c in cols) + (" ..." if len(names) > len(cols) else "")]
    for i in idx:
        t = run.meta["classes"][int(y[i])] if run.task == "classification" else np.round(y[i], 4)
        lines.append(f"{i} | {t} | " + " ".join(f"{X[i, c]:.4g}" for c in cols))
    run.emit("\n".join(lines), numeric=True)


def cmd_size(a):
    run = _run(a.run)
    rows = candidates(run.meta["n_in"], run.task, _n_classes(run), run.meta.get("n_targets", 1),
                      run.meta["sizes"]["train"], meter.context_remaining(run.session), run.mode)
    rec = recommend(rows, run.mode)
    run.emit(table_text(rows, rec, run.meta, run.meta["sizes"]["train"], run.mode))


def cmd_init(a):
    run = _run(a.run)
    if run.mode == "pure" and a.random:
        raise CLIError("pure mode starts from zero weights (no experience); random init is not allowed")
    sizes = parse_arch(a.arch)
    if sizes[0] != run.meta["n_in"]:
        sizes = [run.meta["n_in"]] + sizes if len(sizes) >= 1 and sizes[0] != run.meta["n_in"] else sizes
    _, out_act = output_layer(run.task, _n_classes(run), run.meta.get("n_targets", 1))
    nl = len(sizes) - 1
    if a.act:
        acts = [s.strip() for s in a.act.split(",")]
        if len(acts) == 1 and nl > 1:
            acts = acts * (nl - 1) + [out_act]
    else:
        acts = ["relu"] * (nl - 1) + [out_act]
    layers = mlp_layers(sizes, acts)
    net = Network({**run.base_spec(), "layers": layers}, init_weights(layers, zero=not a.random, seed=a.seed))
    _check_io(run, net)
    v = run.save_net(net, "init", a.note or ("zero init" if not a.random else f"random init seed={a.seed}"), parent="")
    run.emit(f"created {v}: {net.arch_str()}, {net.n_params()} params, {'all zeros' if not a.random else 'random'}\n"
             f"full dump ~{estimate_full_dump_tokens(net):,} tokens")


def cmd_dump(a):
    run = _run(a.run)
    net = run.load_net(a.version)
    v = a.version or run.meta["current"]
    tensors = select_tensors(net, a.layer, [t.strip() for t in a.tensor.split(",")] if a.tensor else None)
    text = build_dump(net, tensors, a.precision, a.rows, a.cols)
    tok = meter.estimate_tokens(text, numeric=True)
    if not check_fits(run.session, tok):
        sizes = ", ".join(f"{t}={net.W[t].size}" for t in tensors[:20])
        raise CLIError(
            f"dump would be ~{tok:,} tokens but only ~{meter.context_remaining(run.session):,} remain in this "
            f"session's context budget.\nTensor sizes: {sizes}\nUse the sandbox fallback: `inspect stats`, "
            "`inspect unit`, `inspect svd`, `inspect probe`, or a narrower `dump --layer/--tensor/--rows/--cols`, "
            "or lower --precision.")
    if a.print or tok <= 1500:
        run.emit(text, numeric=True)
        return
    chunks = chunk_text(text)
    tag = (f"L{a.layer}" if a.layer is not None else (a.tensor.replace(",", "+")[:40] if a.tensor else "all"))
    if a.rows or a.cols:
        tag += "_slice"
    files = []
    for i, c in enumerate(chunks, 1):
        f = run.path / "dumps" / f"{v}_{tag}_part{i:02d}.txt"
        f.write_text(c, encoding="utf-8")
        files.append(f)
    meter.charge_tokens(run.session, tok)  # the agent is expected to Read every part
    run.save()
    run.emit(f"wrote {len(files)} dump file(s), ~{tok:,} tokens total (charged to the context ledger). "
             "Read each part in order:\n" + "\n".join(f"  {f}" for f in files))


def _save_child(run: Run, parent_v: str, net: Network, op: str, note: str) -> str:
    _check_io(run, net)
    return run.save_net(net, op, note, parent=parent_v)


def cmd_write(a):
    run = _run(a.run)
    parent_v = a.base or run.meta["current"]
    if parent_v:
        base = run.load_net(parent_v)
    else:
        raise CLIError("no network to write into; run `wsp init` first")
    src = Path(a.program).read_text(encoding="utf-8")
    try:
        W, upd = run_program(src, base.spec, base.W)
    except ProgramRejected as e:
        raise CLIError(str(e))
    spec = dict(base.spec)
    if upd and "layers" in upd:
        spec["layers"] = [{"in": int(L["in"]), "out": int(L["out"]), "act": L["act"]} for L in upd["layers"]]
    net = Network(spec, W)
    try:
        v = _save_child(run, parent_v, net, "program", a.note or Path(a.program).name)
    except NetworkError as e:
        raise CLIError(f"program output is not a valid network: {e}")
    shutil.copy(a.program, run.path / "programs" / f"{v}.py")
    run.emit(f"created {v} from {parent_v} via {a.program} ({net.arch_str()})\n"
             f"sanity check (free, no data): shapes OK, all finite\n" + _changed_summary(base, net))


def _apply_edit(W: dict, e: dict):
    t = e["tensor"]
    if t not in W:
        raise CLIError(f"unknown tensor {t}")
    idx = _index_from_json(e.get("index"))
    if "value" in e:
        W[t][idx] = e["value"]
    elif "add" in e:
        W[t][idx] += e["add"]
    elif "mul" in e:
        W[t][idx] *= e["mul"]
    else:
        raise CLIError("each edit needs one of value/add/mul")


def cmd_set(a):
    run = _run(a.run)
    parent_v = run.meta["current"]
    base = run.load_net()
    net = base.copy()
    e = {"tensor": a.tensor, "index": a.index.split(",") if a.index else None}
    for k in ("value", "add", "mul"):
        if getattr(a, k) is not None:
            e[k] = getattr(a, k)
    _apply_edit(net.W, e)
    v = _save_child(run, parent_v, net, "set", a.note or json.dumps(e))
    run.emit(f"created {v} from {parent_v}\n" + _changed_summary(base, net))


def cmd_patch(a):
    run = _run(a.run)
    parent_v = run.meta["current"]
    base = run.load_net()
    net = base.copy()
    edits = json.loads(Path(a.edits).read_text(encoding="utf-8"))
    if isinstance(edits, dict):
        edits = edits.get("edits", [edits])
    for e in edits:
        _apply_edit(net.W, e)
    v = _save_child(run, parent_v, net, "patch", a.note or f"{len(edits)} edits from {a.edits}")
    run.emit(f"created {v} from {parent_v} ({len(edits)} edits)\n" + _changed_summary(base, net))


def cmd_resize(a):
    run = _run(a.run)
    parent_v = run.meta["current"]
    base = run.load_net()
    if not base.is_mlp:
        raise CLIError("resize needs an MLP")
    i = a.layer
    if i >= len(base.layers) - 1:
        raise CLIError("only hidden layers can be resized (the output size is fixed by the task)")
    net = base.copy()
    old, new = base.layers[i]["out"], a.units
    W, b, W2 = net.W[f"W{i}"], net.W[f"b{i}"], net.W[f"W{i+1}"]
    if new >= old:
        W = np.hstack([W, np.zeros((W.shape[0], new - old))])
        b = np.concatenate([b, np.zeros(new - old)])
        W2 = np.vstack([W2, np.zeros((new - old, W2.shape[1]))])
    else:
        W, b, W2 = W[:, :new], b[:new], W2[:new, :]
    net.W[f"W{i}"], net.W[f"b{i}"], net.W[f"W{i+1}"] = W, b, W2
    net.spec["layers"][i]["out"] = new
    net.spec["layers"][i + 1]["in"] = new
    v = _save_child(run, parent_v, net, "resize", a.note or f"layer {i}: {old} -> {new} units")
    run.emit(f"created {v}: layer {i} resized {old} -> {new} units "
             f"({'new units are zero' if new > old else 'dropped the last units'}) -> {net.arch_str()}")


def _subset(X, y, n, seed):
    if n is None or n >= len(X):
        return X, y, True
    idx = np.random.default_rng(seed).choice(len(X), n, replace=False)
    return X[idx], y[idx], False


def cmd_eval(a):
    run = _run(a.run)
    v = a.version or run.meta["current"]
    net = run.load_net(v)
    if not net.is_mlp:
        raise CLIError("cannot run an opaque network; use inspect/slice/svd and export it to evaluate elsewhere")
    X, y = run.load_split(a.split)
    X, y, full = _subset(X, y, a.n, a.seed)
    meter.charge_forward(run.session)
    run.save()
    out, (pre, post) = forward(net, X, capture=True)
    pred, prob = to_prediction(net.spec, out)
    k = _n_classes(run)
    m = compute_metrics(run.task, y, pred, prob, k)
    run.log_event({"event": "eval", "version": v, "split": a.split, "n": int(len(X)), "full": full, "metrics": m})
    key = primary_metric(run.task)
    lines = [f"[forward pass {run.session['forward_used']}/{run.session['forward_budget']}] {v} on {a.split} "
             f"({len(X)} samples{'' if full else ', subset'}): " + ", ".join(f"{k_}={v_:.4f}" for k_, v_ in m.items())]
    parent = next((e.get("parent") for e in run.history() if e.get("event") == "create" and e["version"] == v), None)
    scores = run.val_scores()
    if a.split == "val" and parent in scores and full:
        lines.append(f"delta vs parent {parent}: {key} {m[key] - scores[parent]:+.4f}")
    best = run.best_version()
    if best in scores:
        lines.append(f"best so far: {best} val {key}={scores[best]:.4f}")
    if run.task == "classification" and k and k <= 12:
        kk = k if net.layers[-1]["out"] > 1 else 2
        cm = confusion_matrix(y, pred, kk)
        cls = run.meta["classes"]
        lines.append("confusion (rows=true, cols=pred): " + " | ".join(
            f"{cls[i]}: " + " ".join(str(c) for c in cm[i]) for i in range(kk)))
    worst = worst_examples(run.task, y, pred, prob, a.worst)
    if len(worst):
        names = run.meta["feature_names"]
        norm = run.meta.get("input_norm")
        mu = np.asarray(norm["mean"]) if norm else X.mean(axis=0)
        sd = np.asarray(norm["std"]) if norm else np.where(X.std(axis=0) < 1e-12, 1, X.std(axis=0))
        lines.append(f"worst {len(worst)} examples (raw feature values; most extreme features first if many):")
        for i in worst:
            if run.task == "classification":
                cls = run.meta["classes"]
                conf = ""
                if prob is not None:
                    conf = f" p={prob[i]:.2f}" if prob.ndim == 1 else f" p={prob[i, pred[i]]:.2f}"
                head = f"  #{i}: true={cls[int(y[i])]} pred={cls[int(pred[i])]}{conf}"
            else:
                head = f"  #{i}: true={np.round(y[i], 4)} pred={np.round(pred[i], 4)}"
            if len(names) <= a.max_features:
                fidx = range(len(names))
            else:
                fidx = np.argsort(-np.abs((X[i] - mu) / sd))[:a.max_features]
            lines.append(head + " | " + " ".join(f"{names[j]}={X[i, j]:.4g}" for j in fidx))
            if a.activations:
                for li, h in enumerate(post[:-1]):
                    lines.append(f"      h{li}: " + " ".join(f"{x:.2f}" for x in h[i, :16])
                                 + (" ..." if h.shape[1] > 16 else ""))
                lines.append("      out: " + " ".join(f"{x:.3f}" for x in out[i]))
    elif run.task == "classification" and m["accuracy"] == 1.0:
        lines.append("no misclassified examples in this batch")
    run.emit("\n".join(lines), numeric=False)


def cmd_inspect(a):
    run = _run(a.run)
    net = run.load_net(a.version)
    if a.what == "stats":
        run.emit(stats_text(net))
    elif a.what == "unit":
        if a.layer is None or a.unit is None:
            raise CLIError("inspect unit needs --layer and --unit")
        units = [int(u) for u in str(a.unit).split(",")]
        run.emit("\n".join(unit_text(net, a.layer, u, a.top) for u in units))
    elif a.what == "slice":
        if not a.tensor:
            raise CLIError("inspect slice needs --tensor")
        run.emit(slice_text(net, a.tensor, a.rows, a.cols, a.precision), numeric=True)
    elif a.what == "svd":
        if not a.tensor:
            raise CLIError("inspect svd needs --tensor")
        run.emit(svd_text(net, a.tensor, a.k))
    elif a.what == "probe":
        if not net.is_mlp:
            raise CLIError("probe needs a runnable MLP")
        X, y = run.load_split(a.split)
        X, y, _ = _subset(X, y, a.n, a.seed)
        meter.charge_forward(run.session)
        run.save()
        run.log_event({"event": "probe", "version": a.version or run.meta["current"], "split": a.split})
        run.emit(f"[forward pass {run.session['forward_used']}/{run.session['forward_budget']}] "
                 + probe_text(net, X, y, run.task, a.layer, a.max_units))


def cmd_history(a):
    run = _run(a.run)
    scores = run.val_scores()
    key = primary_metric(run.task)
    cur, best = run.meta.get("current"), run.best_version()
    lines = [f"{'ver':<6}{'parent':<8}{'op':<9}{'val ' + key:>12}  arch / note"]
    for e in run.history():
        if e.get("event") != "create":
            continue
        s = scores.get(e["version"])
        mark = (" *current" if e["version"] == cur else "") + (" *best" if e["version"] == best else "")
        lines.append(f"{e['version']:<6}{(e.get('parent') or '-'):<8}{e['op']:<9}"
                     f"{(f'{s:.4f}' if s is not None else '-'):>12}  {e.get('arch')} | {e.get('note', '')[:80]}{mark}")
    run.emit("\n".join(lines))


def cmd_revert(a):
    run = _run(a.run)
    if a.version not in run.versions():
        raise CLIError(f"unknown version {a.version}; have {run.versions()}")
    run.meta["current"] = a.version
    run.save()
    run.log_event({"event": "revert", "version": a.version})
    run.emit(f"current version is now {a.version} (new edits branch from it)")


def cmd_best(a):
    run = _run(a.run)
    b = run.best_version()
    s = run.val_scores().get(b)
    if a.set and b:
        run.meta["current"] = b
        run.save()
    run.emit(f"best={b}" + (f" val {primary_metric(run.task)}={s:.4f}" if s is not None else " (no full val evals yet)")
             + (" (now current)" if a.set else ""))


def cmd_fit(a):
    run = _run(a.run)
    if run.mode != "hybrid":
        raise CLIError("`fit` (gradient training) is disabled in pure weight-space-programming mode. "
                       "Write weights by reasoning (`wsp write`/`patch`/`set`) instead.")
    from .train import fit
    parent_v = a.base or run.meta["current"]
    base = run.load_net(parent_v)
    if not base.is_mlp:
        raise CLIError("fit needs an MLP")
    meter.charge_forward(run.session)
    run.save()
    Xtr, ytr = run.load_split("train")
    Xva, yva = run.load_split("val")
    freeze = [int(f) for f in a.freeze.split(",")] if a.freeze else []
    net, curve, best_ep = fit(base, Xtr, ytr, Xva, yva, epochs=a.epochs, lr=a.lr, batch=a.batch,
                              weight_decay=a.wd, freeze=freeze, patience=a.patience, seed=a.seed)
    v = _save_child(run, parent_v, net, "fit", a.note or f"Adam lr={a.lr} epochs<={a.epochs} freeze={freeze}")
    out, _ = forward(net, Xva)
    pred, prob = to_prediction(net.spec, out)
    m = compute_metrics(run.task, yva, pred, prob, _n_classes(run))
    run.log_event({"event": "eval", "version": v, "split": "val", "n": int(len(Xva)), "full": True, "metrics": m})
    step = max(1, len(curve) // 8)
    lines = [f"[forward pass {run.session['forward_used']}/{run.session['forward_budget']}] gradient fit "
             f"{parent_v} -> {v}: best epoch {best_ep}/{len(curve)} (early stopping on val)",
             "epoch  train_loss  val_" + primary_metric(run.task)]
    for ep, loss, val in curve[::step]:
        lines.append(f"{ep:5d}  {loss:10.4f}  {val:.4f}")
    lines.append(f"{v} val: " + ", ".join(f"{k}={x:.4f}" for k, x in m.items()))
    lines.append(_changed_summary(base, net))
    run.emit("\n".join(lines))


def cmd_import(a):
    from .io_import import import_model
    run = _run(a.run)
    acts = [s.strip() for s in a.act.split(",")] if a.act else None
    net, notes = import_model(a.model, run.base_spec(), acts, trust=a.trust, keep_run_norm=a.keep_run_norm)
    v = run.save_net(net, "import", a.note or f"imported {Path(a.model).name}", parent="")
    if net.is_mlp:
        _check_io(run, net)
    run.meta["source"] = {**(run.meta.get("source") or {}), "imported_model": str(Path(a.model).resolve())}
    run.save()
    run.emit(f"imported {a.model} as {v}: {net.arch_str()}, {net.n_params()} params\n"
             + "\n".join(f"note: {n}" for n in notes) + "\n\n" + stats_text(net))


def cmd_export(a):
    from .io_import import export_model
    run = _run(a.run)
    net = run.load_net(a.version)
    run.emit(export_model(net, a.out, trust=a.trust))


def cmd_report(a):
    from .report import make_report
    run = _run(a.run)
    text = make_report(run, a.version, with_baseline=not a.no_baseline)
    print(text)
    print(f"(written to {run.path / 'report.md'})")


# ------------------------------------------------------------------------------------------------
# argument parsing
# ------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wsp", description="Weight-space programming toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_, description=help_)
        sp.set_defaults(fn=fn)
        return sp

    sp = add("new", cmd_new, "create a run from a data file or a task generator")
    sp.add_argument("name")
    sp.add_argument("--data"); sp.add_argument("--target"); sp.add_argument("--drop")
    sp.add_argument("--task", help="python file defining generate(n, seed) -> (X, y)")
    sp.add_argument("--n", type=int, default=3000, help="samples to generate for --task")
    sp.add_argument("--task-type", choices=["classification", "regression"])
    sp.add_argument("--mode", choices=["pure", "hybrid"], default="pure")
    sp.add_argument("--budget", type=int, default=None, help="forward passes per session (default 50)")
    sp.add_argument("--context-tokens", type=int, default=None, help="context window size (default 1M or $WSP_CONTEXT_TOKENS)")
    g = sp.add_mutually_exclusive_group()
    g.add_argument("--norm", dest="norm", action="store_true", default=None, help="standardize inputs (default for tables)")
    g.add_argument("--no-norm", dest="norm", action="store_false", help="raw inputs (default for tasks)")
    sp.add_argument("--seed", type=int, default=0)

    sp = add("session", cmd_session, "start a new agent session: reset forward-pass budget and context ledger")
    sp.add_argument("run"); sp.add_argument("--budget", type=int); sp.add_argument("--context-tokens", type=int)

    sp = add("status", cmd_status, "show run state, budgets, and whether a full weight dump fits")
    sp.add_argument("run")

    sp = add("profile", cmd_profile, "descriptive statistics of the training data")
    sp.add_argument("run"); sp.add_argument("--max-features", type=int, default=40)

    sp = add("sample", cmd_sample, "show random raw data rows")
    sp.add_argument("run"); sp.add_argument("-n", type=int, default=15)
    sp.add_argument("--split", choices=["train", "val"], default="train")
    sp.add_argument("--max-features", type=int, default=24); sp.add_argument("--seed", type=int, default=0)

    sp = add("size", cmd_size, "candidate architectures with params and context cost")
    sp.add_argument("run")

    sp = add("init", cmd_init, "create v000 with a chosen architecture (zero weights in pure mode)")
    sp.add_argument("run"); sp.add_argument("--arch", required=True, help="e.g. 4-8-3 (input size may be omitted: 8-3)")
    sp.add_argument("--act", help="comma list per layer, or one hidden activation")
    sp.add_argument("--random", action="store_true", help="random init (hybrid only)")
    sp.add_argument("--seed", type=int, default=0); sp.add_argument("--note")

    sp = add("dump", cmd_dump, "write weights as text (chunked files) for reading")
    sp.add_argument("run"); sp.add_argument("--version"); sp.add_argument("--layer", type=int)
    sp.add_argument("--tensor", help="comma list of tensor names"); sp.add_argument("--rows"); sp.add_argument("--cols")
    sp.add_argument("--precision", type=int, default=3); sp.add_argument("--print", action="store_true")

    sp = add("write", cmd_write, "run a weight program (numpy/math only, no data) to create a new version")
    sp.add_argument("run"); sp.add_argument("program"); sp.add_argument("--base", help="version to start from")
    sp.add_argument("--note")

    sp = add("set", cmd_set, "edit tensor entries")
    sp.add_argument("run"); sp.add_argument("--tensor", required=True)
    sp.add_argument("--index", help="e.g. 2,3 or 0:4,1 or :,5"); sp.add_argument("--value", type=float)
    sp.add_argument("--add", type=float); sp.add_argument("--mul", type=float); sp.add_argument("--note")

    sp = add("patch", cmd_patch, "apply a JSON list of edits [{tensor, index, value|add|mul}]")
    sp.add_argument("run"); sp.add_argument("edits"); sp.add_argument("--note")

    sp = add("resize", cmd_resize, "change a hidden layer's width (new units start at zero)")
    sp.add_argument("run"); sp.add_argument("--layer", type=int, required=True)
    sp.add_argument("--units", type=int, required=True); sp.add_argument("--note")

    sp = add("eval", cmd_eval, "METERED forward pass: metrics, confusion, worst examples")
    sp.add_argument("run"); sp.add_argument("--version"); sp.add_argument("--split", choices=["train", "val"], default="val")
    sp.add_argument("--n", type=int, default=None, help="subset size (default: whole split)")
    sp.add_argument("--worst", type=int, default=8); sp.add_argument("--activations", action="store_true")
    sp.add_argument("--max-features", type=int, default=12); sp.add_argument("--seed", type=int, default=0)

    sp = add("inspect", cmd_inspect, "sandbox inspection: stats | unit | slice | svd (free), probe (metered)")
    sp.add_argument("run"); sp.add_argument("what", choices=["stats", "unit", "slice", "svd", "probe"])
    sp.add_argument("--version"); sp.add_argument("--layer", type=int); sp.add_argument("--unit")
    sp.add_argument("--tensor"); sp.add_argument("--rows"); sp.add_argument("--cols")
    sp.add_argument("--precision", type=int, default=3); sp.add_argument("--top", type=int, default=12)
    sp.add_argument("--k", type=int, default=5); sp.add_argument("--split", choices=["train", "val"], default="train")
    sp.add_argument("--n", type=int, default=512); sp.add_argument("--max-units", type=int, default=64)
    sp.add_argument("--seed", type=int, default=0)

    sp = add("history", cmd_history, "list versions with val scores")
    sp.add_argument("run")
    sp = add("revert", cmd_revert, "make an earlier version current")
    sp.add_argument("run"); sp.add_argument("version")
    sp = add("best", cmd_best, "show (and optionally select) the best version by val score")
    sp.add_argument("run"); sp.add_argument("--set", action="store_true")

    sp = add("fit", cmd_fit, "HYBRID ONLY: gradient training (Adam, early stopping on val)")
    sp.add_argument("run"); sp.add_argument("--base"); sp.add_argument("--epochs", type=int, default=200)
    sp.add_argument("--lr", type=float, default=1e-3); sp.add_argument("--batch", type=int, default=64)
    sp.add_argument("--wd", type=float, default=0.0); sp.add_argument("--freeze", help="comma list of layer indices")
    sp.add_argument("--patience", type=int, default=30); sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--note")

    sp = add("import", cmd_import, "import a trained network (.npz/.pkl/.joblib/.pt/.safetensors)")
    sp.add_argument("run"); sp.add_argument("model"); sp.add_argument("--act")
    sp.add_argument("--trust", action="store_true", help="allow loading pickle/joblib files")
    sp.add_argument("--keep-run-norm", action="store_true"); sp.add_argument("--note")

    sp = add("export", cmd_export, "export a version (.npz/.pt/.safetensors/.pkl)")
    sp.add_argument("run"); sp.add_argument("out"); sp.add_argument("--version"); sp.add_argument("--trust", action="store_true")

    sp = add("report", cmd_report, "FINAL: evaluate on the held-out test split and write report.md")
    sp.add_argument("run"); sp.add_argument("--version"); sp.add_argument("--no-baseline", action="store_true")
    return p


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = build_parser().parse_args(argv)
    try:
        args.fn(args)
        return 0
    except (CLIError, NetworkError, meter.BudgetExhausted, PermissionError, FileNotFoundError,
            FileExistsError, KeyError, ValueError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
