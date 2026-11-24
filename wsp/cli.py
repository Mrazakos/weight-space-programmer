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
from . import modalities as M
from .backend import forward, is_runnable, output_act, to_prediction
from .dump import build_dump, check_fits, chunk_text, estimate_full_dump_tokens, select_tensors
from .evaluate import compute_metrics, confusion_matrix, primary_metric, worst_examples
from .inspect_ import (probe_text, slice_text, stats_text, svd_text, tokens_text, unit_text, unit_text_net)
from .netspec import Network, NetworkError, init_weights, mlp_layers, parse_arch
from .program import ProgramRejected, run_program
from .run import Run, resolve, runs_root
from .sizing import (candidates, net_candidates, net_recommend, net_table_text, output_layer, recommend,
                     table_text)

MODES = ("auto", "pure", "hybrid")


class CLIError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
def _run(name: str) -> Run:
    return Run(resolve(name))


def _parse_index(s):
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


def _n_classes(run: Run):
    c = run.meta.get("classes")
    return len(c) if c else None


def _gradients_allowed(run: Run) -> bool:
    return run.mode in ("hybrid", "auto")


def _out_dim(net: Network):
    arch = net.spec.get("arch")
    if arch == "mlp":
        return net.layers[-1]["out"]
    if arch == "net":
        return net.layers[-1]["out"]
    return None


def _check_io(run: Run, net: Network) -> None:
    arch = net.spec.get("arch")
    if arch == "mlp":
        if run.modality != "tabular":
            raise CLIError(f"plain MLPs (--arch) are for tabular data; this run is {run.modality}: use --preset/--layers")
        if net.layers[0]["in"] != run.meta["n_in"]:
            raise CLIError(f"network expects {net.layers[0]['in']} inputs but the run's data has {run.meta['n_in']}")
    elif arch == "net":
        if list(net.spec["input"]["shape"]) != list(run.input_spec()["shape"]):
            raise CLIError(f"network input {net.spec['input']['shape']} != run input {run.input_spec()['shape']}")
    else:
        return
    out = _out_dim(net)
    if run.task == "classification":
        k = _n_classes(run)
        if not (out == k or (k == 2 and out == 1)):
            raise CLIError(f"classification with {k} classes needs {k} outputs (or 1 sigmoid output if binary), got {out}")
    elif out != run.meta.get("n_targets", 1):
        raise CLIError(f"regression with {run.meta.get('n_targets', 1)} target(s) needs that many outputs, got {out}")


def _changed_summary(old: Network, new: Network, limit: int = 30) -> str:
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
    if len(lines) > limit:
        lines = lines[:limit] + [f"  ... {len(lines) - limit} more tensors changed"]
    return "\n".join(lines) if lines else "  (no weights changed)"


def _session_line(run: Run) -> str:
    s = run.session
    return (f"session {s['index']}: forward passes {s['forward_used']}/{s['forward_budget']} used, "
            f"context ~{s['context_used']:,}/{s['context_limit']:,} tokens used "
            f"(window {s['context_window']:,}, 25% reserved)")


def _class_name(run: Run, i) -> str:
    return run.meta["classes"][int(i)] if run.task == "classification" else str(np.round(i, 4))


def _profile(run: Run) -> str:
    X, y = run.load_split("train")
    meta = {"task": run.task, "feature_names": run.meta["feature_names"], "classes": run.meta.get("classes"),
            "description": run.meta.get("description"), "notes": run.meta.get("notes", []),
            "text_level": run.meta.get("text_level")}
    if run.modality == "image":
        return M.image_profile(X, y, meta)
    if run.modality == "text":
        return M.text_profile(run.load_extra("train") or [], X, y, run.vocab(), meta)
    return D.profile_text(X, y, meta)


# ------------------------------------------------------------------------------------------------
# dataset construction
# ------------------------------------------------------------------------------------------------
def _build_dataset(a):
    """Returns (X, y, meta, extras: list|None, info: dict). info holds modality-specific settings."""
    if a.task:
        X, y, meta = D.load_task(a.task, a.n, a.seed) if not _task_returns_raw(a.task) else _load_raw_task(a)
        if a.task_type:
            meta["task"] = a.task_type
        return X, y, meta, meta.pop("extras", None), meta.pop("info", {"modality": "tabular"})
    p = Path(a.data)
    target = [t.strip() for t in (a.target or "").split(",") if t.strip()]
    if p.is_dir():
        paths, labels = M.scan_image_folder(str(p))
        y, task, classes = D.encode_target(__import__("pandas").DataFrame({"label": labels}), ["label"], "classification")
        return None, y, {"task": task, "classes": classes, "feature_names": [], "notes": []}, paths, \
            {"modality": "image", "paths": paths}
    if p.suffix.lower() == ".npz":
        with np.load(p, allow_pickle=False) as z:
            Xa, ya = z["X"], z["y"]
        if Xa.ndim >= 3 or a.modality == "image":
            X = M.array_to_images(Xa)
            import pandas as pd
            task = a.task_type
            y, task, classes = D.encode_target(pd.DataFrame({"y": ya}), ["y"], task)
            return X, y, {"task": task, "classes": classes, "feature_names": [], "notes": []}, None, \
                {"modality": "image"}
    df = D.load_table(str(p))
    if not target:
        raise CLIError("--target COLUMN is required with a table")
    df = df.dropna(subset=target).reset_index(drop=True)
    drop = [d.strip() for d in (a.drop or "").split(",") if d.strip()]
    if a.image_col:
        base = p.parent
        paths = [str((base / v).resolve()) if not Path(v).is_absolute() else v for v in df[a.image_col].astype(str)]
        y, task, classes = D.encode_target(df, target, a.task_type)
        return None, y, {"task": task, "classes": classes, "feature_names": [], "notes": []}, paths, \
            {"modality": "image", "paths": paths}
    text_col = a.text_col
    if text_col is None and a.modality in (None, "auto", "text"):
        text_col = M.detect_text_column(df.drop(columns=drop, errors="ignore"), target)
    if a.modality == "text" and not text_col:
        raise CLIError("no text column found; pass --text-col")
    if text_col and a.modality != "tabular":
        texts = df[text_col].fillna("").astype(str).tolist()
        y, task, classes = D.encode_target(df, target, a.task_type)
        notes = [f"text column {text_col!r}; other feature columns ignored"]
        return None, y, {"task": task, "classes": classes, "feature_names": [], "notes": notes}, texts, \
            {"modality": "text", "text_col": text_col}
    X, y, meta = D.encode_table(df, target, a.task_type, drop)
    return X, y, meta, None, {"modality": "tabular"}


def _task_returns_raw(path: str) -> bool:
    src = Path(path).read_text(encoding="utf-8")
    return "MODALITY" in src


def _load_raw_task(a):
    """Task generators may set MODALITY = 'image' (X: arrays) or 'text' (X: list of strings)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("wsp_task", a.task)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    X, y = mod.generate(a.n, a.seed)
    modality = getattr(mod, "MODALITY", "tabular")
    task = getattr(mod, "TASK_TYPE", "classification")
    y = np.asarray(y)
    classes = [str(c) for c in getattr(mod, "CLASSES", range(int(y.max()) + 1))] if task == "classification" else None
    meta = {"task": task, "classes": classes, "feature_names": [], "notes": [],
            "description": getattr(mod, "DESCRIPTION", "")}
    if modality == "image":
        meta["info"] = {"modality": "image"}
        return M.array_to_images(X), y.astype(int) if task == "classification" else y.astype(float), meta
    if modality == "text":
        meta["extras"] = [str(t) for t in X]
        meta["info"] = {"modality": "text"}
        return None, y.astype(int) if task == "classification" else y.astype(float), meta
    X, y, m2 = D.load_task(a.task, a.n, a.seed)
    return X, y, m2


# ------------------------------------------------------------------------------------------------
# commands
# ------------------------------------------------------------------------------------------------
def cmd_new(a):
    if bool(a.data) == bool(a.task):
        raise CLIError("give exactly one of --data FILE|FOLDER (with --target for tables) or --task task.py")
    X, y, meta, extras, info = _build_dataset(a)
    modality = info["modality"]
    idx = D.split_indices(y, meta["task"], seed=a.seed)
    path = Path(a.name) if ("/" in a.name or "\\" in a.name) else runs_root() / a.name
    run_meta = {"name": path.name, "mode": a.mode, "task": meta["task"], "classes": meta["classes"],
                "feature_names": meta["feature_names"], "modality": modality,
                "source": {"data": str(Path(a.data).resolve()), "target": a.target} if a.data
                else {"task": str(Path(a.task).resolve()), "n": a.n},
                "description": meta.get("description", ""), "notes": meta.get("notes", []),
                "input_norm": None, "output_norm": None}
    extra_splits = None
    vocab = None
    if modality == "image":
        if X is None:
            hw, gray = M.choose_image_format(extras, a.image_size, True if a.grayscale else None)
            X = M.read_images(extras, hw, gray)
        if a.image_size and X.shape[2:] != (a.image_size, a.image_size):
            raise CLIError("--image-size resizing is only applied to image files, not arrays")
        run_meta["input_shape"] = list(X.shape[1:])
        run_meta["n_in"] = int(np.prod(X.shape[1:]))
        run_meta["input_norm"] = M.image_norm_stats(X[idx["train"]]) if a.norm is not False else None
        if extras:
            extra_splits = {k: [extras[i] for i in v] for k, v in idx.items()}
    elif modality == "text":
        texts = extras
        level = a.text_level
        vocab = M.build_vocab([texts[i] for i in idx["train"]], level, a.max_vocab)
        max_len = a.max_len or M.choose_max_len([texts[i] for i in idx["train"]], level)
        X = M.encode_texts(texts, vocab, level, max_len)
        run_meta.update({"input_shape": [max_len], "n_in": max_len, "vocab_size": len(vocab), "text_level": level,
                         "text_col": info.get("text_col")})
        extra_splits = {k: [texts[i] for i in v] for k, v in idx.items()}
    else:
        run_meta["n_in"] = int(X.shape[1])
        norm = (True if a.data else False) if a.norm is None else a.norm
        if norm:
            run_meta["input_norm"] = D.norm_stats(X[idx["train"]])
    ytr = y[idx["train"]]
    run_meta["n_targets"] = 1 if meta["task"] == "classification" or ytr.ndim == 1 else int(ytr.shape[1])
    if meta["task"] == "regression" and (a.norm is not False):
        yy = ytr.reshape(len(ytr), -1)
        sd = yy.std(axis=0)
        run_meta["output_norm"] = {"mean": yy.mean(axis=0).tolist(), "std": np.where(sd < 1e-12, 1, sd).tolist()}
    run_meta["sizes"] = {k: int(len(v)) for k, v in idx.items()}
    run_meta["session"] = meter.new_session(None, a.budget, a.context_tokens)
    splits = {k: (X[v], y[v]) for k, v in idx.items()}
    run = Run.create(path, run_meta, splits, extra_splits)
    if vocab is not None:
        M.save_json(vocab, run.path / "vocab.json")
    if modality == "image" and meta["task"] == "classification":
        M.save_class_means(splits["train"][0], splits["train"][1], meta["classes"], run.path / "dumps" / "class_means.png")
    run.log_event({"event": "new", "source": run_meta["source"], "mode": a.mode, "modality": modality})
    run.emit(f"created run {run.path} (modality={modality}, mode={a.mode}; train/val/test = "
             f"{run_meta['sizes']['train']}/{run_meta['sizes']['val']}/{run_meta['sizes']['test']}; test is held out)\n"
             + _session_line(run) + "\n\n" + _profile(run))


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
    lines = [f"run {run.path}  mode={run.mode}  modality={run.modality}  task={run.task}  "
             f"input shape={run.input_spec()['shape']}"
             + (f"  classes={run.meta['classes']}" if run.meta.get("classes") and len(run.meta["classes"]) <= 20
                else (f"  classes={len(run.meta['classes'])}" if run.meta.get("classes") else "")),
             _session_line(run),
             f"gradient training: {'allowed' if _gradients_allowed(run) else 'DISABLED (pure mode)'}"]
    cur = run.meta.get("current")
    if cur:
        net = run.load_net()
        best = run.best_version()
        scores = run.val_scores()
        key = primary_metric(run.task)
        lines.append(f"current={cur} ({net.arch_str()}, {net.n_params():,} params)"
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
        if not is_runnable(net):
            lines.append("this network is OPAQUE: no forward passes possible")
    else:
        lines.append("no network yet: `wsp size` then `wsp init` (or `wsp import`)")
    run.emit("\n".join(lines))


def cmd_profile(a):
    run = _run(a.run)
    run.emit(_profile(run))


def cmd_sample(a):
    run = _run(a.run)
    X, y = run.load_split(a.split)
    rng = np.random.default_rng(a.seed)
    idx = rng.choice(len(X), size=min(a.n, len(X)), replace=False)
    if run.modality == "image":
        labels = [_class_name(run, y[i]) for i in idx]
        f = M.save_contact_sheet(X[idx], labels, run.path / "dumps" / f"sample_{a.split}_{a.seed}.png")
        extra = run.load_extra(a.split)
        lines = [f"wrote a contact sheet of {len(idx)} random {a.split} images with labels: {f}",
                 "Read it to look at the images." + (f" Class means: {run.path / 'dumps' / 'class_means.png'}"
                                                     if run.task == "classification" else "")]
        if extra:
            lines.append("files: " + ", ".join(f"#{i}={Path(extra[i]).name}" for i in idx[:12]))
        run.emit("\n".join(lines))
        return
    if run.modality == "text":
        texts = run.load_extra(a.split) or []
        lines = [f"{len(idx)} random {a.split} texts"]
        for i in idx:
            lines.append(f"#{i} [{_class_name(run, y[i])}] {texts[i][:a.max_chars]!r}")
        run.emit("\n".join(lines))
        return
    names = run.meta["feature_names"]
    cols = list(range(min(len(names), a.max_features)))
    lines = [f"{len(idx)} random {a.split} rows (raw feature values)",
             "idx | target | " + " ".join(names[c][:10] for c in cols) + (" ..." if len(names) > len(cols) else "")]
    for i in idx:
        lines.append(f"{i} | {_class_name(run, y[i])} | " + " ".join(f"{X[i, c]:.4g}" for c in cols))
    run.emit("\n".join(lines), numeric=True)


def cmd_size(a):
    run = _run(a.run)
    n_train = run.meta["sizes"]["train"]
    if run.modality == "tabular":
        rows = candidates(run.meta["n_in"], run.task, _n_classes(run), run.meta.get("n_targets", 1), n_train,
                          meter.context_remaining(run.session), run.mode)
        rec = recommend(rows, "pure" if run.mode == "pure" else "hybrid")
        txt = table_text(rows, rec, run.meta, n_train, run.mode)
        txt += ("\n - Tabular networks use `init --arch` (numpy MLP, exact hand-editing conventions); "
                "`init --preset mlp --width W --depth D` builds the same thing as a torch layer graph.")
        run.emit(txt)
        return
    rows = net_candidates(run.input_spec(), run.task, _n_classes(run), run.meta.get("n_targets", 1), n_train,
                          meter.context_remaining(run.session))
    rec = net_recommend(rows, run.modality, run.mode, n_train)
    run.emit(net_table_text(rows, rec, run.modality, run.mode, n_train, run.meta["name"]))


def cmd_init(a):
    run = _run(a.run)
    if run.mode == "pure" and a.random:
        raise CLIError("pure mode starts from zero weights (no experience); random init is not allowed")
    out_dim, out_act = output_layer(run.task, _n_classes(run), run.meta.get("n_targets", 1))
    if a.out_act:
        out_act = a.out_act
    given = [x for x in (a.arch, a.preset, a.layers) if x]
    if len(given) != 1:
        raise CLIError("give exactly one of --arch (tabular MLP), --preset NAME, or --layers FILE.json")
    if a.arch:
        if run.modality != "tabular":
            raise CLIError(f"--arch builds a plain MLP for tabular data; this run is {run.modality}: "
                           "use --preset (see `wsp size`) or --layers")
        sizes = parse_arch(a.arch)
        if sizes[0] != run.meta["n_in"]:
            sizes = [run.meta["n_in"]] + sizes
        nl = len(sizes) - 1
        if a.act:
            acts = [s.strip() for s in a.act.split(",")]
            if len(acts) == 1 and nl > 1:
                acts = acts * (nl - 1) + [out_act]
        else:
            acts = ["relu"] * (nl - 1) + [out_act]
        layers = mlp_layers(sizes, acts)
        net = Network({**run.base_spec(), "layers": layers}, init_weights(layers, zero=not a.random, seed=a.seed))
    else:
        from .graph import assign_names, init_params, preset_layers
        if a.preset:
            kw = {k: getattr(a, k) for k in ("width", "depth", "dim", "heads", "blocks", "hidden")
                  if getattr(a, k) is not None}
            layers = preset_layers(a.preset, run.input_spec(), out_dim, out_act, **kw)
        else:
            obj = json.loads(Path(a.layers).read_text(encoding="utf-8"))
            layers = obj["layers"] if isinstance(obj, dict) else obj
        spec = {**run.base_spec(), "arch": "net", "layers": assign_names(layers)}
        net = Network(spec, init_params(spec, zero=not a.random, seed=a.seed))
    _check_io(run, net)
    v = run.save_net(net, "init", a.note or ("zero init" if not a.random else f"random init seed={a.seed}"), parent="")
    run.emit(f"created {v}: {net.arch_str()}\n{net.n_params():,} params, "
             f"{'all zeros' if not a.random else 'random'}{' (LayerNorm gains = 1)' if not a.random and any(k.endswith('.g') for k in net.W) else ''}\n"
             f"full dump ~{estimate_full_dump_tokens(net):,} tokens")


def cmd_dump(a):
    run = _run(a.run)
    net = run.load_net(a.version)
    v = a.version or run.meta["current"]
    layer = a.layer
    if layer is not None and net.is_mlp:
        layer = int(layer)
    tensors = select_tensors(net, layer, [t.strip() for t in a.tensor.split(",")] if a.tensor else None)
    text = build_dump(net, tensors, a.precision, a.rows, a.cols, run.vocab())
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
    tag = (f"L{layer}" if layer is not None else (a.tensor.replace(",", "+")[:40] if a.tensor else "all"))
    if a.rows or a.cols:
        tag += "_slice"
    files = []
    for i, c in enumerate(chunks, 1):
        f = run.path / "dumps" / f"{v}_{tag}_part{i:02d}.txt"
        f.write_text(c, encoding="utf-8")
        files.append(f)
    meter.charge_tokens(run.session, tok)
    run.save()
    run.emit(f"wrote {len(files)} dump file(s), ~{tok:,} tokens total (charged to the context ledger). "
             "Read each part in order:\n" + "\n".join(f"  {f}" for f in files))


def _save_child(run: Run, parent_v: str, net: Network, op: str, note: str) -> str:
    _check_io(run, net)
    return run.save_net(net, op, note, parent=parent_v)


def cmd_write(a):
    run = _run(a.run)
    parent_v = a.base or run.meta["current"]
    if not parent_v:
        raise CLIError("no network to write into; run `wsp init` first")
    base = run.load_net(parent_v)
    src = Path(a.program).read_text(encoding="utf-8")
    prog_spec = dict(base.spec)
    if run.modality == "text":
        prog_spec["vocab"] = run.vocab()
    try:
        W, upd = run_program(src, prog_spec, base.W)
    except ProgramRejected as e:
        raise CLIError(str(e))
    spec = dict(base.spec)
    if upd and "layers" in upd:
        if spec.get("arch") == "net":
            from .graph import assign_names
            spec["layers"] = assign_names(upd["layers"])
        elif spec.get("arch") == "mlp":
            spec["layers"] = [{"in": int(L["in"]), "out": int(L["out"]), "act": L["act"]} for L in upd["layers"]]
        else:
            raise CLIError("this network's architecture cannot be changed by a program")
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
        raise CLIError("resize works on plain MLPs; for layer graphs, write a program that returns "
                       "(W, {'layers': [...]}) with the new layer sizes")
    i = int(a.layer)
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
        return X, y, np.arange(len(X)), True
    idx = np.sort(np.random.default_rng(seed).choice(len(X), n, replace=False))
    return X[idx], y[idx], idx, False


def _describe_example(run: Run, X, i, orig_i, extra, mu, sd, max_features) -> str:
    if run.modality == "image":
        return f"file={Path(extra[orig_i]).name}" if extra else f"row {orig_i}"
    if run.modality == "text":
        t = extra[orig_i] if extra else ""
        return repr(t[:220] + ("..." if len(t) > 220 else ""))
    names = run.meta["feature_names"]
    if len(names) <= max_features:
        fidx = range(len(names))
    else:
        fidx = np.argsort(-np.abs((X[i] - mu) / sd))[:max_features]
    return " ".join(f"{names[j]}={X[i, j]:.4g}" for j in fidx)


def cmd_eval(a):
    run = _run(a.run)
    v = a.version or run.meta["current"]
    net = run.load_net(v)
    if not is_runnable(net):
        raise CLIError("cannot run an opaque network; use inspect/slice/svd, or re-import it as TorchScript")
    X, y = run.load_split(a.split)
    X, y, orig, full = _subset(X, y, a.n, a.seed)
    meter.charge_forward(run.session)
    run.save()
    out, caps = forward(net, X, capture=a.activations)
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
        kk = k if out.shape[1] > 1 else 2
        cm = confusion_matrix(y, pred, kk)
        cls = run.meta["classes"]
        lines.append("confusion (rows=true, cols=pred): " + " | ".join(
            f"{cls[i]}: " + " ".join(str(c) for c in cm[i]) for i in range(kk)))
    elif run.task == "classification":
        cm = confusion_matrix(y, pred, k)
        np.fill_diagonal(cm, 0)
        flat = np.argsort(-cm.ravel())[:8]
        lines.append("most frequent confusions: " + ", ".join(
            f"{run.meta['classes'][i // k]}->{run.meta['classes'][i % k]} x{cm.ravel()[i]}" for i in flat if cm.ravel()[i]))
    worst = worst_examples(run.task, y, pred, prob, a.worst)
    if len(worst):
        extra = run.load_extra(a.split)
        mu = sd = None
        if run.modality == "tabular":
            norm = run.meta.get("input_norm")
            mu = np.asarray(norm["mean"]) if norm else X.mean(axis=0)
            sd = np.asarray(norm["std"]) if norm else np.where(X.std(axis=0) < 1e-12, 1, X.std(axis=0))
        lines.append(f"worst {len(worst)} examples:")
        for i in worst:
            if run.task == "classification":
                conf = ""
                if prob is not None:
                    conf = f" p={prob[i]:.2f}" if prob.ndim == 1 else f" p={prob[i, pred[i]]:.2f}"
                head = f"  #{orig[i]}: true={_class_name(run, y[i])} pred={_class_name(run, pred[i])}{conf}"
            else:
                head = f"  #{orig[i]}: true={np.round(y[i], 4)} pred={np.round(pred[i], 4)}"
            lines.append(head + " | " + _describe_example(run, X, i, orig[i], extra, mu, sd, a.max_features))
            if a.activations and caps:
                for name, h in caps[-3:]:
                    lines.append(f"      {name}: " + " ".join(f"{x:.2f}" for x in h[i, :16])
                                 + (" ..." if h.shape[1] > 16 else ""))
                lines.append("      out: " + " ".join(f"{x:.3f}" for x in out[i, :16]))
        if run.modality == "image":
            labels = [f"t:{_class_name(run, y[i])[:6]} p:{_class_name(run, pred[i])[:6]}" for i in worst]
            f = M.save_contact_sheet(X[worst], labels, run.path / "dumps" / f"{v}_worst_{a.split}.png")
            lines.append(f"worst images (t=true, p=pred): {f}  <- Read it to see them")
    elif run.task == "classification" and m["accuracy"] == 1.0:
        lines.append("no misclassified examples in this batch")
    run.emit("\n".join(lines))


def cmd_inspect(a):
    run = _run(a.run)
    net = run.load_net(a.version)
    if a.what == "stats":
        run.emit(stats_text(net))
    elif a.what == "unit":
        if a.layer is None or a.unit is None:
            raise CLIError("inspect unit needs --layer and --unit")
        units = [int(u) for u in str(a.unit).split(",")]
        if net.spec.get("arch") == "net":
            run.emit("\n".join(unit_text_net(net, a.layer, u, a.top, run.vocab()) for u in units))
        elif net.is_mlp:
            run.emit("\n".join(unit_text(net, int(a.layer), u, a.top) for u in units))
        else:
            raise CLIError("unit inspection needs an MLP or layer graph; use slice/svd")
    elif a.what == "tokens":
        if not a.words:
            raise CLIError("inspect tokens needs --words w1,w2")
        run.emit(tokens_text(net, [w.strip() for w in a.words.split(",")], run.vocab()))
    elif a.what == "slice":
        if not a.tensor:
            raise CLIError("inspect slice needs --tensor")
        run.emit(slice_text(net, a.tensor, a.rows, a.cols, a.precision, run.vocab()), numeric=True)
    elif a.what == "svd":
        if not a.tensor:
            raise CLIError("inspect svd needs --tensor")
        run.emit(svd_text(net, a.tensor, a.k))
    elif a.what == "probe":
        if not is_runnable(net):
            raise CLIError("probe needs a runnable network")
        X, y = run.load_split(a.split)
        X, y, _, _ = _subset(X, y, a.n, a.seed)
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
                     f"{(f'{s:.4f}' if s is not None else '-'):>12}  {str(e.get('arch'))[:90]} | "
                     f"{e.get('note', '')[:80]}{mark}")
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
    if not _gradients_allowed(run):
        raise CLIError("`fit` (gradient training) is disabled in pure weight-space-programming mode. "
                       "Write weights by reasoning (`wsp write`/`patch`/`set`) instead.")
    from .backend import fit
    parent_v = a.base or run.meta["current"]
    base = run.load_net(parent_v)
    if not is_runnable(base):
        raise CLIError("cannot train an opaque network")
    meter.charge_forward(run.session)
    run.save()
    Xtr, ytr = run.load_split("train")
    Xva, yva = run.load_split("val")
    freeze = [f.strip() for f in a.freeze.split(",")] if a.freeze else []
    mlp = base.is_mlp
    epochs = a.epochs or (200 if mlp else 30)
    patience = a.patience or (30 if mlp else 5)
    net, curve, best_ep = fit(base, Xtr, ytr, Xva, yva, epochs=epochs, lr=a.lr, batch=a.batch,
                              weight_decay=a.wd, freeze=freeze, patience=patience, seed=a.seed,
                              max_minutes=a.max_minutes)
    v = _save_child(run, parent_v, net, "fit", a.note or f"Adam lr={a.lr} epochs<={epochs} freeze={freeze}")
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
    lines.append(_changed_summary(base, net, limit=12))
    run.emit("\n".join(lines))


def cmd_import(a):
    from .io_import import import_model
    run = _run(a.run)
    acts = [s.strip() for s in a.act.split(",")] if a.act else None
    net, notes = import_model(a.model, run.base_spec(), acts, trust=a.trust, keep_run_norm=a.keep_run_norm,
                              output_act=a.output_act)
    if net.is_mlp:
        _check_io(run, net)
    v = run.save_net(net, "import", a.note or f"imported {Path(a.model).name}", parent="")
    run.meta["source"] = {**(run.meta.get("source") or {}), "imported_model": str(Path(a.model).resolve())}
    run.save()
    run.emit(f"imported {a.model} as {v}: {net.arch_str()}, {net.n_params():,} params\n"
             + "\n".join(f"note: {n}" for n in notes) + "\n\n" + stats_text(net))


def cmd_export(a):
    from .io_import import export_model
    run = _run(a.run)
    net = run.load_net(a.version)
    run.emit(export_model(net, a.out, trust=a.trust))


def _load_new_inputs(run: Run, path: str, col: str | None):
    """Encode new data with the run's preprocessing. Returns (X, ids)."""
    p = Path(path)
    if run.modality == "image":
        if p.is_dir():
            files = sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in M.IMG_EXT)
        elif p.suffix.lower() in M.IMG_EXT:
            files = [str(p)]
        else:
            import pandas as pd
            df = D.load_table(str(p))
            c = col or df.columns[0]
            files = [str((p.parent / v).resolve()) if not Path(v).is_absolute() else v for v in df[c].astype(str)]
        if not files:
            raise CLIError(f"no images found at {p}")
        shape = run.meta["input_shape"]
        X = M.read_images(files, (shape[1], shape[2]), shape[0] == 1)
        return X, files
    if run.modality == "text":
        if p.suffix.lower() == ".txt":
            texts = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        else:
            df = D.load_table(str(p))
            c = col or run.meta.get("text_col") or df.columns[0]
            texts = df[c].fillna("").astype(str).tolist()
        X = M.encode_texts(texts, run.vocab(), run.meta["text_level"], run.meta["input_shape"][0])
        return X, [t[:80] for t in texts]
    df = D.load_table(str(p))
    norm = run.meta.get("input_norm")
    X, notes = D.encode_features(df, run.meta["feature_names"], norm["mean"] if norm else None)
    for n in notes[:10]:
        print(f"note: {n}", file=sys.stderr)
    return X, list(range(len(df)))


def cmd_predict(a):
    run = _run(a.run)
    net = run.load_net(a.version or run.best_version())
    if not is_runnable(net):
        raise CLIError("this network cannot be run")
    X, ids = _load_new_inputs(run, a.input, a.col)
    out, _ = forward(net, X)
    pred, prob = to_prediction(net.spec, out)
    import pandas as pd
    df = pd.DataFrame({"input": ids})
    if run.task == "classification":
        classes = run.meta["classes"]
        df["prediction"] = [classes[int(i)] for i in pred]
        if prob is not None:
            if prob.ndim == 1:
                df[f"p_{classes[1]}"] = prob
            else:
                for j, c in enumerate(classes):
                    df[f"p_{c}"] = prob[:, j]
    else:
        p2 = np.asarray(pred).reshape(len(pred), -1)
        for j in range(p2.shape[1]):
            df["prediction" if p2.shape[1] == 1 else f"prediction_{j}"] = p2[:, j]
    if a.out:
        df.to_csv(a.out, index=False)
        print(f"wrote {len(df)} predictions to {a.out}")
    else:
        print(df.head(a.show).to_string(index=False))
        if len(df) > a.show:
            print(f"... {len(df) - a.show} more (use --out preds.csv)")


def cmd_report(a):
    from .report import make_report
    run = _run(a.run)
    text = make_report(run, a.version, with_baseline=not a.no_baseline)
    print(text)
    print(f"(written to {run.path / 'report.md'})")


# ------------------------------------------------------------------------------------------------
# argument parsing
# ------------------------------------------------------------------------------------------------
LAYER_HELP = ("layer graph JSON: a list of layers, e.g. [{\"type\":\"conv2d\",\"name\":\"c1\",\"out_ch\":8,\"k\":3,"
              "\"act\":\"relu\"}, {\"type\":\"global_pool\",\"op\":\"mean\"}, {\"type\":\"dense\",\"name\":\"out\","
              "\"out\":10,\"act\":\"softmax\"}]. Types: dense conv2d maxpool2d avgpool2d flatten global_pool "
              "embedding posembed conv1d transformer layernorm act")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wsp", description="Weight-space programming toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_, description=help_)
        sp.set_defaults(fn=fn)
        return sp

    sp = add("new", cmd_new, "create a run from a table, an image folder, an array file, or a task generator")
    sp.add_argument("name")
    sp.add_argument("--data", help="csv/tsv/parquet/json/jsonl/xlsx/npz file, or a folder of images in class subfolders")
    sp.add_argument("--target", help="column(s) to predict (tables)")
    sp.add_argument("--drop", help="comma list of columns to ignore")
    sp.add_argument("--task", help="python file defining generate(n, seed) -> (X, y); set MODALITY='image'|'text' for raw data")
    sp.add_argument("--n", type=int, default=3000, help="samples to generate for --task")
    sp.add_argument("--task-type", choices=["classification", "regression"])
    sp.add_argument("--modality", choices=["auto", "tabular", "image", "text"], default="auto")
    sp.add_argument("--text-col", help="text column (auto-detected if omitted)")
    sp.add_argument("--text-level", choices=["word", "char"], default="word")
    sp.add_argument("--max-len", type=int, help="tokens per text (default: 95th percentile, capped)")
    sp.add_argument("--max-vocab", type=int, default=20000)
    sp.add_argument("--image-col", help="column holding image file paths (relative to the table)")
    sp.add_argument("--image-size", type=int, help="resize images to SIZExSIZE (default: keep if <=64, else 32/64)")
    sp.add_argument("--grayscale", action="store_true")
    sp.add_argument("--mode", choices=MODES, default="auto",
                    help="auto (default): program by reasoning where possible, gradients where needed; "
                         "pure: no gradients ever; hybrid: both, freely")
    sp.add_argument("--budget", type=int, default=None, help="forward passes per session (default 50)")
    sp.add_argument("--context-tokens", type=int, default=None, help="context window size (default 1M or $WSP_CONTEXT_TOKENS)")
    g = sp.add_mutually_exclusive_group()
    g.add_argument("--norm", dest="norm", action="store_true", default=None, help="standardize inputs (default for data files)")
    g.add_argument("--no-norm", dest="norm", action="store_false", help="raw inputs (default for tabular tasks)")
    sp.add_argument("--seed", type=int, default=0)

    sp = add("session", cmd_session, "start a new agent session: reset forward-pass budget and context ledger")
    sp.add_argument("run"); sp.add_argument("--budget", type=int); sp.add_argument("--context-tokens", type=int)

    sp = add("status", cmd_status, "show run state, budgets, and whether a full weight dump fits")
    sp.add_argument("run")

    sp = add("profile", cmd_profile, "descriptive statistics of the training data")
    sp.add_argument("run")

    sp = add("sample", cmd_sample, "show random examples (rows, texts, or an image contact sheet)")
    sp.add_argument("run"); sp.add_argument("-n", type=int, default=15)
    sp.add_argument("--split", choices=["train", "val"], default="train")
    sp.add_argument("--max-features", type=int, default=24); sp.add_argument("--max-chars", type=int, default=300)
    sp.add_argument("--seed", type=int, default=0)

    sp = add("size", cmd_size, "candidate architectures with params and context cost")
    sp.add_argument("run")

    sp = add("init", cmd_init, "create v000: --arch (tabular MLP) | --preset NAME | --layers FILE.json")
    sp.add_argument("run")
    sp.add_argument("--arch", help="tabular MLP sizes, e.g. 4-8-3 (input size may be omitted: 8-3)")
    sp.add_argument("--act", help="MLP activations: comma list per layer, or one hidden activation")
    sp.add_argument("--preset", help="image: linear|cnn  text: bow|cnn1d|transformer  tabular: mlp")
    sp.add_argument("--layers", help=LAYER_HELP)
    for k in ("width", "depth", "dim", "heads", "blocks", "hidden"):
        sp.add_argument(f"--{k}", type=int)
    sp.add_argument("--out-act", help="override the output activation")
    sp.add_argument("--random", action="store_true", help="random init (not in pure mode)")
    sp.add_argument("--seed", type=int, default=0); sp.add_argument("--note")

    sp = add("dump", cmd_dump, "write weights as text (chunked files) for reading")
    sp.add_argument("run"); sp.add_argument("--version"); sp.add_argument("--layer", help="MLP layer index or graph layer name")
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

    sp = add("resize", cmd_resize, "change an MLP hidden layer's width (new units start at zero)")
    sp.add_argument("run"); sp.add_argument("--layer", type=int, required=True)
    sp.add_argument("--units", type=int, required=True); sp.add_argument("--note")

    sp = add("eval", cmd_eval, "METERED forward pass: metrics, confusion, worst examples")
    sp.add_argument("run"); sp.add_argument("--version"); sp.add_argument("--split", choices=["train", "val"], default="val")
    sp.add_argument("--n", type=int, default=None, help="subset size (default: whole split)")
    sp.add_argument("--worst", type=int, default=8); sp.add_argument("--activations", action="store_true")
    sp.add_argument("--max-features", type=int, default=12); sp.add_argument("--seed", type=int, default=0)

    sp = add("inspect", cmd_inspect,
             "sandbox inspection: stats | unit | tokens | slice | svd (free), probe (metered)")
    sp.add_argument("run"); sp.add_argument("what", choices=["stats", "unit", "tokens", "slice", "svd", "probe"])
    sp.add_argument("--version"); sp.add_argument("--layer", help="MLP layer index or graph layer name")
    sp.add_argument("--unit"); sp.add_argument("--words", help="tokens: comma list of words")
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

    sp = add("fit", cmd_fit, "gradient training (auto/hybrid modes only): Adam, early stopping on val")
    sp.add_argument("run"); sp.add_argument("--base"); sp.add_argument("--epochs", type=int)
    sp.add_argument("--lr", type=float, default=1e-3); sp.add_argument("--batch", type=int, default=64)
    sp.add_argument("--wd", type=float, default=0.0)
    sp.add_argument("--freeze", help="comma list: MLP layer indices, or graph layer names / tensor prefixes")
    sp.add_argument("--patience", type=int); sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--max-minutes", type=float, default=10.0); sp.add_argument("--note")

    sp = add("import", cmd_import, "import a trained network (.npz/.pkl/.joblib/.pt/.ts/.safetensors)")
    sp.add_argument("run"); sp.add_argument("model"); sp.add_argument("--act")
    sp.add_argument("--output-act", help="torch modules: identity (logits, default) | softmax | sigmoid")
    sp.add_argument("--trust", action="store_true", help="allow loading pickles / full torch modules")
    sp.add_argument("--keep-run-norm", action="store_true"); sp.add_argument("--note")

    sp = add("export", cmd_export, "export a version (.npz / .pt state_dict / .ts TorchScript / .safetensors / .pkl)")
    sp.add_argument("run"); sp.add_argument("out"); sp.add_argument("--version"); sp.add_argument("--trust", action="store_true")

    sp = add("predict", cmd_predict, "run the best (or a given) version on NEW data with the run's preprocessing")
    sp.add_argument("run"); sp.add_argument("input", help="table / image file or folder / .txt (one text per line)")
    sp.add_argument("--col", help="column holding the text or image paths"); sp.add_argument("--version")
    sp.add_argument("--out", help="write predictions CSV"); sp.add_argument("--show", type=int, default=20)

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
