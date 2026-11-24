"""Loading, encoding, splitting and profiling data (tables or task generators)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

MAX_CATEGORIES = 50


def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".csv", ".txt"):
        return pd.read_csv(p)
    if ext == ".tsv":
        return pd.read_csv(p, sep="\t")
    if ext == ".parquet":
        return pd.read_parquet(p)
    if ext in (".json", ".jsonl"):
        return pd.read_json(p, lines=ext == ".jsonl")
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(p)
    if ext == ".npz":
        with np.load(p, allow_pickle=False) as z:
            if "X" not in z.files or "y" not in z.files:
                raise ValueError("npz data must contain arrays 'X' and 'y'")
            X, y = z["X"], z["y"]
        df = pd.DataFrame(X.reshape(len(X), -1), columns=[f"x{i}" for i in range(X.reshape(len(X), -1).shape[1])])
        if y.ndim == 1:
            df["y"] = y
        else:
            for j in range(y.shape[1]):
                df[f"y{j}"] = y[:, j]
        return df
    raise ValueError(f"unsupported data format {ext}; use csv/tsv/parquet/json/jsonl/xlsx/npz")


def _infer_task(y: pd.Series) -> str:
    t = pd.api.types
    if t.is_bool_dtype(y) or not t.is_numeric_dtype(y):
        return "classification"
    vals = np.asarray(y.dropna().unique(), dtype=float)
    if t.is_integer_dtype(y) and len(vals) <= 20:
        return "classification"
    if t.is_float_dtype(y) and len(vals) <= 10 and np.all(np.mod(vals, 1) == 0):
        return "classification"
    return "regression"


def encode_table(df: pd.DataFrame, target: list, task: str | None = None, drop: list | None = None):
    """Returns X (float), y, meta {task, feature_names, classes, notes}."""
    missing = [t for t in target if t not in df.columns]
    if missing:
        raise ValueError(f"target column(s) {missing} not found; columns are {list(df.columns)}")
    df = df.drop(columns=[c for c in (drop or []) if c in df.columns])
    df = df.dropna(subset=target).reset_index(drop=True)
    notes = []
    y, task, classes = encode_target(df, target, task)

    feats = df.drop(columns=target)
    cols = []
    names = []
    for c in feats.columns:
        s = feats[c]
        if s.dtype == bool:
            s = s.astype(int)
        if pd.api.types.is_numeric_dtype(s):
            if s.isna().any():
                notes.append(f"filled {int(s.isna().sum())} missing values in {c!r} with the median")
                s = s.fillna(s.median())
            cols.append(s.to_numpy(dtype=float)[:, None])
            names.append(str(c))
        else:
            s = s.astype(str).fillna("missing")
            cats = sorted(s.unique().tolist())
            if len(cats) > MAX_CATEGORIES:
                notes.append(f"dropped {c!r}: {len(cats)} categories (> {MAX_CATEGORIES})")
                continue
            for cat in cats:
                cols.append((s == cat).to_numpy(dtype=float)[:, None])
                names.append(f"{c}={cat}")
            notes.append(f"one-hot encoded {c!r} into {len(cats)} columns")
    if not cols:
        raise ValueError("no usable feature columns")
    X = np.hstack(cols)
    return X, y, {"task": task, "feature_names": names, "classes": classes, "target": target, "notes": notes}


def encode_target(df: pd.DataFrame, target: list, task: str | None = None):
    """Returns (y, task, classes). Rows with a missing target must be dropped beforehand."""
    missing = [t for t in target if t not in df.columns]
    if missing:
        raise ValueError(f"target column(s) {missing} not found; columns are {list(df.columns)}")
    ydf = df[target]
    if task is None:
        task = "regression" if len(target) > 1 else _infer_task(ydf[target[0]])
    classes = None
    if task == "classification":
        if len(target) != 1:
            raise ValueError("classification supports a single target column")
        col = ydf[target[0]]
        classes = sorted(col.astype(str).unique().tolist(), key=_natural_key)
        lut = {c: i for i, c in enumerate(classes)}
        y = col.astype(str).map(lut).to_numpy().astype(int)
    else:
        y = ydf.to_numpy().astype(float)
        if y.shape[1] == 1:
            y = y[:, 0]
    return y, task, classes


def encode_features(df: pd.DataFrame, feature_names: list, fill: list | None = None):
    """Encode new rows into an existing run's feature layout (numeric columns and 'col=cat' one-hots)."""
    X = np.zeros((len(df), len(feature_names)))
    notes = []
    for j, name in enumerate(feature_names):
        f = fill[j] if fill else 0.0
        if name in df.columns:
            s = pd.to_numeric(df[name], errors="coerce")
            X[:, j] = s.fillna(f).to_numpy(dtype=float)
        elif "=" in name and name.split("=", 1)[0] in df.columns:
            col, cat = name.split("=", 1)
            X[:, j] = (df[col].astype(str) == cat).to_numpy(dtype=float)
        else:
            X[:, j] = f
            notes.append(f"no column for feature {name!r}; filled with the training mean")
    return X, notes


def _natural_key(s: str):
    try:
        return (0, float(s), s)
    except ValueError:
        return (1, 0.0, s)


def load_task(path: str, n: int, seed: int):
    """Task generator module: must define generate(n, seed) -> (X, y).
    Optional: TASK_TYPE, FEATURE_NAMES, CLASSES, DESCRIPTION."""
    spec = importlib.util.spec_from_file_location("wsp_task", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "generate"):
        raise ValueError(f"{path} must define generate(n, seed) -> (X, y)")
    X, y = mod.generate(n, seed)
    X = np.asarray(X, dtype=float).reshape(len(X), -1)
    y = np.asarray(y)
    task = getattr(mod, "TASK_TYPE", None)
    if task is None:
        task = "classification" if np.issubdtype(y.dtype, np.integer) else "regression"
    names = list(getattr(mod, "FEATURE_NAMES", [f"x{i}" for i in range(X.shape[1])]))
    classes = None
    if task == "classification":
        y = y.astype(int)
        classes = [str(c) for c in getattr(mod, "CLASSES", list(range(int(y.max()) + 1)))]
    else:
        y = y.astype(float)
    return X, y, {"task": task, "feature_names": names, "classes": classes,
                  "description": getattr(mod, "DESCRIPTION", ""), "notes": []}


def split(X, y, task: str, fracs=(0.6, 0.2, 0.2), seed: int = 0):
    idxs = split_indices(y, task, fracs, seed)
    return {k: (X[v], y[v]) for k, v in idxs.items()}


def split_indices(y, task: str, fracs=(0.6, 0.2, 0.2), seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    if task == "classification":
        parts = {"train": [], "val": [], "test": []}
        for c in np.unique(y):
            idx = rng.permutation(np.where(y == c)[0])
            a = int(round(fracs[0] * len(idx)))
            b = a + int(round(fracs[1] * len(idx)))
            parts["train"].append(idx[:a]); parts["val"].append(idx[a:b]); parts["test"].append(idx[b:])
        idxs = {k: rng.permutation(np.concatenate(v)) for k, v in parts.items()}
    else:
        idx = rng.permutation(n)
        a = int(fracs[0] * n)
        b = a + int(fracs[1] * n)
        idxs = {"train": idx[:a], "val": idx[a:b], "test": idx[b:]}
    return idxs


def norm_stats(X: np.ndarray) -> dict:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return {"mean": mean.round(8).tolist(), "std": std.round(8).tolist()}


def profile_text(X, y, meta: dict, max_features: int = 40) -> str:
    names = meta["feature_names"]
    task = meta["task"]
    lines = [f"samples={len(X)}  features={X.shape[1]}  task={task}"]
    if meta.get("description"):
        lines.append(f"description: {meta['description']}")
    if task == "classification":
        classes = meta["classes"]
        counts = np.bincount(y.astype(int), minlength=len(classes))
        lines.append("classes: " + ", ".join(f"{i}={c} (n={counts[i]})" for i, c in enumerate(classes)))
    else:
        y2 = y.reshape(len(y), -1)
        for j in range(y2.shape[1]):
            lines.append(f"target[{j}]: mean={y2[:, j].mean():.4g} std={y2[:, j].std():.4g} "
                         f"min={y2[:, j].min():.4g} max={y2[:, j].max():.4g}")
    # Descriptive per-feature statistics (no model fitting).
    if task == "classification":
        classes = meta["classes"]
        k = len(classes)
        show_k = min(k, 6)
        hdr = f"{'feature':<28}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}  sep  " + " ".join(
            f"mu[{c[:6]}]".rjust(10) for c in classes[:show_k])
        lines.append(hdr)
        rows = []
        for j in range(X.shape[1]):
            x = X[:, j]
            mus = [x[y == c].mean() if np.any(y == c) else np.nan for c in range(k)]
            within = np.nanmean([x[y == c].var() for c in range(k) if np.sum(y == c) > 1] or [0.0])
            between = np.nanvar(mus)
            sep = between / (within + 1e-12)  # Fisher-style separability score (descriptive)
            rows.append((sep, j, x, mus))
        rows.sort(key=lambda r: -r[0])
        for sep, j, x, mus in rows[:max_features]:
            lines.append(f"{names[j][:27]:<28}{x.mean():9.3g}{x.std():9.3g}{x.min():9.3g}{x.max():9.3g} "
                         f"{sep:5.2f}  " + " ".join(f"{m:10.3g}" for m in mus[:show_k]))
        if X.shape[1] > max_features:
            lines.append(f"... {X.shape[1] - max_features} more features (sorted by separability 'sep')")
    else:
        y1 = y.reshape(len(y), -1)[:, 0]
        lines.append(f"{'feature':<28}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}{'corr(y)':>9}")
        rows = []
        for j in range(X.shape[1]):
            x = X[:, j]
            c = np.corrcoef(x, y1)[0, 1] if x.std() > 0 and y1.std() > 0 else 0.0
            rows.append((abs(c), j, x, c))
        rows.sort(key=lambda r: -r[0])
        for _, j, x, c in rows[:max_features]:
            lines.append(f"{names[j][:27]:<28}{x.mean():9.3g}{x.std():9.3g}{x.min():9.3g}{x.max():9.3g}{c:9.3f}")
        if X.shape[1] > max_features:
            lines.append(f"... {X.shape[1] - max_features} more features (sorted by |corr|)")
    uniq = len(np.unique(X, axis=0)) if X.shape[1] <= 64 else None
    if uniq is not None and uniq < len(X):
        lines.append(f"distinct input rows: {uniq}")
    for n in meta.get("notes", []):
        lines.append(f"note: {n}")
    return "\n".join(lines)
