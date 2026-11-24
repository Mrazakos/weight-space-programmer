"""Run directory management: metadata, data splits, network versions, history, output metering.

runs/<name>/
  run.json             mode, task meta, session ledger, current version
  data/train.npz val.npz
  data/heldout/test.npz  (only read by `wsp report`)
  nets/vNNN.json/.npz  every network version
  history.jsonl        create/eval events
  programs/            weight programs written by the agent
  dumps/               weight dumps (chunked text)
  log.md               the agent's hypotheses and decisions
  handoff.md           state for the next improvement session
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from . import meter
from .evaluate import primary_metric
from .netspec import Network


def runs_root() -> Path:
    return Path(os.environ.get("WSP_RUNS", "runs"))


def resolve(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if (p / "run.json").exists():
        return p
    q = runs_root() / name_or_path
    if (q / "run.json").exists():
        return q
    raise FileNotFoundError(f"no run found at {p} or {q} (create one with `python -m wsp new`)")


class Run:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "run.json").read_text(encoding="utf-8"))

    # ---- creation / persistence ---------------------------------------------------------------
    @classmethod
    def create(cls, path: Path, meta: dict, splits: dict) -> "Run":
        path = Path(path)
        if (path / "run.json").exists():
            raise FileExistsError(f"run {path} already exists")
        for d in ("data/heldout", "nets", "programs", "dumps"):
            (path / d).mkdir(parents=True, exist_ok=True)
        np.savez(path / "data" / "train.npz", X=splits["train"][0], y=splits["train"][1])
        np.savez(path / "data" / "val.npz", X=splits["val"][0], y=splits["val"][1])
        np.savez(path / "data" / "heldout" / "test.npz", X=splits["test"][0], y=splits["test"][1])
        meta.setdefault("created", time.strftime("%Y-%m-%d %H:%M:%S"))
        meta.setdefault("current", None)
        (path / "run.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        (path / "log.md").write_text(f"# Run log: {meta['name']}\n\n", encoding="utf-8")
        return cls(path)

    def save(self) -> None:
        (self.path / "run.json").write_text(json.dumps(self.meta, indent=1), encoding="utf-8")

    @property
    def mode(self) -> str:
        return self.meta["mode"]

    @property
    def task(self) -> str:
        return self.meta["task"]

    @property
    def session(self) -> dict:
        if "session" not in self.meta:
            self.meta["session"] = meter.new_session(None, None, None)
        return self.meta["session"]

    # ---- output metering -------------------------------------------------------------------------
    def emit(self, text: str, numeric: bool = False) -> None:
        print(text)
        meter.charge_tokens(self.session, meter.estimate_tokens(text, numeric))
        self.save()
        if meter.context_remaining(self.session) <= 0:
            print("!! CONTEXT BUDGET EXHAUSTED for this session: write handoff.md and stop.")

    # ---- data ----------------------------------------------------------------------------------
    def load_split(self, split: str, allow_test: bool = False):
        if split == "test":
            if not allow_test:
                raise PermissionError("the test split is held out; it is only evaluated by `wsp report`")
            f = self.path / "data" / "heldout" / "test.npz"
        elif split in ("train", "val"):
            f = self.path / "data" / f"{split}.npz"
        else:
            raise ValueError("split must be train or val")
        with np.load(f) as z:
            return z["X"], z["y"]

    # ---- spec meta shared by every network in the run -------------------------------------------
    def base_spec(self) -> dict:
        m = self.meta
        return {
            "arch": "mlp",
            "task": m["task"],
            "classes": m.get("classes"),
            "feature_names": m["feature_names"],
            "input_norm": m.get("input_norm"),
            "output_norm": m.get("output_norm"),
        }

    # ---- versions ------------------------------------------------------------------------------
    def versions(self) -> list:
        return sorted(p.stem for p in (self.path / "nets").glob("v*.json"))

    def load_net(self, version: str | None = None) -> Network:
        v = version or self.meta.get("current")
        if not v:
            raise FileNotFoundError("run has no network yet: use `wsp init` or `wsp import`")
        return Network.load(self.path / "nets" / v)

    def save_net(self, net: Network, op: str, note: str = "", parent: str | None = None) -> str:
        net.validate()
        vs = self.versions()
        nxt = f"v{(int(vs[-1][1:]) + 1) if vs else 0:03d}"
        net.save(self.path / "nets" / nxt)
        self.log_event({"event": "create", "version": nxt, "parent": parent if parent is not None
                        else self.meta.get("current"), "op": op, "note": note,
                        "arch": net.arch_str(), "params": net.n_params()})
        self.meta["current"] = nxt
        self.save()
        return nxt

    def log_event(self, ev: dict) -> None:
        ev = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "session": self.session.get("index"), **ev}
        with open(self.path / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(ev) + "\n")

    def history(self) -> list:
        f = self.path / "history.jsonl"
        if not f.exists():
            return []
        return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]

    def val_scores(self) -> dict:
        """Latest full-val-split primary metric per version."""
        key = primary_metric(self.task)
        scores = {}
        for ev in self.history():
            if ev.get("event") == "eval" and ev.get("split") == "val" and ev.get("full"):
                scores[ev["version"]] = ev["metrics"][key]
        return scores

    def best_version(self) -> str | None:
        scores = self.val_scores()
        if not scores:
            return self.meta.get("current")
        return max(scores, key=lambda v: (scores[v], v))
