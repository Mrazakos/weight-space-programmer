"""Network representation: a JSON spec plus a dict of named numpy tensors.

MLP convention (used everywhere in wsp):
    h_0     = normalize(x)                      (if spec["input_norm"] is set)
    h_{i+1} = act_i(h_i @ W{i} + b{i})          W{i}: (in, out), b{i}: (out,)
    y       = h_L * out_std + out_mean          (regression, if spec["output_norm"] is set)

Imported networks that are not plain MLPs are stored with arch="opaque": their tensors can be
inspected and edited, but wsp cannot run a forward pass on them.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

ACTIVATIONS = (
    "relu", "leaky_relu", "tanh", "sigmoid", "step", "identity", "softmax", "abs", "square", "sin",
)


class NetworkError(ValueError):
    pass


class Network:
    def __init__(self, spec: dict, weights: dict):
        self.spec = spec
        self.W = {k: np.array(v, dtype=np.float64) for k, v in weights.items()}

    # ---- properties -------------------------------------------------------------------------
    @property
    def is_mlp(self) -> bool:
        return self.spec.get("arch") == "mlp"

    @property
    def layers(self) -> list:
        return self.spec.get("layers") or []

    def sizes(self) -> list:
        if not self.is_mlp:
            return []
        return [self.layers[0]["in"]] + [L["out"] for L in self.layers]

    def n_params(self) -> int:
        return int(sum(v.size for v in self.W.values()))

    def arch_str(self) -> str:
        if not self.is_mlp:
            return f"opaque ({len(self.W)} tensors)"
        acts = ",".join(L["act"] for L in self.layers)
        return "-".join(str(s) for s in self.sizes()) + f" ({acts})"

    def copy(self) -> "Network":
        return Network(copy.deepcopy(self.spec), {k: v.copy() for k, v in self.W.items()})

    # ---- validation -------------------------------------------------------------------------
    def validate(self) -> None:
        for k, v in self.W.items():
            if not np.all(np.isfinite(v)):
                raise NetworkError(f"tensor {k} contains NaN/inf")
        if not self.is_mlp:
            return
        layers = self.layers
        if not layers:
            raise NetworkError("MLP has no layers")
        expected = set()
        for i, L in enumerate(layers):
            if L.get("act") not in ACTIVATIONS:
                raise NetworkError(f"layer {i}: unknown activation {L.get('act')!r}; use one of {ACTIVATIONS}")
            if i > 0 and L["in"] != layers[i - 1]["out"]:
                raise NetworkError(f"layer {i}: in={L['in']} but previous layer out={layers[i-1]['out']}")
            if L["act"] == "softmax" and i != len(layers) - 1:
                raise NetworkError("softmax is only allowed on the output layer")
            for name, shape in ((f"W{i}", (L["in"], L["out"])), (f"b{i}", (L["out"],))):
                expected.add(name)
                if name not in self.W:
                    raise NetworkError(f"missing tensor {name}")
                if self.W[name].shape != shape:
                    raise NetworkError(f"{name} has shape {self.W[name].shape}, expected {shape}")
        extra = set(self.W) - expected
        if extra:
            raise NetworkError(f"unexpected tensors for this MLP: {sorted(extra)}")
        norm = self.spec.get("input_norm")
        if norm and len(norm["mean"]) != layers[0]["in"]:
            raise NetworkError("input_norm length does not match network input size")

    # ---- persistence ------------------------------------------------------------------------
    def save(self, base: Path) -> None:
        base = Path(base)
        base.parent.mkdir(parents=True, exist_ok=True)
        Path(str(base) + ".json").write_text(json.dumps(self.spec, indent=1), encoding="utf-8")
        np.savez(str(base) + ".npz", **self.W)

    @classmethod
    def load(cls, base: Path) -> "Network":
        base = Path(base)
        spec = json.loads(Path(str(base) + ".json").read_text(encoding="utf-8"))
        with np.load(str(base) + ".npz") as z:
            W = {k: z[k] for k in z.files}
        return cls(spec, W)


def mlp_layers(sizes: list, acts: list) -> list:
    if len(acts) != len(sizes) - 1:
        raise NetworkError(f"need {len(sizes)-1} activations for sizes {sizes}, got {len(acts)}")
    return [{"in": int(sizes[i]), "out": int(sizes[i + 1]), "act": acts[i]} for i in range(len(sizes) - 1)]


def init_weights(layers: list, zero: bool = True, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    W = {}
    for i, L in enumerate(layers):
        if zero:
            W[f"W{i}"] = np.zeros((L["in"], L["out"]))
        else:
            gain = 2.0 if L["act"] in ("relu", "leaky_relu", "abs") else 1.0
            W[f"W{i}"] = rng.normal(0.0, np.sqrt(gain / L["in"]), (L["in"], L["out"]))
        W[f"b{i}"] = np.zeros(L["out"])
    return W


def parse_arch(arch: str) -> list:
    try:
        return [int(s) for s in arch.replace("x", "-").split("-") if s.strip()]
    except ValueError as e:
        raise NetworkError(f"bad --arch {arch!r}; expected e.g. 4-8-3") from e
