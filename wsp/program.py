"""Weight programs: agent-written Python that constructs/edits weights by reasoning.

A weight program is a file defining

    def program(spec, W):
        # spec: dict (layers, feature_names, classes, input_norm, ...); W: dict name -> np.ndarray copy
        ...
        return W                    # or (W, {"layers": [...]}) to change the architecture

Rules enforced by an AST guard (both modes):
  * only `numpy` and `math` may be imported
  * no file/OS/process access, no exec/eval/getattr tricks, no dunder attribute access
  * the program never receives data: it cannot fit weights to the dataset
Python here is a tool for *expressing* reasoning (e.g. "unit 3 detects x0 > 2"), not for optimization.
"""
from __future__ import annotations

import ast
import builtins
import copy
import math

import numpy as np

ALLOWED_MODULES = {"numpy", "math"}
FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "input", "breakpoint",
    "getattr", "setattr", "delattr", "memoryview", "help", "exit", "quit", "type", "object", "super",
    "classmethod", "staticmethod", "property",
}
FORBIDDEN_ATTRS = {
    "load", "loadtxt", "genfromtxt", "fromfile", "fromregex", "save", "savez", "savez_compressed", "savetxt",
    "tofile", "memmap", "DataSource", "ctypeslib", "lib", "f2py", "distutils", "testing", "ctypes",
    "os", "sys", "frombuffer", "fromstring", "dump", "dumps",
}
SAFE_BUILTINS = {
    n: getattr(builtins, n) for n in (
        "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "int", "isinstance",
        "len", "list", "map", "max", "min", "pow", "print", "range", "reversed", "round", "set", "slice",
        "sorted", "str", "sum", "tuple", "zip", "ValueError", "Exception", "AssertionError", "KeyError",
        "IndexError", "True", "False", "None", "frozenset", "complex", "hash", "iter", "next",
    ) if hasattr(builtins, n)
}


class ProgramRejected(ValueError):
    pass


def check_program(src: str) -> list:
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_MODULES:
                    problems.append(f"line {node.lineno}: import of {a.name!r} not allowed (numpy, math only)")
                elif len(a.name.split(".")) > 1 and set(a.name.split(".")[1:]) & FORBIDDEN_ATTRS:
                    problems.append(f"line {node.lineno}: import of {a.name!r} not allowed")
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")
            if node.level or mod[0] not in ALLOWED_MODULES:
                problems.append(f"line {node.lineno}: import from {node.module!r} not allowed (numpy, math only)")
            elif set(mod[1:]) & FORBIDDEN_ATTRS:
                problems.append(f"line {node.lineno}: import from {node.module!r} not allowed")
            for a in node.names:
                if a.name in FORBIDDEN_ATTRS or a.name == "*":
                    problems.append(f"line {node.lineno}: importing {a.name!r} not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"line {node.lineno}: use of {node.id!r} not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS or node.attr.startswith("__"):
                problems.append(f"line {node.lineno}: attribute .{node.attr} not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            problems.append(f"line {node.lineno}: global/nonlocal not allowed")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "program" for n in tree.body):
        problems.append("file must define a top-level function `program(spec, W)`")
    return problems


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in ALLOWED_MODULES:
        raise ImportError(f"import of {name} is not allowed in weight programs")
    return builtins.__import__(name, globals, locals, fromlist, level)


def run_program(src: str, spec: dict, W: dict):
    problems = check_program(src)
    if problems:
        raise ProgramRejected("weight program rejected:\n  " + "\n  ".join(problems))
    safe = dict(SAFE_BUILTINS)
    safe["__import__"] = _safe_import
    ns = {"__builtins__": safe, "np": np, "numpy": np, "math": math, "__name__": "wsp_program"}
    exec(compile(src, "<weight program>", "exec"), ns)
    result = ns["program"](copy.deepcopy(spec), {k: v.copy() for k, v in W.items()})
    spec_update = None
    if isinstance(result, tuple):
        if len(result) != 2 or not isinstance(result[1], dict):
            raise ProgramRejected("program must return W or (W, {'layers': [...]})")
        result, spec_update = result
        unknown = set(spec_update) - {"layers"}
        if unknown:
            raise ProgramRejected(f"program may only update 'layers' in the spec, got {sorted(unknown)}")
    if not isinstance(result, dict):
        raise ProgramRejected("program must return a dict of tensors")
    out = {}
    for k, v in result.items():
        arr = np.asarray(v, dtype=np.float64)
        out[str(k)] = arr
    return out, spec_update
