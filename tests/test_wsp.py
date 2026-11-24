import json
from pathlib import Path

import numpy as np
import pytest

from wsp import cli, meter
from wsp.forward import forward, predict
from wsp.netspec import Network, init_weights, mlp_layers
from wsp.program import ProgramRejected, check_program, run_program
from wsp.run import Run

ROOT = Path(__file__).resolve().parents[1]
PARITY = ROOT / "examples" / "tasks" / "parity3.py"

PARITY_PROGRAM = """
import numpy as np
def program(spec, W):
    W["W0"][:] = 1.0
    W["b0"][:] = [0.0, -1.0, -2.0]
    W["W1"][:, 0] = [10.0, -20.0, 20.0]
    W["b1"][:] = -5.0
    return W
"""


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setenv("WSP_RUNS", str(tmp_path / "runs"))
    return tmp_path


def wsp(*args):
    return cli.main([str(a) for a in args])


def make_parity(runs, mode="pure", budget=5):
    assert wsp("new", "par", "--task", PARITY, "--n", 400, "--mode", mode, "--budget", budget) == 0
    return Run(runs / "runs" / "par")


# ---- forward pass ---------------------------------------------------------------------------------
def test_forward_matches_hand_computation():
    layers = mlp_layers([2, 2, 1], ["relu", "identity"])
    W = {"W0": np.array([[1.0, -1.0], [2.0, 0.5]]), "b0": np.array([0.0, 1.0]),
         "W1": np.array([[1.0], [-2.0]]), "b1": np.array([0.5])}
    net = Network({"arch": "mlp", "layers": layers, "task": "regression"}, W)
    x = np.array([[1.0, 2.0]])
    h = np.maximum(x @ W["W0"] + W["b0"], 0)      # [5, 1]
    expected = h @ W["W1"] + W["b1"]                # 5 - 2 + 0.5 = 3.5
    out, _ = forward(net, x)
    assert out[0, 0] == pytest.approx(expected[0, 0]) == pytest.approx(3.5)


def test_zero_init_and_validation():
    layers = mlp_layers([3, 4, 2], ["relu", "softmax"])
    net = Network({"arch": "mlp", "layers": layers}, init_weights(layers))
    net.validate()
    assert net.n_params() == 3 * 4 + 4 + 4 * 2 + 2
    assert all(np.all(v == 0) for v in net.W.values())


# ---- program guard --------------------------------------------------------------------------------
@pytest.mark.parametrize("src", [
    "import os\ndef program(spec, W): return W",
    "def program(spec, W):\n    open('x')\n    return W",
    "import sklearn\ndef program(spec, W): return W",
    "import numpy as np\ndef program(spec, W):\n    np.load('data/train.npz')\n    return W",
    "def program(spec, W):\n    return W.__class__",
    "from numpy import load\ndef program(spec, W): return W",
    "def program(spec, W):\n    return eval('W')",
])
def test_guard_rejects_forbidden_code(src):
    assert check_program(src)
    with pytest.raises(ProgramRejected):
        run_program(src, {"layers": []}, {})


def test_guard_accepts_numpy_program():
    assert check_program(PARITY_PROGRAM) == []


# ---- end-to-end pure run --------------------------------------------------------------------------
def test_pure_parity_programmed_exactly(runs, tmp_path):
    run = make_parity(runs)
    assert wsp("init", "par", "--arch", "3-3-1", "--act", "relu,sigmoid") == 0
    prog = tmp_path / "p.py"
    prog.write_text(PARITY_PROGRAM)
    assert wsp("write", "par", prog) == 0
    assert wsp("eval", "par") == 0
    run = Run(run.path)
    assert run.val_scores()["v001"] == 1.0
    assert run.session["forward_used"] == 1


def test_fit_refused_in_pure_mode(runs, capsys):
    make_parity(runs)
    wsp("init", "par", "--arch", "3-1")
    assert wsp("fit", "par") == 2
    assert "disabled in pure" in capsys.readouterr().err


def test_random_init_refused_in_pure_mode(runs):
    make_parity(runs)
    assert wsp("init", "par", "--arch", "3-1", "--random") == 2


def test_forward_budget_blocks_at_zero(runs, capsys):
    make_parity(runs, budget=2)
    wsp("init", "par", "--arch", "3-1")
    assert wsp("eval", "par") == 0
    assert wsp("eval", "par") == 0
    assert wsp("eval", "par") == 2
    assert "budget exhausted" in capsys.readouterr().err
    assert wsp("session", "par", "--budget", 3) == 0   # new session resets the budget
    assert wsp("eval", "par") == 0


def test_dump_refused_past_context_budget(runs, capsys):
    assert wsp("new", "par", "--task", PARITY, "--n", 400, "--context-tokens", 1000) == 0
    wsp("init", "par", "--arch", "3-256-1")
    capsys.readouterr()
    assert wsp("dump", "par") == 2
    assert "sandbox" in capsys.readouterr().err
    assert wsp("inspect", "par", "stats") == 0          # the fallback still works


def test_test_split_hidden_until_report(runs):
    run = make_parity(runs)
    with pytest.raises(PermissionError):
        run.load_split("test")
    wsp("init", "par", "--arch", "3-1")
    assert wsp("report", "par", "--no-baseline") == 0
    assert (run.path / "report.md").exists()
    assert Run(run.path).meta.get("test_seen")


def test_hybrid_fit_improves(runs):
    make_parity(runs, mode="hybrid", budget=10)
    wsp("init", "par", "--arch", "3-8-1", "--act", "relu,sigmoid", "--random", "--seed", 1)
    assert wsp("fit", "par", "--epochs", 400, "--lr", 0.02) == 0
    run = Run(runs / "runs" / "par")
    assert run.val_scores()["v001"] >= 0.9


def test_patch_resize_revert(runs, tmp_path):
    run = make_parity(runs)
    wsp("init", "par", "--arch", "3-3-1", "--act", "relu,sigmoid")
    edits = tmp_path / "e.json"
    edits.write_text(json.dumps([{"tensor": "W0", "index": [":", 0], "value": 1.0},
                                 {"tensor": "b1", "index": [0], "add": -0.5}]))
    assert wsp("patch", "par", edits) == 0
    assert wsp("resize", "par", "--layer", 0, "--units", 5) == 0
    net = Run(run.path).load_net()
    assert net.W["W0"].shape == (3, 5) and net.W["W1"].shape == (5, 1)
    assert np.all(net.W["W0"][:, 0] == 1.0) and np.all(net.W["W0"][:, 3:] == 0)
    assert wsp("revert", "par", "v001") == 0
    assert Run(run.path).meta["current"] == "v001"


# ---- import / export ------------------------------------------------------------------------------
def test_npz_import_roundtrip(runs, tmp_path):
    run = make_parity(runs)
    rng = np.random.default_rng(0)
    torch_style = {"0.weight": rng.normal(size=(4, 3)), "0.bias": rng.normal(size=4),
                   "2.weight": rng.normal(size=(1, 4)), "2.bias": rng.normal(size=1)}
    f = tmp_path / "m.npz"
    np.savez(f, **torch_style)
    assert wsp("import", "par", f, "--act", "relu,sigmoid") == 0
    net = Run(run.path).load_net()
    X = rng.normal(size=(5, 3))
    expected = 1 / (1 + np.exp(-(np.maximum(X @ torch_style["0.weight"].T + torch_style["0.bias"], 0)
                                 @ torch_style["2.weight"].T + torch_style["2.bias"])))
    out, _ = forward(net, X)
    assert np.allclose(out, expected)
    out_f = tmp_path / "back.npz"
    assert wsp("export", "par", out_f) == 0


def test_opaque_import_is_inspect_only(runs, tmp_path):
    make_parity(runs)
    f = tmp_path / "conv.npz"
    np.savez(f, **{"conv.weight": np.ones((2, 1, 3, 3)), "conv.bias": np.zeros(2)})
    assert wsp("import", "par", f) == 0
    assert wsp("eval", "par") == 2
    assert wsp("inspect", "par", "svd", "--tensor", "conv.weight") == 0


def test_sklearn_import_matches_predictions(runs, tmp_path):
    sklearn = pytest.importorskip("sklearn")
    import joblib
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["species"] = [iris.target_names[i] for i in iris.target]
    df = df.drop(columns=["target"])
    csv = tmp_path / "iris.csv"
    df.to_csv(csv, index=False)
    X = df.drop(columns=["species"]).to_numpy()
    pipe = make_pipeline(StandardScaler(), MLPClassifier((6,), max_iter=800, random_state=0)).fit(X, df["species"])
    pkl = tmp_path / "m.pkl"
    joblib.dump(pipe, pkl)

    assert wsp("new", "ir", "--data", csv, "--target", "species", "--no-norm") == 0
    assert wsp("import", "ir", pkl) == 2              # refuses pickles without --trust
    assert wsp("import", "ir", pkl, "--trust") == 0
    net = Run(tmp_path / "runs" / "ir").load_net()
    _, prob = predict(net, X)
    assert np.allclose(prob, pipe.predict_proba(X), atol=1e-6)
    out = tmp_path / "edited.pkl"
    assert wsp("export", "ir", out, "--trust") == 0
    assert np.allclose(joblib.load(out).predict_proba(X), pipe.predict_proba(X), atol=1e-6)


def test_token_estimate_is_conservative():
    assert meter.estimate_tokens("0.123 " * 100, numeric=True) >= 200
