"""Tests for layer-graph networks (images, text), torch modules, export and predict."""
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from wsp import cli
from wsp.backend import forward, predict
from wsp.graph import assign_names, infer, init_params, validate
from wsp.netspec import Network, NetworkError
from wsp.run import Run

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setenv("WSP_RUNS", str(tmp_path / "runs"))
    return tmp_path


def wsp(*args):
    return cli.main([str(a) for a in args])


def make_images(root: Path, n_per_class=30, size=8):
    """Two classes: bright left half vs bright right half."""
    from PIL import Image
    rng = np.random.default_rng(0)
    for cls in ("left", "right"):
        d = root / cls
        d.mkdir(parents=True)
        for i in range(n_per_class):
            a = rng.integers(0, 40, (size, size)).astype(np.uint8)
            if cls == "left":
                a[:, : size // 2] += 200
            else:
                a[:, size // 2:] += 200
            Image.fromarray(a, "L").save(d / f"{i}.png")
    return root


def make_reviews(path: Path, n=400):
    rng = np.random.default_rng(1)
    rows = []
    for _ in range(n):
        y = int(rng.integers(0, 2))
        w = rng.choice(["good", "great", "nice"] if y else ["bad", "awful", "poor"])
        rows.append({"text": f"the item was really {w} and the delivery came on time", "label": "pos" if y else "neg"})
    import pandas as pd
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ---- graph shape inference / validation ------------------------------------------------------------
def test_graph_infer_and_validate():
    spec = {"arch": "net", "input": {"modality": "image", "shape": [3, 16, 16]},
            "layers": assign_names([{"type": "conv2d", "name": "c1", "out_ch": 4, "k": 3, "act": "relu"},
                                    {"type": "maxpool2d", "k": 2},
                                    {"type": "global_pool", "op": "mean"},
                                    {"type": "dense", "name": "out", "out": 5, "act": "softmax"}])}
    rows = infer(spec)
    assert rows[0]["params"]["c1.W"] == (4, 3, 3, 3)
    assert rows[1]["out_shape"] == (4, 8, 8)
    assert rows[-1]["out_shape"] == (5,)
    W = init_params(spec)
    validate(spec, W)
    W["c1.W"] = np.zeros((4, 3, 5, 5))
    with pytest.raises(NetworkError):
        validate(spec, W)


def test_text_net_must_start_with_embedding():
    spec = {"arch": "net", "input": {"modality": "text", "shape": [10], "vocab_size": 50},
            "layers": assign_names([{"type": "dense", "out": 2, "act": "softmax"}])}
    with pytest.raises(NetworkError):
        infer(spec)


def test_transformer_forward_masks_padding():
    spec = {"arch": "net", "task": "classification", "input": {"modality": "text", "shape": [6], "vocab_size": 20},
            "layers": assign_names([{"type": "embedding", "name": "emb", "dim": 8}, {"type": "posembed", "name": "pos"},
                                    {"type": "transformer", "name": "blk", "heads": 2, "ff": 16},
                                    {"type": "global_pool", "op": "mean"},
                                    {"type": "dense", "name": "out", "out": 3, "act": "softmax"}])}
    net = Network(spec, init_params(spec, zero=False, seed=0))
    X = np.array([[3, 4, 5, 0, 0, 0], [3, 4, 5, 0, 0, 0]])
    out, caps = forward(net, X, capture=True)
    assert out.shape == (2, 3) and np.allclose(out.sum(axis=1), 1, atol=1e-5)
    assert np.allclose(out[0], out[1])


# ---- images -----------------------------------------------------------------------------------------
def test_image_run_programmed_template(runs, tmp_path):
    folder = make_images(tmp_path / "imgs")
    assert wsp("new", "img", "--data", folder, "--mode", "pure", "--no-norm") == 0
    run = Run(runs / "runs" / "img")
    assert run.modality == "image" and run.meta["input_shape"] == [1, 8, 8]
    assert wsp("init", "img", "--preset", "linear") == 0
    prog = tmp_path / "tmpl.py"
    prog.write_text(
        "import numpy as np\n"
        "def program(spec, W):\n"
        "    left = np.zeros((8, 8)); left[:, :4] = 1.0\n"
        "    W['out.W'][:, 0] = (left - left[:, ::-1]).ravel() * -1.0   # 1 output: >0 means class 1 ('right')\n"
        "    return W\n")
    assert wsp("write", "img", prog) == 0
    assert wsp("eval", "img") == 0
    assert Run(run.path).val_scores()["v001"] == 1.0
    assert wsp("fit", "img") == 2                       # pure mode
    assert (run.path / "dumps" / "class_means.png").exists()


def test_image_cnn_fit_export_predict(runs, tmp_path):
    folder = make_images(tmp_path / "imgs")
    assert wsp("new", "img", "--data", folder) == 0     # auto mode
    assert wsp("init", "img", "--preset", "cnn", "--width", 4, "--depth", 1, "--random") == 0
    assert wsp("fit", "img", "--epochs", 25, "--lr", 0.01) == 0
    run = Run(runs / "runs" / "img")
    assert run.val_scores()["v001"] >= 0.9
    ts = tmp_path / "m.ts"
    assert wsp("export", "img", ts) == 0
    X, _ = run.load_split("val")
    ours, _ = forward(run.load_net(), X)
    theirs = torch.jit.load(str(ts))(torch.tensor(X)).detach().numpy()
    assert np.allclose(ours, theirs, atol=1e-4)
    out = tmp_path / "preds.csv"
    assert wsp("predict", "img", folder / "left", "--out", out) == 0
    import pandas as pd
    assert (pd.read_csv(out)["prediction"] == "left").mean() >= 0.9


# ---- text -------------------------------------------------------------------------------------------
def test_text_run_lexicon_program(runs, tmp_path):
    csv = make_reviews(tmp_path / "reviews.csv")
    assert wsp("new", "txt", "--data", csv, "--target", "label", "--mode", "pure") == 0
    run = Run(runs / "runs" / "txt")
    assert run.modality == "text" and "good" in run.vocab()
    assert wsp("init", "txt", "--preset", "bow", "--dim", 1) == 0
    prog = tmp_path / "lex.py"
    prog.write_text(
        "def program(spec, W):\n"
        "    idx = {w: i for i, w in enumerate(spec['vocab'])}\n"
        "    for w in ['good', 'great', 'nice']: W['emb.E'][idx[w], 0] = 1.0\n"
        "    for w in ['bad', 'awful', 'poor']: W['emb.E'][idx[w], 0] = -1.0\n"
        "    W['out.W'][0, 0] = 50.0\n"
        "    return W\n")
    assert wsp("write", "txt", prog) == 0
    assert wsp("eval", "txt") == 0
    assert Run(run.path).val_scores()["v001"] == 1.0
    lines = tmp_path / "new.txt"
    lines.write_text("this was great\nawful thing\n")
    assert wsp("predict", "txt", lines) == 0


def test_text_task_generator_and_transformer(runs):
    task = ROOT / "examples" / "tasks" / "negation_sentiment.py"
    assert wsp("new", "neg", "--task", task, "--n", 600) == 0
    assert Run(runs / "runs" / "neg").modality == "text"
    assert wsp("init", "neg", "--preset", "transformer", "--dim", 16, "--blocks", 1, "--heads", 2, "--random") == 0
    assert wsp("fit", "neg", "--epochs", 3) == 0
    assert wsp("inspect", "neg", "probe", "--n", 64) == 0
    assert wsp("dump", "neg", "--layer", "blk1") == 0


# ---- imported torch modules -------------------------------------------------------------------------
def test_torchscript_module_import_edit_fit(runs, tmp_path):
    folder = make_images(tmp_path / "imgs")
    m = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3, padding=1), torch.nn.ReLU(),
                            torch.nn.Flatten(), torch.nn.Linear(2 * 8 * 8, 2))
    f = tmp_path / "ext.pt"
    torch.jit.trace(m, torch.zeros(2, 1, 8, 8)).save(str(f))
    assert wsp("new", "ext", "--data", folder, "--no-norm") == 0
    assert wsp("import", "ext", f) == 2                     # needs --trust
    assert wsp("import", "ext", f, "--trust") == 0
    run = Run(runs / "runs" / "ext")
    net = run.load_net()
    assert net.spec["arch"] == "torch_module" and "3.weight" in net.W
    X, _ = run.load_split("val")
    expected = m(torch.tensor(X, dtype=torch.float32) / 255.0).detach().numpy()
    out, _ = forward(net, X)
    assert np.allclose(out, expected, atol=1e-5)
    edits = tmp_path / "e.json"
    edits.write_text(json.dumps([{"tensor": "3.bias", "index": [0], "add": 100.0}]))
    assert wsp("patch", "ext", edits) == 0
    pred, _ = predict(Run(run.path).load_net(), X)
    assert np.all(pred == 0)                                # the edit is live in the forward pass
    assert wsp("revert", "ext", "v000") == 0
    assert wsp("fit", "ext", "--epochs", 5, "--lr", 0.01, "--freeze", "0") == 0
    after = Run(run.path).load_net()
    assert np.allclose(after.W["0.weight"], net.W["0.weight"])   # frozen layer unchanged
    assert wsp("export", "ext", tmp_path / "edited.pt") == 0


def test_tabular_mlp_torchscript_export_matches(runs, tmp_path):
    task = ROOT / "examples" / "tasks" / "parity3.py"
    assert wsp("new", "par", "--task", task, "--n", 200, "--mode", "hybrid") == 0
    assert wsp("init", "par", "--arch", "3-4-1", "--act", "relu,sigmoid", "--random") == 0
    run = Run(runs / "runs" / "par")
    ts = tmp_path / "par.ts"
    assert wsp("export", "par", ts) == 0
    X, _ = run.load_split("val")
    ours, _ = forward(run.load_net(), X)
    theirs = torch.jit.load(str(ts))(torch.tensor(X, dtype=torch.float32)).detach().numpy()
    assert np.allclose(ours, theirs, atol=1e-5)


def test_tabular_predict_new_rows(runs, tmp_path):
    import pandas as pd
    df = pd.DataFrame({"a": np.arange(60) % 7, "color": ["red", "blue", "green"] * 20,
                       "y": ["hi" if i % 2 else "lo" for i in range(60)]})
    csv = tmp_path / "t.csv"
    df.to_csv(csv, index=False)
    assert wsp("new", "tab", "--data", csv, "--target", "y") == 0
    assert wsp("init", "tab", "--arch", "4-1", "--random") == 0
    new = tmp_path / "new.csv"
    df.drop(columns=["y"]).head(5).to_csv(new, index=False)
    assert wsp("predict", "tab", new, "--out", tmp_path / "p.csv") == 0
    assert len(pd.read_csv(tmp_path / "p.csv")) == 5
