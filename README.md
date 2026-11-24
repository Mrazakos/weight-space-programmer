# Weight-Space Programmer

A Claude Code plugin whose agents **build and improve neural networks by reading and writing their weights directly**. The agents don't hand weights off to an optimizer. They treat the weights as a program: they read them to understand what the network computes, then write or edit them to change it.

It is inspired by *[Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights](https://rikiyat.github.io/blog/weight-space-programming/)* (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, 2026). That post had a language model write networks from all-zero weights with no gradient training, and repair trained policies and language models by editing a small fraction of their weights. This repository is an independent implementation of that workflow for small networks. It is not affiliated with the authors.

## What you get

| Piece | What it does |
|---|---|
| `/wsp-build <data or "task">` | Builds a network for a dataset (csv/tsv/parquet/json/xlsx/npz) or a plain-language task description. The agent **chooses the network size**, then writes the weights. |
| `/wsp-improve <run or model file>` | **Improvement mode.** Works on a network the agents built, or one trained elsewhere (scikit-learn `.pkl`, torch `.pt`, `.safetensors`, `.npz`). It loops until the forward-pass or context budget is spent, and can run several rounds that hand off state between them. |
| `wsp-pure` agent | Follows the blog post's rules: no gradients, all-zero start, metered forward passes, and the weights come only from reasoning. |
| `wsp-hybrid` agent | Can also run gradient training (`wsp fit`). It combines programmed structure, gradient fine-tuning and surgical hand edits. |
| `wsp` toolkit | A numpy CLI that gives both agents a sandbox: weight dumps, weight programs, edits, metered evaluation, inspection, versioning, import/export, and a final report. |

## Install

In Claude Code:

```
/plugin marketplace add Mrazakos/weight-space-programmer
/plugin install weight-space-programmer@weight-space-programmer
```

The agents call the Python toolkit, so install it too (Python ≥ 3.10):

```
pip install "git+https://github.com/Mrazakos/weight-space-programmer"
# optional extras: [sklearn] for scikit-learn import/export, [torch], [safetensors]
pip install "weight-space-programmer[sklearn] @ git+https://github.com/Mrazakos/weight-space-programmer"
```

The skills also install the toolkit automatically if `import wsp` fails.

## Usage

```
/wsp-build data/churn.csv --target churned
/wsp-build "classify whether a 2-D point lies inside the ring 1 <= r <= 2" --name ring
/wsp-build data/houses.csv --target price --mode hybrid --budget 80

/wsp-improve runs/ring --rounds 3
/wsp-improve models/classifier.pkl --data data/val_data.csv --target label --mode pure
```

Each run lives in `./runs/<name>/`. Set `WSP_RUNS` to put runs somewhere else. A run directory holds:

- `nets/vNNN.json/.npz`: every network version
- `history.jsonl`: version tree and evaluations
- `programs/`: the weight program behind each version
- `log.md`: the agent's hypotheses and design notes
- `handoff.md`: state for the next improvement session
- `report.md`: held-out test score next to a gradient baseline with the same architecture

### How the agent sizes the network
`wsp size` lists candidate architectures. For each it shows the parameter count, parameters per training sample, and how many tokens a full weight dump would take against the remaining context budget.

- **Pure agent:** plans the computation as a list of hidden units, each with a stated job (a hyperplane, a threshold, a logic gate, a bump, a radius). The width is the number of jobs. It keeps the full dump small enough to read every weight.
- **Hybrid agent:** targets about 1–2 parameters per sample on tabular data and keeps the network small enough to hand-edit.

Either agent records its decision and the reason in `log.md`.

### Context-window-aware improvement
- **Context ledger:** `wsp` charges every output it prints, and every dump file it writes, to a per-session ledger. The default budget is a 1M-token window minus a 25% reserve. Override it with `WSP_CONTEXT_TOKENS` or `--context-tokens`.
- **Networks that fit:** if a network's full weight dump fits, the agent reads every weight.
- **Networks that don't fit:** the agent uses the **sandbox-inspection fallback**: `inspect stats | svd | unit | probe` and sliced dumps. It locates the responsible component before editing, the same approach the blog post used to edit Qwen.
- **Ending a session:** when the forward-pass budget or the ledger runs out, the agent writes `handoff.md`. The next `/wsp-improve` round starts a fresh agent with a fresh context that resumes from it.

## The `wsp` toolkit

```
python -m wsp new <run> --data F --target COL | --task task.py   [--mode pure|hybrid] [--budget N]
python -m wsp session|status|profile|sample|size <run>
python -m wsp init <run> --arch 8-3 --act relu,softmax          # zero weights (pure)
python -m wsp dump <run> [--layer L] [--tensor W0] [--rows a:b]  # weights as text, chunked
python -m wsp write <run> program.py                             # numpy-only weight program
python -m wsp set|patch|resize <run> ...                         # surgical edits
python -m wsp eval <run> [--activations]                         # METERED forward pass
python -m wsp inspect <run> stats|unit|slice|svd|probe
python -m wsp history|revert|best <run>
python -m wsp fit <run>                                          # hybrid only
python -m wsp import|export <run> model.{pkl,pt,safetensors,npz}
python -m wsp report <run>                                       # held-out test + baseline
```

A weight program is a plain Python file. It never receives data:

```python
import numpy as np

def program(spec, W):
    # 3-bit parity with 3 counting units: h_j = relu(x0+x1+x2 - j)
    W["W0"][:] = 1.0
    W["b0"][:] = [0, -1, -2]
    W["W1"][:, 0] = [10, -20, 20]   # 10*s - 20*relu(s-1) + 20*relu(s-2) alternates with parity
    W["b1"][:] = -5
    return W
```

A demo on `examples/tasks/parity3.py`: this hand-written program scores 100% on held-out data after one forward pass. A gradient-trained network with the same 3-3-1 architecture scores about 79%.

### Rules enforced in pure mode
- `fit` and random initialization are refused.
- Weight programs pass an AST guard: only `numpy` and `math` imports, no file/OS access, no `eval`/`exec`/`getattr`, no dunder attributes. The program receives the spec and the weights, never the data.
- Forward passes (`eval`, `inspect probe`) draw on a per-session budget. The default is 50, the largest budget in the blog post.
- The test split is held out until `report`.

The AST guard enforces the rules for a cooperating agent. **It is not a security sandbox.** Run only weight programs you would run as ordinary Python.

## Examples

```
python examples/make_datasets.py      # iris.csv, breast_cancer.csv, diabetes.csv,
                                      # breast_cancer_mlp.pkl and a deliberately damaged copy
/wsp-build examples/tasks/ring.py ... # or any task description
/wsp-improve examples/data/breast_cancer_mlp_damaged.pkl --data examples/data/breast_cancer.csv --target diagnosis
```

## Limitations
- wsp can only run forward passes on fully connected networks (MLPs). Other imported architectures come in as *opaque*: inspection and edits work, evaluation does not, so export and evaluate them in their own framework.
- `.pt`, `.npz` and `.safetensors` files don't record activations. Pass `--act` when importing them.
- Token counts are estimates (about 2.5 characters per token for numbers, 3.5 for prose).
- Pure weight-space programming works best when the task has structure an agent can reason about. On unstructured, high-dimensional data, expect the hybrid agent or the gradient baseline to win. `report.md` always shows that comparison.

## Development

```
pip install -e ".[dev]"
pytest
claude plugin validate .
```

## License
MIT
