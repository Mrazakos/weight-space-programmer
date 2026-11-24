# Weight-Space Programmer

**Get a working neural network for any prediction task, and see exactly how its weights were made.** Give a Claude Code agent a table, a folder of labelled images, a text dataset, or a plain-language task description. The agent:

- picks the architecture (MLP, CNN, bag-of-words, 1D-CNN, transformer) and the size
- **writes the weights directly** where it can reason about the task
- trains with gradients where it can't
- reads and edits the result
- hands you a validated model you can export and run on new data

It can also **improve networks you already have**: scikit-learn, PyTorch state_dicts, TorchScript, `torch.export` programs, pickled modules, `.safetensors` and `.npz`.

Inspired by *[Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights](https://rikiyat.github.io/blog/weight-space-programming/)* (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, 2026). That post had a language model write networks from all-zero weights with no gradient training, and repair trained policies and language models by editing a small fraction of their weights. This repository is an independent implementation of that workflow. It is not affiliated with the authors.

## What you get

| Piece | What it does |
|---|---|
| `/wsp-build <data or "task">` | Builds a network for tabular data, images (class subfolders), text (auto-detected text column), or a task description. Delivers `report.md`, an exported model and a `predict` command. |
| `/wsp-improve <run or model>` | **Improvement mode** for generated or externally trained networks. It loops until the forward-pass or context budget is spent, then resumes in fresh rounds through `handoff.md`. |
| `wsp-auto` agent (default) | Writes weights by reasoning when the task is programmable (rules, geometry, templates, lexicons). Trains when it is statistical (natural images, free text). Makes surgical hand edits on top of both. |
| `wsp-pure` agent | The blog post's rules: no gradients, an all-zero start, metered forward passes. |
| `wsp-hybrid` agent | Mixes gradients and hand edits freely. |
| `wsp` toolkit | A metered command-line sandbox that the agents drive: data loading, sizing, weight dumps, weight programs, edits, evaluation, inspection, training, import/export, `predict`, and a final report. |

## Install

In Claude Code:

```
/plugin marketplace add Mrazakos/weight-space-programmer
/plugin install weight-space-programmer@weight-space-programmer
```

The agents call the Python toolkit (Python ≥ 3.10). The skills install it automatically if it's missing, or you can install it yourself:

```
pip install "weight-space-programmer[all] @ git+https://github.com/Mrazakos/weight-space-programmer"
```

`[all]` pulls in PyTorch, Pillow, scikit-learn and safetensors. Tabular-only use needs only numpy and pandas: `pip install "git+https://github.com/Mrazakos/weight-space-programmer"`.

## Usage

```
/wsp-build data/churn.csv --target churned                       # tabular
/wsp-build data/photos/                                          # images: photos/<class>/*.jpg
/wsp-build data/tickets.csv --target team                        # text column auto-detected
/wsp-build "is a 2-D point inside the ring 1 <= r <= 2?"         # task description only
/wsp-build data/reviews.csv --target sentiment --mode pure       # no gradients at all

/wsp-improve runs/churn --rounds 3
/wsp-improve models/resnet_small.pt --data data/photos/          # TorchScript / torch.export / pickled module
/wsp-improve models/clf.pkl --data data/val.csv --target label   # scikit-learn MLP or pipeline
```

To use the result:

```
python -m wsp predict runs/churn new_customers.csv --out preds.csv   # same preprocessing as training
python -m wsp export runs/photos model.ts                            # standalone TorchScript (raw images in)
```

Each run lives in `./runs/<name>/`. Set `WSP_RUNS` to put runs elsewhere. A run directory holds:

- `nets/vNNN.*`: every version
- `history.jsonl`: version tree and evaluations
- `programs/`: the weight program behind each version
- `log.md`: hypotheses and design notes
- `handoff.md`: state for the next session
- `report.md`: held-out test score next to a gradient-trained baseline with the same architecture

## How the agents decide

**Architecture and size.** `wsp size` lists candidates for the data type. Each comes with its parameter count, parameters per training sample, and the cost of a full weight dump against the remaining context budget. The agent picks one and records why in `log.md`.

| Data | Programmable choices (hand-written weights) | Learned choices |
|---|---|---|
| Tabular | small MLP whose units are hyperplanes, thresholds, logic gates, bumps, radii | MLP sized to about 1–2 params per sample |
| Images | `linear` templates built from class-mean images, a CNN with designed filters (edges, blobs, colours) | `cnn` width 16–64, depth 2–3 |
| Text | `bow --dim 1` lexicon scorer, per-class lexicons, `conv1d` negation detectors | `bow`, `cnn1d`, `transformer` (by data size) |

You can also describe any custom stack as JSON (`init --layers`). The available layers are `dense`, `conv2d`, `maxpool2d`, `avgpool2d`, `flatten`, `global_pool`, `embedding`, `posembed`, `conv1d`, `transformer`, `layernorm` and `act`.

**Weights.** The auto agent profiles the data and looks at examples. For images it reads the contact sheets wsp renders. Then it chooses:
- **Programmable task:** it writes the weights in a numpy weight program that never sees the data.
- **Statistical task, or when the program falls short:** it trains, optionally starting from its programmed network.
- **Either way:** it then reads the weights, diagnoses the worst examples, and makes targeted fixes that it keeps or reverts.

**Context-window-aware improvement.**
- **Context ledger:** every wsp output is charged to a per-session ledger. The default budget is a 1M-token window minus a 25% reserve; override it with `WSP_CONTEXT_TOKENS` or `--context-tokens`.
- **Networks that fit:** if the full weight dump fits, the agent reads every weight.
- **Networks that don't fit:** it uses **sandbox inspection** instead: `inspect stats | svd | unit | tokens | probe` plus sliced dumps. It locates the component responsible before editing, the same approach the blog post used to edit Qwen.
- **Ending a session:** when a budget runs out, the agent writes `handoff.md`. The next `/wsp-improve` round starts a fresh agent with a fresh context that resumes from it.

## Example: programmed vs trained

Results on test data held out until the end:

| Task | Network | How the weights were made | Test | Same-architecture gradient baseline |
|---|---|---|---|---|
| 3-bit parity | 3-3-1, 16 params | hand-written counting units | 100% | 79% |
| ring 1 ≤ r ≤ 2 (from a sentence) | 2-2-1-1, 11 params, `square`/`abs` | agent-written: computes \|r²−2.5\| exactly | 100% | 98.5% |
| synthetic reviews | bag-of-words, 48 params | 20-word hand-written lexicon | 85.8% val (fails on negation, which a `conv1d` detector fixes) | n/a |
| 8×8 handwritten digits (auto agent, unattended) | CNN, 6,090 params | 16 hand-designed first-layer filters (edges, strokes, blobs), frozen; rest trained | 98.9% | 98.6% |

## The `wsp` toolkit

```
python -m wsp new <run> --data F --target COL | --data IMAGE_FOLDER | --task gen.py   [--mode auto|pure|hybrid]
python -m wsp session|status|profile|sample|size <run>
python -m wsp init <run> --arch 8-3 | --preset cnn --width 16 | --layers layers.json   [--random]
python -m wsp dump <run> [--layer NAME] [--tensor T] [--rows a:b]   # weights as text, chunked
python -m wsp write <run> program.py                                # numpy-only weight program
python -m wsp set|patch|resize <run> ...                            # surgical edits
python -m wsp eval <run> [--activations]                            # METERED forward pass
python -m wsp inspect <run> stats|unit|tokens|slice|svd|probe
python -m wsp fit <run> [--base v] [--freeze names]                 # auto/hybrid modes only
python -m wsp history|revert|best <run>
python -m wsp import|export <run> model.{pkl,pt,pt2,ts,safetensors,npz}
python -m wsp predict <run> new_data [--out preds.csv]
python -m wsp report <run>
```

A weight program is a plain Python file. For text runs, `spec["vocab"]` is available:

```python
def program(spec, W):
    idx = {w: i for i, w in enumerate(spec["vocab"])}
    for w in ["great", "excellent", "love"]:
        W["emb.E"][idx[w], 0] = 1.0       # positive words
    for w in ["awful", "broken", "hate"]:
        W["emb.E"][idx[w], 0] = -1.0      # negative words
    W["out.W"][0, 0] = 40.0                # mean polarity -> confident sigmoid
    return W
```

### Rules enforced in pure mode
- `fit` and random initialization are refused.
- Weight programs pass an AST guard: only `numpy` and `math` imports, no file/OS access, no `eval`/`exec`/`getattr`, no dunder attributes. Programs never receive data.
- Forward passes (`eval`, `inspect probe`) draw on a per-session budget. The default is 50.
- The test split is held out until `report`.

The AST guard enforces the rules for a cooperating agent. **It is not a security sandbox.** Loading pickles, full torch modules or `.pt2` files executes code, so wsp requires `--trust` for them.

## Examples

```
python examples/make_datasets.py
```

This writes into `examples/data/`:
- tables: `iris.csv`, `breast_cancer.csv`, `diabetes.csv`
- images: `digits/` (8×8 digit images in class folders)
- text: `reviews.csv` (synthetic reviews with negation)
- models: a trained and a deliberately damaged scikit-learn model

There are also task generators in `examples/tasks/`:
- `parity3.py`: tabular logic
- `ring.py`: geometry
- `negation_sentiment.py`: text, with `MODALITY='text'`

## Limitations
- **Forward passes on imported models:** these run only for MLPs and for models imported as TorchScript, `torch.export`, or a pickled module. Weights-only state_dicts of other architectures import as *opaque*: editable, but not runnable.
- **Inspecting imported torch modules:** `stats`, `slice` and `svd` work; unit-level inspection and probes don't.
- **Text tokenization:** text uses word- or character-level tokenization built from the training split. Imported text models only make sense if they share that vocabulary.
- **No generation:** there are no sequence-to-sequence or generative tasks; wsp covers classification and regression only.
- **Training speed:** training runs on CPU unless PyTorch sees a GPU. Large image sets are slow, so use `--image-size` and `fit --max-minutes`.
- **Token counts** are estimates.
- **Where pure programming wins:** pure weight programming shines on structured tasks. On natural images or open-ended text, expect the auto agent to train. `report.md` always shows the gradient-baseline comparison.

## Development

```
pip install -e ".[dev]"
pytest
claude plugin validate .
```

## License
MIT
