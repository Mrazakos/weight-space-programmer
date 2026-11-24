---
name: wsp-improve
description: Improve an existing neural network by weight-space programming. The network can be one the wsp agents built, or one trained elsewhere (scikit-learn .pkl, PyTorch .pt state_dict / TorchScript / pickled module, torch.export .pt2, .safetensors, .npz). An agent reads the weights (a full dump if they fit in its context window, sandbox inspection otherwise), diagnoses failures, edits weights, optionally fine-tunes, and loops until the forward-pass or context budget runs out, over as many rounds as requested. Use when the user wants an already-trained or generated network improved or repaired.
argument-hint: <runs/NAME | model file> [--data FILE|FOLDER --target COL] [--mode auto|pure|hybrid] [--budget N] [--rounds R]
---

# /wsp-improve

Run improvement sessions on a network.

Arguments: `$ARGUMENTS`

## 1. Make sure the toolkit is installed
Run `python -c "import wsp"` (or the same with `python3`). If it fails, install the toolkit:
```
python -m pip install "weight-space-programmer[all] @ git+https://github.com/Mrazakos/weight-space-programmer"
```

## 2. Resolve the run
- **Existing run.** If the argument is a run directory (it contains `run.json`) or a name under `./runs/`, use it. Unless `--mode` was given, keep the mode stored in its `run.json`.
- **Model file.** A run with data is needed. Improvement can only be measured against data.
  - **Get the data.** Use `--data FILE --target COL` for tables and text, or `--data FOLDER` for images. If the data is missing, ask the user.
  - **Create the run.** Run `python -m wsp new <name> --data ... --mode <mode> --no-norm --budget <N>`. With `--no-norm`, the model gets raw features, or images as float NCHW in [0,1].
  - **Get consent for files that run code.** Pickles (`.pkl`/`.joblib`), full torch modules (TorchScript or pickled) and `.pt2` files execute code when loaded. Ask the user whether they trust the file before passing `--trust`.
  - **Import the model.** Tell the agent to run `python -m wsp import <name> <model> [--trust]`.
  - **Weights-only files.** A plain state_dict, `.npz` or `.safetensors` file of a non-MLP network imports as *opaque* (editable but not runnable). To make it runnable, ask the user for TorchScript (`torch.jit.script(model).save(...)`) or a `torch.export` program.
  - **MLP weight files.** For these, ask for the activations if they are not obvious, and pass `--act`.
  - **Text models.** These must use the run's tokenization, so warn the user.
- **Defaults.** The default mode is `auto`, the default `--budget` is 50 forward passes per session, and the default `--rounds` is 1.

## 3. Run the improvement sessions
For each round, from 1 to R:
- **Launch the agent** with the Agent tool. Use `subagent_type` `"weight-space-programmer:wsp-auto"`, `"…:wsp-pure"` or `"…:wsp-hybrid"`, matching the mode. If the namespaced names are unavailable, use the bare names. Prompt:
  "Improve workflow. Run: <path>. First run `python -m wsp session <path> --budget <N>`. That resets your forward-pass budget and context ledger and prints the previous handoff. Read the network: use a full dump if `status` says FULL DUMP FITS, otherwise use the sandbox inspection fallback. Improve the network until the forward budget or your context ledger is nearly used up, or progress stalls. Always write `handoff.md`. Do not run `report` unless this is the last round."
- **On the last round,** add: "This is the final round: finish with `best --set`, `report`, and `export` to the original format (or `.ts`)."
- **Each round is a fresh agent with a fresh context window.** `handoff.md` carries the state over. This is how improvement continues beyond one context window.
- **Stop early** if a handoff says progress has stalled with no open hypotheses.

## 4. Relay the result
Report:
- the starting versus best val metric
- the test metric against the gradient baseline
- what changed (hand edits versus fine-tuning)
- the rounds and forward passes used
- the path of the exported model
