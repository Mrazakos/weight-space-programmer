---
name: wsp-improve
description: Improve an existing neural network by weight-space programming. The network can be one the wsp agents built, or one trained elsewhere (.npz, scikit-learn .pkl, torch .pt, .safetensors). An agent reads the weights (a full dump if they fit in its context window, sandbox inspection otherwise), diagnoses failures, and edits the weights, looping until the forward-pass or context budget runs out. Use when the user wants an already-trained or generated network improved or repaired.
argument-hint: <runs/NAME | model file> [--data FILE --target COL] [--mode pure|hybrid] [--budget N] [--rounds R]
---

# /wsp-improve

Run improvement sessions on a network.

Arguments: `$ARGUMENTS`

## 1. Make sure the toolkit is installed
Run `python -c "import wsp"` (or the same with `python3`). If it fails, install the toolkit:
```
python -m pip install "git+https://github.com/Mrazakos/weight-space-programmer"
```

## 2. Resolve the run
- **Existing run.** If the first argument is a run directory (it contains `run.json`) or the name of one under `./runs/`, use it. Unless `--mode` was given, keep the mode stored in its `run.json`.
- **Model file.** If the argument is a model file, a run with data is needed:
  - **Get the data.** Use `--data FILE --target COL`. If either is missing, ask the user. An improvement can only be measured against data.
  - **Create the run.** Run `python -m wsp new <name> --data FILE --target COL --mode <mode> --no-norm --budget <N>`. The imported model brings its own scaler, if it had one.
  - **Import pickles only if trusted.** For `.pkl` or `.joblib`, ask the user whether they trust the file before you pass `--trust`. Loading a pickle executes code.
  - **Import the model.** Tell the agent to run `python -m wsp import <name> <model> [--trust]`.
  - **Activations.** For `.npz`, `.pt` or `.safetensors`, ask the user for the activations if they are not obvious, and pass `--act`.
- **Defaults.** The default mode is `pure`, the default `--budget` is 50 forward passes per session, and the default `--rounds` is 1.

## 3. Run the improvement sessions
For each round, from 1 to R:
- **Launch the agent.** Use the Agent tool with `subagent_type: "weight-space-programmer:wsp-pure"` or `"weight-space-programmer:wsp-hybrid"`. If the namespaced names are not available, use `wsp-pure` or `wsp-hybrid`. Prompt:
  "Workflow B (improve). Run: <path>. First run `python -m wsp session <path> --budget <N>`. That resets your forward-pass budget and context ledger and prints the previous handoff. Read the network: use a full dump if `status` says FULL DUMP FITS, otherwise use the sandbox inspection fallback. Improve the network until the forward budget or your context ledger is nearly used up, or progress stalls. Always write `handoff.md`. Do not run `report` unless this is the last round."
- **On the last round,** add: "This is the final round: finish with `best --set` and `report`."
- **Each round is a fresh agent with a fresh context window.** The ledger resets and `handoff.md` carries the state over. This is how improvement continues past one context window.
- **Stop early** if a round's handoff says progress has stalled with no open hypotheses.

## 4. Relay the result
Report:
- the starting versus best val metric
- the test metric against the gradient baseline (from `report.md`)
- what was changed, from the log's design summary
- the forward passes and rounds used
- export instructions, e.g. `python -m wsp export <run> fixed.pkl --trust` to write the edited weights back into the original scikit-learn pipeline, or `.pt` / `.npz`
