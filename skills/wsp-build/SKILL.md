---
name: wsp-build
description: Build a neural network for a dataset or a plain-language task by weight-space programming. An agent sizes the network, then writes its weights directly. Pure mode allows no gradients; hybrid mode also allows gradient training. Use when the user gives data to predict, or describes a task a network should do, and wants a network created.
argument-hint: <data file | "task description"> [--target COL] [--mode pure|hybrid] [--budget N] [--name RUN]
---

# /wsp-build

Build a network with the weight-space-programmer agents.

Arguments: `$ARGUMENTS`

## 1. Make sure the toolkit is installed
Run `python -c "import wsp; print(wsp.__version__)"`, or the same with `python3`.

If the import fails, install the toolkit:
```
python -m pip install "git+https://github.com/Mrazakos/weight-space-programmer"
```
If you are working inside a clone of the repository, use `python -m pip install -e <repo>` instead.

## 2. Parse the arguments
- **Input.** If the first argument is an existing file (csv/tsv/parquet/json/jsonl/xlsx/npz), it is the **data**. Otherwise, treat the quoted text as a **task description**.
- **`--target COL`.** This names the column to predict. If it is missing for a data file:
  - Read the header, e.g. `python -c "import pandas as pd; print(pd.read_csv('FILE', nrows=3))"`.
  - Pick the obvious target, such as a column named target/label/y/class or the last column.
  - Tell the user which column you picked.
- **`--mode`.** The default is `pure`, which follows the blog post's rules: no gradients. `hybrid` also allows `wsp fit`.
- **`--budget`.** This is the number of forward passes. The default is 50.
- **`--name`.** This is the run name. If it is missing, derive a short name from the data file or the task.

## 3. Launch the agent
Use the Agent tool:
- **Agent type.** Use `subagent_type: "weight-space-programmer:wsp-pure"` for pure mode and `"weight-space-programmer:wsp-hybrid"` for hybrid mode. If those namespaced names are not available, use `wsp-pure` or `wsp-hybrid`.
- **Prompt for a data file:**
  "Workflow A (build). Data: <absolute path>, target: <COL>, run name: <name>, mode: <mode>, forward-pass budget: <N>. Create the run with `python -m wsp new <name> --data <path> --target <COL> --mode <mode> --budget <N>`. Decide the network size yourself and justify it. Finish with `report` and `handoff.md`."
- **Prompt for a task description:** the same, but "Task description: <text>. Write a faithful task generator first, then `new <name> --task <file> --mode <mode> --budget <N>`."
- **Working directory.** Runs are written to `./runs/` in the current working directory. Set `WSP_RUNS` to change this.

## 4. Relay the result
When the agent finishes, summarize for the user:
- the chosen architecture and why
- the val and test metrics, compared with the gradient baseline from `runs/<name>/report.md`
- the forward passes used
- the paths to `report.md`, `log.md` and `handoff.md`
- the `/wsp-improve runs/<name>` command for further improvement
