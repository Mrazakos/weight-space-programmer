---
name: wsp-build
description: Get a working neural network for any prediction task. Give it data (a table, a folder of labelled images, or a table with a text column) or a plain-language task description. An agent picks the architecture (MLP, CNN, bag-of-words, 1D-CNN, transformer) and size, writes the weights directly where it can reason about the task, uses gradient training where it can't, and delivers a validated, exported model. Use when the user wants an ML model or network built for a task.
argument-hint: <data file | image folder | "task description"> [--target COL] [--mode auto|pure|hybrid] [--budget N] [--name RUN]
---

# /wsp-build

Build a network with the weight-space-programmer agents.

Arguments: `$ARGUMENTS`

## 1. Make sure the toolkit is installed
Run `python -c "import wsp, torch, PIL; print(wsp.__version__)"`, or the same with `python3`. If it fails, install everything:
```
python -m pip install "weight-space-programmer[all] @ git+https://github.com/Mrazakos/weight-space-programmer"
```
- **Inside a clone of the repository:** run `python -m pip install -e "<repo>[all]"` instead.
- **Tabular-only work:** torch is optional. If the user declines the PyTorch download, skip it.

## 2. Parse the arguments
- **The input:**
  - **A folder** whose subfolders are classes of images is **image data**.
  - **A file** (csv/tsv/parquet/json/jsonl/xlsx/npz) is **table data**. The toolkit auto-detects a long free-text column as text.
  - **Otherwise** the argument is a **task description**.
- **`--target COL`.** This is required for tables. If it is missing:
  - Read the header, e.g. `python -c "import pandas as pd; print(pd.read_csv('FILE', nrows=3))"`.
  - Pick the obvious target: a column named target/label/y/class, or the last column.
  - Tell the user which column you picked.
- **`--mode`:**
  - `auto` (default): the agent writes weights where the task is programmable and trains where it isn't.
  - `pure`: no gradients at all, following the blog post's rules.
  - `hybrid`: gradients and hand edits used freely.
- **`--budget`.** This is the number of forward passes. The default is 50.
- **`--name`.** This is the run name. Derive a short name if it is missing.

## 3. Launch the agent
Use the Agent tool:
- **Agent type.** Use `subagent_type: "weight-space-programmer:wsp-auto"`, `"weight-space-programmer:wsp-pure"`, or `"weight-space-programmer:wsp-hybrid"`, matching the mode. If the namespaced names are unavailable, use `wsp-auto` / `wsp-pure` / `wsp-hybrid`.
- **Prompt for data:**
  "Build workflow. Data: <absolute path>[, target: <COL>], run name: <name>, mode: <mode>, forward-pass budget: <N>. Create the run with `python -m wsp new <name> --data <path> [--target <COL>] --mode <mode> --budget <N>`. Choose the architecture and size yourself and justify them. Finish with `report`, an exported model (`export <name> runs/<name>/model.ts`), and `handoff.md`."
- **Prompt for a task description:**
  "Task description: <text>. If the description fully specifies the input-to-output mapping, write a faithful task generator (set `MODALITY='text'` or `'image'` for raw text or image inputs) and run `new <name> --task <file> --mode <mode> --budget <N>`. If it needs real-world data (photos, real reviews, measurements), stop and say exactly what data is needed."
- **Working directory.** Runs go to `./runs/`. Set `WSP_RUNS` to change this.

## 4. Relay the result
Summarize for the user:
- the modality and the chosen architecture, with the reason
- how the weights were obtained: hand-written, trained, or both
- val and test metrics against the gradient baseline, from `runs/<name>/report.md`
- the exported model path and how to use it on new data: `python -m wsp predict runs/<name> <new data> --out preds.csv`
- `/wsp-improve runs/<name>` for further improvement
