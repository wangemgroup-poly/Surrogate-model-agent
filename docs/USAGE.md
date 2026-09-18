**English** | [中文](使用说明.md)

# Surrogate Model Agent — full manual

This tool trains a surrogate model on full-wave results you already have in CST, **optimises on
the surrogate**, freezes the proposed designs together with their predictions, and hands them to
CST for verification. It is a local command-line program: **no API key, no network access, no
cloud service**. Run it by hand, or let any LLM assistant with a terminal drive it following
[`../AGENTS.md`](../AGENTS.md).

The online loop follows Liu et al., IEEE TAP 2022 (DOI 10.1109/TAP.2022.3211732), "SB-SADEA".

---

## Contents

1. [What it solves](#1-what-it-solves)
2. [Requirements and installation](#2-requirements-and-installation)
3. [Five-minute start without CST](#3-five-minute-start-without-cst)
4. [The workflow](#4-the-workflow)
5. [Configuration reference](#5-configuration-reference)
6. [Command reference](#6-command-reference)
7. [Inside a task directory](#7-inside-a-task-directory)
8. [Reading the report](#8-reading-the-report)
9. [Steering the search](#9-steering-the-search)
10. [Safety mechanisms](#10-safety-mechanisms)
11. [Troubleshooting](#11-troubleshooting)
12. [Known limitations](#12-known-limitations)
13. [Differences from the paper](#13-differences-from-the-paper)

---

## 1. What it solves

Optimising directly inside CST spends one solver run per trial design. A surrogate trained on
the runs you already have scores thousands of candidates in seconds, so each new solver run goes
to a design that is *predicted* to be good. This tool automates that loop while keeping a hard
separation:

- the surrogate **proposes** — parameters and predictions are written to disk and frozen before
  any solver starts;
- **CST decides** — only measured full-wave results determine whether a design passes;
- every number in the report traces back to a file hash, a run ID and a batch.

## 2. Requirements and installation

| Item | Requirement |
|---|---|
| OS | Windows (CST result library and macro runner are Windows-only) |
| CST | CST Studio Suite 2025, **frequency-domain** solver, parametric project |
| Python | 3.12 (required by the CST 2025 Python result library) |
| Packages | numpy, scipy, scikit-learn, joblib, threadpoolctl |
| Optional | PyTorch, only for the variational-inference BNN (`sb_sadea.surrogate = "bnn"`) |

```bat
setup.cmd                                  REM creates .venv with py -3.12 and installs deps
setup.cmd "C:\path\to\python3.12.exe"      REM when the py launcher is unavailable
```

`setup.cmd` also installs a CUDA 12.6 PyTorch build (falls back to CPU without a GPU); delete
that line if you only use the GP surrogates.

Verify:

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

21 tests should pass, with no CST and no GPU required.

## 3. Five-minute start without CST

```bat
.venv\Scripts\python.exe examples\demo_no_cst.py
```

The script builds a synthetic three-parameter "structure", imports 60 analytic samples as if
they were full-wave results, selects and trains a surrogate, and freezes one SB-SADEA candidate —
the whole surrogate half of the loop, on any machine.

## 4. The workflow

```
init ──► import-cst / import-json ──► train ──► propose ──► validate ──► ingest ──► train
                                                                  (`run` loops automatically)
```

**Setting up a new structure — the order that avoids the common traps:**

1. **Find every project version of the same structure**, not just the newest file. Older
   versions frequently hold better designs; a surrogate trained without them optimises inside a
   blind spot. Use `bounds --project A.cst --project B.cst` to compare their optimiser ranges.
2. **Choose the parameter box**: the union of those optimiser ranges, **and** the parameter
   values of the saved samples themselves. A project's optimiser range is often narrowed after a
   campaign, which silently rejects good older samples at import.
3. **Check band endpoints.** With `"band": [f1, f2]`, both f1 and f2 must be exact frequency
   samples of the stored curve, otherwise extraction fails.
4. **Set `"simulation_budget": 0`** so nothing can start the solver by accident.
5. `init` → `import-cst` → `train` → `propose`. Look at the data volume, the pass rate per
   metric, the best existing design, the model's held-out error and the first candidate — then
   decide on a simulation budget.
6. `run --rounds N --count 1 --budget N`.

## 5. Configuration reference

Template: [`../examples/config_template.json`](../examples/config_template.json).

### Required

| Field | Meaning |
|---|---|
| `schema_version` | Always 1 |
| `project` | Full path to the source `.cst` file |
| `cst_install` | CST installation directory |
| `parameters` | `{"name": [lower, upper]}`. Must cover every independent parameter that varies; do not list fixed ones |
| `metrics` | Output definitions, see below |

### Metric definitions

**Curve metrics** (`"kind": "curve"`):

| Field | Values | Meaning |
|---|---|---|
| `tree` | e.g. `1D Results\\S-Parameters\\S1,1` | CST result tree path |
| `transform` | `real` / `abs` / `db20` | How the stored values are read |
| `reduce` | `max` / `min` / `ripple` / `at` | Reduction over the band; `at` needs `frequency` |
| `band` | `[f1, f2]` | Band in GHz; endpoints must be sample points |
| `op` + `limit` | `<=` `<` `>=` `>` + value | Constraint; omit to report only |
| `scale` | default 1 | Error normalisation used in model selection |
| `penalty_weight` | default 1 | Penalty-fitness weight: how bad a violation is |

**Side-lobe metrics** (`"kind": "sll_phi_cut"`): give `frequency`, `phi`, `port`. The value is
the ratio of the highest to the second-highest local peak of the stored φ-cut.

### Optional

| Field | Default | Meaning |
|---|---|---|
| `simulation_budget` | 0 | Solver starts allowed for the whole task lifetime |
| `cpus` / `timeout_minutes` | 16 / 45 | Settings for isolated-copy solves (source project untouched) |
| `proposal_method` | `random_pool` | `random_pool` / `optimize` / `sb_sadea` |
| `search_domain` | `trust_region` | For `optimize`: trust region or the whole box |
| `sb_sadea` | `{"surrogate":"gp"}` | Also `lambda`, `tau`, `F`, `CR`, `omega`, BNN hyper-parameters |
| `include_global_gp` | false | Add the full-sample GP to the candidate algorithms |
| `trust_radius` / `search_pool` | 0.05 / 5000 | `random_pool` radius and pool size |
| `conservative_sigma` | 1 | `optimize` judges constraints at μ + kσ |
| `extrapolation_ratio` | 2 | Flag a candidate farther than k × median spacing from the data |
| `disagreement_limit` | 2 | Flag when other algorithms disagree by more than k × scale |
| `materialize` | `{"on":"improvement","max":3}` | Auto write-back into the source project; `on` is `improvement` / `pass` / `never` |

## 6. Command reference

All commands: `.venv\Scripts\python.exe agent.py <command> --task <dir> [options]`.
Running `run_agent.cmd` without arguments opens an interactive menu.

| Command | Notes |
|---|---|
| `init --config` | Creates the task and registers file hash, geometry signature, fixed-parameter signature |
| `bounds --project ...` | Read-only; optimiser ranges and their union |
| `inspect` | Run IDs, parameter definitions, result tree items |
| `import-cst [--project ...]` | Extra projects are structure-checked before merging; sample IDs get a version prefix |
| `import-json --file` | External full-wave metrics (same structure and solver settings) |
| `train [--force]` | Reuses the existing model when data and config are unchanged |
| `propose --count N` | Freezes N candidates (`sb_sadea` requires N = 1) |
| `seed --count N` | Latin-hypercube plan when you have no data |
| `validate --budget N` | **Starts CST**: isolated-copy solve, verification, ingest |
| `resume --budget N` | Clears a pause and continues |
| `run --rounds R --count N --budget B` | **Starts CST**: the automatic loop; stops on a compliant design |
| `pause` | Stop after the current solve |
| `discard` | Abandon candidates that have not started |
| `materialize [--design ID]` | **Starts CST**: re-solve a verified design inside the source project and save it |
| `rebaseline` | Re-register a source project that only gained saved runs |
| `report` / `status` | Regenerate the report |

## 7. Inside a task directory

```
tasks/my_task/
├─ config.json          targets, parameter box, method, registered signatures
├─ data.json            every ingested design: parameters, metrics, provenance
├─ state.json           phase, current model, current batch, simulation counter
├─ events.jsonl         append-only decision log
├─ import_summary.json  read/added/repeats/rejected/without-results per project
├─ bounds_advice.json   union-range advice when imported data exceeds the box
├─ models/model_NNN/    model.joblib, selection.json, data_snapshot.json
├─ batches/batch_XXXX/  batch.json (frozen predictions), cst/<candidate>/ (isolated copy)
├─ materialized.json    designs written back into the source project, with per-metric differences
├─ 报告.md / 报告.html    report
└─ tradeoff.svg         constraint trade-off scatter
```

**Never commit a task directory**: it contains absolute paths to your own projects and your
full-wave data. `.gitignore` already excludes it.

## 8. Reading the report

- The **algorithm comparison** is validation-set performance; the test region is scored once,
  after the choice, and never participates in it.
- **Held-out error** is offline error across parameter regions — *not* prospective accuracy on
  new designs. When it is large the report says explicitly that the model may guide sampling but
  cannot certify compliance.
- The **candidate batch table** shows frozen prediction, measured value, the difference, and the
  extrapolation / disagreement warnings.
- **Constraint trade-offs** lists, per metric, the best value overall and the best value among
  designs that satisfy every *other* constraint. A large gap means a real trade-off.
- **Parameter bounds** counts how often the top designs and the agent's own candidates press a
  box boundary. Persistent pressing means the box is limiting the result.
- **Convergence** shows the worst normalised margin over the agent's verified designs and where
  the last improvement happened.

## 9. Steering the search

| Symptom | Lever |
|---|---|
| The search keeps trading away a requirement you care about | `"penalty_weight": 2`–`3` on that metric, then retrain and re-propose |
| Best designs keep pressing the same bound | widen that bound — a box change requires a **new task** |
| Candidates far from data, predictions over-optimistic | raise `conservative_sigma`, or switch `search_domain` from `full` to `trust_region` |
| Rounds are volatile | usually a population spanning unrelated regions; consider a smaller `sb_sadea.lambda` (a deviation from the paper) |
| One metric is predicted badly | ignore it if its margin is large; if it is the bottleneck, split a wide band into sub-bands and model each (the paper's "fine supervision" idea) |

## 10. Safety mechanisms

- The source project must be **closed** (no `Model.lok`) before any copy or solve.
- Identity registered at `init` and re-checked before every solve: geometry files **and** the
  values of every fixed parameter the task does not optimise. Adding saved runs is fine; editing
  a fixed dimension or the geometry is not.
- Candidates solve in **isolated copies**; old results and mesh caches are not copied.
- After each solve: macro return value, `DONE` marker, process exit code, broadband convergence
  in the log, and a parameter read-back from the result file.
- Two budgets apply at once: the task's `simulation_budget` and the call's `--budget`.
- A failed or interrupted solve is **never retried automatically**; the scene is preserved.
- One write lock per task directory.
- Changing budget, CPU count or timeout does not invalidate models or frozen batches; changing
  targets, the box or the method does.

## 11. Troubleshooting

**Import reports "还存在变化的独立参数" (a varying independent parameter is missing)**
Some parameter varies between runs but is not in `parameters`. Add it, or filter those runs out.

**Many samples rejected at import**
They fall outside the task's parameter box. See `bounds_advice.json` for the union range and
create a **new task** with the wider box.

**`import-cst` refuses to merge another project**
The message names the reason: different geometry files, different fixed-parameter definitions,
or different fixed-parameter values. Identical geometry history is not sufficient — fixed
dimensions live in `Parameters.json`, not in the geometry files.

**Encoding error on a non-ASCII project path**
The CST Python result library cannot open such paths. The agent creates a read-only directory
junction under `C:\Users\Public\Documents` automatically; no action needed.

**A solve failed and the candidate is stuck in `running`**
Read `batches/*/cst/*/job.json`, `status.txt` and `project/Result/Model.log`:
- "Full rebuild failed" in the log → that parameter set is geometrically invalid; abandon it.
- Geometry, meshing and port calculation all fine but the process exited → almost always
  external (someone closed CST, a licence conflict). Move the crashed copy aside, set the
  candidate back to `pending`, and let `run` prepare a fresh copy (one budget unit).
The agent never retries by itself so that a repeated failure cannot silently drain your budget.

**`WinError 206` (path too long) during write-back**
CST result trees are deep. Copies already use extended-length paths; if it still fails, move the
task directory closer to the drive root.

**"样本不足以分组验证" (not enough samples to split)**
Fewer than `max(30, 3 × number of parameters)` samples. Import more data, or use `seed` +
`validate` within an explicit budget.

**"数据库样本少于λ或τ" (database smaller than λ or τ)**
Fewer than `4 × number of parameters` samples for SB-SADEA. Add data, or reduce
`sb_sadea.lambda` / `tau` (a documented deviation from the paper).

## 12. Known limitations

- CST parametric **frequency-domain** projects only.
- Metrics come only from the result trees you name and from stored φ-cuts.
- **Uncertainty is not calibrated**: σ is model dispersion, not a probability of passing.
- A model cannot be reused across structures; create a new task.
- Passing at sampled frequencies does not guarantee the whole band.
- "No compliant design found" means *within this box and budget*.
- The surrogate is only as good as its data coverage.

## 13. Differences from the paper

- The Latin-hypercube initialisation is skipped when full-wave data already exists.
- The paper does not state the hidden activation; tanh is used. Uncertainty comes only from
  weight sampling.
- The accepted manuscript prints τ = 4; the text's "at least 4×d training points" is taken to
  mean τ = 4d.
- Equation (11) does not state how the KL term is scaled. Taken literally the network degenerates
  to the neighbour mean on small local sets, so `kl_weight` is exposed (default 1 = literal).
- The default local surrogate is the GP (the paper's GP-ALCB variant, ω = 2); the BNN is optional
  (ω = 14).
- "Fine supervision" is not implemented.
- Penalty fitness follows equations (16)–(17) with per-metric configurable weights.

---

## Licence

Copyright (C) 2026 MiraDaddy, Wangemgroup. Released under the GNU General Public License v3.0
or later; see `LICENSE` and `NOTICE` in the repository root. The program comes with no warranty,
and any redistribution — modified or not — must be under the same licence with source available.
