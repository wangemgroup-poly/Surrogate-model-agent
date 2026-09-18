**English** | [中文](README.zh-CN.md)

# Surrogate Model Agent for CST

*Surrogate-assisted optimisation for any CST parametric frequency-domain model.*

A local, fully offline agent that learns a surrogate model from CST Studio Suite full-wave
results, **optimises on the surrogate**, freezes the proposed designs, and then lets **CST
decide** whether they are real. It is not tied to any particular component: anything whose goals
can be computed from 1D results — filters, couplers, matching networks, waveguide components,
frequency-selective surfaces, antennas — is configured the same way, by naming result trees. It runs as a plain command-line tool — no API keys, no network
access, no cloud service. Any LLM assistant with shell access (Claude Code, Cursor, Copilot
Chat, …) can drive it by reading [`AGENTS.md`](AGENTS.md); you can also run it by hand.

The optimisation loop follows Liu et al., *An Efficient Method for Antenna Design Based on a
Self-Adaptive Bayesian Neural Network Assisted Global Optimization Technique*, IEEE TAP 2022
(DOI 10.1109/TAP.2022.3211732), referred to below as **SB-SADEA**.

---

## What problem this solves

Full-wave optimisation inside CST spends one solver run per trial design. A surrogate model
trained on runs you already have can rank thousands of candidate designs in seconds, so each
new solver run is spent on a design that is *predicted* to be good. The agent automates that
loop while keeping a hard separation:

* the surrogate **proposes** — predictions are frozen to disk before any solver starts;
* CST **disposes** — only measured full-wave results decide whether a design passes;
* every claim in the generated report is traceable to a stored file hash, run ID and batch.

The loop has been exercised on waveguide slot arrays, where it reached a fully compliant design
in 34 solver runs starting from designs that missed the worst-band S11 target by more than 3 dB.
Nothing in the workflow is specific to that problem.

---

## Requirements

| Item | Requirement |
|---|---|
| OS | Windows (the CST result library and the VBA macro runner are Windows-only) |
| CST | CST Studio Suite 2025, **frequency-domain** solver, parametric project |
| Targets | Anything computable from stored 1D results (S-parameters, transmission, gain tables, group delay, efficiency …). An optional metric type reads far-field φ-cuts for antenna side-lobe level |
| Python | 3.12 — required by the CST 2025 Python result library |
| Packages | numpy, scipy, scikit-learn, joblib, threadpoolctl (see `requirements.txt`) |
| Optional | PyTorch, only for the variational-inference BNN surrogate (`sb_sadea.surrogate = "bnn"`) |

No API keys, no accounts, no outbound connections. Everything runs on your machine against
your own CST installation.

## Install

```bat
setup.cmd                                  REM creates .venv with Python 3.12 and installs deps
setup.cmd "C:\path\to\python3.12.exe"      REM if py -3.12 is not on PATH
```

`setup.cmd` also installs a CUDA 12.6 PyTorch build; it falls back to CPU automatically and is
only needed for the optional BNN surrogate. Delete that line from `setup.cmd` to skip it.

Verify the installation:

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

21 tests should pass. They are self-contained: no CST installation and no GPU required.

## Try it without CST

```bat
.venv\Scripts\python.exe examples\demo_no_cst.py
```

The demo builds a synthetic three-parameter "structure", imports 60 analytic samples as if they
were full-wave results, selects and trains a surrogate, and freezes one SB-SADEA candidate —
the whole surrogate half of the loop, on any machine.

---

## The workflow

```
init ──► import-cst / import-json ──► train ──► propose ──► validate ──► (repeat)
                                        ▲                      │
                                        └──── ingest ──────────┘
                             `run` executes this loop automatically
```

| Command | What it does |
|---|---|
| `setup [--project X.cst] [--lang en\|zh]` | **Guided setup**: inspect the project, check it, define targets interactively, advise whether a target needs splitting into sub-bands, write the config and create the task |
| `init --task DIR --config FILE` | Create a task; register the source project's file hash, geometry signature and fixed-parameter signature |
| `bounds --project A.cst [--project B.cst]` | Print each project's optimiser ranges and their union — useful before writing a config |
| `inspect --task DIR` | Dump run IDs, parameter definitions and result tree items of the source project |
| `import-cst --task DIR [--project X.cst ...]` | Read saved runs. Extra `--project` files are merged after a structure check; sample IDs get a version prefix |
| `import-json --task DIR --file rows.json` | Import full-wave metrics prepared elsewhere (same structure and solver settings) |
| `train --task DIR [--force]` | Compare the eligible algorithms on a held-out region, pick one, refit on all data, write `models/model_NNN/` |
| `propose --task DIR --count N` | Freeze N candidates (parameters, predictions, σ, model hash, data hash, config signature) |
| `seed --task DIR --count N` | Latin-hypercube initial sampling plan when you have no data yet |
| `validate --task DIR --budget N` | Solve pending candidates in isolated copies of the project, verify, ingest |
| `resume --task DIR --budget N` | Clear a pause and continue collecting/validating |
| `run --task DIR --rounds R --count N --budget B` | train → propose → validate → train, repeated; stops when a design passes everything |
| `pause --task DIR` | Stop after the current solve finishes |
| `discard --task DIR` | Abandon candidates that have not started solving |
| `materialize --task DIR [--design ID]` | Re-solve one verified design **inside the source project** and save it there so you can inspect curves in CST |
| `rebaseline --task DIR` | Re-register the source project after it gained saved runs, when the geometry is unchanged |
| `report` / `status --task DIR` | Regenerate `报告.md` / `报告.html` |

Run `run_agent.cmd` with no arguments for an interactive menu instead.

### Data in, data out

A task directory is self-describing:

```
tasks/my_task/
├─ config.json            targets, parameter box, method, registered signatures
├─ data.json              every ingested design: parameters, metrics, provenance
├─ state.json             phase, current model, current batch, simulation counter
├─ events.jsonl           append-only log of every decision
├─ models/model_NNN/      model.joblib, selection.json (algorithm comparison), data_snapshot.json
├─ batches/batch_XXXX/    batch.json (frozen predictions), cst/<candidate>/ (isolated project copy)
├─ materialized.json      designs written back into the source project
├─ 报告.md / 报告.html      human-readable report
└─ tradeoff.svg           constraint trade-off scatter (dependency-free SVG)
```

---

## Surrogate models

`train` compares the algorithms that are eligible for the data size, then picks by validation
error normalised per metric `scale`, penalised for *false passes*, breaking near-ties in favour
of the faster model. The test region is only scored after the choice is made.

| Algorithm | Eligibility |
|---|---|
| Local ARD-Matérn Gaussian process (120 nearest points) | ≤ 1800 samples, ≤ 30 inputs, ≤ 16 outputs |
| RBF kernel ridge regression | ≤ 3000 samples |
| Extremely randomised trees | always |
| Bayesian last-layer network (approximate, not a full BNN) | always |
| Full-sample ARD-Matérn Gaussian process | opt-in `include_global_gp`, ≤ 600 samples, ≤ 20 inputs, ≤ 8 outputs |

Data are split by parameter region (KMeans anchors fixed at first training, so one new sample
cannot permute the folds); near-duplicate designs are purged across splits.

## Candidate generation

`proposal_method` selects the strategy:

* **`random_pool`** (default) — sample a trust-region ball around the best known design, score
  every point with the deployed model, pick candidates by predicted improvement, optimistic
  LCB exploration, and model/neighbour disagreement.
* **`optimize`** — run differential evolution directly on the surrogate. `search_domain` is
  `trust_region` or `full`; constraints are judged at μ + `conservative_sigma`·σ so an
  unsupported prediction with a large σ is not selected on its mean alone.
* **`sb_sadea`** — the paper's online loop: take the λ best designs by penalty fitness, generate
  λ children with DE/current-to-best/1, train a local surrogate per child on its τ nearest
  samples, and apply self-adaptive LCB. One full-wave simulation per iteration (`--count 1`).
  `λ = τ = 4d` by default; `surrogate` is `gp` (ω = 2) or `bnn` (ω = 14).

Every frozen candidate carries warnings: its distance to the nearest sample relative to the
median nearest-neighbour spacing (`extrapolation_ratio`), and a cross-check in which the other
algorithms predict the same design (`disagreement_limit`). Warnings never reorder candidates —
they tell you how much to trust a prediction.

## Steering trade-offs

Objectives are constraints with a limit; the penalty fitness is a weighted sum of violations.
Give a metric `"penalty_weight": 2` to tell the search that violating it is twice as bad. This
is the lever to use when the search keeps trading one requirement away for another — in the
slot-array study, weighting the upper-band S11 and the side-lobe level moved the anchor from a
design that sacrificed them to one that kept them, and the loop then closed the remaining gap.

---

## Safety rules the agent enforces

* The source project must be **closed** (no `Model.lok`) before any copy or solve.
* Its identity is registered at `init` and re-checked before every solve: geometry files
  (`Model.mod`, `ModelHistory.json`, solver properties) **and** the values of every fixed
  parameter the task does not optimise. Adding saved runs is fine; editing a fixed dimension or
  the geometry is not.
* Candidates are validated in **isolated copies**; old results and mesh caches are not copied.
* After each solve the agent checks the macro return value, the `DONE` marker, the process exit
  code, broadband convergence in the log, and reads every parameter back from the result file.
* Two budgets apply at once: `simulation_budget` for the whole task and `--budget` for the call.
* A failed or interrupted solve is **not** retried automatically — the scene is preserved so you
  can diagnose it before spending more solver time.
* One write lock per task directory.
* Changing budget, CPU count or timeout does not invalidate trained models or frozen batches;
  changing targets, the parameter box or the method does.

## Limitations — read before trusting a number

* Frequency-domain CST projects only. Other solvers need a new adapter.
* Metrics are read from the result trees you name and from saved φ-cut far-field data. The
  side-lobe level is the ratio of the highest to the second-highest local peak of a stored cut.
* Uncertainty is **not calibrated**. σ is a model dispersion, not a probability of passing.
* A model trained on one structure cannot be reused for another; create a new task.
* Passing at sampled frequency points is not a guarantee between them.
* "No compliant design found" means *not found within this parameter box and budget*.
* The surrogate is only as good as its data coverage: gather results from **every version of the
  same structure** before optimising, not just the newest project file.

## Licence, authorship and citation

Copyright (C) 2026 MiraDaddy, Wangemgroup — <https://github.com/wangemgroup-poly>

Licensed under the **GNU General Public License v3.0 or later** — see [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE). This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. If you redistribute it, modified or not, you must pass on the same freedoms and make
the source available under the same licence.

CST Studio Suite is a product of Dassault Systèmes. No CST component is included here; its
Python libraries are loaded at run time from your own installation.

If you publish results produced with the SB-SADEA proposal method, cite the original paper
(DOI 10.1109/TAP.2022.3211732). This implementation is not a complete reproduction of that
paper; the differences are listed in [`docs/USAGE.md`](docs/USAGE.md) and in
[`CHANGELOG.md`](CHANGELOG.md).

## Documentation map

| | English | 中文 |
|---|---|---|
| Overview | [`README.md`](README.md) | [`README.zh-CN.md`](README.zh-CN.md) |
| Full manual | [`docs/USAGE.md`](docs/USAGE.md) | [`docs/使用说明.md`](docs/使用说明.md) |
| Driving it from an LLM assistant | [`AGENTS.md`](AGENTS.md) | [`docs/AGENTS.zh-CN.md`](docs/AGENTS.zh-CN.md) |
| Changelog | [`CHANGELOG.md`](CHANGELOG.md) | [`docs/CHANGELOG.zh-CN.md`](docs/CHANGELOG.zh-CN.md) |
