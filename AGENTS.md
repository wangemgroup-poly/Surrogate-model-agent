**English** | [中文](docs/AGENTS.zh-CN.md)

# Driving this tool from an LLM assistant

This file tells an AI assistant (Claude Code, Cursor, Copilot Chat, Codex, a local model with a
shell tool — anything that can run commands) how to operate the surrogate model agent on the
user's behalf. **No API key and no network access are required**: the agent is a local Python
CLI that talks to a local CST installation. The assistant supplies judgement and reporting; the
agent supplies the deterministic, auditable workflow.

If you are such an assistant: read this file top to bottom before running anything.

---

## 1. What you are driving

`agent.py` implements: import full-wave data → train a surrogate → **optimise on the surrogate**
→ freeze candidates → validate with CST → ingest → repeat. Every step writes to a task directory
that is the single source of truth. You never need to invent state: read `config.json`,
`state.json`, `data.json`, `events.jsonl`, `报告.md` and the `batches/*/batch.json` files.

**Scope.** It drives any CST parametric frequency-domain project whose goals can be computed from
1D results — filters, couplers, matching networks, waveguide components, frequency-selective
surfaces, antennas. Nothing about the method is antenna-specific; the far-field side-lobe metric
(`sll_phi_cut`) is the one metric type that is. If the user's goals need a quantity the metric
types do not cover, say so rather than approximating it.

Command reference: [`README.md`](README.md). Full manual: [`docs/USAGE.md`](docs/USAGE.md)
(Chinese: [`docs/使用说明.md`](docs/使用说明.md)).

## 2. Rules you must not break

1. **Never start a CST simulation without the user's explicit approval for that specific run,
   including how many solves it may use.** `validate`, `resume`, `run` and `materialize` all
   start the solver. Preparation steps (`init`, `import-*`, `train`, `propose`, `report`,
   `bounds`, `inspect`) never do — run those freely.
2. **The surrogate optimises; CST only verifies.** Do not hand-write a candidate, and do not
   launch CST-internal optimisation. If the built-in proposal looks too conservative or too
   aggressive, say so, propose a change, and wait for approval.
3. **Follow the agent's own workflow.** If you think a step is missing, propose adding it to the
   agent rather than scripting around it. Ad-hoc scripts that bypass `propose`/`validate` lose
   the frozen-prediction audit trail, which is the reason this tool exists.
4. **Never present a predicted value as an achieved result.** Report predictions and measured
   values in separate columns, always with the gap.
5. **Do not edit files inside a task directory by hand** except to recover from a documented
   failure, and say so explicitly when you do.

## 3. Starting a new structure

If the user has not yet defined targets, or you are unsure the project is usable, hand them the
wizard instead of writing a config yourself — it is interactive, so **it must run in a terminal
the user can type into**, not inside your own tool call:

```bat
.venv\Scripts\python.exe agent.py setup --lang en
```

It checks the project, finds sibling versions of the same structure, suggests the parameter box,
lists the available result trees, snaps band endpoints to stored samples, advises on sub-band
splitting, and reports whether there is enough data. It never starts the solver. When the user
would rather you did it, do it yourself as below.

```bat
REM 1. find every project file of the same structure, not only the newest one
REM 2. compare their optimiser ranges
.venv\Scripts\python.exe agent.py bounds --project A.cst --project B.cst
```

Then write a config (copy `examples/config_template.json`) and decide the **parameter box**:

* take the union of the optimiser ranges of all versions you will import, **and**
* widen it to cover the parameter values of the saved runs themselves — a project's optimiser
  range is often narrowed after a campaign, leaving older good samples outside it. Samples
  outside the box are rejected at import, silently shrinking your data.

Set `"simulation_budget": 0` at first so nothing can solve by accident.

```bat
.venv\Scripts\python.exe agent.py init --task tasks\my_task --config my_config.json
.venv\Scripts\python.exe agent.py import-cst --task tasks\my_task --project A.cst --project B.cst
.venv\Scripts\python.exe agent.py train --task tasks\my_task
.venv\Scripts\python.exe agent.py propose --task tasks\my_task --count 1
```

Check `import_summary.json` (read/added/repeats/rejected per project) and `bounds_advice.json`
(written when imported data or optimiser ranges fall outside the task box). Report the data
volume, the pass rate per metric, the best existing design, the model's held-out error and the
first candidate — then ask for a simulation budget.

### Band-limited metrics

If a metric uses `"band": [f1, f2]`, both endpoints must be exact frequency samples of the
result curve, otherwise extraction fails. Verify before creating the task.

### When to split a band into sub-bands

A "worst value over a band" target is only easy to model while the worst value stays in roughly
the same place. Once the worst point jumps between designs — a resonance moving through the band —
the quantity stops being a smooth function of the parameters, the surrogate's error on it grows
past the improvement being chased, and the loop stalls with the whole band at one plateau.

Diagnose it from data you already have, not from intuition:

```python
from simagent import cst
from simagent.setup import worst_frequencies, suggest_segments, segmentation_gain
```

`worst_frequencies` gives the per-design location of the worst value; if those cluster in two or
three separated groups, `suggest_segments` proposes the split, and `segmentation_gain`
cross-validates whole-band against segmented on the user's own designs and returns both errors.
Quote both numbers when you recommend a split — it changes the config signature, so it costs a
retrain, and on a genuinely flat band it buys nothing. Give each sub-band its own metric entry
with its own limit; that is also how you ask for a shape the single limit cannot express.

## 4. Running the loop

```bat
.venv\Scripts\python.exe agent.py run --task tasks\my_task --rounds N --count 1 --budget N
```

Tell the user before it starts: expected wall-clock time, that the source project must stay
closed, and that creating an empty `PAUSE` file in the task directory stops it after the current
solve. Force-closing CST kills the agent's hidden solver session and wastes that budget unit.

While it runs, do not poll in a loop; wait for completion, then summarise:

* a row per round: selection mode, predicted vs measured per metric, fitness, running best;
* prediction error (MAE) per metric, and which metrics are systematically optimistic;
* how many candidates touched a parameter bound — persistent touching means the box is limiting;
* whether any design now passes everything.

## 5. Reading results honestly

* `fitness` is the weighted sum of constraint violations; 0 means every constraint is met.
* A design that misses one constraint by less than the agent's own prediction error for that
  metric is *near*, not *passing*.
* Differences between the isolated batch copy and the same design re-solved inside the source
  project are typically 0.05–0.3 dB. When a margin is smaller than that, say that it is a
  borderline pass.
* If the top designs split into families that each satisfy a different subset of constraints,
  that is a real trade-off. Quantify it (bucket one metric, report the best achievable value of
  the other) before recommending more rounds.

## 6. Steering

| Symptom | Lever |
|---|---|
| Search trades away a requirement the user cares about | add `"penalty_weight": 2`–`3` to that metric, retrain, re-propose |
| Best designs keep pressing the same bound | propose widening that bound; the box change needs a **new task** |
| Candidates far from data, predictions over-optimistic | `conservative_sigma`, or switch `search_domain` from `full` to `trust_region` |
| Rounds are volatile, population spans unrelated regions | fewer, denser data regions; or reduce `sb_sadea.lambda` (a deviation from the paper — get approval) |

Changing targets, the box or the method changes the config signature: retrain and re-propose.
Changing only the budget does not.

## 7. Saving a good design for the user to look at

```bat
.venv\Scripts\python.exe agent.py materialize --task tasks\my_task [--design CANDIDATE_ID]
```

This backs up the source project, writes the parameters into it, solves with the project's own
solver settings and saves, so the user can open the curves in CST. It also runs automatically
inside `run` whenever the database best improves (`materialize.on`, default `improvement`,
`max` 3 per call) — tell the user this will happen and that it costs roughly one solve each.

## 8. When a solve fails

The agent stops and preserves the scene: candidate `state: running`, `job.json` with the exit
code, no `DONE` in `status.txt`. It does **not** retry automatically, and `discard`/`resume`
refuse such a candidate on purpose. Diagnose first — read `job.json`, `status.txt` and
`project/Result/Model.log`; a rebuild failure points at the parameter set, while a crash during
meshing or solving usually points at something external (the user closing CST, a licence
conflict). Then ask the user whether to retry (move the crashed copy aside and let `run` prepare
a fresh one, costing one budget unit) or to abandon the candidate.

## 9. Reporting to the user

Lead with the decision-relevant fact: is there a compliant design, and if not, which constraint
is short and by how much. Then the evidence. Keep predicted and measured values distinct, name
the file or run ID behind every number, and state plainly when something was not verified.
