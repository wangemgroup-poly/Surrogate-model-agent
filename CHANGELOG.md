**English** | [中文](docs/CHANGELOG.zh-CN.md)

# Changelog

All notable changes to this tool. Dates are the dates the behaviour was validated against real
CST projects in a waveguide slot-array study (4, 6, 8, 10 and 12 radiating slots).

## 0.3.0

### Added
- **Multi-project import.** `import-cst --project A.cst --project B.cst` merges saved runs from
  several versions of the same structure; samples from non-source projects get a version prefix.
  Older project versions frequently hold better designs than the newest file, and a surrogate
  trained without them optimises inside a blind spot.
- **`bounds` command** — prints each project's optimiser ranges and their union before you write
  a config.
- **Range advice on import** — `bounds_advice.json` and an event whenever imported optimiser
  ranges *or the parameter values of the saved samples themselves* fall outside the task box. A
  project's optimiser range is often narrowed after a campaign, which silently rejects good older
  samples at import.
- **`import_summary.json`** — read/added/repeats/rejected/without-results per project.
- **Candidate warnings** — each frozen candidate records its distance to the nearest sample
  relative to the median nearest-neighbour spacing (`extrapolation_ratio`) and a cross-check in
  which the other algorithms predict the same design (`disagreement_limit`).
- **Conservative constraints for `optimize`** — the predicted-improvement candidate is judged at
  μ + `conservative_sigma`·σ instead of the mean alone.
- **Report sections** — constraint trade-off table, margin correlation between constraint pairs,
  a dependency-free `tradeoff.svg` scatter, parameter-bound pressure statistics, and a
  convergence summary. A warnings column was added to the candidate table.
- **`materialize` command and rule** — re-solve a verified design inside the source project and
  save it there so its curves can be inspected in CST, with a full backup first and the project's
  own solver settings preserved. Runs automatically inside `run` when the database best improves
  (`materialize.on`: `improvement` / `pass` / `never`, `materialize.max` per call).
- **`rebaseline` command** — re-register a source project that only gained saved runs.
- **Automatic ASCII junction** for non-ASCII project paths; the CST result library cannot open
  them directly and this removes the per-task workaround scripts.
- **Extended-length path support** when copying project backups, so deep CST result trees do not
  hit the Windows 260-character limit.

### Changed
- **Source identity is the structure, not the file.** Tasks register a geometry signature
  (`Model.mod`, `ModelHistory.json`, solver properties) **and** the values of every fixed
  parameter the task does not optimise. Saving further runs into the source project no longer
  invalidates a task; editing a fixed dimension or the geometry still does. Identical geometry
  history is not sufficient on its own: two versions can build the same history with different
  fixed dimensions, because those values live in `Parameters.json`.
- **Config signature excludes execution settings.** Changing `simulation_budget`, `cpus`,
  `timeout_minutes` or the registered hashes no longer invalidates trained models or frozen
  batches; changing targets, the parameter box or the proposal method still does.

### Fixed
- Backup copies of projects with deep result trees failed with `WinError 206`.
- Range advice missed the case where saved samples, rather than optimiser settings, exceeded the
  task box.

## 0.2.0

### Added
- **`proposal_method: "optimize"`** — differential evolution directly on the surrogate, with
  `search_domain` `trust_region` or `full`, minimum distance to existing samples and minimum
  separation between candidates. Full-domain search is refused for the local GP, which only fits
  the 120 points nearest the current centre.
- **`proposal_method: "sb_sadea"`** — the online loop of Liu et al. (IEEE TAP 2022): λ best
  designs as population, DE/current-to-best/1, a local surrogate per child trained on its τ
  nearest samples, self-adaptive LCB, one full-wave simulation per iteration.
- **Penalty fitness** with configurable per-metric `penalty_weight` — the lever for steering a
  search that keeps trading one requirement away for another.
- **Variational-inference BNN** (`sb_sadea.surrogate: "bnn"`, PyTorch, GPU-parallel across
  children), with `kl_weight` exposed because the paper does not state how the KL term is scaled.
  Taken literally the network degenerates to the neighbour mean on small local sets.
- **Optional full-sample GP** (`include_global_gp`) for tasks within its size limits.

### Changed
- Default local surrogate for SB-SADEA is the ARD-Matérn GP (ω = 2), the paper's GP-ALCB variant.
  On a 13-parameter benchmark it was roughly twice as accurate as the paper's BNN at equal cost.

## 0.1.0

Initial release: task directories, CST result import with data audit, region-based train /
validation / test splits with fixed anchors, comparison of local GP / kernel ridge / extremely
randomised trees / Bayesian last-layer network, frozen candidate batches, isolated-copy CST
validation with return-value, convergence and parameter-readback checks, budget limits, pause and
resume, result ingest, model versioning and Markdown/HTML reports.
