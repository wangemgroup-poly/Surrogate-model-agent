"""End-to-end demo of the surrogate half of the loop — no CST installation required.

Builds a synthetic three-parameter "structure", imports 60 analytic samples as if they were
full-wave results, lets the agent select and train a surrogate, and freezes one SB-SADEA
candidate. Nothing here starts a solver: the task keeps simulation_budget = 0 throughout.

Run:  .venv\\Scripts\\python.exe examples\\demo_no_cst.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simagent import engine  # noqa: E402
from simagent.core import arrays, fitness, read  # noqa: E402

PARAMS = {"slot_length": [8.0, 9.6], "slot_width": [0.6, 0.9], "spacing": [1.2, 1.7]}
METRICS = [
    dict(name="worst_S11_dB", kind="curve", tree="demo", transform="db20", reduce="max", op="<=", limit=-14, scale=1),
    dict(name="sidelobe_dB", kind="curve", tree="demo", transform="db20", reduce="max", op="<=", limit=-9, scale=1),
    dict(name="gain_ripple_dB", kind="curve", tree="demo", transform="real", reduce="ripple", op="<", limit=2, scale=0.25),
]


def truth(p):
    """Stand-in for the full-wave solver: smooth, coupled, with a built-in trade-off."""
    a = (p["slot_length"] - 8.9) / 0.8
    b = (p["slot_width"] - 0.74) / 0.15
    c = (p["spacing"] - 1.45) / 0.25
    return dict(
        worst_S11_dB=-16.5 + 6.0 * a ** 2 + 2.5 * b ** 2 + 1.2 * (a * c) ** 2,
        sidelobe_dB=-12.0 + 4.0 * (b + 0.3) ** 2 + 1.5 * c ** 2 - 1.0 * a,
        gain_ripple_dB=1.1 + 1.6 * c ** 2 + 0.8 * abs(a) - 0.4 * b,
    )


def main():
    work = Path(tempfile.mkdtemp(prefix="simagent_demo_"))
    try:
        project = work / "demo_structure.cst"      # a placeholder: init only needs the file to exist
        project.write_bytes(b"demo placeholder, not a real CST project")
        task = work / "demo_task"
        config = dict(schema_version=1, project=str(project), cst_install=str(work),
                      solver="frequency", parameters=PARAMS, metrics=METRICS,
                      simulation_budget=0, proposal_method="sb_sadea", sb_sadea={"surrogate": "gp"})
        print(f"1) init            -> {task}")
        engine.init(task, config)

        rng = np.random.default_rng(7)
        rows = []
        for i in range(60):
            p = {k: float(lo + (hi - lo) * rng.random()) for k, (lo, hi) in PARAMS.items()}
            rows.append(dict(id=f"DEMO_Run_{i+1}", parameters=p, metrics=truth(p),
                             provenance=dict(note="synthetic demo data, not a CST result")))
        rows_file = work / "demo_rows.json"
        rows_file.write_text(json.dumps(dict(rows=rows), ensure_ascii=False), encoding="utf-8")
        print("2) import-json     -> 60 synthetic designs")
        engine.import_json(task, rows_file)

        print("3) train           -> comparing algorithms on a held-out region")
        meta = engine.train(task)
        print(f"   chosen: {meta['label']} | fit {meta['fit_samples']} samples "
              f"| test MAE {[round(v, 3) for v in meta['test']['MAE']]}")

        print("4) propose         -> freezing one SB-SADEA candidate")
        batch = engine.propose(task, 1)
        cand = batch["candidates"][0]
        c = read(Path(task) / "config.json")
        names = [m["name"] for m in c["metrics"]]
        x, y = arrays(read(Path(task) / "data.json")["rows"], c)
        print(f"   candidate {cand['id']} ({cand['role']})")
        for j, n in enumerate(names):
            print(f"     {n:16s} predicted {cand['prediction'][n]:8.3f} ± {cand['std'][j]:.3f}")
        print(f"   predicted fitness {cand['predicted_fitness']:.3f} | best in database {fitness(y, c).min():.3f}"
              f" | warnings: {cand.get('warnings') or 'none'}")

        print("\nWith a real CST project the next step would be:")
        print("   agent.py run --task <task> --rounds N --count 1 --budget N")
        print("which validates each frozen candidate in an isolated copy of the project.")
        print(f"\nReport written to {Path(task) / '报告.md'}")
        print("(temporary demo directory is removed on exit)")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
