"""Experiment 5 (pilot): training-seed variance.

Train 3 seeds at a reduced budget (6000 steps, max 15 min each, one at a time) and rerun experiment 2 on each.
Question: how much do the damage-and-heal results depend on the training seed?
"""
import csv
import json
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np

from common import ROOT
from leafnca.utils import git_sha, stamp

SEEDS, STEPS, MAX_MIN = (1, 2, 3), 6000, 15
OUT = ROOT / "results" / "05_seed_variance"
OUT.mkdir(parents=True, exist_ok=True)

rows = []
for s in SEEDS:
    run = ROOT / "runs" / f"seed{s}"
    if not (run / "loss.csv").exists():  # train only once; reruns reuse the checkpoint
        subprocess.run([sys.executable, "-m", "leafnca.train", "--seed", str(s), "--steps", str(STEPS),
                        "--max_minutes", str(MAX_MIN), "--out", str(run)], check=True)
    sub = f"05_seed_variance/seed{s}"
    subprocess.run([sys.executable, str(ROOT / "experiments" / "02_damage_heal.py")], check=True,
                   env={**os.environ, "LEAFNCA_CKPT": str(run / "model.pt"), "LEAFNCA_OUT": sub})
    steps = sum(1 for _ in open(run / "loss.csv")) - 1
    for r in csv.DictReader(open(OUT / f"seed{s}" / "summary.csv")):
        rows.append({"train_seed": s, "train_steps": steps, **r})

with open(OUT / "summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with open(OUT / "summary.json", "w") as f:
    json.dump({"script": "experiments/05_seed_variance.py", "commit": git_sha(), "status": "pilot",
               "seeds": SEEDS, "steps": STEPS, "max_minutes": MAX_MIN, "table": rows}, f, indent=2)
for r in rows:
    print(r)

damages = list(dict.fromkeys(r["damage"] for r in rows))
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for k, s in enumerate(SEEDS):
    rs = [r for r in rows if r["train_seed"] == s]
    xs = np.arange(len(damages)) + (k - 1) * 0.25
    axes[0].bar(xs, [float(r["share_healed_within_1.5x"]) for r in rs], 0.25, label=f"train seed {s}")
    axes[1].bar(xs, [float(r["mse_after_heal_median"]) for r in rs], 0.25)
    axes[1].scatter(xs, [float(r["mse_before"]) for r in rs], color="k", marker="_", s=120, zorder=3)
for ax, t in zip(axes, ["share healed (<= 1.5x grown MSE)", "median MSE after heal (bar), grown MSE (dash)"]):
    ax.set_xticks(range(len(damages))); ax.set_xticklabels(damages, fontsize=8); ax.set_title(t, fontsize=9)
axes[1].set_yscale("log"); axes[0].legend(fontsize=8)
fig.suptitle(f"PILOT: experiment 2 on 3 training seeds ({STEPS} steps / {MAX_MIN} min cap each)", fontsize=10)
stamp(fig, "experiments/05_seed_variance.py")
fig.savefig(OUT / "seed_variance.png", dpi=130, bbox_inches="tight")
print("wrote", OUT / "seed_variance.png")
