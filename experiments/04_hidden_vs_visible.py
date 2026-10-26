"""Experiment 4: where is the memory? Erase only visible channels, only hidden, or both, in a circular region.

Question: is the pattern stored in the hidden state or rebuilt from neighbours?

Extra condition "RGB only" keeps alpha: erasing alpha also marks the cells dead, so the alive mask then wipes
their hidden state on the next step (measured below as `cells_dead_after_1_step`).
"hidden only" leaves the picture untouched, so its key number is the peak MSE while it recovers, not "healed".
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import Exp, fmt, show
from leafnca import damage as D
from leafnca.metrics import HEAL_FACTOR

N, HEAL, R = 20, 150, 10
CONDITIONS = [("RGB only (alpha kept)", D.RGB), ("visible (RGB+alpha)", D.VISIBLE),
              ("hidden only", D.HIDDEN), ("both", slice(None))]

e = Exp("04_hidden_vs_visible", "experiments/04_hidden_vs_visible.py")
rng = np.random.default_rng(e.seed)
leaf = e.grown()
before = e.mse(leaf).item()
inside = torch.nonzero(e.target[0, 3] > 0.5).cpu().numpy()
centres = [tuple(int(v) for v in inside[rng.integers(len(inside))]) for _ in range(N)]
masks = torch.cat([D.circle_mask(e.H, e.H, cy, cx, R, e.dev) for cy, cx in centres])[:, 0] > 0  # [N,H,W]

rows, curves, examples = [], {}, {}
for k, (name, ch) in enumerate(CONDITIONS):
    xd = torch.cat([D.cut_circle(leaf, cy, cx, R, channels=ch) for cy, cx in centres])
    after = e.mse(xd)
    g = e.gen(1000 * (k + 1))
    x = e.ca(xd, g)
    # share of live region cells that the alive mask removed entirely (all channels zero) after one step
    was_alive = leaf[0, 3] > 0.1
    dead = ((x.abs().sum(1) == 0) & masks & was_alive).float().sum((1, 2)) / (masks & was_alive).float().sum((1, 2))
    curve = [after.cpu().numpy(), e.mse(x).cpu().numpy()]
    for _ in range(HEAL - 1):
        x = e.ca(x, g)
        curve.append(e.mse(x).cpu().numpy())
    curves[name] = np.array(curve)
    examples[name] = (xd[:1], x[:1])
    healed = e.mse(x)
    for i in range(N):
        a, h = after[i].item(), healed[i].item()
        rows.append({"condition": name, "position": i, "mse_after_damage": fmt(a),
                     "mse_peak_during_heal": fmt(curves[name][:, i].max()), "mse_after_heal": fmt(h),
                     "healed_within_1.5x": int(h <= HEAL_FACTOR * before),
                     "cells_dead_after_1_step": fmt(dead[i].item(), 3)})
e.write_csv("per_position.csv", rows)

table = []
for name, _ in CONDITIONS:
    r = [row for row in rows if row["condition"] == name]
    col = lambda c: np.array([row[c] for row in r])  # noqa: E731
    c = curves[name]
    t_heal = [int(np.argmax(c[:, i] <= HEAL_FACTOR * before)) if (c[:, i] <= HEAL_FACTOR * before).any() else None
              for i in range(N)]
    done = [t for t in t_heal if t is not None]
    table.append({"condition": name, "n": N, "radius": R, "mse_before": fmt(before),
                  "mse_after_damage_mean": fmt(col("mse_after_damage").mean()),
                  "mse_peak_during_heal_mean": fmt(col("mse_peak_during_heal").mean()),
                  "mse_after_heal_mean": fmt(col("mse_after_heal").mean()),
                  "share_healed_within_1.5x": fmt(col("healed_within_1.5x").mean(), 3),
                  "median_steps_to_heal": float(np.median(done)) if done else None,
                  "cells_dead_after_1_step_mean": fmt(col("cells_dead_after_1_step").mean(), 3)})
e.write_csv("summary.csv", table)
e.write_json("summary.json", {"heal_steps": HEAL, "positions": N, "radius": R, "table": table})
for t in table:
    print(t)

fig = plt.figure(figsize=(11, 6))
gs = fig.add_gridspec(2, 8, height_ratios=[1, 1.3])
for k, (name, (xd, xh)) in enumerate(examples.items()):
    show(fig.add_subplot(gs[0, 2 * k]), xd, f"{name}")
    show(fig.add_subplot(gs[0, 2 * k + 1]), xh, f"+{HEAL} steps")
ax = fig.add_subplot(gs[1, :])
for name, c in curves.items():
    ax.plot(np.median(c, 1), label=f"{name} (median of {N})")
ax.axhline(before, color="k", lw=1, ls=":", label="grown leaf")
ax.axhline(HEAL_FACTOR * before, color="k", lw=1, ls="--", label="1.5x grown = 'healed'")
ax.set_yscale("log"); ax.set_xlabel("steps after erasing"); ax.set_ylabel("MSE vs target (log)")
ax.set_title(f"Erase a circle (r={R}) in visible, hidden or all channels; {N} positions", fontsize=10)
ax.legend(fontsize=8)
e.save_fig(fig, "hidden_vs_visible.png")
