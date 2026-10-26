"""Experiment 2: damage and heal. Three damage types x 20 random positions, heal 150 steps.

Question: how much does the leaf recover, by damage type?
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import Exp, fmt, show
from leafnca import damage as D
from leafnca.metrics import HEAL_FACTOR

N, HEAL, HOLE_R, ERASE = 20, 150, 8, 0.4

e = Exp("02_damage_heal", "experiments/02_damage_heal.py")
rng = np.random.default_rng(e.seed)
leaf = e.grown()
before = e.mse(leaf).item()
inside = torch.nonzero(e.target[0, 3] > 0.5).cpu().numpy()  # pixels of the leaf blade


def half(i):
    return D.cut_half(leaf, rng.uniform(0, 2 * math.pi))


def hole(i):
    cy, cx = inside[rng.integers(len(inside))]
    return D.cut_circle(leaf, int(cy), int(cx), HOLE_R)


def erase(i):
    return D.erase_random(leaf, ERASE, e.gen(100 + i))


rows, examples, curves = [], {}, {}
for k, (name, fn) in enumerate([("half-cut", half), (f"hole r={HOLE_R}", hole), ("40% random erase", erase)]):
    xd = torch.cat([fn(i) for i in range(N)])
    after = e.mse(xd)
    x, g, curve = xd, e.gen(1000 * (k + 1)), [after.cpu().numpy()]
    for _ in range(HEAL):
        x = e.ca(x, g)
        curve.append(e.mse(x).cpu().numpy())
    healed = e.mse(x)
    curves[name] = np.array(curve)
    examples[name] = (xd[:1], x[:1])
    for i in range(N):
        a, h = after[i].item(), healed[i].item()
        rows.append({"damage": name, "position": i, "mse_before": fmt(before), "mse_after_damage": fmt(a),
                     "mse_after_heal": fmt(h), "recovered_fraction": fmt((a - h) / (a - before), 3),
                     "healed_within_1.5x": int(h <= HEAL_FACTOR * before)})
e.write_csv("per_position.csv", rows)

table = []
for name in curves:
    r = [row for row in rows if row["damage"] == name]
    col = lambda c: np.array([row[c] for row in r])  # noqa: E731
    table.append({"damage": name, "n": N, "mse_before": fmt(before),
                  "mse_after_damage_mean": fmt(col("mse_after_damage").mean()),
                  "mse_after_heal_mean": fmt(col("mse_after_heal").mean()),
                  "mse_after_heal_max": fmt(col("mse_after_heal").max()),
                  "recovered_fraction_median": fmt(np.median(col("recovered_fraction")), 3),
                  "share_healed_within_1.5x": fmt(col("healed_within_1.5x").mean(), 3)})
e.write_csv("summary.csv", table)
e.write_json("summary.json", {"heal_steps": HEAL, "positions_per_type": N, "table": table})
for t in table:
    print(t)

fig = plt.figure(figsize=(10, 6))
gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.3])
for k, (name, (xd, xh)) in enumerate(examples.items()):
    show(fig.add_subplot(gs[0, 2 * k]), xd, f"{name}: damaged")
    show(fig.add_subplot(gs[0, 2 * k + 1]), xh, f"after {HEAL} steps")
ax = fig.add_subplot(gs[1, :])
for name, c in curves.items():
    ax.plot(np.median(c, 1), label=f"{name} (median of {N})")
    ax.fill_between(np.arange(len(c)), c.min(1), c.max(1), alpha=0.15)
ax.axhline(before, color="k", lw=1, ls=":", label="grown leaf (before damage)")
ax.axhline(HEAL_FACTOR * before, color="k", lw=1, ls="--", label="1.5x grown = 'healed'")
ax.set_yscale("log"); ax.set_xlabel("steps after damage"); ax.set_ylabel("MSE vs target (log)")
ax.set_title(f"Recovery from three damage types, {N} random positions each (band = min-max)", fontsize=10)
ax.legend(fontsize=8)
e.save_fig(fig, "damage_heal.png")
