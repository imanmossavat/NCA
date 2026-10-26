"""Experiment 1: grow and hold. Grow 120 steps from one seed, hold 200 more; MSE vs target over time.

Question: does the grown leaf stay stable?
"""
import matplotlib.pyplot as plt
import numpy as np

from common import GROW, Exp, fmt, show
from leafnca import model as M

HOLD, RUNS = 200, 8  # RUNS = independent firing-mask seeds

e = Exp("01_grow_hold", "experiments/01_grow_hold.py")
x = M.seed(RUNS, e.H, device=e.dev)
g = e.gen()
curve, snaps = [e.mse(x).cpu().numpy()], {0: x[:1]}
for t in range(1, GROW + HOLD + 1):
    x = e.ca(x, g)
    curve.append(e.mse(x).cpu().numpy())
    if t in (30, 60, GROW, GROW + 100, GROW + HOLD):
        snaps[t] = x[:1]
curve = np.array(curve)  # [steps+1, RUNS]

grown, end = curve[GROW], curve[GROW + HOLD]
hold_max = curve[GROW:].max(axis=0)
rows = [{"step": t, "mse_mean": fmt(curve[t].mean()), "mse_min": fmt(curve[t].min()),
         "mse_max": fmt(curve[t].max())} for t in range(0, GROW + HOLD + 1, 10)]
e.write_csv("mse_over_time.csv", rows)
summary = {
    "runs": RUNS, "grow_steps": GROW, "hold_steps": HOLD,
    "mse_at_grown_mean": fmt(grown.mean()), "mse_at_end_mean": fmt(end.mean()),
    "mse_max_during_hold_mean": fmt(hold_max.mean()),
    "end_over_grown_ratio_mean": fmt((end / grown).mean(), 3),
    "end_over_grown_ratio_max": fmt((end / grown).max(), 3),
    "runs_stable_within_1.5x": int(((hold_max / grown) <= 1.5).sum()),
}
e.write_json("summary.json", summary)
print(summary)

fig = plt.figure(figsize=(9, 5.2))
gs = fig.add_gridspec(2, len(snaps), height_ratios=[1, 1.4])
for i, (t, s) in enumerate(snaps.items()):
    show(fig.add_subplot(gs[0, i]), s, f"t={t}")
ax = fig.add_subplot(gs[1, :])
steps = np.arange(len(curve))
ax.fill_between(steps, curve.min(1), curve.max(1), alpha=0.25, color="tab:green", label=f"range over {RUNS} seeds")
ax.plot(steps, curve.mean(1), color="tab:green", label="mean")
ax.axvline(GROW, color="0.5", ls="--", lw=1); ax.text(GROW + 3, ax.get_ylim()[1] * 0.9, "hold starts", fontsize=8)
ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("MSE vs target (log)")
ax.set_title(f"Grow {GROW} steps from one cell, then hold {HOLD} steps", fontsize=10)
ax.legend(fontsize=8)
e.save_fig(fig, "grow_hold.png")
