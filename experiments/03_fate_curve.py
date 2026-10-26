"""Experiment 3: luck versus skill (fate curve).

One grown leaf, one damage, then 50 forks that differ only in the random firing mask.
Share healed (MSE within 1.5x the grown-leaf MSE) against steps, for 3 damage sizes.
Control: forks that use the same random seed must give identical outcomes.

Question: how much of the outcome is dice?
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import Exp, fmt, show
from leafnca import damage as D
from leafnca.metrics import HEAL_FACTOR

FORKS, STEPS, RADII = 50, 200, (6, 10, 14)

e = Exp("03_fate_curve", "experiments/03_fate_curve.py")
leaf = e.grown()
ref = e.mse(leaf).item()
cy, cx = e.H // 2, e.H // 2 + 3  # centred just right of the midrib, same place for every size


def fate(xd, g):
    """Run the batch and return share healed at every step plus the final states."""
    x, share = xd, [(e.mse(xd) <= HEAL_FACTOR * ref).float().mean().item()]
    for _ in range(STEPS):
        x = e.ca(x, g)
        share.append((e.mse(x) <= HEAL_FACTOR * ref).float().mean().item())
    return np.array(share), x


rows, curves, finals, damaged = [], {}, {}, {}
for r in RADII:
    xd = D.cut_circle(leaf, cy, cx, r).repeat(FORKS, 1, 1, 1)
    share, xf = fate(xd, e.gen(r))  # one generator: each fork draws different firing masks
    curves[r], finals[r], damaged[r] = share, e.mse(xf).cpu().numpy(), xd[:1]
    first = int(np.argmax(share > 0)) if share.max() > 0 else None
    half = int(np.argmax(share >= 0.5)) if share.max() >= 0.5 else None
    rows.append({"radius": r, "forks": FORKS, "mse_after_damage": fmt(e.mse(xd[:1]).item()),
                 "share_healed_at_50": fmt(share[50], 3), "share_healed_at_100": fmt(share[100], 3),
                 "share_healed_at_200": fmt(share[STEPS], 3), "first_step_any_healed": first,
                 "first_step_half_healed": half, "final_mse_min": fmt(finals[r].min()),
                 "final_mse_max": fmt(finals[r].max())})

# Control: same damage, every fork gets its own generator seeded identically -> identical outcomes.
r = RADII[1]
xd = D.cut_circle(leaf, cy, cx, r)
outs = torch.cat([e.ca.rollout(xd, STEPS, e.gen(r)) for _ in range(FORKS)])
control_max_diff = (outs - outs[:1]).abs().max().item()
control_share = (e.mse(outs) <= HEAL_FACTOR * ref).float().mean().item()
control = {"radius": r, "forks": FORKS, "max_abs_state_diff_between_forks": control_max_diff,
           "share_healed_at_200": control_share, "identical": control_max_diff == 0.0}

e.write_csv("summary.csv", rows)
e.write_csv("share_healed_vs_step.csv",
            [{"step": t, **{f"r{r}": fmt(curves[r][t], 3) for r in RADII}} for t in range(STEPS + 1)])
e.write_json("summary.json", {"reference_grown_mse": fmt(ref), "heal_factor": HEAL_FACTOR,
                              "damage_centre": [cy, cx], "table": rows, "control_same_seed": control})
for row in rows:
    print(row)
print("control", control)

fig = plt.figure(figsize=(10, 5.5))
gs = fig.add_gridspec(2, len(RADII) + 1, height_ratios=[1, 1.3])
for k, r in enumerate(RADII):
    show(fig.add_subplot(gs[0, k]), damaged[r], f"hole r={r}")
ax = fig.add_subplot(gs[0, len(RADII)])
ax.boxplot([finals[r] for r in RADII], tick_labels=[f"r={r}" for r in RADII])
ax.axhline(HEAL_FACTOR * ref, color="k", ls="--", lw=1); ax.set_yscale("log")
ax.set_title(f"final MSE of {FORKS} forks", fontsize=8); ax.tick_params(labelsize=7)
ax = fig.add_subplot(gs[1, :])
for r in RADII:
    ax.plot(curves[r], label=f"hole r={r}, {FORKS} forks differing only in firing dice")
ax.axhline(control_share, color="0.4", ls=":",
           label=f"control r={RADII[1]}: same seed for all forks -> identical (max diff {control_max_diff:g})")
ax.set_ylim(-0.03, 1.03); ax.set_xlabel("steps after damage"); ax.set_ylabel("share healed (<= 1.5x grown MSE)")
ax.set_title("Fate curve: how much of healing is luck?", fontsize=10); ax.legend(fontsize=8)
e.save_fig(fig, "fate_curve.png")
