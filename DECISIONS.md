# Decisions

One line per judgement call, with why.

## Setup
- Python 3.13.9 (miniconda) for the venv, not system Python 3.9.6: current torch wheels need >= 3.10.
- Kept the original `nca.py`, `leaf.py` and `vein_models_preview.png` under `original/` (moved, not deleted) so the starting point stays visible.
- `leaf_preview.png` = right half of `vein_models_preview.png` (left 50% of columns dropped, no analysis), as instructed. It shows a reticulate vein network; the trained target is the simpler procedural `leaf.py` leaf (midrib + 5 side-vein pairs). Flagged for review.
- Dropped the emoji (network download) and smiley targets from `nca.py`: brief says keep it simple; target = procedural leaf or an image path.
- `pyproject.toml` only to set `pythonpath = src` for pytest and allow `pip install -e .`.

## Code fixes while refactoring `nca.py`
- Device/threads were set at import time (global `DEV`); moved to `get_device()`/`setup_threads()` so tests run on CPU and the CLI picks MPS.
- `CA.forward` takes an optional `torch.Generator` for the firing mask: needed for the luck-vs-skill fork experiment and its same-seed control.
- Checkpoint now stores hidden size and fire rate, so `load()` rebuilds the right model; weights saved on CPU for portability.
- LR milestones were hard-coded at 1500/3000 steps; now 40%/75% of `--steps` so a shorter or longer run still decays.
- Gradient normalisation guarded against `None` grads; training stops on a non-finite loss.
- Rollout length range, hidden size, lr, seed, device exposed as flags (the levers the brief allows).
- Hidden size default stays 128 as in `nca.py`, so the "hidden size 128" lever is already applied; it will be skipped in the lever order.
- Training runs on MPS (0.23 s/step vs 1.06 s/step on CPU in a 10-step benchmark); experiments run on CPU so seeded firing masks are exactly reproducible (they take ~2 min in total).
- Demo: each damage now starts from the same held leaf (`nca.py` chained them, so one failed heal contaminated the next damage). Added an untrained-model baseline MSE for the acceptance check.

## Experiments
- Healed = MSE within 1.5x the grown-leaf MSE (brief's fate-curve criterion), used in every experiment for consistency.
- Grown leaf for experiments 2-4 = 120 steps from one seed with firing seed 0; its MSE is the reference.
- Exp 2 "cut half": half-plane through the grid centre at a random angle (20 angles); hole: radius 8 centred on a random blade pixel; erase: 40% of cells, 20 mask seeds.
- Exp 3: hole radii 6/10/14 at a fixed spot just right of the midrib, 200 steps; 50 forks in one batch with one generator (so each fork draws different firing masks). Control: 50 forks each with its own generator seeded identically.
- Exp 4: added a 4th condition "RGB only (alpha kept)": erasing alpha marks cells dead, so the alive mask wipes their hidden state too; without this condition "visible" and "both" would be nearly the same test. Hidden-only erase leaves the picture untouched, so its key number is the peak MSE while recovering.
- Exp 1: 8 firing seeds in one batch, to show spread.

## Training runs (max 4)
- Run 1: defaults (hidden 128, lr 2e-3, rollout 64-96, 4000 steps, 8.4 min on MPS). Final loss 0.0018. Grown MSE 0.0014, stable over 200-step hold (0.0011); untrained baseline 0.100. Holes and 40% erase heal to ~0.0013; half-cut does not (0.051 -> 0.028). Midrib clear, side veins faint and incomplete -> acceptance not met on veins.
- Run 2: lever 1, more steps: 8000 steps, everything else unchanged.
- Run 2 result: 8000 steps, 16.7 min on MPS, final loss 0.0005. Grown MSE 0.00034 (untrained 0.100), after 200-step hold 0.00009; midrib and all 5 side-vein pairs visible; hole and 40% erase heal to ~0.0003. Acceptance met -> stopped after 2 of 4 allowed runs (~25 min training compute). Half-cut still does not regrow: reported as a limit, not chased (training damage is circles of radius <= ~13 px, a half-cut is far outside that).
- Committed checkpoint = run 2 (`results/model.pt`, 36 KB), with its loss curve and training config.
- Experiment 5 (pilot) run because the budget allowed it (~63 min training in total, under the ~2 h cap): seeds 1-3, 6000 steps each (fits the 15-min cap at ~0.125 s/step), sequentially, never two trainings at once. Its checkpoints stay in `runs/` (not committed); only results are committed.
- Experiment 5 reuses `02_damage_heal.py` via env vars (`LEAFNCA_CKPT`, `LEAFNCA_OUT`) instead of copying code.
- No commits during the pilot, so all three seeds carry the same provenance commit.
- REPORT tables are generated into `results/tables.md` by `experiments/make_tables.py`. REPORT.md shows selected columns from it, with values rounded to 2-3 significant figures.
- No NaNs and no MPS problems happened, so no CPU fallback runs were needed.
