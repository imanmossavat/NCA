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
