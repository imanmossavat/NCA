# Status

## What works
- `pytest`: 14 tests green (~10 s, CPU): leaf target, model shape and zero-init no-op, alive masking, determinism,
  damage operators, metrics, a 20-step training smoke test with checkpoint save/load, and a sanity check that an
  untrained rule does not "heal".
- Training on Apple MPS: 2 of 4 allowed main runs used (8.4 + 16.7 min); plus 3 × 12.5 min for the pilot; ~63 min training compute in total. Run 2 meets acceptance: grown MSE 0.00034
  vs 0.100 untrained, midrib and all 5 side-vein pairs visible, stable for 200 extra steps (MSE 0.00009).
  The checkpoint is committed as `results/model.pt` (36 KB).
- Demo: `results/demo.gif`, `results/demo_strip.png`, `results/demo_metrics.json`.
- Experiments 1–4 done in full, with figure, CSV and JSON in `results/0x_*/`, each stamped with script and commit.
- Experiment 5 (pilot, training-seed variance): done as a pilot (3 seeds × 6000 steps, ~12.5 min each). Qualitative results hold across seeds; exact healed shares vary a lot (see REPORT §5).

## What does not work / known gaps
- A leaf cut in half does not regrow (0 of 20 placements heal; about half the error is recovered). Training only
  used circular damage.
- The target is a simple procedural leaf. The reticulate venation in `leaf_preview.png` (cropped from
  `original/vein_models_preview.png`) is **not** what the NCA learned. Needs your review: is the simple target
  OK for the talk, or should the target be swapped for that image?
- Numbers rest on one trained model (seed 0). The 3-seed check is a pilot only.

## How to rerun
```bash
source .venv/bin/activate          # or create it: see README "Run it"
pytest
python -m leafnca.train --steps 8000 --max_minutes 30 --out runs/run2 && cp runs/run2/model.pt results/model.pt
python -m leafnca.demo
for s in 01_grow_hold 02_damage_heal 03_fate_curve 04_hidden_vs_visible; do python experiments/$s.py; done
python experiments/05_seed_variance.py     # optional pilot, ~40 min (trains 3 models)
python experiments/make_tables.py
```
Experiments run on CPU and are deterministic for a given checkpoint. Training on MPS is not bit-for-bit
reproducible.

## Blockers
None.
