# Task: build the "Leaf NCA" repo, run its experiments, push to GitHub. Work autonomously.

Remote: https://github.com/imanmossavat/NCA.git (push to `main`; never force-push; if the remote is non-empty, pull --rebase first).
Machine: MacBook Pro M4, 48 GB RAM. It must stay usable while you work (see "Compute limits").

## Goal
A small, easy-to-explain portfolio demo: a **neural cellular automaton (NCA) that grows a veined leaf from one cell and regrows it after damage**. It is a short preamble to a talk on venation, leaves and self-organisation, and to a literature search of NCA work done with OpenAlex. Keep it simple. Do not add features beyond this brief.

One-sentence story: every cell runs the same tiny rule with no blueprint, and the leaf heals when you cut it.

## Starting material (in this folder)
- `nca.py`: PyTorch NCA (16 channels, Sobel perception, zero-init update, 50% async firing, alive masking, sample pool, damage training, `train` and `demo` commands).
- `leaf.py`: procedural leaf target (PIL only).
- `leaf_preview.png`: what the target should look like. you should make this by cropping the left side of the /Users/imanm/NCA/vein_models_preview.png out, you do not have to do image analysis, just drop 50% of the left most columns. And ask me to review.
`nca.py` has **never been run**. Expect bugs. Fix them, keep the design.

## Repo layout (create it)
```
NCA/
  README.md            # 1-screen story, demo GIF, how to run, honest limits
  requirements.txt     # pinned to the versions you actually installed
  pyproject.toml or setup.cfg (optional, only if needed for pytest imports)
  src/leafnca/         # model.py, targets.py (leaf.py), train.py, damage.py, metrics.py, demo.py
  experiments/         # one script per experiment, each writes to results/<name>/
  tests/               # pytest
  results/             # figures, GIFs, small JSON/CSV, small model checkpoint (<5 MB ok to commit)
  REPORT.md            # plain-language findings with numbers and limits
  DECISIONS.md         # every judgement call you made, one line each, with why
  STATUS.md            # final: what works, what doesn't, how to rerun
```

## Steps
1. `python3 -m venv .venv`, install torch (the standard macOS wheel gives MPS), numpy, pillow, imageio, matplotlib, pytest. Write `requirements.txt` from what worked.
2. Refactor `nca.py` into the package. Keep a thin CLI: `python -m leafnca.train`, `python -m leafnca.demo`.
3. Write the tests (below) and get them green **before** any long training.
4. Train. Acceptance: after training, the grown leaf's MSE vs target is far below an untrained model's, the midrib and most side veins are visible in `demo_strip.png`, and the pattern is stable for 200 extra steps. If not met after the first run, change **one lever at a time**, in this order: more steps, hidden size 128, longer rollouts (64-128), lr. Log each try in DECISIONS.md. Maximum 4 training runs in total.
5. Run the experiments (below). Generate every figure and number by script, never by hand.
6. Write README, REPORT, STATUS. Commit in small steps with clear messages. Push after each major step (tests green, trained model, experiments, report).

## Tests (pytest, fast, no long training)
- leaf target: shape, alpha in [0,1], coverage between 0.3 and 0.6, premultiplied RGB.
- model: output shape equals input shape; a freshly initialised model is a no-op on a seed (zero-init); alive mask kills isolated cells that have no live neighbours; dead stays dead.
- determinism: with the same torch seed, two rollouts are identical (this is the control for the luck experiment).
- damage: mask removes cells only inside the circle; the undamaged region is unchanged.
- metrics: MSE of target vs target is 0; a damaged state has higher error than the target.
- smoke: 20 training steps run on CPU, loss is finite, checkpoint saves and loads.
- sanity: an **untrained** model does not regrow a cut leaf (guards against a metric that always says "healed").

## Experiments (keep each small; all in `experiments/`)
Report every number with the settings, seed and git commit it came from.
1. **Grow and hold.** From one seed, grow 120 steps, hold 200 more. Record MSE vs target over time. Question: does it stay stable?
2. **Damage and heal.** Three damage types: cut half the leaf, a circular hole, 40% random erase. Heal 150 steps. Record MSE before damage, right after, and after healing, for 20 random damage positions each. Question: how much does it recover, by damage type?
3. **Luck versus skill (fate curve).** Take a grown leaf, apply one damage, then fork **50 copies** that differ only in the random firing mask. Report the share that heal to within 1.5x the grown-leaf MSE, against steps. Repeat for 3 damage sizes. Include the control: forking with the same random seed must give identical outcomes. Question: how much of the outcome is dice?
4. **Where is the memory? (hidden versus visible).** Erase only the visible channels (RGB+alpha) of a region, then only the hidden channels, then both. Compare recovery. Question: is the pattern stored in the hidden state or rebuilt from neighbours?
5. **(Only if time allows) Training-seed variance.** Train 3 seeds at a reduced budget (15 min each) and compare experiment 2. Mark as "pilot".
Do experiments 1 to 4 in full. Experiment 5 is optional and must not delay pushing 1 to 4.

## Outputs
- `results/demo.gif`: growth, hold, three damages and heals (captioned frames).
- `results/demo_strip.png`: key frames in a row.
- One figure per experiment, a small table per experiment (CSV and in REPORT.md).
- REPORT.md: for each experiment a heading, the question, the result in one or two sentences, the figure, and an explicit "what this does not show" line. State the limits: one pattern, one architecture, small grid, few seeds. Do not over-claim.
- Provenance: label every figure with who produced it ("generated by experiments/xx.py at commit <sha>").

## Compute limits (do not choke the Mac)
- Use MPS if available, else CPU. `PYTORCH_ENABLE_MPS_FALLBACK=1`. `torch.set_num_threads(4)`.
- Run long jobs in the background with `nice -n 10`. Never run two training jobs at once.
- Max 30 minutes per training run (`--max_minutes 30`); total compute budget about 2 hours. Save checkpoints every 250 steps so nothing is lost.
- If loss becomes NaN or MPS misbehaves, rerun that step on CPU and note it in DECISIONS.md.
- Keep memory low: batch 8, pool 256, grid 64x64 (these are the defaults).
- Do not commit `.venv`, caches or checkpoints over 5 MB. Add a `.gitignore`.

## Autonomy rules (do not ask for my attention)
- Make reasonable decisions yourself and record them in DECISIONS.md.
- Ask me only if: git push is rejected for authentication, a step would use more than the compute budget above, or something would delete my files.
- If a step fails 3 times, write what happened to `STATUS.md` under "Blockers", skip it, and continue with the rest. Push what works.
- Never modify anything outside this repo folder. Never commit secrets.

## Done means
Tests green; trained model committed (small); GIF, strip and four experiment figures in `results/`; README, REPORT, DECISIONS and STATUS written; everything pushed to `main`; a final message of at most 10 lines stating what was achieved, the headline numbers, and any blockers.
