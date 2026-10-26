"""Shared setup for the experiment scripts: load model, grow a leaf, write CSV/JSON/figures with provenance."""
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from leafnca import model as M  # noqa: E402
from leafnca.metrics import mse, to_image  # noqa: E402
from leafnca.targets import load_target  # noqa: E402
from leafnca.utils import ROOT, get_device, git_sha, setup_threads, stamp  # noqa: E402

CKPT = Path(os.environ.get("LEAFNCA_CKPT", ROOT / "results" / "model.pt"))  # env override for dry runs
GROW = 120  # steps from one seed to a grown leaf
DEVICE = "cpu"  # experiments run on CPU: exact reproducibility of seeded firing masks; fast enough at 64x64


class Exp:
    def __init__(self, name, script, seed=0):
        setup_threads()
        torch.set_grad_enabled(False)
        self.name, self.script, self.seed = name, script, seed
        self.dev = get_device(DEVICE)
        self.out = ROOT / "results" / name
        self.out.mkdir(parents=True, exist_ok=True)
        self.ca, ck = M.load(CKPT, self.dev)
        self.target = load_target(ck.get("target", "leaf"), device=self.dev)
        self.H = self.target.shape[-1]
        self.meta = {"script": script, "commit": git_sha(), "seed": seed, "checkpoint": os.path.relpath(CKPT, ROOT),
                     "checkpoint_train_step": ck.get("step"), "device": str(self.dev), "grid": self.H}

    def gen(self, offset=0):
        return torch.Generator(device=self.dev).manual_seed(self.seed + offset)

    def grown(self, steps=GROW, offset=0):
        """Grow one leaf from a single seed. Returns state [1,ch,H,W]."""
        return self.ca.rollout(M.seed(1, self.H, device=self.dev), steps, self.gen(offset))

    def mse(self, x):
        return mse(x, self.target)

    def write_csv(self, fname, rows):
        with open(self.out / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    def write_json(self, fname, data):
        with open(self.out / fname, "w") as f:
            json.dump({**self.meta, **data}, f, indent=2)

    def save_fig(self, fig, fname):
        stamp(fig, self.script)
        fig.savefig(self.out / fname, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("wrote", self.out / fname)


def show(ax, x, title=""):
    ax.imshow(to_image(x), interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def fmt(v, nd=5):
    return float(f"{v:.{nd}g}")


if __name__ == "__main__":
    sys.exit("helper module; run one of the experiment scripts")
