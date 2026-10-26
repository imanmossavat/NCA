"""Sample-pool training with damage.

  python -m leafnca.train --steps 4000 --max_minutes 30 --out runs/run1
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from . import model as M
from .damage import random_circles
from .metrics import mse
from .targets import load_target
from .utils import get_device, git_sha, setup_threads


def train(a):
    setup_threads()
    dev = get_device(a.device)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    target = load_target(a.target, device=dev)
    H = target.shape[-1]
    ca = M.CA(hidden=a.hidden).to(dev)
    start = 0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu")
        ca.load_state_dict(ck["model"]); start = ck.get("step", 0)
    opt = torch.optim.Adam(ca.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(a.steps * 0.4), int(a.steps * 0.75)], 0.3)
    pool = M.seed(a.pool, H, device=dev)
    meta = {"target": a.target, "seed": a.seed, "commit": git_sha(), "device": str(dev),
            "args": vars(a)}
    with open(f"{a.out}/config.json", "w") as f:
        json.dump(meta, f, indent=2)
    log = []
    t0 = time.time()
    step = start
    for i in range(start, start + a.steps):
        idx = torch.randperm(a.pool, device=dev)[:a.batch]
        x = pool[idx]
        order = mse(x, target).argsort(descending=True)  # worst first
        idx, x = idx[order], x[order]
        x[:1] = M.seed(1, H, device=dev)                 # replace worst with fresh seed
        if a.damage > 0:
            x[-a.damage:] = random_circles(x[-a.damage:])  # damaged samples teach regeneration
        for _ in range(np.random.randint(a.min_rollout, a.max_rollout + 1)):
            x = ca(x)
        loss = mse(x, target).mean()
        opt.zero_grad(); loss.backward()
        for p in ca.parameters():
            if p.grad is not None:
                p.grad /= p.grad.norm() + 1e-8           # per-tensor gradient normalization
        opt.step(); sched.step()
        pool[idx] = x.detach()
        step = i + 1
        log.append((step, loss.item(), time.time() - t0))
        if not np.isfinite(log[-1][1]):
            print(f"non-finite loss at step {step}, stopping"); break
        if step % 100 == 0 or i == start:
            print(f"step {step:6d}  loss {np.mean([l for _, l, _ in log[-100:]]):.5f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
        if step % a.ckpt_every == 0:
            M.save(ca, f"{a.out}/model.pt", step=step, **meta)
        if a.max_minutes and (time.time() - t0) > a.max_minutes * 60:
            print(f"time budget reached at step {step}"); break
    M.save(ca, f"{a.out}/model.pt", step=step, **meta)
    with open(f"{a.out}/loss.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "loss", "seconds"]); w.writerows(log)
    print("saved", f"{a.out}/model.pt")
    return ca, log


def parser():
    t = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    t.add_argument("--target", default="leaf"); t.add_argument("--out", default="runs/run1")
    t.add_argument("--steps", type=int, default=4000); t.add_argument("--pool", type=int, default=256)
    t.add_argument("--batch", type=int, default=8); t.add_argument("--damage", type=int, default=3)
    t.add_argument("--hidden", type=int, default=128); t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--min_rollout", type=int, default=64); t.add_argument("--max_rollout", type=int, default=96)
    t.add_argument("--ckpt_every", type=int, default=250); t.add_argument("--resume", default=None)
    t.add_argument("--seed", type=int, default=0); t.add_argument("--device", default=None)
    t.add_argument("--max_minutes", type=float, default=0, help="stop gracefully after this many minutes (0 = no limit)")
    return t


if __name__ == "__main__":
    train(parser().parse_args())
