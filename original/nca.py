"""Growing Neural Cellular Automaton (PyTorch, CPU/GPU).

Reuses building blocks from google-research/self-organising-systems
(isotropic_nca notebook): Sobel/identity filters, perchannel_conv perception,
circle damage masks, and sample-pool training.

Usage:
  python nca.py train --steps 4000 --max_minutes 30 --out run1
  python nca.py demo  --ckpt run1/model.pt --out run1    # growth + damage + regeneration

Target: --target leaf (default, veined leaf, no download) | procedural (smiley), a path to an image, or emoji:<char>.
"""
import argparse, io, os, time
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Mac GPU: fall back to CPU for any unsupported op
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL.Image, PIL.ImageDraw

CH = 16        # state channels: 0-2 RGB, 3 alpha, 4-15 hidden
SIZE = 48      # target size
PAD = 8        # margin so the pattern can be damaged/regrown inside the grid
if torch.cuda.is_available():
    DEV = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEV = torch.device("mps")   # Apple Silicon GPU
else:
    DEV = torch.device("cpu")
torch.set_num_threads(4)        # leave cores free so the laptop stays usable

# ---- perception (from the repo notebook) -----------------------------------
ident = torch.tensor([[0., 0, 0], [0, 1, 0], [0, 0, 0]])
sobel_x = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
FILTERS = torch.stack([ident, sobel_x, sobel_x.T])


def perchannel_conv(x, filters):
    """x: [b,ch,h,w], filters: [n,3,3] -> [b,ch*n,h,w] (depthwise, circular pad)."""
    b, ch, h, w = x.shape
    y = x.reshape(b * ch, 1, h, w)
    y = F.pad(y, [1, 1, 1, 1], "circular")
    y = F.conv2d(y, filters[:, None].to(x.device))
    return y.reshape(b, -1, h, w)


class CA(nn.Module):
    def __init__(self, ch=CH, hidden=128, fire_rate=0.5):
        super().__init__()
        self.ch, self.fire_rate = ch, fire_rate
        self.fc0 = nn.Conv2d(ch * 3, hidden, 1)
        self.fc1 = nn.Conv2d(hidden, ch, 1, bias=False)
        nn.init.zeros_(self.fc1.weight)  # start as "do nothing"

    @staticmethod
    def alive(x):
        return F.max_pool2d(x[:, 3:4], 3, 1, 1) > 0.1

    def forward(self, x):
        pre = self.alive(x)
        dx = self.fc1(F.relu(self.fc0(perchannel_conv(x, FILTERS))))
        fire = (torch.rand(x.shape[0], 1, *x.shape[2:], device=x.device) < self.fire_rate).float()
        x = x + dx * fire
        return x * (pre & self.alive(x)).float()

    def seed(self, n, size):
        x = torch.zeros(n, self.ch, size, size, device=DEV)
        x[:, 3:, size // 2, size // 2] = 1.0
        return x


# ---- targets ---------------------------------------------------------------
def procedural_target(size):
    s = size * 4
    im = PIL.Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = PIL.ImageDraw.Draw(im)
    d.ellipse([4, 4, s - 5, s - 5], fill=(250, 205, 40, 255), outline=(200, 120, 20, 255), width=6)
    d.ellipse([s * .30, s * .28, s * .42, s * .48], fill=(30, 30, 30, 255))
    d.ellipse([s * .58, s * .28, s * .70, s * .48], fill=(30, 30, 30, 255))
    d.arc([s * .22, s * .30, s * .78, s * .78], 20, 160, fill=(150, 30, 30, 255), width=8)
    return im.resize((size, size), PIL.Image.LANCZOS)


def load_target(spec, size=SIZE):
    if spec == "leaf":
        from leaf import make_leaf
        a = make_leaf(size)
        a[..., :3] *= a[..., 3:]
        t = F.pad(torch.tensor(a).permute(2, 0, 1), [PAD] * 4)
        return t[None].to(DEV)
    if spec == "procedural":
        im = procedural_target(size)
    elif spec.startswith("emoji:"):
        import requests
        code = hex(ord(spec[6]))[2:].lower()
        url = f"https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u{code}.png?raw=true"
        im = PIL.Image.open(io.BytesIO(requests.get(url, timeout=30).content)).convert("RGBA")
        im = im.resize((size, size), PIL.Image.LANCZOS)
    else:
        im = PIL.Image.open(spec).convert("RGBA").resize((size, size), PIL.Image.LANCZOS)
    a = np.float32(im) / 255.0
    a[..., :3] *= a[..., 3:]  # premultiplied alpha
    t = torch.tensor(a).permute(2, 0, 1)
    t = F.pad(t, [PAD] * 4)
    return t[None].to(DEV)  # [1,4,H,W]


def to_rgb(x):
    return (1.0 - x[:, 3:4].clamp(0, 1) + x[:, :3]).clamp(0, 1)


def loss_per_sample(x, target):
    return (x[:, :4] - target).square().mean(dim=[1, 2, 3])


# ---- damage ----------------------------------------------------------------
def circle_masks(n, h, w):
    xs = torch.linspace(-1, 1, w, device=DEV)[None, None, :]
    ys = torch.linspace(-1, 1, h, device=DEV)[None, :, None]
    c = torch.empty(2, n, 1, 1, device=DEV).uniform_(-0.5, 0.5)
    r = torch.empty(n, 1, 1, device=DEV).uniform_(0.1, 0.4)
    return ((((xs - c[0]) / r) ** 2 + ((ys - c[1]) / r) ** 2) < 1.0).float()[:, None]


def damage(x):
    return x * (1.0 - circle_masks(x.shape[0], x.shape[2], x.shape[3]))


# ---- train -----------------------------------------------------------------
def train(a):
    os.makedirs(a.out, exist_ok=True)
    target = load_target(a.target)
    H = target.shape[-1]
    ca = CA().to(DEV)
    start = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=DEV)
        ca.load_state_dict(ck["model"]); start = ck.get("step", 0)
    opt = torch.optim.Adam(ca.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [1500, 3000], 0.3)
    pool = ca.seed(a.pool, H)
    log = []
    t0 = time.time()
    for i in range(start, start + a.steps):
        idx = torch.randperm(a.pool, device=DEV)[:a.batch]
        x = pool[idx]
        order = loss_per_sample(x, target).argsort(descending=True)  # worst first
        idx, x = idx[order], x[order]
        x[:1] = ca.seed(1, H)                       # replace worst with fresh seed
        if a.damage > 0:
            x[-a.damage:] = damage(x[-a.damage:])   # damaged samples teach regeneration
        for _ in range(np.random.randint(64, 97)):
            x = ca(x)
        loss = loss_per_sample(x, target).mean()
        opt.zero_grad(); loss.backward()
        for p in ca.parameters():
            p.grad /= p.grad.norm() + 1e-8          # per-tensor gradient normalization
        opt.step(); sched.step()
        pool[idx] = x.detach()
        log.append(loss.item())
        if (i + 1) % 100 == 0 or i == start:
            print(f"step {i+1:6d}  loss {np.mean(log[-100:]):.5f}  {time.time()-t0:.0f}s", flush=True)
        if a.max_minutes and (time.time() - t0) > a.max_minutes * 60:
            print(f"time budget reached at step {i+1}"); a.steps = i + 1 - start; break
        if (i + 1) % a.ckpt_every == 0:
            torch.save({"model": ca.state_dict(), "step": i + 1, "target": a.target}, f"{a.out}/model.pt")
    torch.save({"model": ca.state_dict(), "step": start + a.steps, "target": a.target}, f"{a.out}/model.pt")
    np.save(f"{a.out}/loss.npy", np.array(log))
    print("saved", f"{a.out}/model.pt")


# ---- demo: growth, stability, damage, self-healing -------------------------
def frame(x, caption, scale=6):
    img = to_rgb(x)[0].permute(1, 2, 0).cpu().numpy()
    im = PIL.Image.fromarray((img * 255).astype(np.uint8)).resize(
        (img.shape[1] * scale, img.shape[0] * scale), PIL.Image.NEAREST)
    bar = PIL.Image.new("RGB", (im.width, 20), (255, 255, 255))
    PIL.ImageDraw.Draw(bar).text((4, 4), caption, fill=(0, 0, 0))
    out = PIL.Image.new("RGB", (im.width, im.height + 20), (255, 255, 255))
    out.paste(im, (0, 0)); out.paste(bar, (0, im.height))
    return out


@torch.no_grad()
def demo(a):
    ck = torch.load(a.ckpt, map_location=DEV)
    target = load_target(a.target or ck.get("target", "leaf"))
    H = target.shape[-1]
    ca = CA().to(DEV); ca.load_state_dict(ck["model"]); ca.eval()
    frames, keys, metrics = [], [], {}

    def run(x, n, label, every=3):
        for t in range(n):
            x = ca(x)
            if t % every == 0:
                frames.append(frame(x, f"{label} t={t}"))
        return x

    def err(x):
        return loss_per_sample(x, target).item()

    def cut_right(x):
        x = x.clone(); x[..., H // 2:] = 0; return x

    def cut_circle(x):
        m = torch.zeros(1, 1, H, H, device=DEV)
        yy, xx = torch.meshgrid(torch.arange(H, device=DEV), torch.arange(H, device=DEV), indexing="ij")
        m[0, 0] = (((yy - H // 2 + 6) ** 2 + (xx - H // 2 - 4) ** 2) < 9 ** 2).float()
        return x * (1 - m)

    def erase_random(x):
        return x * (torch.rand(1, 1, H, H, device=DEV) > 0.4).float()

    x = run(ca.seed(1, H), a.grow, "grow"); keys.append(frame(x, "grown")); metrics["grown"] = err(x)
    x = run(x, a.hold, "hold", every=10);  keys.append(frame(x, "stable after hold")); metrics["after_hold(stability)"] = err(x)
    for name, fn in [("half-cut", cut_right), ("circle-cut", cut_circle), ("40%-random-erase", erase_random)]:
        xd = fn(x); keys.append(frame(xd, f"damaged: {name}")); metrics[f"{name} damaged"] = err(xd)
        x = run(xd, a.heal, f"heal {name}"); keys.append(frame(x, f"healed: {name}")); metrics[f"{name} healed"] = err(x)

    os.makedirs(a.out, exist_ok=True)
    import imageio.v2 as imageio
    imageio.mimsave(f"{a.out}/demo.gif", [np.array(f) for f in frames], duration=0.06, loop=0)
    strip = PIL.Image.new("RGB", (sum(k.width for k in keys), keys[0].height), (255, 255, 255))
    xo = 0
    for k in keys:
        strip.paste(k, (xo, 0)); xo += k.width
    strip.save(f"{a.out}/demo_strip.png")
    print("\nMSE vs target (lower = better):")
    for k, v in metrics.items():
        print(f"  {k:28s} {v:.5f}")
    print(f"wrote {a.out}/demo.gif and {a.out}/demo_strip.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train")
    t.add_argument("--target", default="leaf"); t.add_argument("--out", default="run1")
    t.add_argument("--steps", type=int, default=4000); t.add_argument("--pool", type=int, default=256)
    t.add_argument("--batch", type=int, default=8); t.add_argument("--damage", type=int, default=3)
    t.add_argument("--ckpt_every", type=int, default=250); t.add_argument("--resume", default=None)
    t.add_argument("--max_minutes", type=float, default=0, help="stop gracefully after this many minutes (0 = no limit)")
    d = sp.add_parser("demo")
    d.add_argument("--ckpt", required=True); d.add_argument("--target", default=None)
    d.add_argument("--out", default="run1"); d.add_argument("--grow", type=int, default=120)
    d.add_argument("--hold", type=int, default=200); d.add_argument("--heal", type=int, default=150)
    a = p.parse_args()
    train(a) if a.cmd == "train" else demo(a)
