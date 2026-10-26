"""Procedural leaf-with-veins target (PIL only, no downloads)."""
import numpy as np
import PIL.Image
import PIL.ImageDraw
import torch
import torch.nn.functional as F

SIZE = 48  # target size
PAD = 8    # margin so the pattern can be damaged/regrown inside the grid (grid = 64x64)


def make_leaf(size=SIZE, ss=8):
    """Return RGBA float array [size,size,4] (straight alpha): green blade, lighter midrib and side veins."""
    s = size * ss
    im = PIL.Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = PIL.ImageDraw.Draw(im)
    cx, top, bot = s / 2, s * 0.06, s * 0.94
    # blade outline: pointed ellipse-ish shape, widest a little below the middle
    pts_l, pts_r = [], []
    for t in np.linspace(0, 1, 80):
        y = top + t * (bot - top)
        w = (s * 0.36) * np.sin(np.pi * t ** 0.8) ** 0.9
        pts_l.append((cx - w, y)); pts_r.append((cx + w, y))
    d.polygon(pts_l + pts_r[::-1], fill=(76, 160, 64, 255), outline=(40, 100, 40, 255))
    # midrib
    vein = (170, 220, 110, 255)
    d.line([(cx, top), (cx, bot + s * 0.04)], fill=vein, width=int(ss * 1.6))
    # side veins: curve outward and up from the midrib
    for k in range(1, 6):
        y0 = top + (bot - top) * (0.16 + 0.13 * k)
        for sgn in (-1, 1):
            xs = [cx + sgn * (s * 0.25) * (u ** 0.85) for u in np.linspace(0, 1, 20)]
            ys = [y0 - (s * 0.11) * (u ** 1.3) for u in np.linspace(0, 1, 20)]
            d.line(list(zip(xs, ys)), fill=vein, width=int(ss * 1.1))
    im = im.resize((size, size), PIL.Image.LANCZOS)
    return np.float32(im) / 255.0


def load_target(spec="leaf", size=SIZE, pad=PAD, device="cpu"):
    """Target as premultiplied RGBA tensor [1,4,size+2*pad,size+2*pad]. spec: 'leaf' or an image path."""
    if spec == "leaf":
        a = make_leaf(size)
    else:
        a = np.float32(PIL.Image.open(spec).convert("RGBA").resize((size, size), PIL.Image.LANCZOS)) / 255.0
    a[..., :3] *= a[..., 3:]  # premultiplied alpha
    t = F.pad(torch.tensor(a).permute(2, 0, 1), [pad] * 4)
    return t[None].to(device)
