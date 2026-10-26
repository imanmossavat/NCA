"""Damage operators. Each returns a damaged copy of the state x [b,ch,h,w]."""
import math

import torch

VISIBLE = slice(0, 4)   # RGB + alpha
RGB = slice(0, 3)
HIDDEN = slice(4, None)


def circle_mask(h, w, cy, cx, r, device="cpu"):
    """[1,1,h,w] float mask, 1 inside the circle centred at (cy, cx) in pixels."""
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    return (((yy - cy) ** 2 + (xx - cx) ** 2) < r ** 2).float()[None, None]


def random_circle_masks(n, h, w, device="cpu"):
    """Training damage: n circles, random centre in the middle half, random radius (10-40% of half-width)."""
    xs = torch.linspace(-1, 1, w, device=device)[None, None, :]
    ys = torch.linspace(-1, 1, h, device=device)[None, :, None]
    c = torch.empty(2, n, 1, 1, device=device).uniform_(-0.5, 0.5)
    r = torch.empty(n, 1, 1, device=device).uniform_(0.1, 0.4)
    return ((((xs - c[0]) / r) ** 2 + ((ys - c[1]) / r) ** 2) < 1.0).float()[:, None]


def random_circles(x):
    return x * (1.0 - random_circle_masks(x.shape[0], x.shape[2], x.shape[3], x.device))


def cut_circle(x, cy, cx, r, channels=slice(None)):
    """Zero the given channels inside a circle (default: all channels)."""
    m = circle_mask(x.shape[2], x.shape[3], cy, cx, r, x.device)
    x = x.clone()
    x[:, channels] = x[:, channels] * (1.0 - m)
    return x


def cut_half(x, angle=0.0):
    """Remove the half-plane on one side of a line through the grid centre. angle=0 removes the right half."""
    h, w = x.shape[2:]
    yy, xx = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij")
    side = (xx - (w - 1) / 2) * math.cos(angle) + (yy - (h - 1) / 2) * math.sin(angle)
    return x * (side <= 0).float()[None, None]


def erase_random(x, frac=0.4, generator=None):
    """Zero a random `frac` of cells (all channels)."""
    keep = torch.rand((1, 1, *x.shape[2:]), generator=generator, device=x.device) >= frac
    return x * keep.float()
