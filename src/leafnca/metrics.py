"""Error metrics and rendering."""
import torch

HEAL_FACTOR = 1.5  # "healed" = within 1.5x the grown-leaf MSE


def mse(x, target):
    """Per-sample MSE between the visible channels (RGBA) and the target. Returns [b]."""
    return (x[:, :4] - target).square().mean(dim=[1, 2, 3])


def healed(x, target, reference_mse, factor=HEAL_FACTOR):
    return mse(x, target) <= factor * reference_mse


def to_rgb(x):
    """Composite premultiplied RGBA onto white. Returns [b,3,h,w] in [0,1]."""
    return (1.0 - x[:, 3:4].clamp(0, 1) + x[:, :3]).clamp(0, 1)


def to_image(x):
    """First sample as a uint8 HxWx3 numpy array."""
    return (to_rgb(x[:1])[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
