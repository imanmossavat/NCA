"""The NCA update rule: Sobel perception -> tiny MLP -> residual update, async firing, alive masking.

Building blocks follow google-research/self-organising-systems (Growing NCA).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CH = 16  # state channels: 0-2 RGB (premultiplied), 3 alpha, 4-15 hidden
ALIVE_THRESHOLD = 0.1

ident = torch.tensor([[0., 0, 0], [0, 1, 0], [0, 0, 0]])
sobel_x = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
FILTERS = torch.stack([ident, sobel_x, sobel_x.T])


def perchannel_conv(x, filters):
    """x: [b,ch,h,w], filters: [n,3,3] -> [b,ch*n,h,w] (depthwise, circular pad)."""
    b, ch, h, w = x.shape
    y = x.reshape(b * ch, 1, h, w)
    y = F.pad(y, [1, 1, 1, 1], "circular")
    y = F.conv2d(y, filters[:, None].to(x.device, x.dtype))
    return y.reshape(b, -1, h, w)


def alive(x):
    """A cell is alive if it or any of its 8 neighbours has alpha > 0.1."""
    return F.max_pool2d(x[:, 3:4], 3, 1, 1) > ALIVE_THRESHOLD


class CA(nn.Module):
    def __init__(self, ch=CH, hidden=128, fire_rate=0.5):
        super().__init__()
        self.ch, self.hidden, self.fire_rate = ch, hidden, fire_rate
        self.fc0 = nn.Conv2d(ch * 3, hidden, 1)
        self.fc1 = nn.Conv2d(hidden, ch, 1, bias=False)
        nn.init.zeros_(self.fc1.weight)  # start as "do nothing"

    def forward(self, x, generator=None):
        pre = alive(x)
        dx = self.fc1(F.relu(self.fc0(perchannel_conv(x, FILTERS))))
        shape = (x.shape[0], 1, *x.shape[2:])
        fire = (torch.rand(shape, device=x.device, generator=generator) < self.fire_rate).to(x.dtype)
        x = x + dx * fire
        return x * (pre & alive(x)).to(x.dtype)

    def rollout(self, x, steps, generator=None):
        for _ in range(steps):
            x = self(x, generator)
        return x


def seed(n, size, ch=CH, device="cpu"):
    """One live cell in the centre: alpha and hidden channels set to 1."""
    x = torch.zeros(n, ch, size, size, device=device)
    x[:, 3:, size // 2, size // 2] = 1.0
    return x


def save(ca, path, **extra):
    torch.save({"model": {k: v.cpu() for k, v in ca.state_dict().items()},
                "hidden": ca.hidden, "fire_rate": ca.fire_rate, **extra}, path)


def load(path, device="cpu"):
    ck = torch.load(path, map_location="cpu")
    ca = CA(hidden=ck.get("hidden", 128), fire_rate=ck.get("fire_rate", 0.5))
    ca.load_state_dict(ck["model"])
    return ca.to(device).eval(), ck
