import argparse

import numpy as np
import pytest
import torch

from leafnca import damage as D
from leafnca import model as M
from leafnca.metrics import healed, mse
from leafnca.targets import PAD, SIZE, load_target, make_leaf
from leafnca.train import parser, train

GRID = SIZE + 2 * PAD


@pytest.fixture
def target():
    return load_target("leaf")


def grown_state(target):
    """A state that looks exactly like the target (hidden channels zero)."""
    x = torch.zeros(1, M.CH, GRID, GRID)
    x[:, :4] = target
    return x


def random_ca():
    torch.manual_seed(1)
    ca = M.CA()
    torch.nn.init.normal_(ca.fc1.weight, std=0.1)
    return ca


# ---- leaf target -----------------------------------------------------------
def test_leaf_shape_alpha_coverage():
    a = make_leaf(SIZE)
    assert a.shape == (SIZE, SIZE, 4)
    assert a[..., 3].min() >= 0 and a[..., 3].max() <= 1
    cov = (a[..., 3] > 0.5).mean()
    assert 0.3 <= cov <= 0.6


def test_target_premultiplied(target):
    assert target.shape == (1, 4, GRID, GRID)
    assert torch.all(target[:, :3] <= target[:, 3:4] + 1e-6)


# ---- model -----------------------------------------------------------------
def test_output_shape():
    x = torch.randn(2, M.CH, 32, 32)
    assert random_ca()(x).shape == x.shape


def test_fresh_model_is_noop_on_seed():
    x = M.seed(1, GRID)
    assert torch.equal(M.CA()(x), x)


def test_alive_mask_kills_isolated_cell():
    x = torch.zeros(1, M.CH, 16, 16)
    x[:, 3, 8, 8] = 0.05   # below the alive threshold, no live neighbours
    x[:, 4:, 8, 8] = 1.0
    assert random_ca()(x).abs().sum() == 0


def test_dead_stays_dead():
    ca = random_ca()
    x = torch.zeros(1, M.CH, 16, 16)
    for _ in range(10):
        x = ca(x)
    assert x.abs().sum() == 0
    # and far from a seed, cells stay dead after a few steps (life spreads at most 1 cell per step)
    x = M.seed(1, GRID)
    for _ in range(5):
        x = ca(x)
    assert x[..., :GRID // 2 - 8, :].abs().sum() == 0


def test_rollout_determinism():
    ca = random_ca()
    outs = []
    for _ in range(2):
        torch.manual_seed(123)
        outs.append(ca.rollout(M.seed(2, GRID), 30))
    assert torch.equal(outs[0], outs[1])
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    assert torch.equal(ca.rollout(M.seed(1, GRID), 30, g1), ca.rollout(M.seed(1, GRID), 30, g2))


# ---- damage ----------------------------------------------------------------
def test_circle_damage_only_inside(target):
    x = grown_state(target) + 0.5
    m = D.circle_mask(GRID, GRID, 30, 34, 8)
    y = D.cut_circle(x, 30, 34, 8)
    inside, outside = m[0, 0] > 0, m[0, 0] == 0
    assert torch.all(y[0][:, inside] == 0)
    assert torch.equal(y[0][:, outside], x[0][:, outside])


def test_channel_damage(target):
    x = grown_state(target) + 0.5
    y = D.cut_circle(x, 32, 32, 8, channels=D.HIDDEN)
    assert torch.equal(y[:, :4], x[:, :4])
    assert y[:, 4:, 32, 32].abs().sum() == 0


def test_cut_half_and_erase(target):
    x = grown_state(target)
    y = D.cut_half(x)
    assert y[..., GRID // 2:].abs().sum() == 0 and torch.equal(y[..., :GRID // 2], x[..., :GRID // 2])
    y = D.erase_random(x + 1, 0.4, torch.Generator().manual_seed(0))
    frac = (y[0, 0] == 0).float().mean().item()
    assert 0.3 < frac < 0.5


# ---- metrics ---------------------------------------------------------------
def test_metrics(target):
    x = grown_state(target)
    assert mse(x, target).item() == 0
    assert mse(D.cut_circle(x, 32, 32, 10), target).item() > 0


# ---- smoke -----------------------------------------------------------------
def test_training_smoke(tmp_path):
    a = parser().parse_args(["--steps", "20", "--pool", "16", "--batch", "4", "--min_rollout", "8",
                             "--max_rollout", "12", "--hidden", "32", "--device", "cpu",
                             "--out", str(tmp_path)])
    _, log = train(a)
    assert len(log) == 20 and all(np.isfinite(l) for _, l, _ in log)
    ca, ck = M.load(tmp_path / "model.pt")
    assert ck["step"] == 20 and ca.hidden == 32
    x = M.seed(1, GRID)
    torch.manual_seed(0)
    assert ca(x).shape == x.shape


# ---- sanity: untrained models do not "heal" --------------------------------
@pytest.mark.parametrize("make", [M.CA, random_ca])
def test_untrained_does_not_regrow(target, make):
    ca = make()
    x = grown_state(target)
    ref = mse(x, target) + 1e-4  # tiny reference, like a well-trained model's grown MSE
    xd = D.cut_half(x)
    torch.manual_seed(0)
    with torch.no_grad():
        xh = ca.rollout(xd, 100)
    assert not healed(xh, target, ref).item()
    assert mse(xh, target).item() > 0.5 * mse(xd, target).item()
