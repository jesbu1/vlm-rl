import numpy as np
import pytest
import torch

from vlmrl import algo


def test_center():
    assert np.allclose(algo.group_advantages([[1, 0, 0, 1]], "center")[0], [0.5, -0.5, -0.5, 0.5])


def test_zscore():
    a = algo.group_advantages([[1, 0, 0, 1]], "zscore")[0]
    assert np.allclose(a, [1, -1, -1, 1], atol=1e-4)


def test_rloo():
    r = algo.group_advantages([[1, 0, 0, 0]], "rloo")[0]
    assert np.isclose(r[0], 1.0) and np.allclose(r[1:], -1 / 3)


def test_advantages_sum_to_zero_per_group():
    for mode in ("center", "zscore"):
        for g in algo.group_advantages([[0.3, 2, -1], [5, 5, 6, 1]], mode):
            assert abs(sum(g)) < 1e-9


def test_bad_mode():
    with pytest.raises(ValueError):
        algo.group_advantages([[1, 2]], "nope")


def test_group_filter():
    assert not algo.keep_group([1, 1, 1])
    assert not algo.keep_group([0.0, 0.0])
    assert algo.keep_group([1, 0, 1])


def test_reinforce_gradient():
    logp = torch.tensor([[-1.0, -2.0, -3.0]], requires_grad=True)
    adv = torch.full_like(logp, 2.0)
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    loss, stats = algo.policy_loss_sum(logp, adv, mask)
    loss.backward()
    assert torch.allclose(logp.grad, torch.tensor([[-2.0, -2.0, 0.0]]))  # masked tokens get no grad
    assert np.isclose(loss.item(), 2 * (1 + 2))
    assert stats == {}


def test_ppo_at_ratio_one_matches_reinforce():
    logp = torch.randn(2, 5, requires_grad=True)
    adv = torch.randn(2, 1).expand(2, 5)
    mask = torch.ones(2, 5)
    algo.policy_loss_sum(logp, adv, mask)[0].backward()
    g1 = logp.grad.clone()
    lp2 = logp.detach().clone().requires_grad_(True)
    algo.policy_loss_sum(lp2, adv, mask, old_logp=lp2.detach())[0].backward()
    assert torch.allclose(lp2.grad, g1, atol=1e-6)


def test_ppo_clipping_zeroes_gradient_outside_trust_region():
    old = torch.zeros(1, 2)
    logp = torch.tensor([[0.5, -0.5]], requires_grad=True)  # ratio e^.5 = 1.65 (clipped for A>0), e^-.5
    adv = torch.ones(1, 2)
    loss, stats = algo.policy_loss_sum(logp, adv, torch.ones(1, 2), old_logp=old, clip=(0.2, 0.28))
    loss.backward()
    assert logp.grad[0, 0] == 0 and logp.grad[0, 1] != 0
    assert stats["clipfrac_sum"] == 2.0  # both outside [0.8, 1.28]


def test_token_sum_normalised_globally_equals_concatenated_mean():
    """Summing per micro-batch then dividing by the global token count == one big token-mean."""
    torch.manual_seed(0)
    lps = [torch.randn(1, 7), torch.randn(1, 3)]
    masks = [torch.ones(1, 7), torch.ones(1, 3)]
    advs = [torch.full((1, 7), 1.0), torch.full((1, 3), -2.0)]
    n = sum(m.sum() for m in masks)
    split = sum(algo.policy_loss_sum(l, a, m)[0] for l, a, m in zip(lps, advs, masks)) / n
    full = (-(torch.cat(advs, 1) * torch.cat(lps, 1))).mean()
    assert torch.isclose(split, full)
