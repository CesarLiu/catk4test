import torch

from src.responsibility.risk import cvar, value_at_risk


def test_cvar_alpha_near_one_is_max():
    samples = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0])
    result = cvar(samples, alpha=0.999)
    assert torch.isclose(result, samples.max(), atol=1e-3)


def test_cvar_alpha_near_zero_is_close_to_mean():
    samples = torch.arange(1, 101).float()  # 1..100, fine-grained so alpha~0 tail ~= whole set
    result = cvar(samples, alpha=1e-6)
    assert torch.isclose(result, samples.mean(), atol=1.0)


def test_cvar_monotonic_in_alpha():
    samples = torch.tensor([0.0, 1.0, 2.0, 3.0, 10.0])
    low = cvar(samples, alpha=0.1)
    mid = cvar(samples, alpha=0.5)
    high = cvar(samples, alpha=0.9)
    assert low <= mid + 1e-6
    assert mid <= high + 1e-6


def test_cvar_single_sample():
    samples = torch.tensor([[7.0]])
    result = cvar(samples, alpha=0.5)
    assert torch.isclose(result, torch.tensor([7.0]))


def test_cvar_batched():
    samples = torch.stack(
        [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([10.0, 20.0, 30.0])]
    )
    result = cvar(samples, alpha=0.999)
    assert torch.allclose(result, torch.tensor([3.0, 30.0]), atol=1e-2)


def test_value_at_risk_matches_quantile():
    samples = torch.arange(0, 11).float()
    assert torch.isclose(value_at_risk(samples, 0.5), torch.quantile(samples, 0.5))
