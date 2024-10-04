import pytest
import torch

from shared.signature import Zhang2DSignature


@pytest.fixture
def zs() -> Zhang2DSignature:
    return Zhang2DSignature()


# def test_vectorization(zs) -> None:
#     tol = torch.Tensor([1e-10])
#     n = 5
#     tensor = torch.randn(size=(n, n))

#     brute_force = zs._brute_force(tensor=tensor)

#     assert brute_force[0] - zs._sum_expression(tensor=tensor, n=n, m=n) < tol
#     assert brute_force[1] - zs._sum_expression_hat(tensor=tensor, n=n, m=n) < tol


def test_dim(zs) -> None:
    batch, channels, n, n = 1, 8, 32, 32
    x = torch.ones(size=(batch, channels, n, n))

    assert zs.calculate_batch_sig(x=x, depth=1).shape[1] == zs.calculate_feature_dim(
        channels=channels, depth=1
    )
    assert zs.calculate_batch_sig(x=x, depth=2).shape[1] == zs.calculate_feature_dim(
        channels=channels, depth=2
    )
