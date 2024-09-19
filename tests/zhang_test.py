import numpy as np
import pytest

from shared.signature import Zhang2DSignature


@pytest.fixture
def zs() -> Zhang2DSignature:
    return Zhang2DSignature()


def test_vectorization(zs) -> None:
    tol = 1e-10
    n = 5
    arr = np.random.randn(n, n)

    brute_force = zs._brute_force(arr=arr)

    assert brute_force[0] - zs._sum_expression(arr=arr, n=n, m=n) < tol
    assert brute_force[1] - zs._sum_expression_hat(arr=arr, n=n, m=n) < tol
