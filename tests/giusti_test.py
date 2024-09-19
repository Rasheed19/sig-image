import numpy as np
import pytest

from shared.signature import Giusti2DSignature


@pytest.fixture
def gs() -> Giusti2DSignature:
    return Giusti2DSignature()


def test_permutation(gs) -> None:
    perm = [gs._permutation(x=x, deg=deg) for deg in [1, 2] for x in [1, 2]]

    assert perm == [1, 2, 2, 1]


def test_injection(gs) -> None:
    perm = [gs._injection(x=x, deg=deg) for deg in [1, 2, 3] for x in [1, 2]]

    assert perm == [1, 2, 1, 3, 2, 3]


def test_vectorization(gs) -> None:
    tol = 1e-10
    n = 5
    arr1 = np.random.randn(n, n)
    arr2 = np.random.randn(n, n)

    brute_force = gs._brute_force(arr1=arr1, arr2=arr2)
    vectorized = gs._sum_expression(arr1=arr1, arr2=arr2)

    assert brute_force - vectorized < tol
