import math
import minipypy as mini


def assert_close_list(actual, expected, tol=1e-5):
    assert len(actual) == len(expected)

    for a, e in zip(actual, expected):
        assert abs(a - e) < tol


def test_sqrt_forward():
    x = mini.Tensor([1.0, 4.0, 9.0, 16.0])

    y = x.sqrt()

    assert_close_list(y.cpu(), [1.0, 2.0, 3.0, 4.0])


def test_sqrt_backward():
    x = mini.Tensor([1.0, 4.0, 9.0, 16.0], requires_grad=True)

    y = x.sqrt()
    loss = y.sum()
    loss.backward()

    expected = [
        1.0 / (2.0 * math.sqrt(1.0)),
        1.0 / (2.0 * math.sqrt(4.0)),
        1.0 / (2.0 * math.sqrt(9.0)),
        1.0 / (2.0 * math.sqrt(16.0)),
    ]

    assert_close_list(x.grad().cpu(), expected, tol=1e-4)
