import minipypy as mini


def test_tensor_mul_scalar_backward():
    x = mini.Tensor([1.0, 2.0, 3.0], requires_grad=True)

    y = x * 0.5
    z = y.sum()
    z.backward()

    assert x.grad().cpu() == [0.5, 0.5, 0.5]


def test_scalar_mul_tensor_backward():
    x = mini.Tensor([1.0, 2.0, 3.0], requires_grad=True)

    y = 0.5 * x
    z = y.sum()
    z.backward()

    assert x.grad().cpu() == [0.5, 0.5, 0.5]


def test_tensor_add_scalar_backward():
    x = mini.Tensor([1.0, 2.0, 3.0], requires_grad=True)

    y = x + 2.0
    z = y.sum()
    z.backward()

    assert x.grad().cpu() == [1.0, 1.0, 1.0]


def test_scalar_add_tensor_backward():
    x = mini.Tensor([1.0, 2.0, 3.0], requires_grad=True)

    y = 2.0 + x
    z = y.sum()
    z.backward()

    assert x.grad().cpu() == [1.0, 1.0, 1.0]


def test_tensor_sub_scalar_backward():
    x = mini.Tensor([1.0, 2.0, 3.0], requires_grad=True)

    y = x - 2.0
    z = y.sum()
    z.backward()

    assert x.grad().cpu() == [1.0, 1.0, 1.0]


def test_scalar_sub_tensor_backward():
    x = mini.Tensor([1.0, 2.0, 3.0], requires_grad=True)

    y = 2.0 - x
    z = y.sum()
    z.backward()

    assert x.grad().cpu() == [-1.0, -1.0, -1.0]


def test_tensor_div_scalar_backward():
    x = mini.Tensor([2.0, 4.0, 8.0], requires_grad=True)

    y = x / 2.0
    z = y.sum()
    z.backward()

    assert x.grad().cpu() == [0.5, 0.5, 0.5]


def test_scalar_div_tensor_backward():
    x = mini.Tensor([2.0, 4.0, 8.0], requires_grad=True)

    y = 8.0 / x
    z = y.sum()
    z.backward()

    grad = x.grad().cpu()

    # d(8/x)/dx = -8/x^2
    assert abs(grad[0] - (-2.0)) < 1e-5
    assert abs(grad[1] - (-0.5)) < 1e-5
    assert abs(grad[2] - (-0.125)) < 1e-5
