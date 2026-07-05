import minipypy as mini


def test_tensor_mul_scalar():
    x = mini.Tensor([1.0, 2.0, 3.0])

    y = x * 0.5
    z = 0.5 * x

    assert y.cpu() == [0.5, 1.0, 1.5]
    assert z.cpu() == [0.5, 1.0, 1.5]


def test_tensor_add_scalar():
    x = mini.Tensor([1.0, 2.0, 3.0])

    y = x + 2.0
    z = 2.0 + x

    assert y.cpu() == [3.0, 4.0, 5.0]
    assert z.cpu() == [3.0, 4.0, 5.0]


def test_tensor_sub_scalar():
    x = mini.Tensor([1.0, 2.0, 3.0])

    y = x - 2.0
    z = 2.0 - x

    assert y.cpu() == [-1.0, 0.0, 1.0]
    assert z.cpu() == [1.0, 0.0, -1.0]


def test_tensor_div_scalar():
    x = mini.Tensor([1.0, 2.0, 4.0])

    y = x / 2.0
    z = 8.0 / x

    assert y.cpu() == [0.5, 1.0, 2.0]
    assert z.cpu() == [8.0, 4.0, 2.0]


def test_scalar_ops_preserve_shape():
    x = mini.Tensor([[1.0, 2.0], [3.0, 4.0]])

    assert (x * 2.0).cpu() == [2.0, 4.0, 6.0, 8.0]
    assert (x + 2.0).cpu() == [3.0, 4.0, 5.0, 6.0]
    assert (x - 2.0).cpu() == [-1.0, 0.0, 1.0, 2.0]
    assert (x / 2.0).cpu() == [0.5, 1.0, 1.5, 2.0]
