import minipypy as mini


def test_add_backward():
    x = mini.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    z = (x + x).sum()
    z.backward()
    assert x.grad().cpu() == [2.0, 2.0, 2.0]


def test_sub_backward():
    x = mini.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    z = (x - x).sum()
    z.backward()
    assert x.grad().cpu() == [0.0, 0.0, 0.0]


def test_mul_backward():
    x = mini.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    z = (x * x).sum()
    z.backward()
    assert x.grad().cpu() == [2.0, 4.0, 6.0]


def test_div_backward():
    x = mini.Tensor([2.0, 4.0, 8.0], requires_grad=True)
    y = mini.Tensor([2.0, 2.0, 2.0], requires_grad=True)

    z = (x / y).sum()
    z.backward()

    assert x.grad().cpu() == [0.5, 0.5, 0.5]
    assert y.grad().cpu() == [-0.5, -1.0, -2.0]


def test_sum_backward():
    x = mini.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    z = x.sum()
    z.backward()
    assert x.grad().cpu() == [1.0, 1.0, 1.0]


def test_mean_backward():
    x = mini.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    z = x.mean()
    z.backward()

    grad = x.grad().cpu()
    assert abs(grad[0] - 1.0 / 3.0) < 1e-5
    assert abs(grad[1] - 1.0 / 3.0) < 1e-5
    assert abs(grad[2] - 1.0 / 3.0) < 1e-5


def test_matmul_2d_backward():
    A = mini.Tensor([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0]], requires_grad=True)

    B = mini.Tensor([[7.0, 8.0],
                     [9.0, 10.0],
                     [11.0, 12.0]], requires_grad=True)

    z = (A @ B).sum()
    z.backward()

    assert A.grad().cpu() == [15.0, 19.0, 23.0,
                              15.0, 19.0, 23.0]

    assert B.grad().cpu() == [5.0, 5.0,
                              7.0, 7.0,
                              9.0, 9.0]


def test_broadcasted_matmul_backward_A_broadcast():
    A = mini.Tensor([[[1.0, 2.0],
                      [3.0, 4.0]]], requires_grad=True)

    B = mini.Tensor([[[5.0, 6.0],
                      [7.0, 8.0]],
                     [[50.0, 60.0],
                      [70.0, 80.0]]], requires_grad=True)

    z = (A @ B).sum()
    z.backward()

    assert A.grad().cpu() == [121.0, 165.0,
                              121.0, 165.0]

    assert B.grad().cpu() == [4.0, 4.0,
                              6.0, 6.0,
                              4.0, 4.0,
                              6.0, 6.0]


def test_broadcasted_matmul_backward_B_broadcast():
    A = mini.Tensor([[[1.0, 2.0],
                      [3.0, 4.0]],
                     [[10.0, 20.0],
                      [30.0, 40.0]]], requires_grad=True)

    B = mini.Tensor([[[5.0, 6.0],
                      [7.0, 8.0]]], requires_grad=True)

    z = (A @ B).sum()
    z.backward()

    assert A.grad().cpu() == [11.0, 15.0,
                              11.0, 15.0,
                              11.0, 15.0,
                              11.0, 15.0]

    assert B.grad().cpu() == [44.0, 44.0,
                              66.0, 66.0]

def test_add_broadcast_backward():
    x = mini.Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    y = mini.Tensor([[10.0], [20.0]], requires_grad=True)

    z = (x + y).sum()
    z.backward()

    assert x.grad().cpu() == [2.0, 2.0, 2.0]
    assert y.grad().cpu() == [3.0, 3.0]


def test_sub_broadcast_backward():
    x = mini.Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    y = mini.Tensor([[10.0], [20.0]], requires_grad=True)

    z = (x - y).sum()
    z.backward()

    assert x.grad().cpu() == [2.0, 2.0, 2.0]
    assert y.grad().cpu() == [-3.0, -3.0]


def test_mul_broadcast_backward():
    x = mini.Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    y = mini.Tensor([[10.0], [20.0]], requires_grad=True)

    z = (x * y).sum()
    z.backward()

    assert x.grad().cpu() == [30.0, 30.0, 30.0]
    assert y.grad().cpu() == [6.0, 6.0]


def test_div_broadcast_backward():
    x = mini.Tensor([[10.0, 20.0, 30.0]], requires_grad=True)
    y = mini.Tensor([[10.0], [20.0]], requires_grad=True)

    z = (x / y).sum()
    z.backward()

    gx = x.grad().cpu()
    gy = y.grad().cpu()

    assert abs(gx[0] - 0.15) < 1e-5
    assert abs(gx[1] - 0.15) < 1e-5
    assert abs(gx[2] - 0.15) < 1e-5

    assert abs(gy[0] - (-0.6)) < 1e-5
    assert abs(gy[1] - (-0.15)) < 1e-5
