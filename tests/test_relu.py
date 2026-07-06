import minipypy as mini


def test_relu_forward():
    x = mini.Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    y = x.relu()

    assert y.cpu() == [0.0, 0.0, 0.0, 1.0, 2.0]


def test_relu_backward_sum():
    x = mini.Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)

    y = x.relu()
    z = y.sum()
    z.backward()

    assert x.grad().cpu() == [0.0, 0.0, 0.0, 1.0, 1.0]


def test_functional_relu():
    x = mini.Tensor([-1.0, 0.0, 2.0])

    y = mini.nn.functional.relu(x)

    assert y.cpu() == [0.0, 0.0, 2.0]


def test_relu_module():
    x = mini.Tensor([-1.0, 0.0, 2.0])

    relu = mini.nn.ReLU()
    y = relu(x)

    assert y.cpu() == [0.0, 0.0, 2.0]


def test_relu_inside_expression_backward():
    x = mini.Tensor([-1.0, 2.0, 3.0], requires_grad=True)

    y = (x.relu() * x).sum()
    y.backward()

    # y = relu(x) * x
    # for x < 0: y = 0
    # for x > 0: y = x * x, dy/dx = 2x
    assert x.grad().cpu() == [0.0, 4.0, 6.0]
