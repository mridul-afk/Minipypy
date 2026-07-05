import minipypy as mini


def test_detach_removes_grad_history():
    x = mini.Tensor([1.0, 2.0, 3.0], requires_grad=True)

    y = (x * x).detach()

    assert y.cpu() == [1.0, 4.0, 9.0]
    assert y.grad().cpu() == [0.0, 0.0, 0.0]


def test_requires_grad_inplace_enables_backward():
    x = mini.Tensor([1.0, 2.0, 3.0])

    x.requires_grad_(True)

    z = (x * x).sum()
    z.backward()

    assert x.grad().cpu() == [2.0, 4.0, 6.0]


def test_training_style_parameter_update():
    x = mini.Tensor([1.0, 2.0, 3.0], requires_grad=True)

    loss = (x * x).sum()
    loss.backward()

    lr = 0.1
    x = (x - x.grad() * lr).detach().requires_grad_(True)

    values = x.cpu()

    assert abs(values[0] - 0.8) < 1e-5
    assert abs(values[1] - 1.6) < 1e-5
    assert abs(values[2] - 2.4) < 1e-5


def test_linear_regression_learns_weight_no_bias():
    x = mini.Tensor([[1.0], [2.0], [3.0], [4.0]])
    y = mini.Tensor([[2.0], [4.0], [6.0], [8.0]])

    W = mini.Tensor([[0.1]], requires_grad=True)

    lr = 0.01

    initial_loss_value = None

    for epoch in range(101):
        pred = x @ W
        error = pred - y
        loss = (error * error).mean()

        if epoch == 0:
            initial_loss_value = loss.cpu()[0]

        loss.backward()

        W = (W - W.grad() * lr).detach().requires_grad_(True)

    final_pred = x @ W
    final_error = final_pred - y
    final_loss = (final_error * final_error).mean()

    final_loss_value = final_loss.cpu()[0]
    final_w = W.cpu()[0]

    assert final_loss_value < initial_loss_value
    assert abs(final_w - 2.0) < 0.01


def test_linear_regression_with_bias_loss_decreases():
    x = mini.Tensor([[1.0], [2.0], [3.0], [4.0]])
    y = mini.Tensor([[2.0], [4.0], [6.0], [8.0]])

    W = mini.Tensor([[0.1]], requires_grad=True)
    b = mini.Tensor([[0.0]], requires_grad=True)

    lr = 0.01

    initial_loss_value = None

    for epoch in range(101):
        pred = x @ W + b
        error = pred - y
        loss = (error * error).mean()

        if epoch == 0:
            initial_loss_value = loss.cpu()[0]

        loss.backward()

        W = (W - W.grad() * lr).detach().requires_grad_(True)
        b = (b - b.grad() * lr).detach().requires_grad_(True)

    final_pred = x @ W + b
    final_error = final_pred - y
    final_loss = (final_error * final_error).mean()

    final_loss_value = final_loss.cpu()[0]

    assert final_loss_value < initial_loss_value
    assert final_loss_value < 1.0
