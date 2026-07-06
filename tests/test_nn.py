import minipypy as mini


def test_linear_parameters():
    layer = mini.nn.Linear(1, 1)

    params = layer.parameters()

    assert len(params) == 2
    assert params[0] is layer.W
    assert params[1] is layer.b


def test_linear_forward_single_output():
    x = mini.Tensor([[1.0], [2.0], [3.0]])

    layer = mini.nn.Linear(1, 1)

    # force known weights
    layer.W = mini.Tensor([[2.0]], requires_grad=True)
    layer.b = mini.Tensor([[1.0]], requires_grad=True)

    out = layer(x)

    assert out.cpu() == [3.0, 5.0, 7.0]


def test_mse_loss_functional():
    pred = mini.Tensor([[1.0], [2.0], [3.0]])
    target = mini.Tensor([[1.0], [4.0], [5.0]])

    loss = mini.nn.functional.mse_loss(pred, target)

    # errors: 0, -2, -2
    # squared: 0, 4, 4
    # mean: 8 / 3
    assert abs(loss.cpu()[0] - (8.0 / 3.0)) < 1e-5


def test_mse_loss_module():
    pred = mini.Tensor([[1.0], [2.0], [3.0]])
    target = mini.Tensor([[1.0], [4.0], [5.0]])

    loss_fn = mini.nn.MSELoss()
    loss = loss_fn(pred, target)

    assert abs(loss.cpu()[0] - (8.0 / 3.0)) < 1e-5


def test_linear_training_reduces_loss():
    x = mini.Tensor([[1.0], [2.0], [3.0], [4.0]])
    y = mini.Tensor([[2.0], [4.0], [6.0], [8.0]])

    model = mini.nn.Linear(1, 1)
    loss_fn = mini.nn.MSELoss()

    lr = 0.01

    pred = model(x)
    initial_loss = loss_fn(pred, y).cpu()[0]

    for _ in range(101):
        pred = model(x)
        loss = loss_fn(pred, y)

        model.zero_grad()
        loss.backward()
        model.step(lr)

    pred = model(x)
    final_loss = loss_fn(pred, y).cpu()[0]

    assert final_loss < initial_loss
    assert final_loss < 1.0
