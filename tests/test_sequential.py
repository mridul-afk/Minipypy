import minipypy as mini


def test_sequential_forward():
    model = mini.nn.Sequential(
        mini.nn.Linear(1, 2),
        mini.nn.ReLU(),
        mini.nn.Linear(2, 1),
    )

    x = mini.Tensor([[1.0], [2.0], [3.0]])

    out = model(x)

    assert len(out.cpu()) == 3


def test_sequential_len_and_getitem():
    model = mini.nn.Sequential(
        mini.nn.Linear(1, 2),
        mini.nn.ReLU(),
        mini.nn.Linear(2, 1),
    )

    assert len(model) == 3
    assert isinstance(model[0], mini.nn.Linear)
    assert isinstance(model[1], mini.nn.ReLU)
    assert isinstance(model[2], mini.nn.Linear)


def test_sequential_parameters():
    model = mini.nn.Sequential(
        mini.nn.Linear(1, 2),
        mini.nn.ReLU(),
        mini.nn.Linear(2, 1),
    )

    params = model.parameters()

    # first Linear has W,b
    # ReLU has no params
    # second Linear has W,b
    assert len(params) == 4


def test_sequential_training_reduces_loss():
    x = mini.Tensor([[1.0], [2.0], [3.0], [4.0]])
    y = mini.Tensor([[2.0], [4.0], [6.0], [8.0]])

    model = mini.nn.Sequential(
        mini.nn.Linear(1, 4),
        mini.nn.ReLU(),
        mini.nn.Linear(4, 1),
    )

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
