import minipypy as mini


def test_sgd_linear_training_reduces_loss():
    x = mini.Tensor([[1.0], [2.0], [3.0], [4.0]])
    y = mini.Tensor([[2.0], [4.0], [6.0], [8.0]])

    model = mini.nn.Linear(1, 1)
    loss_fn = mini.nn.MSELoss()
    optimizer = mini.optim.SGD(model, lr=0.01)

    initial_loss = loss_fn(model(x), y).cpu()[0]

    for _ in range(101):
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = loss_fn(model(x), y).cpu()[0]

    assert final_loss < initial_loss
    assert final_loss < 1.0


def test_sgd_sequential_training_reduces_loss():
    x = mini.Tensor([[1.0], [2.0], [3.0], [4.0]])
    y = mini.Tensor([[2.0], [4.0], [6.0], [8.0]])

    model = mini.nn.Sequential(
        mini.nn.Linear(1, 4),
        mini.nn.ReLU(),
        mini.nn.Linear(4, 1),
    )

    loss_fn = mini.nn.MSELoss()
    optimizer = mini.optim.SGD(model, lr=0.01)

    initial_loss = loss_fn(model(x), y).cpu()[0]

    for _ in range(101):
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = loss_fn(model(x), y).cpu()[0]

    assert final_loss < initial_loss
