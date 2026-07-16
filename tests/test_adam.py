import minipypy as mini


def test_adam_optimizer_reduces_mse_loss():
    x = mini.Tensor([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ])

    y = mini.Tensor([
        [2.0],
        [4.0],
        [6.0],
        [8.0],
    ])

    model = mini.nn.Linear(1, 1)
    loss_fn = mini.nn.MSELoss()
    optimizer = mini.optim.Adam(model, lr=0.05)

    initial_loss = loss_fn(model(x), y).cpu()[0]

    for _ in range(150):
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = loss_fn(model(x), y).cpu()[0]

    assert final_loss < initial_loss


def test_adam_optimizer_reduces_cross_entropy_loss():
    x = mini.Tensor([
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [0.0, 2.0],
    ])

    y = mini.Tensor([0.0, 0.0, 1.0, 1.0])

    model = mini.nn.Linear(2, 2)
    loss_fn = mini.nn.CrossEntropyLoss()
    optimizer = mini.optim.Adam(model, lr=0.05)

    initial_loss = loss_fn(model(x), y).cpu()[0]

    for _ in range(100):
        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = loss_fn(model(x), y).cpu()[0]

    assert final_loss < initial_loss


def test_adam_optimizer_reduces_bce_with_logits_loss():
    x = mini.Tensor([
        [1.0],
        [2.0],
        [-1.0],
        [-2.0],
    ])

    y = mini.Tensor([
        [1.0],
        [1.0],
        [0.0],
        [0.0],
    ])

    model = mini.nn.Linear(1, 1)
    loss_fn = mini.nn.BCEWithLogitsLoss()
    optimizer = mini.optim.Adam(model, lr=0.05)

    initial_loss = loss_fn(model(x), y).cpu()[0]

    for _ in range(100):
        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = loss_fn(model(x), y).cpu()[0]

    assert final_loss < initial_loss
