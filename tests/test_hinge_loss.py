import minipypy as mini


def test_hinge_loss_functional_forward():
    pred = mini.Tensor([[2.0], [-1.0], [0.5]])
    target = mini.Tensor([[1.0], [-1.0], [1.0]])

    loss = mini.nn.functional.hinge_loss(pred, target)

    # sample 1: relu(1 - 1*2)    = relu(-1)  = 0
    # sample 2: relu(1 - -1*-1)  = relu(0)   = 0
    # sample 3: relu(1 - 1*0.5)  = relu(0.5) = 0.5
    # mean = 0.5 / 3
    assert abs(loss.cpu()[0] - (0.5 / 3.0)) < 1e-5


def test_hinge_loss_module_forward():
    pred = mini.Tensor([[2.0], [-1.0], [0.5]])
    target = mini.Tensor([[1.0], [-1.0], [1.0]])

    loss_fn = mini.nn.HingeLoss()
    loss = loss_fn(pred, target)

    assert abs(loss.cpu()[0] - (0.5 / 3.0)) < 1e-5


def test_hinge_loss_backward():
    pred = mini.Tensor([[2.0], [-1.0], [0.5]], requires_grad=True)
    target = mini.Tensor([[1.0], [-1.0], [1.0]])

    loss = mini.nn.functional.hinge_loss(pred, target)
    loss.backward()

    grad = pred.grad().cpu()

    # loss = mean(relu(1 - y * pred))
    #
    # if margin <= 0:
    #   grad = 0
    #
    # if margin > 0:
    #   grad = -target / N
    #
    # sample 1 margin = -1  -> grad 0
    # sample 2 margin = 0   -> grad 0 because ReLU grad at 0 is 0
    # sample 3 margin = 0.5 -> grad -1/3
    assert abs(grad[0] - 0.0) < 1e-5
    assert abs(grad[1] - 0.0) < 1e-5
    assert abs(grad[2] - (-1.0 / 3.0)) < 1e-5


def test_hinge_loss_training_reduces_loss():
    x = mini.Tensor([[1.0], [2.0], [-1.0], [-2.0]])
    y = mini.Tensor([[1.0], [1.0], [-1.0], [-1.0]])

    model = mini.nn.Linear(1, 1)
    loss_fn = mini.nn.HingeLoss()
    optimizer = mini.optim.SGD(model, lr=0.05)

    initial_loss = loss_fn(model(x), y).cpu()[0]

    for _ in range(100):
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = loss_fn(model(x), y).cpu()[0]

    assert final_loss < initial_loss
