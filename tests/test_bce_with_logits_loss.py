import math
import minipypy as mini


def assert_close(a, b, tol=1e-5):
    assert abs(a - b) < tol


def assert_close_list(actual, expected, tol=1e-5):
    assert len(actual) == len(expected)

    for a, e in zip(actual, expected):
        assert abs(a - e) < tol


def stable_bce_with_logits_single(x, y):
    return max(x, 0.0) - x * y + math.log(1.0 + math.exp(-abs(x)))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def test_bce_with_logits_forward_target_one():
    logits = mini.Tensor([2.0])
    target = mini.Tensor([1.0])

    loss = mini.nn.functional.binary_cross_entropy_with_logits(logits, target)

    expected = stable_bce_with_logits_single(2.0, 1.0)

    assert_close(loss.cpu()[0], expected)


def test_bce_with_logits_forward_target_zero():
    logits = mini.Tensor([2.0])
    target = mini.Tensor([0.0])

    loss = mini.nn.functional.binary_cross_entropy_with_logits(logits, target)

    expected = stable_bce_with_logits_single(2.0, 0.0)

    assert_close(loss.cpu()[0], expected)


def test_bce_with_logits_forward_vector_mean():
    logits = mini.Tensor([2.0, -1.0, 0.0])
    target = mini.Tensor([1.0, 0.0, 1.0])

    loss = mini.nn.functional.binary_cross_entropy_with_logits(logits, target)

    expected = (
        stable_bce_with_logits_single(2.0, 1.0)
        + stable_bce_with_logits_single(-1.0, 0.0)
        + stable_bce_with_logits_single(0.0, 1.0)
    ) / 3.0

    assert_close(loss.cpu()[0], expected)


def test_bce_with_logits_module_forward():
    logits = mini.Tensor([2.0, -1.0, 0.0])
    target = mini.Tensor([1.0, 0.0, 1.0])

    loss_fn = mini.nn.BCEWithLogitsLoss()
    loss = loss_fn(logits, target)

    expected = (
        stable_bce_with_logits_single(2.0, 1.0)
        + stable_bce_with_logits_single(-1.0, 0.0)
        + stable_bce_with_logits_single(0.0, 1.0)
    ) / 3.0

    assert_close(loss.cpu()[0], expected)


def test_bce_with_logits_backward_vector():
    logits = mini.Tensor([2.0, -1.0, 0.0], requires_grad=True)
    target = mini.Tensor([1.0, 0.0, 1.0])

    loss = mini.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()

    grad = logits.grad().cpu()

    expected = [
        (sigmoid(2.0) - 1.0) / 3.0,
        (sigmoid(-1.0) - 0.0) / 3.0,
        (sigmoid(0.0) - 1.0) / 3.0,
    ]

    assert_close_list(grad, expected, tol=1e-4)


def test_bce_with_logits_backward_2d():
    logits = mini.Tensor([
        [2.0, -1.0],
        [0.0, 3.0],
    ], requires_grad=True)

    target = mini.Tensor([
        [1.0, 0.0],
        [1.0, 1.0],
    ])

    loss = mini.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()

    grad = logits.grad().cpu()

    expected = [
        (sigmoid(2.0) - 1.0) / 4.0,
        (sigmoid(-1.0) - 0.0) / 4.0,
        (sigmoid(0.0) - 1.0) / 4.0,
        (sigmoid(3.0) - 1.0) / 4.0,
    ]

    assert_close_list(grad, expected, tol=1e-4)


def test_bce_with_logits_training_reduces_loss():
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
    optimizer = mini.optim.SGD(model, lr=0.1)

    initial_loss = loss_fn(model(x), y).cpu()[0]

    for _ in range(100):
        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = loss_fn(model(x), y).cpu()[0]

    assert final_loss < initial_loss
