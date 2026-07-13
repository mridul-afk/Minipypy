import math
import minipypy as mini


def assert_close(a, b, tol=1e-5):
    assert abs(a - b) < tol


def assert_close_list(actual, expected, tol=1e-5):
    assert len(actual) == len(expected)

    for a, e in zip(actual, expected):
        assert abs(a - e) < tol


def stable_cross_entropy_single(logits, label):
    max_logit = max(logits)
    sum_exp = sum(math.exp(x - max_logit) for x in logits)

    return -logits[label] + max_logit + math.log(sum_exp)


def softmax(logits):
    max_logit = max(logits)
    exps = [math.exp(x - max_logit) for x in logits]
    total = sum(exps)
    return [x / total for x in exps]


def test_cross_entropy_forward_single_sample():
    logits = mini.Tensor([[1.0, 2.0, 3.0]])
    target = mini.Tensor([2.0])

    loss = mini.nn.functional.cross_entropy(logits, target)

    expected = stable_cross_entropy_single([1.0, 2.0, 3.0], 2)

    assert_close(loss.cpu()[0], expected)


def test_cross_entropy_forward_batch_mean():
    logits = mini.Tensor([
        [1.0, 2.0, 3.0],
        [3.0, 1.0, 2.0],
    ])

    target = mini.Tensor([2.0, 0.0])

    loss = mini.nn.functional.cross_entropy(logits, target)

    loss_0 = stable_cross_entropy_single([1.0, 2.0, 3.0], 2)
    loss_1 = stable_cross_entropy_single([3.0, 1.0, 2.0], 0)

    expected = (loss_0 + loss_1) / 2.0

    assert_close(loss.cpu()[0], expected)


def test_cross_entropy_module_forward():
    logits = mini.Tensor([[1.0, 2.0, 3.0]])
    target = mini.Tensor([2.0])

    loss_fn = mini.nn.CrossEntropyLoss()
    loss = loss_fn(logits, target)

    expected = stable_cross_entropy_single([1.0, 2.0, 3.0], 2)

    assert_close(loss.cpu()[0], expected)


def test_cross_entropy_backward_single_sample():
    logits = mini.Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    target = mini.Tensor([2.0])

    loss = mini.nn.functional.cross_entropy(logits, target)
    loss.backward()

    grad = logits.grad().cpu()

    probs = softmax([1.0, 2.0, 3.0])

    expected = [
        probs[0],
        probs[1],
        probs[2] - 1.0,
    ]

    assert_close_list(grad, expected, tol=1e-4)


def test_cross_entropy_backward_batch_mean():
    logits = mini.Tensor([
        [1.0, 2.0, 3.0],
        [3.0, 1.0, 2.0],
    ], requires_grad=True)

    target = mini.Tensor([2.0, 0.0])

    loss = mini.nn.functional.cross_entropy(logits, target)
    loss.backward()

    grad = logits.grad().cpu()

    probs_0 = softmax([1.0, 2.0, 3.0])
    probs_1 = softmax([3.0, 1.0, 2.0])

    expected = [
        probs_0[0] / 2.0,
        probs_0[1] / 2.0,
        (probs_0[2] - 1.0) / 2.0,

        (probs_1[0] - 1.0) / 2.0,
        probs_1[1] / 2.0,
        probs_1[2] / 2.0,
    ]

    assert_close_list(grad, expected, tol=1e-4)


def test_cross_entropy_training_reduces_loss():
    x = mini.Tensor([
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [0.0, 2.0],
    ])

    # class 0 for x-axis samples
    # class 1 for y-axis samples
    y = mini.Tensor([0.0, 0.0, 1.0, 1.0])

    model = mini.nn.Linear(2, 2)
    loss_fn = mini.nn.CrossEntropyLoss()
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


def test_cross_entropy_accepts_target_shape_batch_1():
    logits = mini.Tensor([[1.0, 2.0, 3.0]])
    target = mini.Tensor([[2.0]])

    loss = mini.nn.functional.cross_entropy(logits, target)

    expected = stable_cross_entropy_single([1.0, 2.0, 3.0], 2)

    assert_close(loss.cpu()[0], expected)
