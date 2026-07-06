import math
import minipypy as mini


def assert_close_list(actual, expected, tol=1e-5):
    assert len(actual) == len(expected)

    for a, e in zip(actual, expected):
        assert abs(a - e) < tol


def test_softmax_1d_forward_sum_to_one():
    x = mini.Tensor([1.0, 2.0, 3.0])

    y = x.softmax(dim=0)

    assert abs(sum(y.cpu()) - 1.0) < 1e-5


def test_softmax_2d_dim1_rows_sum_to_one():
    x = mini.Tensor([
        [1.0, 2.0, 3.0],
        [1.0, 3.0, 2.0],
    ])

    y = x.softmax(dim=1)
    values = y.cpu()

    row1 = values[0:3]
    row2 = values[3:6]

    assert abs(sum(row1) - 1.0) < 1e-5
    assert abs(sum(row2) - 1.0) < 1e-5


def test_softmax_2d_dim0_columns_sum_to_one():
    x = mini.Tensor([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
    ])

    y = x.softmax(dim=0)
    values = y.cpu()

    # Shape is [2, 3], flattened row-major:
    # row0: values[0:3]
    # row1: values[3:6]
    #
    # For dim=0, each column should sum to 1.
    col0 = values[0] + values[3]
    col1 = values[1] + values[4]
    col2 = values[2] + values[5]

    assert abs(col0 - 1.0) < 1e-5
    assert abs(col1 - 1.0) < 1e-5
    assert abs(col2 - 1.0) < 1e-5


def test_softmax_3d_last_dim_sums_to_one():
    x = mini.Tensor([
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
        ],
        [
            [1.0, 3.0, 2.0],
            [4.0, 5.0, 6.0],
        ],
    ])

    y = x.softmax(dim=-1)
    values = y.cpu()

    # Shape [2, 2, 3]
    # last dim groups are contiguous chunks of 3.
    for i in range(0, len(values), 3):
        group = values[i:i + 3]
        assert abs(sum(group) - 1.0) < 1e-5


def test_softmax_forward_known_values():
    x = mini.Tensor([[1.0, 2.0, 3.0]])

    y = x.softmax(dim=1)
    values = y.cpu()

    exps = [
        math.exp(1.0 - 3.0),
        math.exp(2.0 - 3.0),
        math.exp(3.0 - 3.0),
    ]

    total = sum(exps)
    expected = [v / total for v in exps]

    assert_close_list(values, expected)


def test_functional_softmax():
    x = mini.Tensor([[1.0, 2.0, 3.0]])

    y = mini.nn.functional.softmax(x, dim=1)

    assert abs(sum(y.cpu()) - 1.0) < 1e-5


def test_softmax_module():
    x = mini.Tensor([[1.0, 2.0, 3.0]])

    softmax = mini.nn.Softmax(dim=1)
    y = softmax(x)

    assert abs(sum(y.cpu()) - 1.0) < 1e-5


def test_softmax_backward_sum_is_zero():
    # For y = softmax(x), sum(y) is always 1.
    # So:
    #   z = sum(softmax(x))
    #   dz/dx = 0
    x = mini.Tensor([[1.0, 2.0, 3.0]], requires_grad=True)

    y = x.softmax(dim=1)
    z = y.sum()
    z.backward()

    grad = x.grad().cpu()

    assert_close_list(grad, [0.0, 0.0, 0.0], tol=1e-4)


def test_softmax_backward_weighted_sum_dim1():
    x = mini.Tensor([[1.0, 2.0, 3.0]], requires_grad=True)

    y = x.softmax(dim=1)

    weights = mini.Tensor([[1.0, 2.0, 3.0]])
    z = (y * weights).sum()
    z.backward()

    grad = x.grad().cpu()
    y_vals = y.cpu()
    weights_vals = weights.cpu()

    dot = sum(y_vals[i] * weights_vals[i] for i in range(3))

    expected = [
        y_vals[i] * (weights_vals[i] - dot)
        for i in range(3)
    ]

    assert_close_list(grad, expected, tol=1e-4)


def test_softmax_backward_weighted_sum_dim0():
    x = mini.Tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ], requires_grad=True)

    y = x.softmax(dim=0)

    weights = mini.Tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    z = (y * weights).sum()
    z.backward()

    grad = x.grad().cpu()
    y_vals = y.cpu()
    w_vals = weights.cpu()

    # For dim=0, each column is a softmax group.
    # Shape [2, 2], flattened:
    # [row0col0, row0col1, row1col0, row1col1]
    #
    # Column 0 indices: 0, 2
    # Column 1 indices: 1, 3

    expected = [0.0, 0.0, 0.0, 0.0]

    for group in ([0, 2], [1, 3]):
        dot = sum(y_vals[i] * w_vals[i] for i in group)

        for i in group:
            expected[i] = y_vals[i] * (w_vals[i] - dot)

    assert_close_list(grad, expected, tol=1e-4)
