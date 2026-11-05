import minipypy as mini


def test_1d_tensor_metadata():
    x = mini.Tensor([1.0, 2.0, 3.0, 4.0])

    assert x.shape() == [4]
    assert x.stride() == [1]
    assert x.ndim() == 1
    assert x.numel() == 4


def test_2d_tensor_metadata():
    x = mini.Tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    assert x.shape() == [2, 3]
    assert x.stride() == [3, 1]
    assert x.ndim() == 2
    assert x.numel() == 6


def test_3d_tensor_metadata():
    x = mini.Tensor([
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        [
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
        ],
    ])

    assert x.shape() == [2, 3, 2]
    assert x.stride() == [6, 2, 1]
    assert x.ndim() == 3
    assert x.numel() == 12


def test_4d_tensor_metadata():
    x = mini.Tensor([[[[1.0, 2.0, 3.0]]]])

    assert x.shape() == [1, 1, 1, 3]
    assert x.stride() == [3, 3, 3, 1]
    assert x.ndim() == 4
    assert x.numel() == 3


def test_5d_tensor_metadata():
    x = mini.Tensor([[[[[1.0, 2.0]]]]])

    assert x.shape() == [1, 1, 1, 1, 2]
    assert x.stride() == [2, 2, 2, 2, 1]
    assert x.ndim() == 5
    assert x.numel() == 2


import pytest
import minipypy as mini


def test_3d_tensor_flattened_values():
    x = mini.Tensor([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
    ])

    assert x.shape() == [2, 2, 2]
    assert x.cpu() == [
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
        7.0, 8.0,
    ]


def test_4d_tensor_flattened_values():
    x = mini.Tensor([
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    ])

    assert x.shape() == [1, 2, 2, 2]
    assert x.cpu() == [
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
        7.0, 8.0,
    ]


def test_ragged_2d_tensor_rejected():
    with pytest.raises(RuntimeError, match="Ragged tensor"):
        mini.Tensor([
            [1.0, 2.0],
            [3.0],
        ])


def test_ragged_3d_tensor_rejected():
    with pytest.raises(RuntimeError, match="Ragged tensor"):
        mini.Tensor([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0]],
        ])

def test_3d_reshape():
    x = mini.Tensor([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
    ])

    y = x.reshape([4, 2])

    assert y.shape() == [4, 2]
    assert y.numel() == 8
    assert y.cpu() == x.cpu()


def test_4d_reshape():
    x = mini.Tensor([
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    ])

    y = x.reshape([2, 4])

    assert y.shape() == [2, 4]
    assert y.numel() == 8
    assert y.cpu() == x.cpu()


def test_nd_reshape_to_1d():
    x = mini.Tensor([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
    ])

    y = x.reshape([8])

    assert y.shape() == [8]
    assert y.cpu() == [
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    ]


def test_1d_reshape_to_3d():
    x = mini.Tensor([
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    ])

    y = x.reshape([2, 2, 2])

    assert y.shape() == [2, 2, 2]
    assert y.stride() == [4, 2, 1]
    assert y.cpu() == x.cpu()


def test_invalid_reshape_rejected():
    x = mini.Tensor([
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0,
    ])

    with pytest.raises(RuntimeError):
        x.reshape([2, 2])

def test_reshape_autograd_3d_to_2d():
    x = mini.Tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        requires_grad=True,
    )

    y = x.reshape([4, 2])
    loss = y.sum()
    loss.backward()

    grad = x.grad()

    assert grad.shape() == [2, 2, 2]
    assert grad.cpu() == [1.0] * 8


def test_reshape_autograd_1d_to_3d():
    x = mini.Tensor(
        [1.0, 2.0, 3.0, 4.0,
         5.0, 6.0, 7.0, 8.0],
        requires_grad=True,
    )

    y = x.reshape([2, 2, 2])
    loss = y.sum()
    loss.backward()

    grad = x.grad()

    assert grad.shape() == [8]
    assert grad.cpu() == [1.0] * 8


def test_reshape_autograd_with_operation():
    x = mini.Tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        requires_grad=True,
    )

    y = x.reshape([4, 2])
    z = y * 2.0
    loss = z.sum()

    loss.backward()

    grad = x.grad()

    assert grad.shape() == [2, 2, 2]
    assert grad.cpu() == [2.0] * 8

def test_reshape_autograd_after_intermediate():
    x = mini.Tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        requires_grad=True,
    )

    y = x * 3.0
    z = y.reshape([4, 2])

    loss = z.sum()
    loss.backward()

    grad = x.grad()

    assert grad.shape() == [2, 2, 2]
    assert grad.cpu() == [3.0] * 8

def test_flatten_autograd_nd():
    x = mini.Tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        requires_grad=True,
    )

    y = x.flatten()

    assert y.shape() == [8]

    loss = y.sum()
    loss.backward()

    grad = x.grad()

    assert grad.shape() == [2, 2, 2]
    assert grad.cpu() == [1.0] * 8

# ============================================================
# Stage 1.3 — N-D Permute
# ============================================================

def test_permute_3d_shape():
    x = mini.Tensor([
        [[1.0, 2.0],
         [3.0, 4.0],
         [5.0, 6.0]],

        [[7.0, 8.0],
         [9.0, 10.0],
         [11.0, 12.0]],
    ])

    # [2, 3, 2] -> [3, 2, 2]
    y = x.permute([1, 0, 2])

    assert y.shape() == [3, 2, 2]
    assert y.numel() == 12


def test_permute_3d_values():
    x = mini.Tensor([
        [[1.0, 2.0],
         [3.0, 4.0],
         [5.0, 6.0]],

        [[7.0, 8.0],
         [9.0, 10.0],
         [11.0, 12.0]],
    ])

    y = x.permute([1, 0, 2])

    assert y.cpu() == [
        1.0, 2.0,
        7.0, 8.0,

        3.0, 4.0,
        9.0, 10.0,

        5.0, 6.0,
        11.0, 12.0,
    ]


def test_permute_4d_shape():
    x = mini.Tensor(
        list(range(1, 25)),
        [2, 3, 2, 2]
    )

    # [2,3,2,2] -> [2,2,2,3]
    y = x.permute([2, 0, 3, 1])

    assert y.shape() == [2, 2, 2, 3]
    assert y.numel() == 24


def test_permute_identity():
    x = mini.Tensor([
        [[1.0, 2.0],
         [3.0, 4.0]]
    ])

    y = x.permute([0, 1, 2])

    assert y.shape() == x.shape()
    assert y.cpu() == x.cpu()

def test_permute_wrong_number_of_dims_rejected():
    x = mini.Tensor([
        [[1.0, 2.0],
         [3.0, 4.0]]
    ])

    with pytest.raises(RuntimeError):
        x.permute([0, 1])


def test_permute_duplicate_dim_rejected():
    x = mini.Tensor([
        [[1.0, 2.0],
         [3.0, 4.0]]
    ])

    with pytest.raises(RuntimeError):
        x.permute([0, 0, 2])


def test_permute_out_of_range_dim_rejected():
    x = mini.Tensor([
        [[1.0, 2.0],
         [3.0, 4.0]]
    ])

    with pytest.raises(RuntimeError):
        x.permute([0, 1, 3])

# ============================================================
# Stage 1.3 — Permute Autograd
# ============================================================

def test_permute_autograd_3d():
    x = mini.Tensor(
        [
            [[1.0, 2.0],
             [3.0, 4.0],
             [5.0, 6.0]],

            [[7.0, 8.0],
             [9.0, 10.0],
             [11.0, 12.0]],
        ],
        requires_grad=True,
    )

    y = x.permute([1, 0, 2])

    loss = y.sum()
    loss.backward()

    grad = x.grad()

    assert grad.shape() == [2, 3, 2]
    assert grad.cpu() == [1.0] * 12


def test_permute_autograd_with_operation():
    x = mini.Tensor(
        [
            [[1.0, 2.0],
             [3.0, 4.0]],

            [[5.0, 6.0],
             [7.0, 8.0]],
        ],
        requires_grad=True,
    )

    y = x.permute([1, 0, 2])
    z = y * 3.0

    loss = z.sum()
    loss.backward()

    grad = x.grad()

    assert grad.shape() == [2, 2, 2]
    assert grad.cpu() == [3.0] * 8


def test_permute_autograd_after_intermediate():
    x = mini.Tensor(
        [
            [[1.0, 2.0],
             [3.0, 4.0]],

            [[5.0, 6.0],
             [7.0, 8.0]],
        ],
        requires_grad=True,
    )

    y = x * 2.0
    z = y.permute([1, 0, 2])

    loss = z.sum()
    loss.backward()

    grad = x.grad()

    assert grad.shape() == [2, 2, 2]
    assert grad.cpu() == [2.0] * 8

def test_permute_autograd_nonuniform_gradient():
    x = mini.Tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [2, 3],
        requires_grad=True,
    )

    weights = mini.Tensor(
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        [3, 2],
    )

    y = x.permute([1, 0])

    loss = (y * weights).sum()
    loss.backward()

    assert x.grad().cpu() == [
        10.0, 30.0, 50.0,
        20.0, 40.0, 60.0,
    ]
