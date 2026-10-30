import minipypy as mp


def test_clone_preserves_values():
    x = mp.Tensor([1.0, 2.0, 3.0, 4.0])

    y = x.clone()

    assert y.cpu() == x.cpu()


def test_clone_preserves_shape():
    x = mp.Tensor(
        [1.0, 2.0, 3.0, 4.0],
        [2, 2]
    )

    y = x.clone()

    assert y.shape() == [2, 2]
    assert y.numel() == 4


def test_clone_is_independent_tensor():
    x = mp.Tensor([1.0, 2.0, 3.0])

    y = x.clone()

    # Creating another tensor operation from y should not
    # modify the original tensor.
    z = y + 10.0

    assert x.cpu() == [1.0, 2.0, 3.0]
    assert y.cpu() == [1.0, 2.0, 3.0]
    assert z.cpu() == [11.0, 12.0, 13.0]
