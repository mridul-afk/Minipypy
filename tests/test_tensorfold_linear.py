import minipypy as mini


def test_tensorfold_linear_forward_shape():
    layer = mini.nn.TensorFoldLinear(4, 3, rank=2)

    x = mini.Tensor([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
    ])

    out = layer(x)

    assert out.shape() == [2, 3]


def test_tensorfold_linear_parameters():
    layer = mini.nn.TensorFoldLinear(4, 3, rank=2)

    params = layer.parameters()

    assert len(params) == 3

    assert params[0].shape() == [4, 2]
    assert params[1].shape() == [2, 3]
    assert params[2].shape() == [1, 3]


def test_tensorfold_linear_named_parameters():
    layer = mini.nn.TensorFoldLinear(4, 3, rank=2)

    named = layer.named_parameters()

    assert len(named) == 3

    assert named[0][1] == "U"
    assert named[1][1] == "V"
    assert named[2][1] == "b"


def test_tensorfold_linear_parameter_count():
    layer = mini.nn.TensorFoldLinear(784, 10, rank=4)

    assert layer.dense_parameter_count() == 7850
    assert layer.parameter_count() == 3186

    compression = layer.compression_ratio()

    assert compression > 2.0


def test_tensorfold_linear_training_reduces_cross_entropy_loss():
    x = mini.Tensor([
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
    ])

    y = mini.Tensor([0.0, 0.0, 1.0, 1.0])

    model = mini.nn.TensorFoldLinear(4, 2, rank=2)

    loss_fn = mini.nn.CrossEntropyLoss()
    optimizer = mini.optim.Adam(model, lr=0.01)

    initial_loss = loss_fn(model(x), y).cpu()[0]

    for _ in range(100):
        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = loss_fn(model(x), y).cpu()[0]

    assert final_loss < initial_loss
