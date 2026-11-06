
import numpy as np
import torch
import minipypy as mini


def make_tensor(data, requires_grad=False):
    return mini.Tensor(
        data.flatten().tolist(),
        list(data.shape),
        requires_grad=requires_grad,
    )


def compare(name, actual, expected, shape, rtol=1e-4, atol=1e-4):
    actual = np.asarray(actual, dtype=np.float32).reshape(shape)
    expected = expected.detach().cpu().numpy()

    np.testing.assert_allclose(
        actual, expected, rtol=rtol, atol=atol
    )

    print(f"PASS: {name}")


def test_reshape_permute_matmul():
    rng = np.random.default_rng(42)
    data = rng.uniform(-1, 1, (2, 3, 4)).astype(np.float32)
    weights = rng.uniform(-1, 1, (2, 3, 5)).astype(np.float32)

    x = make_tensor(data)
    w = make_tensor(weights)

    y = x.reshape([2, 3, 4]).permute([0, 2, 1])
    z = y.matmul(w)

    tx = torch.tensor(data)
    tw = torch.tensor(weights)

    tz = tx.reshape(2, 3, 4).permute(0, 2, 1).matmul(tw)

    assert z.shape() == list(tz.shape)
    compare("reshape -> permute -> matmul", z.cpu(), tz, tz.shape)


def test_broadcast_reduction():
    rng = np.random.default_rng(42)
    a = rng.uniform(-1, 1, (2, 1, 4)).astype(np.float32)
    b = rng.uniform(-1, 1, (1, 3, 4)).astype(np.float32)

    x = make_tensor(a)
    y = make_tensor(b)

    z = (x + y).sum(axis=[2]).mean(axis=[1])

    tx = torch.tensor(a)
    ty = torch.tensor(b)

    tz = (tx + ty).sum(dim=2).mean(dim=1)

    assert z.shape() == list(tz.shape)
    compare("broadcast -> sum -> mean", z.cpu(), tz, tz.shape)


def test_matmul_mean_backward():
    rng = np.random.default_rng(42)
    a = rng.uniform(-1, 1, (2, 3, 4)).astype(np.float32)
    b = rng.uniform(-1, 1, (2, 4, 5)).astype(np.float32)

    x = make_tensor(a, requires_grad=True)
    w = make_tensor(b, requires_grad=True)

    loss = x.matmul(w).mean(axis=[0, 2]).sum()
    loss.backward()

    tx = torch.tensor(a, requires_grad=True)
    tw = torch.tensor(b, requires_grad=True)

    tloss = tx.matmul(tw).mean(dim=(0, 2)).sum()
    tloss.backward()

    compare("matmul -> mean: grad x", x.grad().cpu(), tx.grad, a.shape)
    compare("matmul -> mean: grad w", w.grad().cpu(), tw.grad, b.shape)


def test_permute_reduction_backward():
    rng = np.random.default_rng(42)
    a = rng.uniform(-1, 1, (2, 3, 4)).astype(np.float32)
    weights = rng.uniform(-1, 1, (4, 2)).astype(np.float32)

    x = make_tensor(a, requires_grad=True)
    w = make_tensor(weights)

    y = x.permute([2, 0, 1])
    z = y.sum(axis=[2])
    loss = (z * w).sum()
    loss.backward()

    tx = torch.tensor(a, requires_grad=True)
    tw = torch.tensor(weights)

    ty = tx.permute(2, 0, 1)
    tz = ty.sum(dim=2)
    tloss = (tz * tw).sum()
    tloss.backward()

    compare(
        "permute -> reduction: backward",
        x.grad().cpu(),
        tx.grad,
        a.shape,
    )


def test_end_to_end_graph():
    rng = np.random.default_rng(42)

    a = rng.uniform(-1, 1, (2, 3, 4)).astype(np.float32)
    b = rng.uniform(-1, 1, (2, 4, 5)).astype(np.float32)
    bias = rng.uniform(-1, 1, (1, 1, 5)).astype(np.float32)

    x = make_tensor(a, requires_grad=True)
    w = make_tensor(b, requires_grad=True)
    c = make_tensor(bias, requires_grad=True)

    y = x.matmul(w)
    y = (y + c).relu()
    y = y.permute([0, 2, 1])
    y = y.mean(axis=[2], keepdims=True)
    loss = y.sum()
    loss.backward()

    tx = torch.tensor(a, requires_grad=True)
    tw = torch.tensor(b, requires_grad=True)
    tc = torch.tensor(bias, requires_grad=True)

    ty = tx.matmul(tw)
    ty = (ty + tc).relu()
    ty = ty.permute(0, 2, 1)
    ty = ty.mean(dim=2, keepdim=True)
    tloss = ty.sum()
    tloss.backward()

    compare("end-to-end: forward", y.cpu(), ty, ty.shape)
    compare("end-to-end: grad x", x.grad().cpu(), tx.grad, a.shape)
    compare("end-to-end: grad w", w.grad().cpu(), tw.grad, b.shape)
    compare("end-to-end: grad bias", c.grad().cpu(), tc.grad, bias.shape)


if __name__ == "__main__":
    tests = [
        test_reshape_permute_matmul,
        test_broadcast_reduction,
        test_matmul_mean_backward,
        test_permute_reduction_backward,
        test_end_to_end_graph,
    ]

    for test in tests:
        print(f"\nRunning {test.__name__}")
        test()

    print("\nAll N-D integration checks passed.")
