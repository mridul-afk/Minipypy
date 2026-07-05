import minipypy as mini


print("Starting MiniPyPy linear regression training with bias...")

x = mini.Tensor([[1.0], [2.0], [3.0], [4.0]])
y = mini.Tensor([[2.0], [4.0], [6.0], [8.0]])

W = mini.Tensor([[0.1]], requires_grad=True)
b = mini.Tensor([[0.0]], requires_grad=True)

lr = 0.01

for epoch in range(501):
    pred = x @ W + b

    error = pred - y
    loss = (error * error).mean()

    loss.backward()

    if epoch % 100 == 0:
        print("epoch:", epoch)
        print("loss:", loss)
        print("W:", W)
        print("W grad:", W.grad())
        print("b:", b)
        print("b grad:", b.grad())
        print()

    W = (W - W.grad() * lr).detach().requires_grad_(True)
    b = (b - b.grad() * lr).detach().requires_grad_(True)

print("Training finished.")
print("Final W:", W)
print("Final b:", b)
