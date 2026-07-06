import minipypy as mini


print("Training using MiniPyPy nn API...")

x = mini.Tensor([[1.0], [2.0], [3.0], [4.0]])
y = mini.Tensor([[2.0], [4.0], [6.0], [8.0]])

model = mini.nn.Linear(1, 1)
loss_fn = mini.nn.MSELoss()

lr = 0.01

for epoch in range(501):
    pred = model(x)
    loss = loss_fn(pred, y)

    model.zero_grad()
    loss.backward()
    model.step(lr)

    if epoch % 100 == 0:
        print("epoch:", epoch)
        print("loss:", loss)
        print("W:", model.W)
        print("b:", model.b)
        print()

print("Training finished.")
print("Final W:", model.W)
print("Final b:", model.b)
