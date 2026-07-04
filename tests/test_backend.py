import minipypy as mini

x = mini.Tensor([1.0, 2.0, 3.0], requires_grad=True)
z = (x * x).sum()
z.backward()
print(x.grad())  # [2, 4, 6]

x = mini.Tensor([2.0], requires_grad=True)
z = (x * x).sum()
z.backward()
print(x.grad())  # [4]
