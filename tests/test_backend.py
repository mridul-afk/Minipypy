import minipypy as mini

x = mini.Tensor([2.0], requires_grad=True)

print(x)
print(x.grad())

x.zero_grad()
print(x.grad())
