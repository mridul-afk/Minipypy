import minipypy as mini
x = mini.Tensor([1.0,2.0,3.0], requires_grad=True)

y = (x*x).mean()

y.backward()

print(x.grad())
