import minipypy as mini

a = mini.Tensor([1.0, 2.0, 3.0, 4.0])
b = mini.Tensor([10.0, 20.0, 30.0, 40.0])

c = a + b
d = a * b
e = b - a
f = b / a

print(c)
print(d)
print(e)
print(f.cpu())
