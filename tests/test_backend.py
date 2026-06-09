## Matmul Test


import minipypy as mini

a = mini.Tensor(
    [1, 2,
     3, 4],
    [2, 2]
)

b = mini.Tensor(
    [5, 6,
     7, 8,
     9, 10],
    [2, 3]
)

c = a.matmul(b)

d = a @ b

print(d.cpu())

print(c.cpu())
