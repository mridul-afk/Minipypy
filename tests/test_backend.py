import minipypy as mini
a = mini.Tensor(
    [1,2,
     3,4],
    [2,2]
)

print(a)

b = mini.Tensor(
    list(range(24)),
    [2,3,4]
)

print(b)

c = mini.Tensor(
    list(range(48)),
    [2,2,3,4]
)

print(c)
