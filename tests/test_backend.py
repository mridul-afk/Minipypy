import minipypy as mini

A = mini.Tensor([
    [[1, 2, 3],
     [4, 5, 6]],

    [[7, 8, 9],
     [10, 11, 12]]
])

B = mini.Tensor([
    [[1, 2],
     [3, 4],
     [5, 6]],

    [[7, 8],
     [9, 10],
     [11, 12]]
])

print(A @ B)
print((A @ B).shape())
